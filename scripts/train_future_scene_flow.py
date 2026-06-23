#!/usr/bin/env python3
"""Train the pretrained-TrackCraft3r future scene-flow predictor.

This is the alternate architecture to ``scripts/train_future_3d_tracks.py``:
instead of a from-scratch model, it reuses the frozen pretrained TrackCraft3r
(Wan2.1 DiT + LoRA + dual VAEs) and only learns future mask latents (+ the I/O
projections / head by default), supervised on the same curated dense clips.

Example (single 24 GB GPU)::

    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    /home/homanga/miniconda3/envs/3dflow/bin/python scripts/train_future_scene_flow.py \
      --root data/something_something \
      --tracks-name anchor_tracks32_curated_dense \
      --manifest sam3_anchor_masks/manifest_curated_dense.json \
      --output-dir data/something_something/future_track_training/pretrained_tc3r_10f_to_32f \
      --obs-frames 10 --total-frames 32 \
      --image-size 224 384 \
      --trainable mask io head \
      --batch-size 1 --grad-accum 8 --grad-checkpoint \
      --epochs 20 --steps-per-epoch 600 --val-steps 32 --amp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models_pretrained  # noqa: E402,F401  (activates vendored-import isolation)
from models_pretrained.future_scene_flow import FutureSceneFlowConfig, FutureSceneFlowModel  # noqa: E402
from models_pretrained.future_scene_flow.dataset import (  # noqa: E402
    FutureSceneFlowDataset,
    build_track_index,
    future_collate,
    split_items,
)
from models_pretrained.future_scene_flow.losses import decoded_metrics, latent_loss  # noqa: E402

DEFAULT_CKPT = REPO_ROOT / "models_pretrained" / "checkpoints" / "trackcraft3r" / "model.safetensors"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/something_something"))
    p.add_argument("--tracks-name", default="anchor_tracks32_curated_dense")
    p.add_argument("--manifest", type=Path, default=Path("sam3_anchor_masks/manifest_curated_dense.json"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--obs-frames", type=int, default=10)
    p.add_argument("--total-frames", type=int, default=32)
    p.add_argument("--image-size", type=int, nargs=2, default=(224, 384), metavar=("H", "W"))
    p.add_argument("--lora-rank", type=int, default=1024)
    p.add_argument("--trainable", nargs="+", default=["mask", "io", "head"],
                   choices=["mask", "io", "head", "lora", "vae"])
    p.add_argument("--lora-scope", default="attn", choices=["all", "attn"],
                   help="When training LoRA: 'attn' = self-attn q,k,v,o only "
                        "(~0.38B, fits 24 GB); 'all' = every LoRA tensor (~1.0B).")
    p.add_argument("--lora-last-n", type=int, default=0,
                   help="Restrict LoRA training to the last N DiT blocks (0=all).")
    p.add_argument("--no-vis", action="store_true", help="Disable the visibility head.")
    p.add_argument("--future-weight", type=float, default=1.0)
    p.add_argument("--obs-weight", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=600)
    p.add_argument("--val-steps", type=int, default=32)
    p.add_argument("--samples-per-clip", type=int, default=2)
    p.add_argument("--limit-clips", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--amp", action="store_true", help="bf16 autocast for the DiT forward.")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--metric-every", type=int, default=100, help="Decode metrics every N steps.")
    return p.parse_args()


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def _save(model, optimizer, path, **extra):
    """Save only the trainable params (small) plus config + base checkpoint ref."""
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(
        {
            "trainable_state": trainable,
            "config": model.config_dict,
            "optimizer": optimizer.state_dict(),
            **extra,
        },
        path,
    )


@torch.no_grad()
def evaluate(model, loader, device, max_steps, obs_frames, amp) -> dict[str, float]:
    model.eval()
    rows = []
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        batch = _to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            query = model(batch["rgb_obs"], batch["pj_obs"])
            xyz, _ = model.split_latent(query)
            target = model.encode_target_delta(batch["target_delta"])
            loss = latent_loss(xyz, target, obs_frames)
        m = decoded_metrics(model, query, batch, obs_frames)
        m["loss"] = float(loss.item())
        rows.append(m)
    if not rows:
        return {"loss": float("nan"), "ade_m": float("nan"), "fde_m": float("nan"),
                "ade_future_m": float("nan"), "fde_future_m": float("nan")}
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for the pretrained TrackCraft3r model.")
    device = torch.device("cuda")

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir or (root / "future_track_training" / time.strftime("%Y%m%d_%H%M%S"))
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = build_track_index(
        root=root, tracks_name=args.tracks_name, manifest=args.manifest,
        require_masks=True, limit_clips=args.limit_clips,
    )
    if len(items) < 2:
        raise SystemExit(f"need >=2 dense clips with masks; found {len(items)}")
    train_items, val_items = split_items(items, args.val_fraction, args.seed)
    print(f"indexed {len(items)} clips: train={len(train_items)} val={len(val_items)}")

    ds_kw = dict(
        obs_frames=args.obs_frames, total_frames=args.total_frames,
        image_size=tuple(args.image_size),
    )
    train_ds = FutureSceneFlowDataset(train_items, samples_per_clip=args.samples_per_clip,
                                      seed=args.seed, **ds_kw)
    val_ds = FutureSceneFlowDataset(val_items, samples_per_clip=1,
                                    seed=args.seed + 100_000, **ds_kw)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, collate_fn=future_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            drop_last=False, collate_fn=future_collate)

    config = FutureSceneFlowConfig(
        checkpoint_path=str(args.checkpoint_path.expanduser().resolve()),
        lora_rank=args.lora_rank,
        height=args.image_size[0], width=args.image_size[1],
        obs_frames=args.obs_frames, total_frames=args.total_frames,
        trainable=tuple(args.trainable),
        lora_scope=args.lora_scope,
        lora_last_n=args.lora_last_n,
        predict_vis=not args.no_vis,
    )
    model = FutureSceneFlowModel(config)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"trainable params: {n_train/1e6:.2f}M  (groups={args.trainable})")

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    run_cfg = vars(args).copy()
    run_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in run_cfg.items()}
    run_cfg["model"] = model.config_dict
    run_cfg["output_dir"] = str(output_dir)
    run_cfg["trainable_params_M"] = n_train / 1e6
    (output_dir / "config.json").write_text(json.dumps(run_cfg, indent=2))

    best = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            if step > args.steps_per_epoch:
                break
            batch = _to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                query = model(batch["rgb_obs"], batch["pj_obs"],
                              use_gradient_checkpointing=args.grad_checkpoint)
                xyz, _ = model.split_latent(query)
                target = model.encode_target_delta(batch["target_delta"])
                loss = latent_loss(xyz, target, args.obs_frames,
                                   future_weight=args.future_weight,
                                   obs_weight=args.obs_weight)
            (loss / args.grad_accum).backward()
            losses.append(float(loss.item()))

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if step == 1 or step % args.metric_every == 0:
                m = decoded_metrics(model, query.detach(), batch, args.obs_frames)
                print(f"epoch {epoch:03d} step {step:04d} loss={loss.item():.4f} "
                      f"ade_fut_m={m['ade_future_m']:.4f} fde_fut_m={m['fde_future_m']:.4f}")

        val = evaluate(model, val_loader, device, args.val_steps, args.obs_frames, args.amp)
        tl = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch {epoch:03d} done train_loss={tl:.4f} val_loss={val['loss']:.4f} "
              f"val_ade_fut_m={val['ade_future_m']:.4f} val_fde_fut_m={val['fde_future_m']:.4f} "
              f"time={time.time()-start:.1f}s")

        _save(model, optimizer, output_dir / "last.pt", epoch=epoch,
              global_step=global_step, val_metrics=val)
        if np.isfinite(val["ade_future_m"]) and val["ade_future_m"] < best:
            best = val["ade_future_m"]
            _save(model, optimizer, output_dir / "best.pt", epoch=epoch,
                  global_step=global_step, val_metrics=val)
        if args.save_every > 0 and epoch % args.save_every == 0:
            _save(model, optimizer, output_dir / f"epoch_{epoch:03d}.pt", epoch=epoch,
                  global_step=global_step, val_metrics=val)

    print(f"wrote checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
