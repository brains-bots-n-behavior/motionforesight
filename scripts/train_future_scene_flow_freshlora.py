#!/usr/bin/env python3
"""Train the fresh-LoRA future scene-flow variant.

Identical pipeline to ``train_future_scene_flow.py`` but trains a small fresh
LoRA adapter (default rank 32, ~30 M params) stacked on the frozen pretrained
model, instead of the checkpoint's existing rank-1024 LoRA. See
``models_pretrained/future_scene_flow/model_fresh_lora.py``.

Example (single 24 GB GPU)::

    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    /home/homanga/miniconda3/envs/3dflow/bin/python scripts/train_future_scene_flow_freshlora.py \
      --root data/something_something \
      --tracks-name anchor_tracks32_curated_dense \
      --manifest sam3_anchor_masks/manifest_curated_dense.json \
      --output-dir data/something_something/future_track_training/pretrained_tc3r_freshlora_10f_to_32f \
      --image-size 160 288 --fresh-lora-rank 32 \
      --batch-size 1 --grad-accum 8 --grad-checkpoint \
      --epochs 20 --steps-per-epoch 600 --val-steps 32 --amp --save-every 5
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import models_pretrained  # noqa: E402,F401
from models_pretrained.future_scene_flow.model_fresh_lora import (  # noqa: E402
    FreshLoRAConfig,
    FreshLoRAFutureSceneFlowModel,
)
from models_pretrained.future_scene_flow.dataset import (  # noqa: E402
    FutureSceneFlowDataset,
    build_track_index,
    future_collate,
    split_items,
)
from models_pretrained.future_scene_flow.losses import (  # noqa: E402
    decoded_loss,
    decoded_metrics,
    latent_loss,
)

# Reuse the loop helpers from the base trainer (DRY).
import train_future_scene_flow as base  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device, max_steps, obs_frames, amp, loss_space, loss_scale):
    """Val loop reporting decoded ADE/FDE plus the chosen-space val loss."""
    model.eval()
    rows = []
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        batch = base._to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            query = model(batch["rgb_obs"], batch["pj_obs"])
            if loss_space == "decoded":
                loss = decoded_loss(model, query, batch, obs_frames,
                                    scale=loss_scale, use_checkpoint=False)
            else:
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
    p.add_argument("--image-size", type=int, nargs=2, default=(160, 288), metavar=("H", "W"))
    # fresh-LoRA knobs
    p.add_argument("--fresh-lora-rank", type=int, default=32)
    p.add_argument("--fresh-lora-alpha", type=int, default=32)
    p.add_argument("--fresh-lora-scope", default="attn", choices=["attn", "all"])
    p.add_argument("--extra-trainable", nargs="*", default=["io", "head"],
                   choices=["io", "head"], help="On top of mask + fresh LoRA.")
    p.add_argument("--no-vis", action="store_true")
    p.add_argument("--loss-space", default="latent", choices=["latent", "decoded"],
                   help="'latent' = MSE on the predicted latent (cheap). "
                        "'decoded' = TrackCraft3r-faithful: VAE-decode then "
                        "coordinate-space MSE x scale (heavier, optimizes ADE).")
    p.add_argument("--loss-scale", type=float, default=10.0,
                   help="Coordinate-space MSE scale for --loss-space decoded (TC3r uses 10).")
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
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--metric-every", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda")

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir or (root / "future_track_training" / time.strftime("%Y%m%d_%H%M%S"))
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = build_track_index(root=root, tracks_name=args.tracks_name, manifest=args.manifest,
                              require_masks=True, limit_clips=args.limit_clips)
    if len(items) < 2:
        raise SystemExit(f"need >=2 dense clips with masks; found {len(items)}")
    train_items, val_items = split_items(items, args.val_fraction, args.seed)
    print(f"indexed {len(items)} clips: train={len(train_items)} val={len(val_items)}")

    ds_kw = dict(obs_frames=args.obs_frames, total_frames=args.total_frames,
                 image_size=tuple(args.image_size))
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

    config = FreshLoRAConfig(
        checkpoint_path=str(args.checkpoint_path.expanduser().resolve()),
        height=args.image_size[0], width=args.image_size[1],
        obs_frames=args.obs_frames, total_frames=args.total_frames,
        trainable=tuple(["mask"] + list(args.extra_trainable)),
        predict_vis=not args.no_vis,
        fresh_lora_rank=args.fresh_lora_rank,
        fresh_lora_alpha=args.fresh_lora_alpha,
        fresh_lora_scope=args.fresh_lora_scope,
    )
    model = FreshLoRAFutureSceneFlowModel(config)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"trainable params: {n_train/1e6:.2f}M  (fresh rank={args.fresh_lora_rank}, "
          f"scope={args.fresh_lora_scope}, extra={args.extra_trainable})")

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    run_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_cfg["variant"] = "fresh_lora"
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
            batch = base._to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                query = model(batch["rgb_obs"], batch["pj_obs"],
                              use_gradient_checkpointing=args.grad_checkpoint)
                if args.loss_space == "decoded":
                    loss = decoded_loss(model, query, batch, args.obs_frames,
                                        future_weight=args.future_weight,
                                        obs_weight=args.obs_weight, scale=args.loss_scale,
                                        use_checkpoint=args.grad_checkpoint)
                else:
                    xyz, _ = model.split_latent(query)
                    target = model.encode_target_delta(batch["target_delta"])
                    loss = latent_loss(xyz, target, args.obs_frames,
                                       future_weight=args.future_weight, obs_weight=args.obs_weight)
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

        val = evaluate(model, val_loader, device, args.val_steps, args.obs_frames,
                       args.amp, args.loss_space, args.loss_scale)
        tl = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch {epoch:03d} done train_loss={tl:.4f} val_loss={val['loss']:.4f} "
              f"val_ade_fut_m={val['ade_future_m']:.4f} val_fde_fut_m={val['fde_future_m']:.4f} "
              f"time={time.time()-start:.1f}s")

        base._save(model, optimizer, output_dir / "last.pt", epoch=epoch,
                   global_step=global_step, val_metrics=val)
        if np.isfinite(val["ade_future_m"]) and val["ade_future_m"] < best:
            best = val["ade_future_m"]
            base._save(model, optimizer, output_dir / "best.pt", epoch=epoch,
                       global_step=global_step, val_metrics=val)
        if args.save_every > 0 and epoch % args.save_every == 0:
            base._save(model, optimizer, output_dir / f"epoch_{epoch:03d}.pt", epoch=epoch,
                       global_step=global_step, val_metrics=val)

    print(f"wrote checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
