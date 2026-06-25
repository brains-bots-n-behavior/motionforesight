#!/usr/bin/env python3
"""Train the fresh-LoRA future scene-flow model (the best variant) on the SPARSE
TrackCraft3r data, with camera-motion subtraction to the last observed frame.

Same model/loss family as train_future_scene_flow_freshlora.py (fresh rank-32
LoRA + I/O/head + mask latents, TrackCraft3r-faithful decoded loss), but:
  - reads `*_sparse.npz` (computes Pj from depth on the fly),
  - supervises sparsely at the query points (visibility-masked),
  - expresses Pj / anchor / GT in the last-observed camera frame so predicted
    future tracks have ego-motion removed (``--subtract-camera-motion``, default on).

Example::

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python scripts/train_future_scene_flow_freshlora_sparse.py \
      --root data/something_something --tracks-name anchor_tracks32_next2000_sparse \
      --output-dir data/something_something/future_track_training/freshlora_sparse_camsub_320x576 \
      --image-size 320 576 --batch-size 1 --grad-accum 8 --grad-checkpoint \
      --epochs 15 --steps-per-epoch 600 --val-steps 24 --amp --save-every 2
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import models_pretrained  # noqa: E402,F401
from models_pretrained.future_scene_flow.model_fresh_lora import (  # noqa: E402
    FreshLoRAConfig, FreshLoRAFutureSceneFlowModel)
from models_pretrained.future_scene_flow.sparse_dataset import (  # noqa: E402
    SparseTrackDataset, build_sparse_index, split_items, sparse_collate)
from models_pretrained.future_scene_flow.losses import sparse_decoded_loss, sparse_metrics  # noqa: E402
import train_future_scene_flow as base  # noqa: E402

DEFAULT_CKPT = REPO / "models_pretrained/checkpoints/trackcraft3r/model.safetensors"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/something_something"))
    p.add_argument("--tracks-name", default="anchor_tracks32_next2000_sparse")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--obs-frames", type=int, default=10)
    p.add_argument("--total-frames", type=int, default=32)
    p.add_argument("--image-size", type=int, nargs=2, default=(320, 576), metavar=("H", "W"))
    p.add_argument("--fresh-lora-rank", type=int, default=32)
    p.add_argument("--fresh-lora-scope", default="attn", choices=["attn", "all"])
    p.add_argument("--extra-trainable", nargs="*", default=["io", "head"], choices=["io", "head"])
    p.add_argument("--subtract-camera-motion", dest="subcam", action="store_true", default=True)
    p.add_argument("--no-subtract-camera-motion", dest="subcam", action="store_false")
    p.add_argument("--loss-scale", type=float, default=10.0)
    p.add_argument("--future-weight", type=float, default=1.0)
    p.add_argument("--obs-weight", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--steps-per-epoch", type=int, default=600)
    p.add_argument("--val-steps", type=int, default=24)
    p.add_argument("--samples-per-clip", type=int, default=1)
    p.add_argument("--limit-clips", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--metric-every", type=int, default=100)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, max_steps, obs, amp, scale):
    model.eval(); rows = []
    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        batch = base._to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            q = model(batch["rgb_obs"], batch["pj_obs"])
            loss = sparse_decoded_loss(model, q, batch, obs, scale=scale, use_checkpoint=False)
        m = sparse_metrics(model, q, batch, obs); m["loss"] = float(loss.item()); rows.append(m)
    if not rows:
        return {"loss": float("nan"), "ade_future_m": float("nan"), "fde_future_m": float("nan")}
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda")
    out = args.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)

    items = build_sparse_index(args.root.expanduser().resolve(), args.tracks_name, args.limit_clips)
    if len(items) < 2:
        raise SystemExit(f"need >=2 sparse clips; found {len(items)}")
    tr, va = split_items(items, args.val_fraction, args.seed)
    print(f"indexed {len(items)} sparse clips: train={len(tr)} val={len(va)} | camsub={args.subcam}")
    ds_kw = dict(obs_frames=args.obs_frames, total_frames=args.total_frames,
                 image_size=tuple(args.image_size), subtract_camera_motion=args.subcam)
    train_ds = SparseTrackDataset(tr, samples_per_clip=args.samples_per_clip, seed=args.seed, **ds_kw)
    val_ds = SparseTrackDataset(va, samples_per_clip=1, seed=args.seed + 100000, **ds_kw)
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, drop_last=True, collate_fn=sparse_collate)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True, collate_fn=sparse_collate)

    cfg = FreshLoRAConfig(checkpoint_path=str(args.checkpoint_path.expanduser().resolve()),
                          height=args.image_size[0], width=args.image_size[1],
                          obs_frames=args.obs_frames, total_frames=args.total_frames,
                          trainable=tuple(["mask"] + list(args.extra_trainable)),
                          fresh_lora_rank=args.fresh_lora_rank, fresh_lora_scope=args.fresh_lora_scope)
    model = FreshLoRAFutureSceneFlowModel(cfg)
    nt = sum(p.numel() for p in model.trainable_parameters())
    print(f"trainable params: {nt/1e6:.2f}M")
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    rc = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    rc.update(variant="fresh_lora_sparse", subtract_camera_motion=args.subcam,
              model=model.config_dict, output_dir=str(out), trainable_params_M=nt / 1e6)
    (out / "config.json").write_text(json.dumps(rc, indent=2))

    # auto-resume from last.pt if present (robust to mid-run OOM crashes)
    best = float("inf"); gstep = 0; start_epoch = 1
    if (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["trainable_state"], strict=False)
        try:
            opt.load_state_dict(st["optimizer"])
        except Exception:
            pass
        start_epoch = int(st.get("epoch", 0)) + 1
        gstep = int(st.get("global_step", 0))
        if (out / "best.pt").exists():
            bst = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
            best = float(bst.get("val_metrics", {}).get("ade_future_m", float("inf")))
        print(f"resumed from epoch {start_epoch-1} (best ade={best:.4f}); continuing at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); losses = []; start = time.time(); opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tl, 1):
            if step > args.steps_per_epoch:
                break
            batch = base._to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                q = model(batch["rgb_obs"], batch["pj_obs"], use_gradient_checkpointing=args.grad_checkpoint)
                loss = sparse_decoded_loss(model, q, batch, args.obs_frames, future_weight=args.future_weight,
                                           obs_weight=args.obs_weight, scale=args.loss_scale,
                                           use_checkpoint=args.grad_checkpoint)
            (loss / args.grad_accum).backward(); losses.append(float(loss.item()))
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); gstep += 1
            if step == 1 or step % args.metric_every == 0:
                m = sparse_metrics(model, q.detach(), batch, args.obs_frames)
                print(f"epoch {epoch:03d} step {step:04d} loss={loss.item():.4f} "
                      f"ade_fut_m={m['ade_future_m']:.4f} fde_fut_m={m['fde_future_m']:.4f}", flush=True)
        val = evaluate(model, vl, device, args.val_steps, args.obs_frames, args.amp, args.loss_scale)
        print(f"epoch {epoch:03d} done train_loss={np.mean(losses):.4f} val_loss={val['loss']:.4f} "
              f"val_ade_fut_m={val['ade_future_m']:.4f} val_fde_fut_m={val['fde_future_m']:.4f} "
              f"time={time.time()-start:.1f}s", flush=True)
        base._save(model, opt, out / "last.pt", epoch=epoch, global_step=gstep, val_metrics=val)
        if np.isfinite(val["ade_future_m"]) and val["ade_future_m"] < best:
            best = val["ade_future_m"]; base._save(model, opt, out / "best.pt", epoch=epoch, global_step=gstep, val_metrics=val)
        if args.save_every > 0 and epoch % args.save_every == 0:
            base._save(model, opt, out / f"epoch_{epoch:03d}.pt", epoch=epoch, global_step=gstep, val_metrics=val)
    print(f"wrote checkpoints to {out}")


if __name__ == "__main__":
    main()
