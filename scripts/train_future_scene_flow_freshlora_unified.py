#!/usr/bin/env python3
"""Train the fresh-LoRA future scene-flow model (best variant) on ALL curated
Something-Something data: dense (`anchor_tracks32_curated_dense`) + sparse
(`anchor_tracks32_next2000_sparse`) combined, with camera-motion subtraction to
the last observed frame.

Uses the unified dataset (query-point supervision for both formats) + the
TrackCraft3r-faithful decoded loss, auto-resumes from last.pt.

Example::

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    python scripts/train_future_scene_flow_freshlora_unified.py \
      --output-dir data/something_something/future_track_training/freshlora_unified_camsub_320x576 \
      --image-size 320 576 --num-points 48 \
      --batch-size 1 --grad-accum 8 --grad-checkpoint \
      --epochs 12 --steps-per-epoch 800 --val-steps 32 --amp --save-every 1
"""
from __future__ import annotations
import argparse, json, os, sys, time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import models_pretrained  # noqa: E402,F401
from models_pretrained.future_scene_flow.model_fresh_lora import (  # noqa: E402
    FreshLoRAConfig, FreshLoRAFutureSceneFlowModel)
from models_pretrained.future_scene_flow.unified_dataset import (  # noqa: E402
    UnifiedTrackDataset, build_unified_index, split_items)
from models_pretrained.future_scene_flow.sparse_dataset import sparse_collate  # noqa: E402
from models_pretrained.future_scene_flow.losses import sparse_decoded_loss, sparse_metrics  # noqa: E402
import train_future_scene_flow as base  # noqa: E402

DEFAULT_CKPT = REPO / "models_pretrained/checkpoints/trackcraft3r/model.safetensors"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/something_something"))
    p.add_argument("--dense-tracks-name", default="anchor_tracks32_curated_dense")
    p.add_argument("--dense-manifest", type=Path, default=Path("sam3_anchor_masks/manifest_curated_dense.json"))
    p.add_argument("--sparse-tracks-name", default="anchor_tracks32_next2000_sparse")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--obs-frames", type=int, default=10)
    p.add_argument("--total-frames", type=int, default=32)
    p.add_argument("--image-size", type=int, nargs=2, default=(320, 576), metavar=("H", "W"))
    p.add_argument("--num-points", type=int, default=256, help="points sampled per dense clip from the SAM mask")
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
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--steps-per-epoch", type=int, default=800)
    p.add_argument("--val-steps", type=int, default=32)
    p.add_argument("--samples-per-clip", type=int, default=1)
    p.add_argument("--limit-dense", type=int, default=0)
    p.add_argument("--limit-sparse", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save-every", type=int, default=1)
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

    # ---- distributed (torchrun) setup ----
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp_on = world > 1
    if ddp_on:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    is_main = rank == 0
    device = torch.device("cuda")
    pr = (lambda *a, **k: print(*a, **k)) if is_main else (lambda *a, **k: None)

    out = args.output_dir.expanduser().resolve()
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
    if ddp_on:
        dist.barrier()

    items = build_unified_index(args.root.expanduser().resolve(), args.dense_tracks_name,
                                args.dense_manifest, args.sparse_tracks_name,
                                limit_dense=args.limit_dense, limit_sparse=args.limit_sparse)
    nd = sum(1 for it in items if it.fmt == "dense"); ns = len(items) - nd
    if len(items) < 2:
        raise SystemExit(f"need >=2 clips; found {len(items)}")
    tr, va = split_items(items, args.val_fraction, args.seed)
    pr(f"indexed {len(items)} clips (dense={nd} sparse={ns}): train={len(tr)} val={len(va)} "
       f"| camsub={args.subcam} | world={world}", flush=True)
    ds_kw = dict(obs_frames=args.obs_frames, total_frames=args.total_frames,
                 image_size=tuple(args.image_size), num_points=args.num_points,
                 subtract_camera_motion=args.subcam)
    train_ds = UnifiedTrackDataset(tr, samples_per_clip=args.samples_per_clip, seed=args.seed, **ds_kw)
    val_ds = UnifiedTrackDataset(va, samples_per_clip=1, seed=args.seed + 100000, **ds_kw)
    train_sampler = (DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
                     if ddp_on else None)
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                    sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
                    drop_last=True, collate_fn=sparse_collate)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True, collate_fn=sparse_collate)

    cfg = FreshLoRAConfig(checkpoint_path=str(args.checkpoint_path.expanduser().resolve()),
                          height=args.image_size[0], width=args.image_size[1],
                          obs_frames=args.obs_frames, total_frames=args.total_frames,
                          trainable=tuple(["mask"] + list(args.extra_trainable)),
                          fresh_lora_rank=args.fresh_lora_rank, fresh_lora_scope=args.fresh_lora_scope)
    model = FreshLoRAFutureSceneFlowModel(cfg)
    nt = sum(p.numel() for p in model.trainable_parameters())
    pr(f"trainable params: {nt/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # DDP wrapper for the forward; `model` stays the raw module for decode/save.
    # All trainable params are used every step (no unused), so
    # find_unused_parameters=False; grad accumulation uses no_sync() below.
    fwd = DDP(model, device_ids=[local_rank], output_device=local_rank,
              find_unused_parameters=False) if ddp_on else model

    if is_main:
        rc = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        rc.update(variant="fresh_lora_unified", subtract_camera_motion=args.subcam, world_size=world,
                  n_dense=nd, n_sparse=ns, model=model.config_dict, output_dir=str(out), trainable_params_M=nt / 1e6)
        (out / "config.json").write_text(json.dumps(rc, indent=2))

    best = float("inf"); gstep = 0; start_epoch = 1
    if (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["trainable_state"], strict=False)
        try:
            opt.load_state_dict(st["optimizer"])
        except Exception:
            pass
        start_epoch = int(st.get("epoch", 0)) + 1; gstep = int(st.get("global_step", 0))
        if (out / "best.pt").exists():
            best = float(torch.load(out / "best.pt", map_location="cpu", weights_only=False)
                         .get("val_metrics", {}).get("ade_future_m", float("inf")))
        pr(f"resumed from epoch {start_epoch-1} (best ade={best:.4f}); continuing at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        if ddp_on:
            train_sampler.set_epoch(epoch)
        model.train(); losses = []; start = time.time(); opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tl, 1):
            if step > args.steps_per_epoch:
                break
            batch = base._to_device(batch, device)
            sync = (step % args.grad_accum == 0)
            no_sync = fwd.no_sync() if (ddp_on and not sync) else nullcontext()
            with no_sync:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                    q = fwd(batch["rgb_obs"], batch["pj_obs"], use_gradient_checkpointing=args.grad_checkpoint)
                    loss = sparse_decoded_loss(model, q, batch, args.obs_frames, future_weight=args.future_weight,
                                               obs_weight=args.obs_weight, scale=args.loss_scale,
                                               use_checkpoint=args.grad_checkpoint)
                (loss / args.grad_accum).backward()
            losses.append(float(loss.item()))
            if sync:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); gstep += 1
            if is_main and (step == 1 or step % args.metric_every == 0):
                m = sparse_metrics(model, q.detach(), batch, args.obs_frames)
                pr(f"epoch {epoch:03d} step {step:04d} loss={loss.item():.4f} "
                   f"ade_fut_m={m['ade_future_m']:.4f} fde_fut_m={m['fde_future_m']:.4f}", flush=True)

        # validation + checkpointing on rank 0 only
        if is_main:
            val = evaluate(model, vl, device, args.val_steps, args.obs_frames, args.amp, args.loss_scale)
            pr(f"epoch {epoch:03d} done train_loss={np.mean(losses):.4f} val_loss={val['loss']:.4f} "
               f"val_ade_fut_m={val['ade_future_m']:.4f} val_fde_fut_m={val['fde_future_m']:.4f} "
               f"time={time.time()-start:.1f}s", flush=True)
            base._save(model, opt, out / "last.pt", epoch=epoch, global_step=gstep, val_metrics=val)
            if np.isfinite(val["ade_future_m"]) and val["ade_future_m"] < best:
                best = val["ade_future_m"]; base._save(model, opt, out / "best.pt", epoch=epoch, global_step=gstep, val_metrics=val)
            if args.save_every > 0 and epoch % args.save_every == 0:
                base._save(model, opt, out / f"epoch_{epoch:03d}.pt", epoch=epoch, global_step=gstep, val_metrics=val)
        if ddp_on:
            dist.barrier()
    pr(f"wrote checkpoints to {out}")
    if ddp_on:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
