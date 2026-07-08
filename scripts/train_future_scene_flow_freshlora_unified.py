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


@torch.no_grad()
def _render_pred_vs_gt(model, sample, obs, total, line_width=2):
    """Inference on one clip -> (T,H,W,3) uint8 frames: GT (left) | pred (right),
    both projected into the last-observed camera (ego-motion removed). Returns
    (frames, future_ade_m). Mirrors comparison/render_unified_viewer."""
    from render_future_track_prediction_viewer import _add_label, _draw_trails, _point_colors

    def project_cam(X, intr, W, H):
        fx, fy, cx, cy = intr
        z = np.clip(X[..., 2], 1e-6, None)
        u = fx * X[..., 0] / z + cx
        v = fy * X[..., 1] / z + cy
        uv = np.stack([u, v], -1).astype(np.float32)
        valid = (X[..., 2] > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return uv, valid

    dev = model.device
    b = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in sample.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q = model(b["rgb_obs"], b["pj_obs"])
        xyz, _ = model.split_latent(q)
        delta = model.decode_xyz(xyz).float()
    h, w = sample["rgb_obs"].shape[-2], sample["rgb_obs"].shape[-1]
    p0 = b["p0_t0_norm"].float(); mean = b["pj_mean"].float(); sc = b["pj_scale"].float()
    pred_m = model.denormalize(model.reconstruct(delta, p0), mean, sc)[0].cpu().numpy()  # 3,T,h,w
    quv = sample["query_uv_model"].numpy()
    xs = np.clip(quv[:, 0].round().astype(int), 0, w - 1)
    ys = np.clip(quv[:, 1].round().astype(int), 0, h - 1)
    pred_pts = pred_m[:, :, ys, xs].transpose(2, 1, 0)                                   # N,T,3
    gt_pts = (sample["gt_tracks_norm"].numpy() * float(sc) + sample["pj_mean"].numpy()).transpose(1, 0, 2)
    vis = sample["visibility"].numpy().T                                                 # N,T
    intr = sample["intr_model"].numpy()
    d = np.linalg.norm(pred_pts[:, obs:] - gt_pts[:, obs:], axis=-1); vm = vis[:, obs:]
    ade = float((d * vm).sum() / (vm.sum() + 1e-6))
    rgb = ((sample["rgb_obs"].numpy().transpose(0, 2, 3, 1) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    last_obs = rgb[obs - 1]
    # Both trails start at the GT last-observed (obs-1) position, which is where
    # the tracked point actually is in the displayed obs-1 frame (on the object).
    # NOTE: the SAM mask is defined on frame 0; the object moves over the observed
    # frames, so obs-1 positions need NOT lie in the frame-0 mask — that is
    # expected object motion, not a rendering error.
    anchor = gt_pts[:, obs - 1:obs]
    gt_uv, gt_v = project_cam(np.concatenate([anchor, gt_pts[:, obs:]], 1), intr, w, h)
    pr_uv, pr_v = project_cam(np.concatenate([anchor, pred_pts[:, obs:]], 1), intr, w, h)
    uvn = np.stack([xs / max(1, w - 1) * 2 - 1, ys / max(1, h - 1) * 2 - 1], -1).astype(np.float32)
    colors = _point_colors(uvn)
    fut = total - obs
    frames = []
    for t in range(obs):
        f = rgb[t].copy(); _add_label(f, f"observed {t + 1}/{obs}")
        frames.append(np.concatenate([f, f], axis=1))
    for k in range(fut):
        gf = _draw_trails(last_obs, gt_uv, gt_v, colors, k + 1, f"GT future {k + 1}/{fut}", line_width)
        pf = _draw_trails(last_obs, pr_uv, pr_v, colors, k + 1,
                          f"pred {k + 1}/{fut} | ADE {ade * 100:.1f}cm", line_width)
        frames.append(np.concatenate([gf, pf], axis=1))
    return np.stack(frames), ade


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
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (rank 0 only).")
    p.add_argument("--wandb-project", default="future-scene-flow")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-entity", default=None)
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
    use_wandb = False  # set True on rank 0 below if --wandb
    viz_samples = {}   # fixed train/val clips for per-epoch wandb video

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
        if args.wandb:
            import wandb
            rc.update(effective_batch=args.batch_size * args.grad_accum * world,
                      clips_per_epoch=args.steps_per_epoch * world * args.batch_size)
            # Persistent unique run id per output-dir: a genuine training resume
            # (from last.pt) continues the SAME wandb run; a fresh start — including
            # a restart after a crash before any epoch finished — makes a NEW run.
            # (Reusing a fixed id across a crash makes wandb drop the new step-0
            #  logs because the dead run already advanced past them.)
            _wid_file = out / "wandb_run_id.txt"
            _resume_train = (out / "last.pt").exists() and _wid_file.exists()
            if _resume_train:
                _wid = _wid_file.read_text().strip()
            else:
                _wid = wandb.util.generate_id()
                _wid_file.write_text(_wid)
            wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                       entity=args.wandb_entity, dir=str(out), config=rc,
                       id=_wid, resume="allow" if _resume_train else None)
            use_wandb = True
            (out / "viz").mkdir(exist_ok=True)
            _vkw = dict(obs_frames=args.obs_frames, total_frames=args.total_frames,
                        image_size=tuple(args.image_size), num_points=min(64, args.num_points),
                        subtract_camera_motion=args.subcam, samples_per_clip=1, seed=args.seed + 777)
            if tr:
                viz_samples["train"] = UnifiedTrackDataset([tr[0]], **_vkw)[0]
            if va:
                viz_samples["val"] = UnifiedTrackDataset([va[0]], **_vkw)[0]

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
                if use_wandb:
                    wandb.log({"train/loss": float(loss.item()),
                               "train/ade_fut_m": m["ade_future_m"],
                               "train/fde_fut_m": m["fde_future_m"],
                               "epoch": epoch, "step": step}, step=gstep)

        # validation + checkpointing on rank 0 only
        if is_main:
            val = evaluate(model, vl, device, args.val_steps, args.obs_frames, args.amp, args.loss_scale)
            pr(f"epoch {epoch:03d} done train_loss={np.mean(losses):.4f} val_loss={val['loss']:.4f} "
               f"val_ade_fut_m={val['ade_future_m']:.4f} val_fde_fut_m={val['fde_future_m']:.4f} "
               f"time={time.time()-start:.1f}s", flush=True)
            if use_wandb:
                wandb.log({"val/loss": val["loss"], "val/ade_fut_m": val["ade_future_m"],
                           "val/fde_fut_m": val["fde_future_m"],
                           "train/epoch_loss": float(np.mean(losses)) if losses else float("nan"),
                           "epoch": epoch, "epoch_time_s": time.time() - start}, step=gstep)
            if use_wandb and viz_samples:
                import imageio.v2 as _imageio
                vlog = {}
                for _tag, _samp in viz_samples.items():
                    try:
                        _fr, _vade = _render_pred_vs_gt(model, _samp, args.obs_frames, args.total_frames)
                        _vp = out / "viz" / f"{_tag}_epoch{epoch:03d}.mp4"
                        _imageio.mimwrite(str(_vp), list(_fr), fps=8, quality=8, macro_block_size=None)
                        vlog[f"viz/{_tag}"] = wandb.Video(str(_vp), fps=8, format="mp4",
                                                          caption=f"{_tag} {_samp['video_id']} | future-ADE {_vade*100:.1f}cm")
                    except Exception as _e:
                        pr(f"[wandb viz] {_tag} failed: {_e}")
                if vlog:
                    wandb.log(vlog, step=gstep)
            base._save(model, opt, out / "last.pt", epoch=epoch, global_step=gstep, val_metrics=val)
            if np.isfinite(val["ade_future_m"]) and val["ade_future_m"] < best:
                best = val["ade_future_m"]; base._save(model, opt, out / "best.pt", epoch=epoch, global_step=gstep, val_metrics=val)
            if args.save_every > 0 and epoch % args.save_every == 0:
                base._save(model, opt, out / f"epoch_{epoch:03d}.pt", epoch=epoch, global_step=gstep, val_metrics=val)
        if ddp_on:
            dist.barrier()
    pr(f"wrote checkpoints to {out}")
    if use_wandb:
        wandb.finish()
    if ddp_on:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
