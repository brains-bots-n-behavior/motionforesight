#!/usr/bin/env python3
"""Prepare a shared comparison set: same val clips, same query points, GT + OUR
model's predicted 3D tracks. Run in the 3dflow env.

Per clip writes a small .npz the MolmoMotion script (separate env) consumes:
    video_id, dense_path, action, obs_frames, total_frames
    query_xy_dense   (P,2)  query pixel coords at the dense track resolution (frame-0 indexed)
    gt_3d_cam0       (T,P,3) GT 3D tracks in cam-0 (= world; w2c[0]=I) meters
    ours_3d_cam0     (T,P,3) OUR model's predicted 3D tracks (same points)
    w2c              (T,4,4) per-frame world->cam extrinsics
    intr_dense       (4,)    fx,fy,cx,cy scaled to the dense rgb resolution
    dense_h, dense_w
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import models_pretrained  # noqa: F401
from models_pretrained.future_scene_flow.dataset import build_track_index, split_items, _load_union_mask  # noqa: E402
from render_future_scene_flow_viewer import build_model_from_ckpt, predict_dense  # noqa: E402
from models_pretrained.future_scene_flow.dataset import FutureSceneFlowDataset  # noqa: E402
import cv2  # noqa: E402

DEFAULT_CKPT = REPO / "models_pretrained/checkpoints/trackcraft3r/model.safetensors"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/something_something"))
    p.add_argument("--tracks-name", default="anchor_tracks32_curated_dense")
    p.add_argument("--manifest", type=Path, default=Path("sam3_anchor_masks/manifest_curated_dense.json"))
    p.add_argument("--checkpoint", type=Path, required=True, help="our best.pt")
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--val-count", type=int, default=20)
    p.add_argument("--num-points", type=int, default=8, help="= MolmoMotion config.num_points")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    model, _ = build_model_from_ckpt(args.checkpoint, args.base_checkpoint)
    cfg = model.config
    obs, total, mh, mw = cfg.obs_frames, cfg.total_frames, cfg.height, cfg.width

    items = build_track_index(args.root.expanduser().resolve(), tracks_name=args.tracks_name,
                              manifest=args.manifest, require_masks=True)
    _, val_items = split_items(items, 0.1, args.seed)
    val_items = val_items[: args.val_count] if args.val_count > 0 else val_items
    ds = FutureSceneFlowDataset(val_items, obs_frames=obs, total_frames=total,
                                image_size=(mh, mw), samples_per_clip=1, seed=args.seed + 500_000)

    n = 0
    for idx, item in enumerate(val_items):
        if (out / f"{item.video_id}.npz").exists():
            print(f"[{idx+1}/{len(val_items)}] {item.video_id} (skip, exists)")
            continue
        try:
            _process_clip(idx, item, model, ds, obs, total, mh, mw, args, out, len(val_items))
            n += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[{idx+1}/{len(val_items)}] {item.video_id} OOM -> skipped (rerun to resume)", flush=True)
    print(f"wrote {n} prep files this pass to {out}")


def _process_clip(idx, item, model, ds, obs, total, mh, mw, args, out, ntot):
    if True:
        dense = np.load(item.dense_path)
        track_map = dense["track_map"][:total].astype(np.float32)   # T,Hd,Wd,3 (cam0)
        T, Hd, Wd, _ = track_map.shape
        user = np.load(item.dense_path.with_name(item.dense_path.name.replace("_dense.npz", "_user.npz")),
                       allow_pickle=True)
        w2c = user["extrinsics_w2c"][:total].astype(np.float32)
        oh, ow = user["depth_map"].shape[1:3]
        fx, fy, cx, cy = user["fx_fy_cx_cy"].astype(np.float64)
        intr_dense = np.array([fx * Wd / ow, fy * Hd / oh, cx * Wd / ow, cy * Hd / oh], dtype=np.float64)

        # sample query points at dense res: object mask & finite over all frames
        obj = _load_union_mask(item.mask_paths, (Hd, Wd))
        valid = np.isfinite(track_map).all(axis=(0, 3))            # Hd,Wd
        cand = obj & valid
        if cand.sum() < args.num_points:
            cand = valid
        ys, xs = np.where(cand)
        rng = np.random.default_rng(args.seed + idx)
        sel = rng.choice(ys.size, size=min(args.num_points, ys.size), replace=ys.size < args.num_points)
        ys, xs = ys[sel], xs[sel]
        gt_3d = track_map[:, ys, xs, :]                            # T,P,3 (cam0)

        # OUR model prediction at the same points (sample its dense pred at model res)
        sample = ds[idx]
        pred_m = predict_dense(model, sample)                     # 3,T,mh,mw (cam0 meters)
        xs_m = np.clip((xs.astype(np.float64) * mw / Wd).round().astype(int), 0, mw - 1)
        ys_m = np.clip((ys.astype(np.float64) * mh / Hd).round().astype(int), 0, mh - 1)
        ours_3d = pred_m[:, :, ys_m, xs_m].transpose(1, 2, 0)     # T,P,3 (cam0)

        np.savez(out / f"{item.video_id}.npz",
                 video_id=item.video_id, dense_path=str(item.dense_path), action=item.text,
                 obs_frames=obs, total_frames=total,
                 query_xy_dense=np.stack([xs, ys], axis=-1).astype(np.float32),
                 gt_3d_cam0=gt_3d.astype(np.float32), ours_3d_cam0=ours_3d.astype(np.float32),
                 w2c=w2c, intr_dense=intr_dense, dense_h=Hd, dense_w=Wd)
        print(f"[{idx+1}/{ntot}] {item.video_id}  P={len(xs)}  action='{item.text[:40]}'", flush=True)


if __name__ == "__main__":
    main()
