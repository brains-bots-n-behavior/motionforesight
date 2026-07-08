#!/usr/bin/env python3
"""Full-video GT-vs-pred renderer.

Plays the ACTUAL clip (all `total` frames) with the tracks glued to the moving
object, instead of freezing the future tracks on the last-observed image.

The model predicts in the camera-subtracted (last-observed) frame. To overlay on
the real, moving video we transform tracks back to world (via inv(w2c[obs-1]))
and, for each frame t, project the trajectory-so-far into frame t's REAL camera
pose w2c[t]. Left = GT (TrackCraft3r tracks), right = model prediction. Observed
frames 0..obs-1 and future frames obs..total-1 are all shown on their real image.
"""
from __future__ import annotations
import argparse, html, json, sys
from pathlib import Path
import numpy as np
import cv2
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import models_pretrained  # noqa: E402,F401
from models_pretrained.future_scene_flow.unified_dataset import (  # noqa: E402
    UnifiedTrackDataset, build_unified_index, split_items)
from models_pretrained.future_scene_flow.sparse_dataset import _apply_rigid  # noqa: E402
from render_future_scene_flow_viewer import build_model_from_ckpt  # noqa: E402
from render_future_track_prediction_viewer import _draw_trails, _point_colors, _write_video  # noqa: E402

DEFAULT_BASE = REPO / "models_pretrained/checkpoints/trackcraft3r/model.safetensors"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-checkpoint", default=str(DEFAULT_BASE))
    ap.add_argument("--root", required=True)
    ap.add_argument("--dense-tracks-name", required=True)
    ap.add_argument("--dense-manifest", required=True)
    ap.add_argument("--sparse-tracks-name", default="")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--count", type=int, default=8, help="how many clips (0=all)")
    ap.add_argument("--split", choices=["all", "val", "train"], default="all")
    ap.add_argument("--num-points", type=int, default=64)
    ap.add_argument("--grid-stride", type=int, default=0)
    ap.add_argument("--max-points", type=int, default=4000)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--line-width", type=int, default=2)
    ap.add_argument("--video-ext", default=".webm")
    ap.add_argument("--video-codec", default="VP80")
    return ap.parse_args()


def _proj(Xw, w2c_t, intr, W, H):
    """world XYZ (N,3) -> pixel uv (N,2) in frame t + valid mask, via w2c_t + intr."""
    Xc = _apply_rigid(w2c_t, Xw)
    fx, fy, cx, cy = intr
    z = np.clip(Xc[:, 2], 1e-6, None)
    u = fx * Xc[:, 0] / z + cx
    v = fy * Xc[:, 1] / z + cy
    uv = np.stack([u, v], -1).astype(np.float32)
    valid = (Xc[:, 2] > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return uv, valid


def _label(img, text):
    cv2.rectangle(img, (10, 10), (12 + 9 * len(text), 34), (0, 0, 0), -1)
    cv2.putText(img, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    out = Path(a.output_dir); (out / "videos").mkdir(parents=True, exist_ok=True)
    model, st = build_model_from_ckpt(Path(a.checkpoint), Path(a.base_checkpoint)); model.eval()
    cfg = model.config
    obs, total, h, w = cfg.obs_frames, cfg.total_frames, cfg.height, cfg.width
    dev = model.device

    items = build_unified_index(Path(a.root), a.dense_tracks_name, Path(a.dense_manifest), a.sparse_tracks_name)
    items = [it for it in items if it.fmt == "dense"]
    if a.split != "all":
        train, val = split_items(items, a.val_fraction, a.seed)
        items = val if a.split == "val" else train
    if a.count > 0:
        items = items[:a.count]
    ds = UnifiedTrackDataset(items, obs_frames=obs, total_frames=total, image_size=(h, w),
                             num_points=a.num_points, samples_per_clip=1, seed=a.seed + 500000,
                             subtract_camera_motion=True, dense_grid_stride=a.grid_stride,
                             max_points=a.max_points)
    viewer = []
    for idx, item in enumerate(items):
        s = ds[idx]
        b = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            q = model(b["rgb_obs"], b["pj_obs"])
            xyz, _ = model.split_latent(q)
            delta = model.decode_xyz(xyz).float()
        p0 = b["p0_t0_norm"].float(); mean = b["pj_mean"].float(); sc = b["pj_scale"].float()
        pred_m = model.denormalize(model.reconstruct(delta, p0), mean, sc)[0].cpu().numpy()  # 3,T,h,w
        quv = s["query_uv_model"].numpy()
        xs = np.clip(quv[:, 0].round().astype(int), 0, w - 1)
        ys = np.clip(quv[:, 1].round().astype(int), 0, h - 1)
        pred_lastobs = pred_m[:, :, ys, xs].transpose(2, 1, 0)                       # N,T,3 last-obs cam
        gt_lastobs = (s["gt_tracks_norm"].numpy() * float(sc) + s["pj_mean"].numpy()).transpose(1, 0, 2)
        intr = s["intr_model"].numpy()

        # full extrinsics + all-T real frames
        user = np.load(str(item.path).replace("_dense.npz", "_user.npz"), allow_pickle=True)
        w2c = user["extrinsics_w2c"][:total].astype(np.float64)                      # T,4,4 (w2c[0]=I)
        dense = np.load(item.path)
        rgb_all = np.stack([cv2.resize(dense["rgb"][t], (w, h)) for t in range(total)]).astype(np.uint8)

        inv_obs = np.linalg.inv(w2c[obs - 1])
        gt_world = _apply_rigid(inv_obs, gt_lastobs)                                 # N,T,3 world
        pred_world = _apply_rigid(inv_obs, pred_lastobs)
        uvn = np.stack([xs / max(1, w - 1) * 2 - 1, ys / max(1, h - 1) * 2 - 1], -1).astype(np.float32)
        colors = _point_colors(uvn)

        frames = []
        for t in range(total):
            # project trajectory-so-far (0..t) into THIS frame's real camera
            g_uv = np.stack([_proj(gt_world[:, k], w2c[t], intr, w, h)[0] for k in range(t + 1)], 1)
            g_v = np.stack([_proj(gt_world[:, k], w2c[t], intr, w, h)[1] for k in range(t + 1)], 1)
            p_uv = np.stack([_proj(pred_world[:, k], w2c[t], intr, w, h)[0] for k in range(t + 1)], 1)
            p_v = np.stack([_proj(pred_world[:, k], w2c[t], intr, w, h)[1] for k in range(t + 1)], 1)
            phase = "obs" if t < obs else "future"
            gL = _label(_draw_trails(rgb_all[t], g_uv, g_v, colors, t, "", a.line_width), f"GT - {phase} {t+1}/{total}")
            pL = _label(_draw_trails(rgb_all[t], p_uv, p_v, colors, t, "", a.line_width), f"pred - {phase} {t+1}/{total}")
            frames.append(np.concatenate([gL, pL], axis=1))
        vp = out / "videos" / f"{item.video_id}{a.video_ext}"
        _write_video(vp, frames, a.fps, a.video_codec)
        # future ADE at query points (meters)
        d = np.linalg.norm(pred_lastobs[:, obs:] - gt_lastobs[:, obs:], axis=-1)
        ade = float(d.mean())
        viewer.append({"videoId": item.video_id, "title": item.text, "ade": ade,
                       "numPoints": int(len(xs)), "video": f"videos/{vp.name}"})
        print(f"[{idx+1}/{len(items)}] {item.video_id} full-video ({total} frames) ade={ade*100:.1f}cm", flush=True)

    css = ("body{margin:0;background:#101214;color:#eee;font-family:system-ui}main{padding:16px;display:grid;"
           "grid-template-columns:1fr;gap:16px}.card{border:1px solid #2a3037;border-radius:8px;padding:8px}"
           "video{width:100%;border-radius:4px;background:#000}")
    cards = "".join(f'<div class="card"><h3>{html.escape(it["videoId"])} · {it["numPoints"]}pts · '
                    f'future-ADE {it["ade"]*100:.1f}cm — GT (left) | prediction (right)</h3>'
                    f'<video src="{it["video"]}" controls muted loop autoplay></video></div>'
                    for it in sorted(viewer, key=lambda x: x["ade"]))
    (out / "index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>Full-video GT vs pred</title>'
                                    f'<style>{css}</style><main>{cards}</main>')
    (out / "manifest.json").write_text(json.dumps({"items": viewer}, indent=2))
    print(f"wrote {out/'index.html'} ({len(viewer)} clips)")


if __name__ == "__main__":
    main()
