#!/usr/bin/env python3
"""Static 2D HTML viewer of zero-shot OOD predictions (all clips, one model load).

For each processed clip + SAM mask: run the model, project the predicted future
tracks into the last-observed image, render a trail video on the masked object,
and write an index.html. No GT (zero-shot).

    python scripts/comparison/ood_render_html.py \
      --checkpoint .../best.pt --proc-dir zero-shot-eval/processed \
      --mask-dir zero-shot-eval/masks --out-dir zero-shot-eval/viewer --grid-stride 4
"""
from __future__ import annotations
import argparse, glob, html, json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "scripts"), str(REPO / "scripts/comparison")):
    if p not in sys.path:
        sys.path.insert(0, p)
import models_pretrained  # noqa: E402,F401
from render_future_scene_flow_viewer import build_model_from_ckpt  # noqa: E402
from render_future_track_prediction_viewer import _add_label, _draw_trails, _write_video  # noqa: E402
from ood_common import ood_predict  # noqa: E402

DEFAULT_BASE = REPO / "models_pretrained/checkpoints/trackcraft3r/model.safetensors"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-checkpoint", default=str(DEFAULT_BASE))
    ap.add_argument("--proc-dir", default="zero-shot-eval/processed")
    ap.add_argument("--mask-dir", default="zero-shot-eval/masks")
    ap.add_argument("--out-dir", default="zero-shot-eval/viewer")
    ap.add_argument("--grid-stride", type=int, default=4)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--line-width", type=int, default=2)
    ap.add_argument("--video-ext", default=".webm")
    ap.add_argument("--video-codec", default="VP80")
    return ap.parse_args()


def project_cam(X, intr, W, H):
    fx, fy, cx, cy = intr
    z = np.clip(X[..., 2], 1e-6, None)
    u = fx * X[..., 0] / z + cx; v = fy * X[..., 1] / z + cy
    uv = np.stack([u, v], -1).astype(np.float32)
    valid = (X[..., 2] > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return uv, valid


def main():
    a = parse_args()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    out = Path(a.out_dir); (out / "videos").mkdir(parents=True, exist_ok=True)
    model, st = build_model_from_ckpt(Path(a.checkpoint), Path(a.base_checkpoint)); model.eval()

    items = []
    for f in sorted(glob.glob(os.path.join(a.proc_dir, "*_user.npz"))):
        vid = os.path.basename(f).replace("_user.npz", "")
        mask = Path(a.mask_dir) / f"{vid}_mask.png"
        if not mask.exists():
            print(f"  {vid}: no mask, skip"); continue
        r = ood_predict(model, f, mask, grid_stride=a.grid_stride)
        obs, total, mw, mh = r["obs"], r["total"], r["mw"], r["mh"]
        pred = r["pred_pts"]                                          # N,T,3 cam-lastobs
        anchor = pred[:, obs - 1:obs]
        xyz = np.concatenate([anchor, pred[:, obs:total]], 1)        # N,1+fut,3
        uv, val = project_cam(xyz, r["intr_m"], mw, mh)
        rgb8 = r["rgb8"]; last = rgb8[obs - 1]; fut = total - obs; cols = r["track_col"]
        frames = []
        for t in range(obs):
            im = rgb8[t].copy(); _add_label(im, f"observed {t+1}/{obs}"); frames.append(im)
        for k in range(fut):
            frames.append(_draw_trails(last, uv, val, cols, k + 1, f"pred future {k+1}/{fut}", a.line_width))
        vp = out / "videos" / f"{vid}{a.video_ext}"
        _write_video(vp, frames, a.fps, a.video_codec)
        items.append({"videoId": vid, "numPoints": r["N"], "obs": obs, "total": total,
                      "video": f"videos/{vp.name}"})
        print(f"  {vid}: {r['N']} masked pts -> {vp.name}", flush=True)

    css = """:root{color-scheme:dark}body{margin:0;background:#101214;color:#eee;font-family:system-ui,sans-serif}
    header{padding:14px 20px;border-bottom:1px solid #2a3037}main{padding:16px 20px;display:grid;
    grid-template-columns:repeat(2,1fr);gap:16px}.card{border:1px solid #2a3037;background:#181b1f;border-radius:8px;padding:8px}
    .card h2{font-size:14px;margin:4px 6px}video{width:100%;background:#000;border-radius:4px}@media(max-width:800px){main{grid-template-columns:1fr}}"""
    cards = "".join(
        f'<div class="card"><h2>{html.escape(it["videoId"])} · {it["numPoints"]} mask pts · obs {it["obs"]}/{it["total"]}</h2>'
        f'<video src="{it["video"]}" controls muted loop autoplay></video></div>' for it in items)
    (out / "index.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>Zero-shot OOD predictions</title>'
        f'<style>{css}</style></head><body><header><h1>Zero-shot OOD future tracks (epoch {st.get("epoch")})</h1>'
        f'<p>predicted future trajectories on the SAM-masked object, projected into the last observed frame</p></header>'
        f'<main>{cards}</main></body></html>')
    (out / "manifest.json").write_text(json.dumps({"items": items}, indent=2))
    print(f"wrote {out/'index.html'} ({len(items)} clips)")


if __name__ == "__main__":
    main()
