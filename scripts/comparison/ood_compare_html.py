#!/usr/bin/env python3
"""Side-by-side OOD viewer: our model vs MolmoMotion (3dflow env).

For each OOD clip, samples both models at the SAME tracked mask points, projects
their predicted future tracks into the last-observed frame, and writes an HTML
with our-vs-MolmoMotion trail videos. Zero-shot (no GT).
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
from render_future_track_prediction_viewer import _add_label, _draw_trails, _point_colors, _write_video  # noqa: E402
from ood_common import ood_predict  # noqa: E402

DEFAULT_BASE = REPO / "models_pretrained/checkpoints/trackcraft3r/model.safetensors"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-checkpoint", default=str(DEFAULT_BASE))
    ap.add_argument("--proc-dir", default="zero-shot-eval/processed")
    ap.add_argument("--mask-dir", default="zero-shot-eval/masks")
    ap.add_argument("--prep-dir", default="zero-shot-eval/molmo/prep")
    ap.add_argument("--molmo-dir", default="zero-shot-eval/molmo/results")
    ap.add_argument("--out-dir", default="zero-shot-eval/compare")
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


def render_panel(rgb8, uv, val, cols, obs, fut, label, lw, fps, codec, path):
    last = rgb8[obs - 1]; frames = []
    for t in range(obs):
        im = rgb8[t].copy(); _add_label(im, f"observed {t+1}/{obs}"); frames.append(im)
    for k in range(fut):
        frames.append(_draw_trails(last, uv, val, cols, k + 1, f"{label} {k+1}/{fut}", lw))
    _write_video(path, frames, fps, codec)


def main():
    a = parse_args()
    import torch
    out = Path(a.out_dir); (out / "videos").mkdir(parents=True, exist_ok=True)
    model, st = build_model_from_ckpt(Path(a.checkpoint), Path(a.base_checkpoint)); model.eval()
    items = []
    for f in sorted(glob.glob(os.path.join(a.proc_dir, "*_user.npz"))):
        vid = os.path.basename(f).replace("_user.npz", "")
        mres = Path(a.molmo_dir) / f"{vid}.npz"; pres = Path(a.prep_dir) / f"{vid}.npz"
        mask = Path(a.mask_dir) / f"{vid}_mask.png"
        if not (mres.exists() and pres.exists() and mask.exists()):
            print(f"  {vid}: missing molmo/prep/mask, skip"); continue
        r = ood_predict(model, f, mask, grid_stride=2)
        obs, total, mw, mh = r["obs"], r["total"], r["mw"], r["mh"]
        Hd, Wd = np.load(f)["rgb"].shape[1:3]
        molmo = np.load(mres, allow_pickle=True); prep = np.load(pres, allow_pickle=True)
        qxy = molmo["query_xy0"].astype(np.float64)                    # P,2 @ (Hd,Wd) frame0
        xm = np.clip((qxy[:, 0] * mw / Wd).round().astype(int), 0, mw - 1)
        ym = np.clip((qxy[:, 1] * mh / Hd).round().astype(int), 0, mh - 1)
        # ours: sample dense pred at the same points -> cam-lastobs
        ours = r["pred_m"][:, :, ym, xm].transpose(2, 1, 0)            # P,T,3 cam9
        ours_traj = ours[:, obs - 1:total]                            # P,1+fut,3
        # molmo: future in cam-at-t0 (=cam9); anchor at tracked frame-9 pos (cam9)
        w2c9 = prep["w2c"][obs - 1].astype(np.float64)
        a9 = (w2c9[:3, :3] @ prep["pts3d_world"][obs - 1].T).T + w2c9[:3, 3]   # P,3 cam9
        molmo_traj = np.concatenate([a9[:, None], molmo["molmo_camt0"].astype(np.float64)], 1)  # P,1+fut,3
        fut = total - obs
        uvn = np.stack([xm / max(1, mw - 1) * 2 - 1, ym / max(1, mh - 1) * 2 - 1], -1).astype(np.float32)
        cols = _point_colors(uvn)
        uo, vo = project_cam(ours_traj, r["intr_m"], mw, mh)
        um, vm = project_cam(molmo_traj, r["intr_m"], mw, mh)
        ext = a.video_ext
        po = out / "videos" / f"{vid}_ours{ext}"; pm = out / "videos" / f"{vid}_molmo{ext}"
        render_panel(r["rgb8"], uo, vo, cols, obs, fut, "Ours", a.line_width, a.fps, a.video_codec, po)
        render_panel(r["rgb8"], um, vm, cols, obs, fut, "MolmoMotion", a.line_width, a.fps, a.video_codec, pm)
        items.append({"videoId": vid, "action": str(molmo["action"]), "numPoints": int(len(xm)),
                      "obs": obs, "total": total, "ours": f"videos/{po.name}", "molmo": f"videos/{pm.name}"})
        print(f"  {vid}: ours+molmo rendered ({len(xm)} pts)", flush=True)

    css = """:root{color-scheme:dark}body{margin:0;background:#101214;color:#eee;font-family:system-ui,sans-serif}
    header{padding:14px 20px;border-bottom:1px solid #2a3037}main{padding:16px 20px}
    .card{border:1px solid #2a3037;background:#181b1f;border-radius:8px;margin-bottom:16px;overflow:hidden}
    .meta{padding:8px 12px;border-bottom:1px solid #2a3037}.meta h2{font-size:14px;margin:0}
    .v{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#2a3037}.p{background:#090a0b;padding:8px}
    .p h3{font-size:11px;text-transform:uppercase;color:#a6afb8;margin:0 0 6px}video{width:100%;background:#000;border-radius:4px}
    @media(max-width:800px){.v{grid-template-columns:1fr}}"""
    cards = "".join(
        f'<div class="card"><div class="meta"><h2>{html.escape(it["videoId"])} · action: "{html.escape(it["action"])}" · {it["numPoints"]} pts</h2></div>'
        f'<div class="v"><div class="p"><h3>Ours (TrackCraft3r-based, camsub)</h3><video src="{it["ours"]}" controls muted loop autoplay></video></div>'
        f'<div class="p"><h3>MolmoMotion-4B</h3><video src="{it["molmo"]}" controls muted loop autoplay></video></div></div></div>'
        for it in items)
    (out / "index.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>OOD: Ours vs MolmoMotion</title><style>{css}</style></head>'
        f'<body><header><h1>Zero-shot OOD: Ours vs MolmoMotion-4B (epoch {st.get("epoch")})</h1>'
        f'<p>predicted future tracks on the same SAM-mask points, projected into the last observed frame. '
        f'MolmoMotion gets CoTracker 3D history + an action caption; ours gets 10 RGB+depth frames.</p></header>'
        f'<main>{cards}</main></body></html>')
    (out / "manifest.json").write_text(json.dumps({"items": items}, indent=2))
    print(f"wrote {out/'index.html'} ({len(items)} clips)")


if __name__ == "__main__":
    main()
