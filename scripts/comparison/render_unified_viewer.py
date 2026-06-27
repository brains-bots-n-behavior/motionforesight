#!/usr/bin/env python3
"""Eval + HTML viewer for the unified camera-subtracted model.

The model predicts future 3D tracks in the LAST-OBSERVED-frame camera (ego-motion
removed), supervised at SAM-mask points. So we project predictions/GT into the
last-observed image using intrinsics only (no extrinsics), render GT-vs-pred trail
videos, and report ADE/FDE (meters) on the same val split used for training.
"""
from __future__ import annotations
import argparse, html, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import models_pretrained  # noqa: E402,F401
from models_pretrained.future_scene_flow.unified_dataset import (  # noqa: E402
    UnifiedTrackDataset, build_unified_index, split_items)
from render_future_scene_flow_viewer import build_model_from_ckpt  # noqa: E402
from render_future_track_prediction_viewer import _add_label, _draw_trails, _point_colors, _write_video  # noqa: E402

DEFAULT_BASE = REPO / "models_pretrained/checkpoints/trackcraft3r/model.safetensors"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-checkpoint", default=str(DEFAULT_BASE))
    ap.add_argument("--root", default="data/something_something")
    ap.add_argument("--dense-tracks-name", default="anchor_tracks32_curated_dense")
    ap.add_argument("--dense-manifest", default="sam3_anchor_masks/manifest_curated_dense.json")
    ap.add_argument("--sparse-tracks-name", default="anchor_tracks32_next2000_sparse")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--val-count", type=int, default=24)
    ap.add_argument("--val-start", type=int, default=0, help="offset into the val split")
    ap.add_argument("--train-count", type=int, default=0, help="also render this many train clips")
    ap.add_argument("--num-points", type=int, default=64, help="random points per dense clip (viz)")
    ap.add_argument("--grid-stride", type=int, default=0,
                    help=">0: dense pixel grid (every Nth px) inside the mask for dense clips")
    ap.add_argument("--max-points", type=int, default=4000)
    ap.add_argument("--dense-only", action="store_true", help="only render dense clips (needed for grid GT)")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--line-width", type=int, default=2)
    ap.add_argument("--video-ext", default=".webm")
    ap.add_argument("--video-codec", default="VP80")
    return ap.parse_args()


def project_cam(X, intr, W, H):
    """camera-frame XYZ (...,3) -> pixel uv (...,2) + valid mask, pinhole."""
    fx, fy, cx, cy = intr
    z = np.clip(X[..., 2], 1e-6, None)
    u = fx * X[..., 0] / z + cx
    v = fy * X[..., 1] / z + cy
    uv = np.stack([u, v], -1).astype(np.float32)
    valid = (X[..., 2] > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return uv, valid


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    out = Path(a.output_dir).resolve(); (out / "videos").mkdir(parents=True, exist_ok=True)
    model, state = build_model_from_ckpt(Path(a.checkpoint), Path(a.base_checkpoint))
    model.eval()
    cfg = model.config
    obs, total, h, w = cfg.obs_frames, cfg.total_frames, cfg.height, cfg.width

    items = build_unified_index(Path(a.root), a.dense_tracks_name, Path(a.dense_manifest), a.sparse_tracks_name)
    train, val = split_items(items, a.val_fraction, a.seed)
    if a.dense_only:
        train = [it for it in train if it.fmt == "dense"]
        val = [it for it in val if it.fmt == "dense"]
    val_sel = val[a.val_start : a.val_start + a.val_count]
    train_sel = train[: a.train_count]
    sel_items = train_sel + val_sel
    splits = ["train"] * len(train_sel) + ["val"] * len(val_sel)
    ds = UnifiedTrackDataset(sel_items, obs_frames=obs, total_frames=total, image_size=(h, w),
                             num_points=a.num_points, samples_per_clip=1, seed=a.seed + 500000,
                             subtract_camera_motion=True,
                             dense_grid_stride=a.grid_stride, max_points=a.max_points)

    viewer = []
    agg = {"train": [], "val": []}
    for idx in range(len(sel_items)):
        split = splits[idx]
        s = ds[idx]
        dev = model.device
        b = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            q = model(b["rgb_obs"], b["pj_obs"])
            xyz, _ = model.split_latent(q)
            delta = model.decode_xyz(xyz).float()
        p0 = b["p0_t0_norm"].float(); mean = b["pj_mean"].float(); sc = b["pj_scale"].float()
        pred_m = model.denormalize(model.reconstruct(delta, p0), mean, sc)[0].cpu().numpy()  # 3,T,h,w
        # sample at query points
        quv = s["query_uv_model"].numpy(); xs = np.clip(quv[:, 0].round().astype(int), 0, w - 1)
        ys = np.clip(quv[:, 1].round().astype(int), 0, h - 1)
        pred_pts = pred_m[:, :, ys, xs].transpose(2, 1, 0)            # N,T,3 (cam-lastobs)
        gt_pts = (s["gt_tracks_norm"].numpy() * float(sc) + s["pj_mean"].numpy()).transpose(1, 0, 2)  # N,T,3
        vis = s["visibility"].numpy().T                              # N,T
        intr = s["intr_model"].numpy()

        # ADE/FDE (future, vis-masked)
        d = np.linalg.norm(pred_pts[:, obs:] - gt_pts[:, obs:], axis=-1)   # N,Fut
        vm = vis[:, obs:]
        ade = float((d * vm).sum() / (vm.sum() + 1e-6))
        fde = float((d[:, -1] * vis[:, -1]).sum() / (vis[:, -1].sum() + 1e-6))
        agg[split].append((ade, fde))

        # render: anchor both trails at GT last-observed point
        rgb = ((s["rgb_obs"].numpy().transpose(0, 2, 3, 1) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)  # obs,h,w,3
        last_obs = rgb[obs - 1]
        anchor = gt_pts[:, obs - 1:obs]
        gt_xyz = np.concatenate([anchor, gt_pts[:, obs:]], 1)        # N,1+Fut,3
        pred_xyz = np.concatenate([anchor, pred_pts[:, obs:]], 1)
        gt_uv, gt_v = project_cam(gt_xyz, intr, w, h)
        pr_uv, pr_v = project_cam(pred_xyz, intr, w, h)
        uvn = np.stack([xs / max(1, w - 1) * 2 - 1, ys / max(1, h - 1) * 2 - 1], -1).astype(np.float32)
        colors = _point_colors(uvn)
        fut = total - obs
        gtf, prf = [], []
        for t in range(obs):
            f = rgb[t].copy(); _add_label(f, f"observed {t+1}/{obs}"); gtf.append(f.copy()); prf.append(f.copy())
        for k in range(fut):
            gtf.append(_draw_trails(last_obs, gt_uv, gt_v, colors, k + 1, f"GT future {k+1}/{fut}", a.line_width))
            prf.append(_draw_trails(last_obs, pr_uv, pr_v, colors, k + 1,
                                    f"pred future {k+1}/{fut} | ADE {ade*100:.1f}cm", a.line_width))
        ext = a.video_ext
        gp = out / "videos" / f"{s['video_id']}_gt{ext}"; pp = out / "videos" / f"{s['video_id']}_pred{ext}"
        _write_video(gp, gtf, a.fps, a.video_codec); _write_video(pp, prf, a.fps, a.video_codec)
        viewer.append({"videoId": s["video_id"], "title": s["text"], "numPoints": int(len(xs)),
                       "split": split, "ade": ade, "fde": fde,
                       "gtVideo": f"videos/{gp.name}", "predVideo": f"videos/{pp.name}"})
        print(f"[{idx+1}/{len(sel_items)}] {split} {s['video_id']} ade={ade*100:.1f}cm "
              f"fde={fde*100:.1f}cm ({len(xs)} pts)", flush=True)

    def _astr(k):
        if not agg[k]:
            return f"{k}: n/a"
        a_ = np.mean([x[0] for x in agg[k]]) * 100; f_ = np.mean([x[1] for x in agg[k]]) * 100
        return f"{k}: ADE {a_:.2f}cm · FDE {f_:.2f}cm ({len(agg[k])} clips)"
    header = " | ".join(_astr(k) for k in ("train", "val") if agg[k])
    _write_html(out, viewer, header, state.get("epoch"))
    print(f"\n{header}\nwrote {out/'index.html'}")


def _write_html(out, items, agg, epoch):
    css = """:root{color-scheme:dark}body{margin:0;background:#101214;color:#f3f5f7;font-family:system-ui,sans-serif}
    header{position:sticky;top:0;background:rgba(16,18,20,.95);border-bottom:1px solid #2a3037;padding:14px 20px}
    h1{font-size:17px;margin:0 0 6px}header p{margin:0;color:#a6afb8;font-size:13px}main{padding:16px 20px}
    .card{border:1px solid #2a3037;background:#181b1f;border-radius:8px;margin-bottom:16px;overflow:hidden}
    .meta{padding:10px 14px;border-bottom:1px solid #2a3037}.meta h2{font-size:14px;margin:0}.meta p{font-size:12px;color:#a6afb8;margin:4px 0 0}
    .v{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#2a3037}.pane{background:#090a0b;padding:8px}
    .pane h3{font-size:11px;text-transform:uppercase;color:#a6afb8;margin:0 0 6px}video{width:100%;background:#000;border-radius:4px}"""
    cards = []
    for it in sorted(items, key=lambda x: (x.get("split", "val"), x["ade"])):
        tag = it.get("split", "val").upper()
        cards.append(f"""<article class="card"><div class="meta"><h2>[{tag}] {html.escape(it['title'])}</h2>
<p>{html.escape(it['videoId'])} · {it['numPoints']} pts · ADE {it['ade']*100:.1f}cm · FDE {it['fde']*100:.1f}cm</p></div>
<div class="v"><section class="pane"><h3>Ground Truth</h3><video src="{it['gtVideo']}" controls muted loop></video></section>
<section class="pane"><h3>Prediction (camera-subtracted)</h3><video src="{it['predVideo']}" controls muted loop></video></section></div></article>""")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unified camera-subtracted future tracks</title><style>{css}</style></head><body>
<header><h1>Future 3D tracks — unified (dense+sparse), camera-motion subtracted · epoch {epoch}</h1>
<p>{agg} · projected into the last-observed camera (ego-motion removed) · mask points</p></header>
<main>{chr(10).join(cards)}</main></body></html>"""
    (out / "index.html").write_text(doc)
    (out / "manifest.json").write_text(json.dumps({"items": items}, indent=2))


if __name__ == "__main__":
    main()
