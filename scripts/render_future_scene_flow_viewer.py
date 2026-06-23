#!/usr/bin/env python3
"""Eval + HTML viewer for the pretrained-TrackCraft3r future scene-flow models.

Sibling of ``render_future_track_prediction_viewer.py`` (the from-scratch
viewer).  Loads a checkpoint from ``train_future_scene_flow.py`` or
``train_future_scene_flow_freshlora.py``, predicts future 3D tracks, samples
object-mask points, projects GT vs prediction into the last-observed camera,
renders side-by-side trail videos, computes ADE/FDE (meters), and writes an
``index.html`` with per-clip + aggregate metrics.

Example::

    CUDA_VISIBLE_DEVICES=1 python scripts/render_future_scene_flow_viewer.py \
      --checkpoint data/.../pretrained_tc3r_freshlora_10f_to_32f/best.pt \
      --output-dir data/something_something/future_scene_flow_viewer_freshlora \
      --train-count 12 --val-count 0
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import models_pretrained  # noqa: E402,F401  (vendored-import isolation)
from models_pretrained.future_scene_flow import FutureSceneFlowConfig, FutureSceneFlowModel  # noqa: E402
from models_pretrained.future_scene_flow.model_fresh_lora import (  # noqa: E402
    FreshLoRAConfig,
    FreshLoRAFutureSceneFlowModel,
)
from models_pretrained.future_scene_flow.dataset import (  # noqa: E402
    FutureSceneFlowDataset,
    build_track_index,
    split_items,
)
# Reuse the rendering helpers from the from-scratch viewer.
from render_future_track_prediction_viewer import (  # noqa: E402
    _add_label,
    _draw_trails,
    _point_colors,
    _project,
    _write_video,
)

DEFAULT_CKPT = REPO_ROOT / "models_pretrained" / "checkpoints" / "trackcraft3r" / "model.safetensors"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/something_something"))
    p.add_argument("--tracks-name", default="anchor_tracks32_curated_dense")
    p.add_argument("--manifest", type=Path, default=Path("sam3_anchor_masks/manifest_curated_dense.json"))
    p.add_argument("--checkpoint", type=Path, required=True, help="best.pt/last.pt from a future-scene-flow run.")
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--output-dir", type=Path, default=Path("data/something_something/future_scene_flow_viewer"))
    p.add_argument("--train-count", type=int, default=12)
    p.add_argument("--val-count", type=int, default=0, help="Val clips to render (0 = all).")
    p.add_argument("--num-points", type=int, default=48)
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--line-width", type=int, default=2)
    p.add_argument("--video-ext", default=".webm")
    p.add_argument("--video-codec", default="VP80")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def build_model_from_ckpt(checkpoint: Path, base_ckpt: Path) -> FutureSceneFlowModel:
    """Reconstruct the right model variant and load its fine-tuned params."""
    # Copy first to avoid racing a live training job mid-write.
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        shutil.copy(checkpoint, tmp.name)
        tmp_path = tmp.name
    state = torch.load(tmp_path, map_location="cpu", weights_only=False)
    Path(tmp_path).unlink(missing_ok=True)

    c = state["config"]
    is_fresh = "fresh_lora_rank" in c
    common = dict(
        checkpoint_path=str(base_ckpt.expanduser().resolve()),
        lora_rank=c.get("lora_rank", 1024),
        height=c["height"], width=c["width"],
        obs_frames=c["obs_frames"], total_frames=c["total_frames"],
        trainable=tuple(c.get("trainable", ["mask", "io", "head"])),
        predict_vis=c.get("predict_vis", True),
    )
    if is_fresh:
        cfg = FreshLoRAConfig(
            **common,
            fresh_lora_rank=c["fresh_lora_rank"],
            fresh_lora_alpha=c.get("fresh_lora_alpha", c["fresh_lora_rank"]),
            fresh_lora_scope=c.get("fresh_lora_scope", "attn"),
        )
        model = FreshLoRAFutureSceneFlowModel(cfg)
    else:
        cfg = FutureSceneFlowConfig(**common, lora_scope=c.get("lora_scope", "attn"),
                                    lora_last_n=c.get("lora_last_n", 0))
        model = FutureSceneFlowModel(cfg)
    missing, unexpected = model.load_state_dict(state["trainable_state"], strict=False)
    print(f"loaded {len(state['trainable_state'])} fine-tuned tensors "
          f"(variant={'fresh' if is_fresh else 'lora'}, unexpected={len(unexpected)}, "
          f"epoch={state.get('epoch')})")
    model.eval()
    return model, state


@torch.no_grad()
def predict_dense(model, sample) -> np.ndarray:
    """Return predicted dense track in metric cam-0 space: (3, T, h, w)."""
    dev = model.device
    rgb = sample["rgb_obs"].unsqueeze(0).to(dev)
    pj = sample["pj_obs"].unsqueeze(0).to(dev)
    p0 = sample["p0_t0_norm"].unsqueeze(0).to(dev).float()
    mean = sample["pj_mean"].unsqueeze(0).to(dev).float()
    scale = sample["pj_scale"].unsqueeze(0).to(dev).float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q = model(rgb, pj)
        xyz, _ = model.split_latent(q)
        delta = model.decode_xyz(xyz).float()
    pred_norm = model.reconstruct(delta, p0)
    pred_m = model.denormalize(pred_norm, mean, scale)
    return pred_m[0].cpu().numpy()  # 3,T,h,w


def _sample_points(obj_mask, valid_all, num_points, rng):
    cand = obj_mask & valid_all
    if cand.sum() < 8:
        cand = valid_all
    ys, xs = np.where(cand)
    if ys.size == 0:
        return None, None
    sel = rng.choice(ys.size, size=min(num_points, ys.size), replace=False)
    return ys[sel], xs[sel]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    out_dir = args.output_dir.expanduser().resolve()
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)

    model, state = build_model_from_ckpt(args.checkpoint, args.base_checkpoint)
    cfg = model.config
    obs, total = cfg.obs_frames, cfg.total_frames
    h, w = cfg.height, cfg.width

    items = build_track_index(args.root.expanduser().resolve(), tracks_name=args.tracks_name,
                              manifest=args.manifest, require_masks=True)
    train_items, val_items = split_items(items, 0.1, args.seed)
    selected = [("train", it) for it in train_items[: args.train_count]]
    sel_val = val_items if args.val_count <= 0 else val_items[: args.val_count]
    selected += [("val", it) for it in sel_val]
    sel_items = [it for _, it in selected]

    ds = FutureSceneFlowDataset(sel_items, obs_frames=obs, total_frames=total,
                                image_size=(h, w), samples_per_clip=1, seed=args.seed + 500_000)
    rng = np.random.default_rng(args.seed)

    viewer_items: list[dict] = []
    agg = {"train": [], "val": []}
    for idx, (split_name, item) in enumerate(selected):
        sample = ds[idx]
        pred_m = predict_dense(model, sample)                                  # 3,T,h,w
        mean = sample["pj_mean"].numpy(); scale = float(sample["pj_scale"])
        gt_m = (sample["track_norm"].numpy().transpose(1, 0, 2, 3) * scale
                + mean[:, None, None, None])                                   # 3,T,h,w
        valid = sample["valid"].numpy()                                        # T,h,w
        obj = sample["obj_mask"].numpy()                                       # h,w

        ys, xs = _sample_points(obj, valid.all(axis=0), args.num_points, rng)
        if ys is None:
            print(f"[skip] {item.video_id}: no valid points")
            continue
        gt_pts = gt_m[:, :, ys, xs].transpose(2, 1, 0)                         # K,T,3
        pred_pts = pred_m[:, :, ys, xs].transpose(2, 1, 0)                     # K,T,3

        # ADE/FDE in meters on future frames.
        d = np.linalg.norm(pred_pts[:, obs:] - gt_pts[:, obs:], axis=-1)       # K, future
        ade = float(d.mean()); fde = float(d[:, -1].mean())
        agg[split_name].append((ade, fde))

        # Render: anchor both trails at the GT last-observed point.
        dense = np.load(item.dense_path)
        rgb = dense["rgb"][:total].astype(np.uint8)
        rh, rw = rgb.shape[1:3]
        user = np.load(item.dense_path.with_name(item.dense_path.name.replace("_dense.npz", "_user.npz")),
                       allow_pickle=True)
        oh, ow = user["depth_map"].shape[1:3]
        fx, fy, cx, cy = user["fx_fy_cx_cy"].astype(np.float64)
        intr = np.array([fx * rw / ow, fy * rh / oh, cx * rw / ow, cy * rh / oh])
        w2c = user["extrinsics_w2c"][obs - 1]

        anchor = gt_pts[:, obs - 1:obs, :]
        gt_xyz = np.concatenate([anchor, gt_pts[:, obs:, :]], axis=1)
        pred_xyz = np.concatenate([anchor, pred_pts[:, obs:, :]], axis=1)
        gt_uv, gt_v = _project(gt_xyz, w2c, intr, rw, rh)
        pred_uv, pred_v = _project(pred_xyz, w2c, intr, rw, rh)
        uv_norm = np.stack([xs / max(1, w - 1) * 2 - 1, ys / max(1, h - 1) * 2 - 1], axis=-1)
        colors = _point_colors(uv_norm.astype(np.float32))
        last_obs = rgb[obs - 1]
        future = total - obs

        gt_frames, pred_frames = [], []
        for t in range(obs):
            f = rgb[t].copy(); _add_label(f, f"observed {t+1}/{obs}")
            gt_frames.append(f.copy()); pred_frames.append(f.copy())
        for k in range(future):
            gt_frames.append(_draw_trails(last_obs, gt_uv, gt_v, colors, k + 1,
                                          f"GT future {k+1}/{future}", args.line_width))
            pred_frames.append(_draw_trails(last_obs, pred_uv, pred_v, colors, k + 1,
                                            f"pred future {k+1}/{future} | ADE {ade*100:.1f}cm",
                                            args.line_width))
        ext = args.video_ext if args.video_ext.startswith(".") else f".{args.video_ext}"
        gt_path = out_dir / "videos" / f"{split_name}_{item.video_id}_gt{ext}"
        pred_path = out_dir / "videos" / f"{split_name}_{item.video_id}_pred{ext}"
        _write_video(gt_path, gt_frames, args.fps, args.video_codec)
        _write_video(pred_path, pred_frames, args.fps, args.video_codec)

        viewer_items.append({
            "split": split_name, "videoId": item.video_id, "title": item.text,
            "numPoints": int(ys.size), "obsFrames": obs, "totalFrames": total,
            "adeM": ade, "fdeM": fde,
            "gtVideo": f"videos/{gt_path.name}", "predVideo": f"videos/{pred_path.name}",
        })
        print(f"[{idx+1:02d}/{len(selected)}] {split_name} {item.video_id} ade={ade*100:.1f}cm fde={fde*100:.1f}cm")

    _write_html(out_dir, viewer_items, agg, state)
    print(f"wrote {out_dir/'index.html'} ({len(viewer_items)} clips)")
    for sp in ("train", "val"):
        if agg[sp]:
            a = np.mean([x[0] for x in agg[sp]]); f = np.mean([x[1] for x in agg[sp]])
            print(f"  {sp}: ADE={a*100:.2f}cm FDE={f*100:.2f}cm over {len(agg[sp])} clips")


def _write_html(out_dir, items, agg, state):
    def _agg(sp):
        if not agg[sp]:
            return "n/a"
        a = np.mean([x[0] for x in agg[sp]]) * 100
        f = np.mean([x[1] for x in agg[sp]]) * 100
        return f"ADE {a:.2f}cm · FDE {f:.2f}cm ({len(agg[sp])} clips)"

    css = """
    :root{color-scheme:dark;--bg:#101214;--panel:#181b1f;--text:#f3f5f7;--muted:#a6afb8;--line:#2a3037;--accent:#65b8ff}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,sans-serif}
    header{position:sticky;top:0;z-index:5;background:rgba(16,18,20,.94);border-bottom:1px solid var(--line);padding:14px 20px}
    h1{font-size:18px;margin:0 0 6px}header p{margin:0;color:var(--muted);font-size:13px}
    main{padding:18px 20px 34px}.grid{display:grid;grid-template-columns:1fr;gap:18px}.card{border:1px solid var(--line);background:var(--panel);border-radius:8px;overflow:hidden}
    .meta{display:flex;gap:12px;align-items:baseline;justify-content:space-between;padding:12px 14px;border-bottom:1px solid var(--line)}
    .meta h2{font-size:15px;line-height:1.3;margin:0}.meta p{font-size:12px;color:var(--muted);margin:4px 0 0}.tag{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--accent)}
    .videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--line)}.pane{background:#090a0b;padding:10px}.pane h3{font-size:12px;letter-spacing:.06em;text-transform:uppercase;margin:0 0 8px;color:var(--muted)}
    video{display:block;width:100%;background:#000;border-radius:4px}@media(max-width:820px){.videos{grid-template-columns:1fr}.meta{display:block}}
    """
    cards = []
    for it in sorted(items, key=lambda x: (x["split"], x["adeM"])):
        cards.append(
            f"""<article class="card">
  <div class="meta"><div><span class="tag">{html.escape(it['split'])}</span><h2>{html.escape(it['title'])}</h2>
  <p>{html.escape(it['videoId'])} · {it['numPoints']} pts · obs {it['obsFrames']}/{it['totalFrames']} · ADE {it['adeM']*100:.1f}cm · FDE {it['fdeM']*100:.1f}cm</p></div></div>
  <div class="videos">
    <section class="pane"><h3>Ground Truth</h3><video src="{html.escape(it['gtVideo'])}" controls muted loop preload="metadata"></video></section>
    <section class="pane"><h3>Prediction</h3><video src="{html.escape(it['predVideo'])}" controls muted loop preload="metadata"></video></section>
  </div>
</article>""")
    epoch = state.get("epoch")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Future Scene-Flow (pretrained TrackCraft3r)</title><style>{css}</style></head>
<body><header><h1>Future 3D Scene-Flow — pretrained TrackCraft3r</h1>
<p>checkpoint epoch {epoch} · <b>train</b> {_agg('train')} · <b>val</b> {_agg('val')} · future-frame ADE/FDE in cam-0 meters</p></header>
<main><section class="grid">{chr(10).join(cards)}</section></main></body></html>"""
    (out_dir / "index.html").write_text(doc)
    (out_dir / "manifest.json").write_text(json.dumps({"items": items}, indent=2))


if __name__ == "__main__":
    main()
