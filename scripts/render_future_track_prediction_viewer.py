#!/usr/bin/env python3
"""Render GT-vs-pred future 3D track overlays from a trained predictor."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.future_3d_tracks.dataset import (  # noqa: E402
    DenseTrackDataset,
    build_track_index,
    split_items,
)
from models.future_3d_tracks.model import FutureTrackPredictor, FutureTrackPredictorConfig  # noqa: E402
from models.future_3d_tracks.text_adaln_model import (  # noqa: E402
    TextAdaLNFutureTrackPredictor,
    TextAdaLNFutureTrackPredictorConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/something_something"))
    parser.add_argument("--tracks-name", default="anchor_tracks32_500")
    parser.add_argument("--manifest", type=Path, default=Path("sam3_anchor_masks/manifest_500.json"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/something_something/future_track_training/full_10f_to_32f/best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/something_something/future_track_prediction_viewer"),
    )
    parser.add_argument("--train-count", type=int, default=20)
    parser.add_argument("--val-count", type=int, default=0, help="Validation clips to render. 0 means all validation clips.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--obs-frames", type=int, default=None)
    parser.add_argument("--total-frames", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--text-dim", type=int, default=None)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--video-ext", default=".webm", help="Browser-facing video extension.")
    parser.add_argument("--video-codec", default="VP80", help="OpenCV fourcc used for rendered videos.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _read_run_config(checkpoint: Path) -> dict:
    config_path = checkpoint.parent / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}


def _scale_intrinsics_for_model(intrinsics: np.ndarray, orig_w: int, orig_h: int, model_w: int, model_h: int) -> np.ndarray:
    fx, fy, cx, cy = intrinsics.astype(np.float64)
    return np.array(
        [
            fx * (model_w / orig_w),
            fy * (model_h / orig_h),
            cx * (model_w / orig_w),
            cy * (model_h / orig_h),
        ],
        dtype=np.float64,
    )


def _project(points_cam0: np.ndarray, w2c: np.ndarray, intr: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    flat = points_cam0.reshape(-1, 3).astype(np.float64)
    ones = np.ones((flat.shape[0], 1), dtype=np.float64)
    cam = (w2c @ np.concatenate([flat, ones], axis=1).T).T[:, :3]
    z = cam[:, 2]
    fx, fy, cx, cy = intr
    uv = np.empty((flat.shape[0], 2), dtype=np.float64)
    uv[:, 0] = cam[:, 0] / np.maximum(z, 1e-8) * fx + cx
    uv[:, 1] = cam[:, 1] / np.maximum(z, 1e-8) * fy + cy
    valid = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv.reshape(*points_cam0.shape[:-1], 2).astype(np.float32), valid.reshape(points_cam0.shape[:-1])


def _point_colors(point_uv: np.ndarray) -> np.ndarray:
    y = np.clip((point_uv[:, 1] + 1.0) * 0.5, 0.0, 1.0)[:, None]
    top = np.array([20, 90, 255], dtype=np.float32)
    bottom = np.array([255, 45, 20], dtype=np.float32)
    return (top[None] * (1.0 - y) + bottom[None] * y).astype(np.uint8)


def _add_label(frame: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (12, 12), (30 + tw, 40 + th), (0, 0, 0), -1)
    cv2.putText(frame, text, (22, 34 + th // 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_trails(
    background: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    colors: np.ndarray,
    end_step: int,
    label: str,
    line_width: int,
) -> np.ndarray:
    frame = background.copy()
    max_step = min(end_step, uv.shape[1] - 1)
    for point_idx in range(uv.shape[0]):
        color = tuple(int(c) for c in colors[point_idx])
        for t in range(1, max_step + 1):
            if not (valid[point_idx, t - 1] and valid[point_idx, t]):
                continue
            p0 = tuple(np.round(uv[point_idx, t - 1]).astype(int))
            p1 = tuple(np.round(uv[point_idx, t]).astype(int))
            cv2.line(frame, p0, p1, color, line_width, cv2.LINE_AA)
        if valid[point_idx, max_step]:
            head = tuple(np.round(uv[point_idx, max_step]).astype(int))
            cv2.circle(frame, head, max(2, line_width + 1), color, -1, cv2.LINE_AA)
    _add_label(frame, label)
    return frame


def _write_video(path: Path, frames_rgb: list[np.ndarray], fps: float, codec: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path} with codec {codec}")
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _render_pair(
    item,
    sample: dict,
    pred_norm: np.ndarray,
    out_dir: Path,
    split_name: str,
    obs_frames: int,
    total_frames: int,
    fps: float,
    line_width: int,
    video_ext: str,
    video_codec: str,
    force: bool,
) -> dict:
    uid = item.video_id
    if not video_ext.startswith("."):
        video_ext = f".{video_ext}"
    gt_path = out_dir / "videos" / f"{split_name}_{uid}_gt{video_ext}"
    pred_path = out_dir / "videos" / f"{split_name}_{uid}_pred{video_ext}"
    if gt_path.exists() and pred_path.exists() and not force:
        return {"gtVideo": f"videos/{gt_path.name}", "predVideo": f"videos/{pred_path.name}"}

    dense = np.load(item.dense_path)
    rgb = dense["rgb"][:total_frames].astype(np.uint8)
    height, width = rgb.shape[1:3]

    user_npz = item.dense_path.with_name(item.dense_path.name.replace("_dense.npz", "_user.npz"))
    user = np.load(user_npz, allow_pickle=True)
    orig_h, orig_w = user["depth_map"].shape[1:3]
    intr = _scale_intrinsics_for_model(user["fx_fy_cx_cy"], orig_w, orig_h, width, height)
    w2c = user["extrinsics_w2c"][obs_frames - 1]

    center = sample["track_center"].numpy()
    scale = float(sample["track_scale"].item())
    observed = sample["observed_tracks"].numpy() * scale + center[None, None, :]
    gt_future = sample["future_tracks"].numpy() * scale + center[None, None, :]
    pred_future = pred_norm * scale + center[None, None, :]

    gt_xyz = np.concatenate([observed[:, -1:, :], gt_future], axis=1)
    pred_xyz = np.concatenate([observed[:, -1:, :], pred_future], axis=1)
    gt_uv, gt_valid = _project(gt_xyz, w2c, intr, width, height)
    pred_uv, pred_valid = _project(pred_xyz, w2c, intr, width, height)
    colors = _point_colors(sample["point_uv"].numpy())
    last_observed = rgb[obs_frames - 1]

    gt_frames: list[np.ndarray] = []
    pred_frames: list[np.ndarray] = []
    for t in range(obs_frames):
        obs = rgb[t].copy()
        _add_label(obs, f"observed {t + 1}/{obs_frames}")
        gt_frames.append(obs.copy())
        pred_frames.append(obs.copy())
    future_frames = total_frames - obs_frames
    for k in range(future_frames):
        gt_frames.append(
            _draw_trails(
                last_observed,
                gt_uv,
                gt_valid,
                colors,
                k + 1,
                f"GT future {k + 1}/{future_frames}",
                line_width,
            )
        )
        pred_frames.append(
            _draw_trails(
                last_observed,
                pred_uv,
                pred_valid,
                colors,
                k + 1,
                f"prediction future {k + 1}/{future_frames}",
                line_width,
            )
        )

    _write_video(gt_path, gt_frames, fps, video_codec)
    _write_video(pred_path, pred_frames, fps, video_codec)
    return {"gtVideo": f"videos/{gt_path.name}", "predVideo": f"videos/{pred_path.name}"}


def _write_html(out_dir: Path, items: list[dict]) -> None:
    css = """
    :root{color-scheme:dark;--bg:#101214;--panel:#181b1f;--text:#f3f5f7;--muted:#a6afb8;--line:#2a3037;--accent:#65b8ff}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,sans-serif}
    header{position:sticky;top:0;z-index:5;background:rgba(16,18,20,.94);border-bottom:1px solid var(--line);padding:14px 20px}
    h1{font-size:18px;margin:0 0 6px}header p{margin:0;color:var(--muted);font-size:13px}
    main{padding:18px 20px 34px}.grid{display:grid;grid-template-columns:1fr;gap:18px}.card{border:1px solid var(--line);background:var(--panel);border-radius:8px;overflow:hidden}
    .meta{display:flex;gap:12px;align-items:baseline;justify-content:space-between;padding:12px 14px;border-bottom:1px solid var(--line)}
    .meta h2{font-size:15px;line-height:1.3;margin:0}.meta p{font-size:12px;color:var(--muted);margin:4px 0 0}.tag{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--accent)}
    .videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--line)}.pane{background:#090a0b;padding:10px}.pane h3{font-size:12px;letter-spacing:.06em;text-transform:uppercase;margin:0 0 8px;color:var(--muted)}
    video{display:block;width:100%;aspect-ratio:832/480;background:#000;border-radius:4px}@media(max-width:820px){.videos{grid-template-columns:1fr}.meta{display:block}}
    """
    cards = []
    for item in items:
        title = html.escape(item["title"])
        cards.append(
            f"""<article class="card">
  <div class="meta"><div><span class="tag">{html.escape(item['split'])}</span><h2>{title}</h2><p>{html.escape(item['videoId'])} · {item['numPoints']} points · observed {item['obsFrames']} / total {item['totalFrames']} frames</p></div></div>
  <div class="videos">
    <section class="pane"><h3>Ground Truth</h3><video src="{html.escape(item['gtVideo'])}" controls muted loop preload="metadata"></video></section>
    <section class="pane"><h3>Prediction</h3><video src="{html.escape(item['predVideo'])}" controls muted loop preload="metadata"></video></section>
  </div>
</article>"""
        )
    body = "\n".join(cards)
    doc = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Future 3D Track Prediction Viewer</title><style>{css}</style></head>
<body><header><h1>Future 3D Track Predictions</h1><p>First 10 RGB frames, then future trajectories overlaid on observed frame 10. Ground truth and predictions are projected into the same last-observed camera.</p></header><main><section class="grid">{body}</section></main></body>
</html>
"""
    (out_dir / "index.html").write_text(doc)
    (out_dir / "manifest.json").write_text(json.dumps({"items": items}, indent=2))


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = _read_run_config(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model_variant = ckpt.get("model_variant", run_config.get("model_variant", "baseline"))
    if model_variant == "text-adaln":
        model_config = TextAdaLNFutureTrackPredictorConfig(**ckpt["config"])
        text_vocab = ckpt.get("text_vocab") or run_config.get("text_vocab")
        if not text_vocab:
            raise SystemExit("text-adaln checkpoint is missing text_vocab")
    else:
        model_config = FutureTrackPredictorConfig(**ckpt["config"])
        text_vocab = None
    seed = args.seed if args.seed is not None else int(run_config.get("seed", 7))
    val_fraction = args.val_fraction if args.val_fraction is not None else float(run_config.get("val_fraction", 0.1))
    obs_frames = args.obs_frames if args.obs_frames is not None else int(run_config.get("obs_frames", model_config.obs_frames))
    total_frames = args.total_frames if args.total_frames is not None else int(run_config.get("total_frames", obs_frames + model_config.future_frames))
    num_points = args.num_points if args.num_points is not None else int(run_config.get("num_points", 256))
    image_size = tuple(args.image_size or run_config.get("image_size", [128, 224]))
    text_dim = args.text_dim if args.text_dim is not None else int(run_config.get("text_dim", getattr(model_config, "text_dim", 256)))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    model = TextAdaLNFutureTrackPredictor(model_config) if model_variant == "text-adaln" else FutureTrackPredictor(model_config)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    items = build_track_index(root, tracks_name=args.tracks_name, manifest=args.manifest, require_masks=True)
    train_items, val_items = split_items(items, val_fraction, seed)
    selected = [("train", item) for item in train_items[: args.train_count]]
    selected_val = val_items if args.val_count <= 0 else val_items[: args.val_count]
    selected.extend(("val", item) for item in selected_val)
    selected_items = [item for _, item in selected]
    dataset = DenseTrackDataset(
        selected_items,
        obs_frames=obs_frames,
        total_frames=total_frames,
        num_points=num_points,
        image_size=image_size,
        text_dim=text_dim,
        text_vocab=text_vocab,
        max_text_tokens=getattr(model_config, "max_text_tokens", 16),
        samples_per_clip=1,
        seed=seed + 500_000,
    )

    viewer_items: list[dict] = []
    with torch.no_grad():
        for idx, (split_name, item) in enumerate(selected):
            sample = dataset[idx]
            batch = {
                "frames": sample["frames"].unsqueeze(0).to(device),
                "observed_tracks": sample["observed_tracks"].unsqueeze(0).to(device),
                "point_uv": sample["point_uv"].unsqueeze(0).to(device),
            }
            if model_variant == "text-adaln":
                batch["text_tokens"] = sample["text_tokens"].unsqueeze(0).to(device)
                batch["text_mask"] = sample["text_mask"].unsqueeze(0).to(device)
            else:
                batch["text_bow"] = sample["text_bow"].unsqueeze(0).to(device)
            pred = model(**batch).squeeze(0).detach().cpu().numpy()
            videos = _render_pair(
                item,
                sample,
                pred,
                out_dir,
                split_name,
                obs_frames,
                total_frames,
                args.fps,
                args.line_width,
                args.video_ext,
                args.video_codec,
                args.force,
            )
            viewer_items.append(
                {
                    "split": split_name,
                    "videoId": item.video_id,
                    "title": item.text,
                    "numPoints": num_points,
                    "obsFrames": obs_frames,
                    "totalFrames": total_frames,
                    **videos,
                }
            )
            print(f"[{idx + 1:02d}/{len(selected)}] {split_name} {item.video_id}")

    _write_html(out_dir, viewer_items)
    print(f"wrote {out_dir / 'index.html'} ({len(viewer_items)} clips)")


if __name__ == "__main__":
    main()
