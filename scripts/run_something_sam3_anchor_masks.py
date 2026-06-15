#!/usr/bin/env python3
"""Create SAM3 anchor-frame masks for Something-Something videos.

The anchor frame is chosen by scanning for the first frame where SAM3 finds a
hand mask, then moving a few frames forward. The script writes an anchor-starting
MP4 clip so downstream TrackCraft3r inference can still start at frame 0.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


PALETTE = [
    (255, 64, 64),
    (64, 190, 255),
    (96, 230, 96),
    (255, 195, 64),
    (190, 96, 255),
    (64, 230, 190),
]
HAND_COLOR = (70, 160, 255)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/something_something"))
    p.add_argument(
        "--video-root",
        type=Path,
        default=Path("~/Downloads/20bn-something-something-v2"),
        help="Directory containing <id>.webm Something-Something videos.",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("~/Downloads/20bn-something-something-download-package-labels/labels/train.json"),
    )
    p.add_argument("--sam3-root", type=Path, default=Path("external/sam3"))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--confidence-threshold", type=float, default=0.25)
    p.add_argument("--hand-prompt", default="hand")
    p.add_argument("--scan-step", type=int, default=2)
    p.add_argument("--anchor-offset", type=int, default=3)
    p.add_argument("--min-hand-area", type=int, default=150)
    p.add_argument("--min-frames-after-anchor", type=int, default=20)
    p.add_argument("--clip-max-frames", type=int, default=48)
    p.add_argument("--limit", type=int, default=4, help="Maximum candidate videos to scan. Use 0 for all.")
    p.add_argument("--start-index", type=int, default=0, help="Skip this many matching candidates before processing.")
    p.add_argument("--num-shards", type=int, default=1, help="Split matching candidates into this many shards.")
    p.add_argument("--shard-index", type=int, default=0, help="Process only this shard index.")
    p.add_argument("--manifest-name", default="manifest.json")
    p.add_argument("--selected-list-name", default="anchor_selected_videos.txt")
    p.add_argument("--no-viewer", action="store_true", help="Skip writing the HTML mask viewer.")
    p.add_argument("--video-id", action="append", default=[])
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value[:140] or "clip"


def relpath(path: Path, start: Path) -> str:
    return path.resolve().relative_to(start.resolve()).as_posix()


def alpha_blend(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = base.astype(np.float32).copy()
    out[mask] = out[mask] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"could not read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 12.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def sam_masks(processor: Sam3Processor, frame_rgb: np.ndarray, prompt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.inference_mode():
        state = processor.set_image(Image.fromarray(frame_rgb))
        state = processor.set_text_prompt(prompt=prompt, state=state)
    masks = state["masks"].detach().cpu().numpy()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    boxes = state["boxes"].detach().cpu().numpy()
    scores = state["scores"].detach().cpu().numpy()
    return masks.astype(bool), boxes, scores


def find_hand_anchor(
    processor: Sam3Processor,
    video_path: Path,
    total_frames: int,
    scan_step: int,
    anchor_offset: int,
    min_hand_area: int,
    min_frames_after_anchor: int,
    hand_prompt: str,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    last_start = max(0, total_frames - min_frames_after_anchor)
    for frame_index in range(0, max(1, last_start + 1), max(1, scan_step)):
        frame_rgb = read_frame(video_path, frame_index)
        masks, boxes, scores = sam_masks(processor, frame_rgb, hand_prompt)
        areas = np.array([int(m.sum()) for m in masks], dtype=np.int64)
        keep = np.where(areas >= min_hand_area)[0]
        if len(keep) == 0:
            continue
        best = int(keep[np.argmax(scores[keep])])
        anchor = min(frame_index + anchor_offset, last_start)
        anchor_rgb = read_frame(video_path, anchor)
        return anchor, frame_index, anchor_rgb, masks[[best]], boxes[[best]], scores[[best]]
    raise RuntimeError("no SAM3 hand mask found before the usable tracking window")


def write_anchor_clip(src: Path, dst: Path, start_frame: int, max_frames: int, fps: float) -> int:
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError(f"could not read anchor frame {start_frame} from {src}")
    h, w = first.shape[:2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 12.0,
        (w, h),
    )
    count = 0
    frame = first
    while frame is not None and count < max_frames:
        writer.write(frame)
        count += 1
        ok, frame = cap.read()
        if not ok:
            break
    writer.release()
    cap.release()
    if count == 0:
        raise RuntimeError(f"wrote zero frames to {dst}")
    return count


def choose_items(
    labels: list[dict],
    video_root: Path,
    video_ids: list[str],
    limit: int,
    min_frames: int,
    start_index: int,
    num_shards: int,
    shard_index: int,
) -> list[dict]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= shard_index < num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < --num-shards")
    wanted = set(video_ids)
    chosen = []
    match_index = 0
    for item in labels:
        if wanted and item["id"] not in wanted:
            continue
        if not item.get("placeholders"):
            continue
        video_path = video_root / f"{item['id']}.webm"
        if not video_path.exists():
            continue
        try:
            info = video_info(video_path)
        except RuntimeError:
            continue
        if info["frames"] < min_frames:
            continue
        if match_index < start_index:
            match_index += 1
            continue
        if not wanted and match_index % num_shards != shard_index:
            match_index += 1
            continue
        item = dict(item)
        item["video_path"] = str(video_path)
        item["video_info"] = info
        chosen.append(item)
        match_index += 1
        if not wanted and limit > 0 and len(chosen) >= limit:
            break
    return chosen


def run_one(
    processor: Sam3Processor,
    item: dict,
    root: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    video_id = item["id"]
    source = Path(item["video_path"])
    info = item["video_info"]
    clip_stem = slugify(f"{video_id}_anchor")
    clip_dir = out_dir / "clips" / clip_stem
    summary_path = clip_dir / "summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text())

    anchor_frame, hand_frame, anchor_rgb, hand_masks, hand_boxes, hand_scores = find_hand_anchor(
        processor=processor,
        video_path=source,
        total_frames=info["frames"],
        scan_step=args.scan_step,
        anchor_offset=args.anchor_offset,
        min_hand_area=args.min_hand_area,
        min_frames_after_anchor=args.min_frames_after_anchor,
        hand_prompt=args.hand_prompt,
    )

    anchor_clip = root / "anchor_clips" / f"{clip_stem}.mp4"
    clip_frames = write_anchor_clip(
        src=source,
        dst=anchor_clip,
        start_frame=anchor_frame,
        max_frames=args.clip_max_frames,
        fps=info["fps"],
    )

    clip_dir.mkdir(parents=True, exist_ok=True)
    anchor_frame_path = clip_dir / "anchor_frame.jpg"
    save_rgb(anchor_frame_path, anchor_rgb)

    overlay = anchor_rgb.copy()
    hand_instances = []
    for idx, mask in enumerate(hand_masks):
        mask_path = clip_dir / f"hand_mask_{idx:02d}.png"
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        overlay = alpha_blend(overlay, mask, HAND_COLOR, alpha=0.32)
        hand_instances.append(
            {
                "index": idx,
                "mask_path": relpath(mask_path, root),
                "score": float(hand_scores[idx]),
                "box_xyxy": [float(v) for v in hand_boxes[idx].tolist()],
                "mask_area": int(mask.sum()),
            }
        )

    object_instances = []
    for prompt in item.get("placeholders", []):
        masks, boxes, scores = sam_masks(processor, anchor_rgb, prompt)
        for idx, mask in enumerate(masks):
            if int(mask.sum()) < 20:
                continue
            mask_path = clip_dir / f"mask_{len(object_instances):02d}.png"
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            color = PALETTE[len(object_instances) % len(PALETTE)]
            overlay = alpha_blend(overlay, mask, color, alpha=0.45)
            x0, y0, x1, y1 = boxes[idx].tolist()
            cv2.rectangle(
                overlay,
                (int(round(x0)), int(round(y0))),
                (int(round(x1)), int(round(y1))),
                color,
                2,
            )
            cv2.putText(
                overlay,
                f"{prompt[:14]} {float(scores[idx]):.2f}",
                (int(round(x0)), max(18, int(round(y0)) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )
            object_instances.append(
                {
                    "index": len(object_instances),
                    "prompt": prompt,
                    "mask_path": relpath(mask_path, root),
                    "score": float(scores[idx]),
                    "box_xyxy": [float(v) for v in boxes[idx].tolist()],
                    "mask_area": int(mask.sum()),
                }
            )

    overlay_path = clip_dir / "overlay.png"
    save_rgb(overlay_path, overlay)
    summary = {
        "video_uid": video_id,
        "video_title": item.get("label", ""),
        "label": item.get("label", ""),
        "template": item.get("template", ""),
        "placeholders": item.get("placeholders", []),
        "clip_path": relpath(anchor_clip, root),
        "source_video_path": str(source),
        "frame_path": relpath(anchor_frame_path, root),
        "overlay_path": relpath(overlay_path, root),
        "prompt": ", ".join(item.get("placeholders", [])),
        "segment_text": item.get("label", ""),
        "source_total_frames": info["frames"],
        "source_fps": info["fps"],
        "hand_prompt": args.hand_prompt,
        "first_hand_frame": hand_frame,
        "anchor_frame": anchor_frame,
        "anchor_offset": args.anchor_offset,
        "anchor_clip_frames": clip_frames,
        "num_instances": len(object_instances),
        "instances": object_instances,
        "hand_instances": hand_instances,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def build_viewer(out_dir: Path, root: Path, summaries: list[dict]) -> None:
    cards = []
    for item in summaries:
        title = html.escape(item.get("label") or item.get("video_uid") or "video")
        overlay = html.escape(Path("..", item["overlay_path"]).as_posix())
        frame = html.escape(Path("..", item["frame_path"]).as_posix())
        prompts = html.escape(", ".join(item.get("placeholders", [])))
        cards.append(
            f"""
      <article class="card">
        <a href="{overlay}" target="_blank" rel="noreferrer"><img src="{overlay}" alt=""></a>
        <div class="meta">
          <h2>{title}</h2>
          <p>objects: <strong>{prompts}</strong></p>
          <p>hand frame {item.get("first_hand_frame")} -> anchor frame {item.get("anchor_frame")} &middot; {item.get("num_instances", 0)} object mask(s)</p>
          <p><a href="{frame}" target="_blank" rel="noreferrer">anchor frame</a></p>
        </div>
      </article>"""
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Something-Something SAM3 Anchor Masks</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background:#f6f6f3; color:#202124; }}
    * {{ box-sizing: border-box; }} body {{ margin: 0; }}
    header {{ padding: 22px clamp(18px,4vw,48px); background:#fff; border-bottom:1px solid #d8d8d2; }}
    h1 {{ margin:0; font-size:clamp(22px,3vw,32px); letter-spacing:0; }}
    main {{ padding:22px clamp(18px,4vw,48px) 48px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(min(100%,330px),1fr)); gap:18px; }}
    .card {{ overflow:hidden; border:1px solid #d7d7d0; border-radius:8px; background:#fff; }}
    img {{ display:block; width:100%; aspect-ratio:16/9; object-fit:contain; background:#101112; }}
    .meta {{ padding:12px 14px 14px; }} h2 {{ margin:0 0 8px; font-size:15px; line-height:1.3; letter-spacing:0; }}
    p {{ margin:5px 0 0; color:#686b6f; font-size:13px; line-height:1.4; }} a {{ color:#246b70; }}
  </style>
</head>
<body>
  <header><h1>Something-Something SAM3 Anchor Masks</h1></header>
  <main><section class="grid">{''.join(cards)}</section></main>
</body>
</html>
"""
    viewer = out_dir / "viewer"
    viewer.mkdir(parents=True, exist_ok=True)
    (viewer / "index.html").write_text(page)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    sam3_root = args.sam3_root.expanduser().resolve()
    out_dir = root / "sam3_anchor_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = json.loads(labels_path.read_text())
    selected = choose_items(
        labels=labels,
        video_root=video_root,
        video_ids=args.video_id,
        limit=args.limit,
        min_frames=args.min_frames_after_anchor + args.anchor_offset + 1,
        start_index=args.start_index,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if not selected:
        raise SystemExit("no matching Something-Something videos found")

    bpe_path = sam3_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    print(f"loading SAM3 on {args.device}")
    model = build_sam3_image_model(
        bpe_path=str(bpe_path) if bpe_path.exists() else None,
        checkpoint_path=args.checkpoint,
        load_from_HF=args.checkpoint is None,
        device=args.device,
        eval_mode=True,
    )
    processor = Sam3Processor(
        model=model,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    summaries = []
    selected_list = []
    for idx, item in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] {item['id']} :: {item['label']}")
        try:
            summary = run_one(processor, item, root, out_dir, args)
        except Exception as exc:
            summary = {
                "video_uid": item["id"],
                "video_title": item.get("label", ""),
                "label": item.get("label", ""),
                "placeholders": item.get("placeholders", []),
                "source_video_path": item.get("video_path", ""),
                "error": str(exc),
                "num_instances": 0,
            }
            print(f"  error: {exc}")
        summaries.append(summary)
        if summary.get("clip_path") and summary.get("num_instances", 0) > 0:
            selected_list.append(root / summary["clip_path"])

    manifest_path = out_dir / args.manifest_name
    selected_list_path = root / args.selected_list_name
    manifest_path.write_text(json.dumps(summaries, indent=2))
    selected_list_path.write_text(
        "".join(f"{path}\n" for path in selected_list)
    )
    if not args.no_viewer:
        build_viewer(out_dir=out_dir, root=root, summaries=[s for s in summaries if "overlay_path" in s])
        print(f"wrote {out_dir / 'viewer' / 'index.html'}")
    print(f"wrote {manifest_path}")
    print(f"wrote {selected_list_path} ({len(selected_list)} clips)")


if __name__ == "__main__":
    main()
