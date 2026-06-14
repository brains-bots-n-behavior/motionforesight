#!/usr/bin/env python3
"""Run SAM3 text-prompt masks on first frames of Action100M segment clips."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 on first frames of Action100M segment clips."
    )
    parser.add_argument("--root", default="data/action100m", help="Action100M data root")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Segment manifest JSON. Defaults to <root>/segments/segments_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <root>/sam3_first_frame_masks",
    )
    parser.add_argument(
        "--sam3-root",
        default="../../external/sam3",
        help="SAM3 checkout, used for the BPE vocab path",
    )
    parser.add_argument("--checkpoint", default=None, help="Optional local SAM3 checkpoint")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None, help="Optional max clips")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run entries even when summary.json already exists",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value[:140] or "clip"


def relpath(path: Path, start: Path) -> str:
    return path.resolve().relative_to(start.resolve()).as_posix()


def extract_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def colorize_mask(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask] = np.array(color, dtype=np.uint8)
    return canvas


def alpha_blend(base: np.ndarray, overlay: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = base.astype(np.float32).copy()
    mask = overlay.sum(axis=2) > 0
    out[mask] = out[mask] * (1.0 - alpha) + overlay[mask].astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def run_one(
    processor: Sam3Processor,
    item: dict,
    out_dir: Path,
    root: Path,
    confidence_threshold: float,
) -> dict:
    clip_path = Path(item["clip_path"])
    prompt = item.get("segment_text") or item.get("label") or item.get("video_title") or "object"
    stem = slugify(clip_path.stem)
    clip_out = out_dir / "clips" / stem
    frame_path = clip_out / "first_frame.jpg"
    overlay_path = clip_out / "overlay.png"
    summary_path = clip_out / "summary.json"
    clip_out.mkdir(parents=True, exist_ok=True)

    frame_rgb = extract_first_frame(clip_path)
    save_rgb(frame_path, frame_rgb)
    pil_image = Image.fromarray(frame_rgb)

    with torch.inference_mode():
        state = processor.set_image(pil_image)
        state = processor.set_text_prompt(prompt=prompt, state=state)

    masks = state["masks"].detach().cpu().numpy()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    masks = masks.astype(bool)
    boxes = state["boxes"].detach().cpu().numpy()
    scores = state["scores"].detach().cpu().numpy()

    overlay = frame_rgb.copy()
    instances = []
    for idx, mask in enumerate(masks):
        mask_path = clip_out / f"mask_{idx:02d}.png"
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)

        color = PALETTE[idx % len(PALETTE)]
        overlay = alpha_blend(overlay, colorize_mask(mask, color))
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
            f"{idx}:{scores[idx]:.2f}",
            (int(round(x0)), max(18, int(round(y0)) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        instances.append(
            {
                "index": idx,
                "mask_path": relpath(mask_path, root),
                "score": float(scores[idx]),
                "box_xyxy": [float(v) for v in boxes[idx].tolist()],
                "mask_area": int(mask.sum()),
            }
        )

    save_rgb(overlay_path, overlay)
    summary = {
        "video_uid": item.get("video_uid"),
        "video_title": item.get("video_title"),
        "clip_path": relpath(clip_path, root),
        "frame_path": relpath(frame_path, root),
        "overlay_path": relpath(overlay_path, root),
        "prompt": prompt,
        "segment_text": item.get("segment_text"),
        "start": item.get("start"),
        "end": item.get("end"),
        "duration": item.get("duration"),
        "level": item.get("level"),
        "confidence_threshold": confidence_threshold,
        "num_instances": len(instances),
        "instances": instances,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def build_viewer(out_dir: Path, root: Path, summaries: list[dict]) -> None:
    cards = []
    for item in summaries:
        prompt = html.escape(item.get("prompt") or "")
        title = html.escape(item.get("segment_text") or item.get("prompt") or "Segment")
        video_title = html.escape(item.get("video_title") or "")
        time_text = ""
        if item.get("start") is not None and item.get("end") is not None:
            time_text = f"{float(item['start']):.1f}s-{float(item['end']):.1f}s"
        overlay = html.escape(Path("..", item["overlay_path"]).as_posix())
        frame = html.escape(Path("..", item["frame_path"]).as_posix())
        cards.append(
            f"""
      <article class="card">
        <a href="{overlay}" target="_blank" rel="noreferrer">
          <img src="{overlay}" alt="SAM3 mask overlay for {title}" loading="lazy">
        </a>
        <div class="meta">
          <h2>{title}</h2>
          <p>{video_title}</p>
          <p>{html.escape(time_text)} &middot; prompt: <strong>{prompt}</strong></p>
          <p>{int(item.get("num_instances", 0))} mask(s) &middot; <a href="{frame}" target="_blank" rel="noreferrer">first frame</a></p>
        </div>
      </article>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Action100M SAM3 First-Frame Masks</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f5f2;
      color: #202124;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    header {{
      padding: 24px clamp(18px, 4vw, 48px) 14px;
      border-bottom: 1px solid #d8d6cf;
      background: #fff;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: clamp(22px, 3vw, 34px);
      line-height: 1.1;
      letter-spacing: 0;
    }}
    header p {{
      margin: 0;
      max-width: 880px;
      color: #5a5d60;
      font-size: 14px;
      line-height: 1.45;
    }}
    main {{ padding: 22px clamp(18px, 4vw, 48px) 48px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(min(100%, 330px), 1fr));
      gap: 18px;
      align-items: start;
    }}
    .card {{
      overflow: hidden;
      border: 1px solid #d5d5ce;
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 1px 2px rgba(20, 20, 20, 0.06);
    }}
    img {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #111;
      object-fit: contain;
    }}
    .meta {{ padding: 12px 14px 14px; }}
    h2 {{
      margin: 0 0 8px;
      font-size: 15px;
      line-height: 1.28;
      letter-spacing: 0;
    }}
    .meta p {{
      margin: 4px 0 0;
      color: #686b6f;
      font-size: 13px;
      line-height: 1.4;
    }}
    a {{ color: #246b70; }}
    strong {{ color: #303336; font-weight: 650; }}
  </style>
</head>
<body>
  <header>
    <h1>Action100M SAM3 First-Frame Masks</h1>
    <p>SAM3 text-prompt masks on the first frame of each segment clip. Each prompt is the segment text from the Action100M segment manifest.</p>
  </header>
  <main>
    <section class="grid">
{''.join(cards)}
    </section>
  </main>
</body>
</html>
"""
    viewer_dir = out_dir / "viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    (viewer_dir / "index.html").write_text(page)


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else root / "segments" / "segments_manifest.json"
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "sam3_first_frame_masks"
    sam3_root = Path(args.sam3_root).expanduser().resolve()
    bpe_path = sam3_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    items = json.loads(manifest_path.read_text())
    if args.limit is not None:
        items = items[: args.limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading SAM3 image model on {args.device}")
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

    summaries: list[dict] = []
    for idx, item in enumerate(items, start=1):
        clip_path = Path(item["clip_path"])
        stem = slugify(clip_path.stem)
        summary_path = out_dir / "clips" / stem / "summary.json"
        if summary_path.exists() and not args.force:
            summaries.append(json.loads(summary_path.read_text()))
            print(f"[{idx}/{len(items)}] skip {stem}")
            continue

        print(f"[{idx}/{len(items)}] {stem} :: {item.get('segment_text')}")
        try:
            summaries.append(
                run_one(
                    processor=processor,
                    item=item,
                    out_dir=out_dir,
                    root=root,
                    confidence_threshold=args.confidence_threshold,
                )
            )
        except Exception as exc:
            error = {
                "video_uid": item.get("video_uid"),
                "video_title": item.get("video_title"),
                "clip_path": relpath(clip_path, root),
                "prompt": item.get("segment_text"),
                "segment_text": item.get("segment_text"),
                "error": str(exc),
                "num_instances": 0,
            }
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(error, indent=2))
            summaries.append(error)
            print(f"  error: {exc}")

    (out_dir / "manifest.json").write_text(json.dumps(summaries, indent=2))
    build_viewer(out_dir=out_dir, root=root, summaries=summaries)
    print(f"wrote {out_dir / 'viewer' / 'index.html'}")


if __name__ == "__main__":
    main()
