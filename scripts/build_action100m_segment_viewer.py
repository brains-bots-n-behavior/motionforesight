#!/usr/bin/env python3
"""Cut Action100M annotated segments and build a local HTML clip grid."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
from pathlib import Path


def _text_for_node(node: dict) -> str:
    gpt = node.get("gpt") or {}
    action = gpt.get("action") or {}
    summary = gpt.get("summary") or {}
    return (
        action.get("brief")
        or summary.get("brief")
        or node.get("plm_action")
        or "Untitled segment"
    )


def _select_segments(nodes: list[dict], per_video: int, min_duration: float, max_duration: float) -> list[dict]:
    candidates = []
    for node in nodes:
        start = float(node.get("start", 0.0))
        end = float(node.get("end", 0.0))
        duration = end - start
        if not (min_duration <= duration <= max_duration):
            continue
        if int(node.get("level", -1)) < 3:
            continue
        text = _text_for_node(node)
        normalized_text = text.strip().lower()
        if (
            normalized_text in {"n/a", "na", "none", "null"}
            or normalized_text.startswith("na - no actions")
        ):
            continue
        candidates.append(
            {
                "node_id": node.get("node_id"),
                "level": int(node.get("level", -1)),
                "start": start,
                "end": end,
                "duration": duration,
                "text": text,
            }
        )

    candidates.sort(key=lambda x: (abs(x["duration"] - 10.0), x["level"], x["start"]))
    chosen = []
    for cand in candidates:
        overlaps = any(
            max(cand["start"], prev["start"]) < min(cand["end"], prev["end"])
            for prev in chosen
        )
        if overlaps:
            continue
        chosen.append(cand)
        if len(chosen) >= per_video:
            break

    return sorted(chosen, key=lambda x: x["start"])


def _cut_clip(ffmpeg: str, src: Path, dst: Path, start: float, duration: float, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.3f}",
        "-vf",
        "scale='min(854,iw)':-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if dst.exists() and dst.stat().st_size == 0:
            dst.unlink()
        raise


def _write_html(items: list[dict], out_path: Path) -> None:
    cards = []
    for item in items:
        rel = os.path.relpath(item["clip_path"], out_path.parent)
        cards.append(
            f"""
      <article class="card">
        <video controls preload="metadata" src="{html.escape(rel)}"></video>
        <div class="meta">
          <h2>{html.escape(item["segment_text"])}</h2>
          <p>{html.escape(item["video_title"])}</p>
          <p>{item["start"]:.1f}s-{item["end"]:.1f}s &middot; {item["duration"]:.1f}s &middot; level {item["level"]}</p>
        </div>
      </article>"""
        )
    body = "\n".join(cards) if cards else '<p class="empty">No segment clips were generated.</p>'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Action100M Segment Clips</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f4f1;
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
      max-width: 820px;
      color: #5a5d60;
      font-size: 14px;
      line-height: 1.45;
    }}
    main {{
      padding: 22px clamp(18px, 4vw, 48px) 48px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr));
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
    video {{
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
    .meta p, .empty {{
      margin: 4px 0 0;
      color: #686b6f;
      font-size: 13px;
      line-height: 1.4;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Action100M Segment Clips</h1>
    <p>Short temporal segments cut from the locally downloaded Action100M-preview videos. These clips are better units for visual inspection and 3D point tracking than full source videos.</p>
  </header>
  <main>
    <section class="grid">
{body}
    </section>
  </main>
</body>
</html>
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/action100m"))
    parser.add_argument("--per-video", type=int, default=3)
    parser.add_argument("--min-duration", type=float, default=5.0)
    parser.add_argument("--max-duration", type=float, default=18.0)
    parser.add_argument(
        "--exclude-video-uid",
        action="append",
        default=[],
        help="Video UID to omit from the segment viewer. Can be passed multiple times.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--min-source-size-mb", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    annotations_dir = root / "annotations"
    clips_dir = root / "segments" / "clips"
    manifest_path = root / "segments" / "segments_manifest.json"
    html_path = root / "segments" / "viewer" / "index.html"

    items = []
    excluded_uids = set(args.exclude_video_uid)
    for ann_path in sorted(annotations_dir.glob("*.json")):
        data = json.loads(ann_path.read_text())
        uid = data.get("video_uid") or ann_path.stem
        if uid in excluded_uids:
            print(f"skip {uid}: excluded")
            continue
        src = root / "raw" / f"{uid}.mp4"
        if not src.exists():
            print(f"skip {uid}: missing {src}")
            continue
        if src.stat().st_size < args.min_source_size_mb * 1024 * 1024:
            print(f"skip {uid}: source file is too small ({src.stat().st_size} bytes)")
            continue
        title = (data.get("metadata") or {}).get("title") or uid
        chosen = _select_segments(data.get("nodes") or [], args.per_video, args.min_duration, args.max_duration)
        for idx, segment in enumerate(chosen, start=1):
            safe_uid = uid.replace("/", "_")
            clip_path = clips_dir / f"{safe_uid}_seg{idx:02d}_{segment['start']:.1f}-{segment['end']:.1f}.mp4"
            try:
                _cut_clip(args.ffmpeg, src, clip_path, segment["start"], segment["duration"], args.overwrite)
            except subprocess.CalledProcessError as exc:
                print(f"skip {uid}: ffmpeg failed while cutting {clip_path.name} ({exc.returncode})")
                break
            items.append(
                {
                    "video_uid": uid,
                    "video_title": title,
                    "clip_path": str(clip_path),
                    "segment_text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"],
                    "level": segment["level"],
                    "node_id": segment["node_id"],
                }
            )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(items, indent=2))
    _write_html(items, html_path)
    print(f"wrote {manifest_path}")
    print(f"wrote {html_path}")
    print(f"clips: {len(items)}")


if __name__ == "__main__":
    main()
