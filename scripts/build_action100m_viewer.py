#!/usr/bin/env python3
"""Build a local HTML grid for downloaded Action100M preview videos."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".m4v"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/action100m"),
        help="Action100M local folder containing raw/ and selected_videos.json.",
    )
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    root = args.root.resolve()
    raw_dir = root / "raw"
    out_path = root / "viewer" / "index.html"
    manifest_path = root / "selected_videos.json"

    selected = {}
    if manifest_path.exists():
        selected = {row["video_uid"]: row for row in json.loads(manifest_path.read_text())}

    videos = []
    for path in sorted(raw_dir.iterdir() if raw_dir.exists() else []):
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        if ".f" in path.stem and path.with_name(f"{path.stem.split('.')[0]}.mp4").exists():
            continue
        uid = path.stem.split(".")[0]
        meta = selected.get(uid, {})
        videos.append(
            {
                "uid": uid,
                "title": meta.get("title") or uid,
                "duration": meta.get("duration_seconds"),
                "path": path,
                "rel": os.path.relpath(path, out_path.parent),
            }
        )
        if len(videos) >= args.limit:
            break

    cards = []
    for item in videos:
        duration = f"{int(item['duration'])}s" if item["duration"] else "duration unknown"
        cards.append(
            f"""
      <article class="card">
        <video controls preload="metadata" src="{html.escape(str(item['rel']))}"></video>
        <div class="meta">
          <h2>{html.escape(item['title'])}</h2>
          <p>{html.escape(item['uid'])} &middot; {html.escape(duration)}</p>
        </div>
      </article>"""
        )

    body = "\n".join(cards) if cards else '<p class="empty">No downloaded videos found in raw/ yet.</p>'
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Action100M Preview Videos</title>
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
      background: #ffffff;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: clamp(22px, 3vw, 34px);
      line-height: 1.1;
      letter-spacing: 0;
    }}
    header p {{
      margin: 0;
      max-width: 780px;
      color: #5a5d60;
      font-size: 14px;
      line-height: 1.45;
    }}
    main {{
      padding: 22px clamp(18px, 4vw, 48px) 48px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(min(100%, 360px), 1fr));
      gap: 18px;
      align-items: start;
    }}
    .card {{
      overflow: hidden;
      border: 1px solid #d5d5ce;
      border-radius: 8px;
      background: #ffffff;
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
      margin: 0 0 6px;
      font-size: 15px;
      line-height: 1.28;
      letter-spacing: 0;
    }}
    .meta p, .empty {{
      margin: 0;
      color: #686b6f;
      font-size: 13px;
      line-height: 1.4;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Action100M Preview Videos</h1>
    <p>Downloaded samples from the Action100M-preview split, staged for local inspection and 3D point tracking.</p>
  </header>
  <main>
    <section class="grid">
{body}
    </section>
  </main>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text)
    print(out_path)


if __name__ == "__main__":
    main()
