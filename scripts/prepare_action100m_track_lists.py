#!/usr/bin/env python3
"""Prepare resumable TrackCraft3r video lists from SAM3 mask outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _clip_id(clip_path: str) -> str:
    return Path(clip_path).stem.split(".")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/action100m"))
    parser.add_argument("--sam-manifest", default="sam3_first_frame_masks/manifest.json")
    parser.add_argument("--tracks-name", default=None, help="Optional tracks dir; skips existing *_dense.npz.")
    parser.add_argument("--min-masks", type=int, default=1)
    parser.add_argument("--max-masks", type=int, default=3)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output-prefix", default="mask_trace32")
    parser.add_argument("--require-clip", action="store_true", help="Skip entries whose clip file is missing.")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = json.loads((root / args.sam_manifest).read_text())
    tracks_dir = root / args.tracks_name if args.tracks_name else None

    videos: list[Path] = []
    for entry in manifest:
        num_masks = int(entry.get("num_instances", 0))
        if not (args.min_masks <= num_masks <= args.max_masks):
            continue
        clip = root / entry["clip_path"]
        if args.require_clip and not clip.exists():
            continue
        if tracks_dir and (tracks_dir / f"{_clip_id(entry['clip_path'])}_dense.npz").exists():
            continue
        videos.append(clip)

    selected_path = root / f"{args.output_prefix}_selected_videos.txt"
    selected_path.write_text("\n".join(str(p) for p in videos) + ("\n" if videos else ""))
    print(f"wrote {len(videos)} videos -> {selected_path}")

    if args.num_shards > 1:
        for shard in range(args.num_shards):
            shard_videos = videos[shard :: args.num_shards]
            shard_path = root / f"{args.output_prefix}_gpu{shard}.txt"
            shard_path.write_text("\n".join(str(p) for p in shard_videos) + ("\n" if shard_videos else ""))
            print(f"wrote {len(shard_videos)} videos -> {shard_path}")


if __name__ == "__main__":
    main()
