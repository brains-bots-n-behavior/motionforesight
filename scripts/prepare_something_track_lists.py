#!/usr/bin/env python3
"""Merge Something-Something SAM3 shard manifests into tracking lists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/something_something"))
    parser.add_argument(
        "--manifest-glob",
        default="sam3_anchor_masks/manifest_shard*.json",
        help="Glob relative to --root for sharded SAM3 manifests.",
    )
    parser.add_argument(
        "--merged-manifest",
        default="sam3_anchor_masks/manifest_merged.json",
        help="Output merged manifest path relative to --root.",
    )
    parser.add_argument(
        "--selected-name",
        default="anchor_selected_videos_track.txt",
        help="Combined selected video list relative to --root.",
    )
    parser.add_argument(
        "--shard-prefix",
        default="anchor_selected_videos_track_gpu",
        help="Shard list prefix relative to --root; writes <prefix><i>.txt.",
    )
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--min-masks", type=int, default=1)
    parser.add_argument("--max-masks", type=int, default=0, help="0 means no upper bound.")
    parser.add_argument("--min-anchor-frames", type=int, default=32)
    return parser.parse_args()


def _dedupe_entries(paths: list[Path]) -> list[dict]:
    entries: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        for entry in json.loads(path.read_text()):
            key = str(entry.get("video_uid") or entry.get("clip_path") or entry.get("source_video_path"))
            if not key or key in seen:
                continue
            seen.add(key)
            entries.append(entry)
    return entries


def _is_trackable(entry: dict, min_masks: int, max_masks: int, min_anchor_frames: int) -> bool:
    if not entry.get("clip_path"):
        return False
    num_masks = int(entry.get("num_instances", 0))
    if num_masks < min_masks:
        return False
    if max_masks > 0 and num_masks > max_masks:
        return False
    if int(entry.get("anchor_clip_frames", 0)) < min_anchor_frames:
        return False
    return True


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")

    root = args.root.resolve()
    manifest_paths = sorted(root.glob(args.manifest_glob))
    if not manifest_paths:
        raise SystemExit(f"no manifests matched {root / args.manifest_glob}")

    entries = _dedupe_entries(manifest_paths)
    trackable = [
        entry
        for entry in entries
        if _is_trackable(
            entry,
            min_masks=args.min_masks,
            max_masks=args.max_masks,
            min_anchor_frames=args.min_anchor_frames,
        )
    ]
    selected = [(root / entry["clip_path"]).resolve().as_posix() for entry in trackable]

    merged_manifest = root / args.merged_manifest
    merged_manifest.parent.mkdir(parents=True, exist_ok=True)
    merged_manifest.write_text(json.dumps(entries, indent=2))

    selected_path = root / args.selected_name
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text("\n".join(selected) + ("\n" if selected else ""))

    shard_paths = []
    for shard_idx in range(args.num_shards):
        shard = [
            path
            for idx, path in enumerate(selected)
            if idx % args.num_shards == shard_idx
        ]
        shard_path = root / f"{args.shard_prefix}{shard_idx}.txt"
        shard_path.write_text("\n".join(shard) + ("\n" if shard else ""))
        shard_paths.append(shard_path)

    print(f"input manifests: {len(manifest_paths)}")
    print(f"merged entries: {len(entries)} -> {merged_manifest}")
    print(f"trackable clips: {len(selected)} -> {selected_path}")
    for shard_path in shard_paths:
        count = sum(1 for line in shard_path.read_text().splitlines() if line.strip())
        print(f"shard {shard_path.name}: {count}")


if __name__ == "__main__":
    main()
