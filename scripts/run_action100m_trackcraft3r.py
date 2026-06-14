#!/usr/bin/env python3
"""Run the TrackCraft3r DA3 -> NPZ -> dense tracking pipeline on local samples."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _video_uid(path: Path) -> str:
    return path.stem.split(".")[0]


def _run(cmd: list[str], cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    print("\n$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/action100m"))
    parser.add_argument("--trackcraft-root", type=Path, default=Path("../../external/TrackCraft3r"))
    parser.add_argument("--da3-root", type=Path, default=Path("../../external/depth-anything-3"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--limit", type=int, default=0, help="0 means all merged MP4s.")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    trackcraft_root = args.trackcraft_root.resolve()
    da3_root = args.da3_root.resolve()
    checkpoint = trackcraft_root / "checkpoints" / "trackcraft3r" / "model.safetensors"
    wan_cache = trackcraft_root / "checkpoints" / "wan_models"

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{trackcraft_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["MODELSCOPE_CACHE"] = str(wan_cache)

    videos = [
        p for p in sorted((root / "raw").glob("*.mp4"))
        if ".f" not in p.stem
    ]
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"no merged MP4s found under {root / 'raw'}")

    for video in videos:
        uid = _video_uid(video)
        preproc = root / "preproc" / f"{uid}_da3"
        user_npz = root / "tracks" / f"{uid}_user.npz"
        dense_npz = root / "tracks" / f"{uid}_dense.npz"

        if not (preproc / "depth.npy").exists():
            _run(
                [
                    args.python,
                    "scripts/preprocess_da3.py",
                    "--video_path",
                    str(video),
                    "--output_dir",
                    str(preproc),
                    "--da3_root",
                    str(da3_root),
                    "--process_res",
                    str(args.process_res),
                    "--chunk_size",
                    str(args.chunk_size),
                    "--device",
                    args.device,
                ],
                cwd=trackcraft_root,
                env=env,
                dry_run=args.dry_run,
            )

        if not user_npz.exists():
            _run(
                [
                    args.python,
                    "scripts/build_user_npz.py",
                    "--video_path",
                    str(video),
                    "--depth_npy",
                    str(preproc / "depth.npy"),
                    "--extrinsics_npy",
                    str(preproc / "extrinsics.npy"),
                    "--intrinsics_npy",
                    str(preproc / "intrinsics.npy"),
                    "--depth_convention",
                    "z",
                    "--extrinsics_convention",
                    "w2c",
                    "--output_npz",
                    str(user_npz),
                ],
                cwd=trackcraft_root,
                env=env,
                dry_run=args.dry_run,
            )

        if not dense_npz.exists():
            _run(
                [
                    args.python,
                    "scripts/inference_user_video.py",
                    "--checkpoint_path",
                    str(checkpoint),
                    "--input_npz",
                    str(user_npz),
                    "--output_npz",
                    str(dense_npz),
                    "--num_frames",
                    str(args.num_frames),
                    "--frame_stride",
                    str(args.frame_stride),
                    "--device",
                    args.device,
                ],
                cwd=trackcraft_root,
                env=env,
                dry_run=args.dry_run,
            )

        print(f"\nready: {dense_npz}")


if __name__ == "__main__":
    main()
