#!/usr/bin/env python3
"""Run TrackCraft3r dense inference over many user NPZs with one model load."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def _video_uid(path: Path) -> str:
    return path.stem.split(".")[0]


def _frame0_query_grid(height: int, width: int, stride: int = 8) -> np.ndarray:
    vv, uu = np.meshgrid(
        np.arange(0, height, stride),
        np.arange(0, width, stride),
        indexing="ij",
    )
    return np.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1).astype(np.float32)


def _load_video_paths(args: argparse.Namespace) -> list[Path]:
    if args.video_list:
        return [
            Path(line.strip()).resolve()
            for line in args.video_list.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return [Path(p).resolve() for p in args.video]


def _load_npz_for_predict(path: Path, load_npz_data, num_frames: int, frame_stride: int):
    in_npz = np.load(path, allow_pickle=True)
    total = in_npz["images_jpeg_bytes"].shape[0]
    span = num_frames * frame_stride
    if span > total:
        raise ValueError(
            f"need {span} frames (= num_frames {num_frames} x frame_stride {frame_stride}) "
            f"but {path} only has {total}"
        )

    tmp_path = None
    if "tracks_XYZ" not in in_npz.files:
        dummy_track = np.zeros((total, 1, 3), dtype=np.float32)
        dummy_vis = np.ones((total, 1), dtype=bool)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            data = {k: in_npz[k] for k in in_npz.files}
            data["tracks_XYZ"] = dummy_track
            data["visibility"] = dummy_vis
            np.savez_compressed(tmp.name, **data)
            tmp_path = tmp.name
        npz_path = tmp_path
    else:
        npz_path = str(path)

    try:
        loaded = load_npz_data(npz_path, num_frames=num_frames, frame_stride=frame_stride)
    finally:
        if tmp_path:
            os.unlink(tmp_path)

    depth_map = in_npz["depth_map"][:span:frame_stride]
    return loaded, depth_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/action100m"))
    parser.add_argument("--trackcraft-root", type=Path, default=Path("../../external/TrackCraft3r"))
    parser.add_argument("--tracks-name", default="tracks")
    parser.add_argument("--video-list", type=Path, default=None)
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--lora-rank", type=int, default=1024)
    parser.add_argument("--lora-target-modules", default="q,k,v,o,ffn.0,ffn.2")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--regression-timestep", type=int, default=-1)
    parser.add_argument("--track-latent-length", type=int, default=12)
    parser.add_argument("--resize-mode", default="stretch", choices=["pad", "stretch"])
    parser.add_argument("--diag-max-depth", type=float, default=80.0)
    parser.add_argument("--pj-norm-percentile-lo", type=float, default=2.0)
    parser.add_argument("--pj-norm-percentile-hi", type=float, default=98.0)
    args = parser.parse_args()

    root = args.root.resolve()
    trackcraft_root = args.trackcraft_root.resolve()
    tracks_dir = root / args.tracks_name
    checkpoint = args.checkpoint or trackcraft_root / "checkpoints" / "trackcraft3r" / "model.safetensors"
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["MODELSCOPE_CACHE"] = str(trackcraft_root / "checkpoints" / "wan_models")
    sys.path.insert(0, str(trackcraft_root))

    import torch
    from evaluation.dust3r_eval_utils import load_npz_data
    from evaluation.wan_scene_flow_predictor import WanSceneFlowPredictor

    videos = _load_video_paths(args)
    if not videos:
        raise SystemExit("no videos provided")

    predictor = WanSceneFlowPredictor(
        checkpoint_path=str(checkpoint),
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
        height=args.height,
        width=args.width,
        device=args.device,
        regression_timestep=args.regression_timestep,
        track_latent_length=args.track_latent_length,
        resize_mode=args.resize_mode,
        diag_max_depth=args.diag_max_depth,
        pj_norm_percentile_lo=args.pj_norm_percentile_lo,
        pj_norm_percentile_hi=args.pj_norm_percentile_hi,
    )

    failures: list[tuple[Path, str]] = []
    for index, video in enumerate(videos, start=1):
        uid = _video_uid(video)
        user_npz = tracks_dir / f"{uid}_user.npz"
        dense_npz = tracks_dir / f"{uid}_dense.npz"
        if dense_npz.exists():
            print(f"[{index}/{len(videos)}] skip {dense_npz}")
            continue
        try:
            (video_list, _, _, intrinsics, _, _, _, extrinsics_w2c), depth_map = _load_npz_for_predict(
                user_npz,
                load_npz_data,
                args.num_frames,
                args.frame_stride,
            )
            height, width = video_list[0].height, video_list[0].width
            query_uv = _frame0_query_grid(height, width, stride=max(1, min(height, width) // 16))
            vis_dummy = np.ones((args.num_frames, query_uv.shape[0]), dtype=bool)
            print(f"[{index}/{len(videos)}] dense {uid}")
            with torch.no_grad():
                predictor.predict(
                    video_list,
                    query_uv,
                    vis_dummy,
                    intrinsics,
                    depth_map=depth_map,
                    extrinsics_w2c=extrinsics_w2c,
                )
            out = {
                "track_map": predictor._last_row_dense.astype(np.float32),
                "recon_map": predictor._last_pj_input.astype(np.float32),
                "rgb": predictor._last_rgb_frames,
            }
            dense_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dense_npz, **out)
            print(f"ready: {dense_npz}")
        except Exception as exc:
            if not args.keep_going:
                raise
            failures.append((video, str(exc)))
            print(f"failed: {video}\n  {exc}", file=sys.stderr)

    if failures:
        print("\nfailures:", file=sys.stderr)
        for video, error in failures:
            print(f"- {video}: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
