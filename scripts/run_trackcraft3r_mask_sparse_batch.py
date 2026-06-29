#!/usr/bin/env python3
"""Run TrackCraft3r on SAM-mask query points and save sparse tracks only.

TrackCraft3r's predictor accepts arbitrary frame-0 query UV points and returns
tracks with shape (T, M, 3).  The model still decodes a dense scene-flow field
internally, but this script avoids writing the heavy dense `track_map` and
`recon_map` arrays to disk.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _video_uid(path: Path) -> str:
    return path.stem.split(".")[0]


def _load_video_paths(args: argparse.Namespace) -> list[Path]:
    if args.video_list:
        videos = [
            Path(line.strip()).resolve()
            for line in args.video_list.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        videos = [Path(p).resolve() for p in args.video]
    return videos[: args.limit] if args.limit > 0 else videos


def _load_manifest(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("items", data.get("clips", []))
    out: dict[str, dict] = {}
    for item in data:
        clip_path = item.get("clip_path")
        if clip_path:
            out[Path(clip_path).stem] = item
        video_uid = item.get("video_uid")
        if video_uid:
            out[f"{video_uid}_anchor"] = item
    return out


def _load_union_mask(mask_paths: list[Path]) -> np.ndarray:
    masks = []
    for path in mask_paths:
        if path.exists():
            masks.append(np.asarray(Image.open(path).convert("L")) > 0)
    if not masks:
        raise ValueError("no SAM mask images found")
    return np.logical_or.reduce(masks)


def _sample_mask_grid(mask: np.ndarray, max_points: int, min_step: int) -> list[tuple[int, int]]:
    """Sample mask-grid points using the same defaults as the HTML trace viewer."""

    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []

    area = len(xs)
    step = max(min_step, int(math.sqrt(max(1, area) / max(1, max_points))))
    points: list[tuple[int, int]] = []
    offset = step // 2
    for y in range(offset, h, step):
        for x in range(offset, w, step):
            y0, y1 = max(0, y - step // 2), min(h, y + step // 2 + 1)
            x0, x1 = max(0, x - step // 2), min(w, x + step // 2 + 1)
            patch_points = np.argwhere(mask[y0:y1, x0:x1])
            if len(patch_points) == 0:
                continue
            py, px = patch_points[len(patch_points) // 2]
            points.append((x0 + int(px), y0 + int(py)))

    if len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = [points[i] for i in keep]
    return points


def _manifest_mask_paths(root: Path, meta: dict) -> list[Path]:
    mask_paths = []
    for inst in meta.get("instances", []):
        rel = inst.get("mask_path")
        if rel:
            mask_paths.append(root / rel)
    return mask_paths


def _query_points_from_masks(
    root: Path,
    uid: str,
    meta: dict,
    image_width: int,
    image_height: int,
    max_points: int,
    min_step: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], list[Path]]:
    mask_paths = _manifest_mask_paths(root, meta)
    if not mask_paths:
        mask_dir = root / "sam3_anchor_masks" / "clips" / uid
        mask_paths = sorted(mask_dir.glob("mask_*.png"))
    mask = _load_union_mask(mask_paths)
    points = _sample_mask_grid(mask, max_points=max_points, min_step=min_step)
    if not points:
        raise ValueError("SAM mask has no sampled query points")

    mask_h, mask_w = mask.shape
    mask_uv = np.asarray(points, dtype=np.float32)
    query_uv = np.empty_like(mask_uv)
    query_uv[:, 0] = mask_uv[:, 0] / max(1, mask_w - 1) * max(1, image_width - 1)
    query_uv[:, 1] = mask_uv[:, 1] / max(1, mask_h - 1) * max(1, image_height - 1)
    return query_uv.astype(np.float32), mask_uv, (mask_h, mask_w), mask_paths


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
    source_extrinsics = in_npz["extrinsics_w2c"][:span:frame_stride]
    return loaded, depth_map, source_extrinsics


def _sample_visibility(predictor, valid_oob_mask: np.ndarray) -> np.ndarray | None:
    vis_dense = getattr(predictor, "_last_vis_dense", None)
    q_uv_model = getattr(predictor, "_last_query_uv_model", None)
    if vis_dense is None or q_uv_model is None:
        return None
    h_vis, w_vis = vis_dense.shape[-2], vis_dense.shape[-1]
    q_uv_model = q_uv_model[valid_oob_mask]
    u_int = np.clip(q_uv_model[:, 0].astype(np.int64), 0, w_vis - 1)
    v_int = np.clip(q_uv_model[:, 1].astype(np.int64), 0, h_vis - 1)
    return vis_dense[:, v_int, u_int].astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/something_something"))
    parser.add_argument("--trackcraft-root", type=Path, default=Path("external/TrackCraft3r"))
    parser.add_argument(
        "--tracks-name",
        default="anchor_tracks32_all",
        help="Directory under --root containing *_user.npz files.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output directory under --root for *_sparse.npz. Defaults to --tracks-name.",
    )
    parser.add_argument("--sam-manifest", default="sam3_anchor_masks/manifest_all.json")
    parser.add_argument("--video-list", type=Path, default=None)
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=260)
    parser.add_argument("--min-step", type=int, default=16)
    parser.add_argument("--min-points", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-rgb", action="store_true", help="Also save TrackCraft-resized RGB frames.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    trackcraft_root = args.trackcraft_root.resolve()
    tracks_dir = root / args.tracks_name
    out_dir = root / (args.output_name or args.tracks_name)
    manifest_path = root / args.sam_manifest
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
    manifest = _load_manifest(manifest_path)

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
        sparse_npz = out_dir / f"{uid}_sparse.npz"
        if sparse_npz.exists() and not args.force:
            print(f"[{index}/{len(videos)}] skip {sparse_npz}")
            continue
        try:
            if not user_npz.exists():
                raise FileNotFoundError(f"missing user NPZ: {user_npz}")
            meta = manifest.get(uid, {})
            if not meta:
                raise ValueError(f"missing manifest entry for {uid}")
            (video_list, _, _, intrinsics, _, _, _, extrinsics_w2c), depth_map, source_extrinsics = _load_npz_for_predict(
                user_npz,
                load_npz_data,
                args.num_frames,
                args.frame_stride,
            )
            image_width, image_height = video_list[0].width, video_list[0].height
            query_uv, mask_uv, mask_shape, mask_paths = _query_points_from_masks(
                root=root,
                uid=uid,
                meta=meta,
                image_width=image_width,
                image_height=image_height,
                max_points=args.max_points,
                min_step=args.min_step,
            )
            if len(query_uv) < args.min_points:
                raise ValueError(f"only sampled {len(query_uv)} mask points")
            vis_dummy = np.ones((args.num_frames, query_uv.shape[0]), dtype=bool)
            print(f"[{index}/{len(videos)}] sparse {uid}: {len(query_uv)} mask query points")
            with torch.no_grad():
                tracks_xyz = predictor.predict(
                    video_list,
                    query_uv,
                    vis_dummy,
                    intrinsics,
                    depth_map=depth_map,
                    extrinsics_w2c=extrinsics_w2c,
                )

            oob_mask = getattr(predictor, "_last_oob_mask", np.ones(len(query_uv), dtype=bool))
            visibility_pred = _sample_visibility(predictor, oob_mask)
            out = {
                "tracks_xyz": tracks_xyz.astype(np.float32),
                "query_uv": query_uv[oob_mask].astype(np.float32),
                "query_uv_mask": mask_uv[oob_mask].astype(np.float32),
                "oob_mask": oob_mask.astype(bool),
                "visibility": (
                    visibility_pred
                    if visibility_pred is not None
                    else np.ones((tracks_xyz.shape[0], tracks_xyz.shape[1]), dtype=bool)
                ),
                "fx_fy_cx_cy": np.asarray(intrinsics, dtype=np.float32),
                "extrinsics_w2c": np.asarray(extrinsics_w2c, dtype=np.float32),
                "source_extrinsics_w2c": np.asarray(source_extrinsics, dtype=np.float32),
                "mask_shape_hw": np.asarray(mask_shape, dtype=np.int32),
                "image_size_hw": np.asarray([image_height, image_width], dtype=np.int32),
                "video_path": np.asarray(str(video)),
                "user_npz": np.asarray(str(user_npz)),
                "mask_paths": np.asarray([str(path) for path in mask_paths]),
                "text": np.asarray(meta.get("segment_text") or meta.get("label") or uid),
            }
            if args.save_rgb:
                out["rgb"] = predictor._last_rgb_frames.astype(np.uint8)
            sparse_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sparse_npz, **out)
            print(f"ready: {sparse_npz} ({tracks_xyz.shape[1]} points)")
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
