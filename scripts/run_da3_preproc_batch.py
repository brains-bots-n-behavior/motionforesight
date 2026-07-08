#!/usr/bin/env python3
"""Batched DA3 depth + user-NPZ builder: load the DA3 model ONCE, loop a clip list.

Output-identical replacement for stage-3's per-clip pair
``preprocess_da3_chunked.py`` + ``external/TrackCraft3r/scripts/build_user_npz.py``,
which reload the 6.5 GB DA3 model for EVERY clip (~75% of wall-clock is reload +
subprocess overhead). This loads the model once and reuses it across every clip
in ``--video-list``.

Writes exactly the same files as the per-clip path:
    <root>/<preproc-name>/<uid>_da3/{depth,extrinsics,intrinsics}.npy
    <root>/<tracks-name>/<uid>_user.npz   keys: images_jpeg_bytes, depth_map,
                                          extrinsics_w2c, fx_fy_cx_cy

Faithfulness notes (must match the originals so the dataset stays consistent):
  * DA3 inference + post-processing is copied verbatim from
    preprocess_da3_chunked.py (chunked multi-view inference, scipy zoom order=1
    depth resize, extrinsics 3x4->4x4 pad, intrinsics rescale to original res).
  * user.npz assembly is copied from build_user_npz.py with the SAME conventions
    the orchestrator uses: depth_convention='z', extrinsics_convention='w2c',
    intrinsics_resolution=None (frame-0 normalized w2c, JPEG quality 95).
  * process_res=336 and chunk_size=24 MUST match the already-processed clips:
    chunk_size changes which frames co-attend within a clip -> changes output.
  * NO cross-clip batching: DA3 is a multi-view model (batch dim pinned to 1; the
    N frames in one call are N views of ONE scene that cross-attend). Mixing
    clips would corrupt depth/poses.

Resumable: skips any clip whose <uid>_user.npz already exists; reuses existing
depth/extrinsics/intrinsics.npy when present (only the user.npz is missing).
All outputs are written to a temp path then os.replace()'d into place, so a
killed worker never leaves a truncated depth.npy / user.npz behind.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom


# --------------------------------------------------------------------------- #
# Frame reading (matches cv2 BGR->RGB decode used by BOTH original scripts)
# --------------------------------------------------------------------------- #
def read_frames_rgb(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 failed to open {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return frames  # list of (H, W, 3) uint8 RGB


# --------------------------------------------------------------------------- #
# DA3 inference + post-proc — verbatim port of preprocess_da3_chunked.py:100-146
# --------------------------------------------------------------------------- #
def da3_depth_extr_intr(model, pil_images, process_res, chunk_size, device):
    T = len(pil_images)
    orig_W, orig_H = pil_images[0].size
    cs = chunk_size if chunk_size and chunk_size > 0 else T
    depth_chunks, extr_chunks, intr_chunks = [], [], []
    with torch.no_grad():
        for start in range(0, T, cs):
            end = min(T, start + cs)
            pred = model.inference(
                image=pil_images[start:end],
                process_res=process_res,
                process_res_method="upper_bound_resize",
                export_dir=None,
            )
            depth_chunks.append(pred.depth.astype(np.float32))
            extr_chunks.append(pred.extrinsics.astype(np.float32))
            intr_chunks.append(pred.intrinsics.astype(np.float32))
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    depth_proc = np.concatenate(depth_chunks, axis=0)   # (T, H_proc, W_proc) z-depth
    extr_3x4 = np.concatenate(extr_chunks, axis=0)      # (T, 3, 4) W2C
    intr_proc = np.concatenate(intr_chunks, axis=0)     # (T, 3, 3) at proc res
    H_proc, W_proc = depth_proc.shape[1], depth_proc.shape[2]

    # 1. Resize depth to original image resolution.
    if (H_proc, W_proc) != (orig_H, orig_W):
        sh, sw = orig_H / H_proc, orig_W / W_proc
        depth = np.stack([
            zoom(depth_proc[t], (sh, sw), order=1) for t in range(T)
        ]).astype(np.float32)
    else:
        depth = depth_proc.astype(np.float32)

    # 2. Pad extrinsics (T, 3, 4) -> (T, 4, 4).
    extr = np.zeros((T, 4, 4), dtype=np.float32)
    extr[:, :3, :] = extr_3x4
    extr[:, 3, 3] = 1.0

    # 3. Rescale intrinsics from processing res to original res.
    sx, sy = orig_W / W_proc, orig_H / H_proc
    intr = intr_proc.copy().astype(np.float32)
    intr[:, 0, 0] *= sx
    intr[:, 1, 1] *= sy
    intr[:, 0, 2] *= sx
    intr[:, 1, 2] *= sy
    return depth, extr, intr


# --------------------------------------------------------------------------- #
# user.npz assembly — verbatim port of build_user_npz.py main() for the
# orchestrator's fixed conventions: depth='z', extrinsics='w2c', intr_res=None.
# --------------------------------------------------------------------------- #
def _frame_zero_normalize_w2c(extrinsics_w2c):
    inv0 = np.linalg.inv(extrinsics_w2c[0])
    return np.stack([
        extrinsics_w2c[t] @ inv0 for t in range(extrinsics_w2c.shape[0])
    ]).astype(np.float32)


def build_user_npz(frames_rgb, depth, extr, intr, out_npz: Path):
    depth_full = np.asarray(depth).astype(np.float32)
    extr_full = np.asarray(extr).astype(np.float64)
    K_in = np.asarray(intr).astype(np.float64)

    # Per-frame (T,3,3) DA3 intrinsics -> use frame 0 (sub-pixel variation).
    if K_in.ndim == 3 and K_in.shape[1:] == (3, 3):
        K_in = K_in[0]
    if K_in.shape == (3, 3):
        fx, fy, cx, cy = K_in[0, 0], K_in[1, 1], K_in[0, 2], K_in[1, 2]
    elif K_in.shape == (4,):
        fx, fy, cx, cy = K_in
    else:
        raise ValueError(f"intrinsics shape {K_in.shape} not supported")

    # extrinsics_convention == 'w2c' -> no inversion, frame-0 normalize.
    extr_w2c = _frame_zero_normalize_w2c(extr_full)

    T = min(len(frames_rgb), depth_full.shape[0], extr_w2c.shape[0])
    frames_sel = frames_rgb[:T]
    depth_sel = depth_full[:T].copy()
    extr_sel = extr_w2c[:T].copy()

    H_img, W_img = frames_sel[0].shape[:2]
    H_d, W_d = depth_sel.shape[1], depth_sel.shape[2]
    if (H_img, W_img) != (H_d, W_d):
        raise ValueError(f"image res {(H_img, W_img)} != depth res {(H_d, W_d)}")

    fx_fy_cx_cy = np.array([fx, fy, cx, cy], dtype=np.float64)

    # depth_convention == 'z' -> no undistort.
    enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    images_jpeg_bytes = []
    for fr in frames_sel:
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, enc_params)
        assert ok, "JPEG encode failed"
        images_jpeg_bytes.append(buf.tobytes())
    images_jpeg_bytes = np.array(images_jpeg_bytes, dtype=object)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.parent / f".{out_npz.stem}.tmp{os.getpid()}.npz"
    np.savez_compressed(
        tmp,
        images_jpeg_bytes=images_jpeg_bytes,
        depth_map=depth_sel.astype(np.float32),
        extrinsics_w2c=extr_sel.astype(np.float32),
        fx_fy_cx_cy=fx_fy_cx_cy,
    )
    os.replace(tmp, out_npz)
    return T, H_img, W_img


def _atomic_save(path: Path, arr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.stem}.tmp{os.getpid()}.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--preproc-name", default="anchor_preproc")
    ap.add_argument("--tracks-name", default="anchor_tracks32")
    ap.add_argument("--video-list", type=Path, required=True)
    ap.add_argument("--da3-root", type=Path, default=Path("external/depth-anything-3"))
    ap.add_argument("--model-name", default="depth-anything/DA3NESTED-GIANT-LARGE")
    ap.add_argument("--process-res", type=int, default=336)
    ap.add_argument("--chunk-size", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep-going", action="store_true",
                    help="Continue to the next clip if one fails.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # Respect per-worker thread caps (set by the launcher env) so many workers
    # don't oversubscribe the CPU. Does not affect outputs.
    _nt = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
    if _nt > 0:
        try:
            cv2.setNumThreads(_nt)
        except Exception:
            pass
        try:
            torch.set_num_threads(_nt)
        except Exception:
            pass

    root = args.root.resolve()
    if args.da3_root:
        sys.path.insert(0, str(Path(args.da3_root).resolve() / "src"))
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError:
        sys.exit("Failed to import depth_anything_3 — pass --da3-root <repo>.")

    videos = [
        Path(line.strip()).resolve()
        for line in args.video_list.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"no videos in {args.video_list}")

    print(f"[da3-batch] {len(videos)} clips | device={args.device} "
          f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES','?')} "
          f"process_res={args.process_res} chunk_size={args.chunk_size}", flush=True)
    print(f"[da3-batch] loading {args.model_name} ONCE ...", flush=True)
    t0 = time.time()
    model = DepthAnything3.from_pretrained(args.model_name).to(args.device).eval()
    print(f"[da3-batch] model loaded in {time.time()-t0:.1f}s", flush=True)

    done = skipped = failed = 0
    for i, video in enumerate(videos, 1):
        uid = video.stem.split(".")[0]
        preproc = root / args.preproc_name / f"{uid}_da3"
        user_npz = root / args.tracks_name / f"{uid}_user.npz"
        depth_p = preproc / "depth.npy"
        extr_p = preproc / "extrinsics.npy"
        intr_p = preproc / "intrinsics.npy"

        if user_npz.exists():
            skipped += 1
            if skipped % 200 == 0:
                print(f"[{i}/{len(videos)}] skip (exists): {skipped} skipped so far", flush=True)
            continue

        tc = time.time()
        try:
            frames_rgb = read_frames_rgb(video)
            if depth_p.exists() and extr_p.exists() and intr_p.exists():
                depth = np.load(depth_p)
                extr = np.load(extr_p)
                intr = np.load(intr_p)
                stage = "reuse-da3"
            else:
                pil = [Image.fromarray(f) for f in frames_rgb]
                depth, extr, intr = da3_depth_extr_intr(
                    model, pil, args.process_res, args.chunk_size, args.device)
                _atomic_save(depth_p, depth)
                _atomic_save(extr_p, extr)
                _atomic_save(intr_p, intr)
                stage = "da3"
            T, H, W = build_user_npz(frames_rgb, depth, extr, intr, user_npz)
            done += 1
            print(f"[{i}/{len(videos)}] {uid} {stage} T={T} {W}x{H} "
                  f"{time.time()-tc:.2f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{i}/{len(videos)}] FAILED {uid}: {exc}", file=sys.stderr, flush=True)
            if not args.keep_going:
                raise

    print(f"[da3-batch] done={done} skipped={skipped} failed={failed}", flush=True)
    if failed and not args.keep_going:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
