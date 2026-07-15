#!/usr/bin/env python3
"""Preprocess raw OOD videos into model-ready inputs (3dflow env).

For each video: sample N frames, run Depth-Anything-3 (depth + camera
+ intrinsics, the same estimator used for the training data), and save a
`_user.npz` matching the training convention:

    rgb            : (N, H, W, 3) uint8         resized RGB frames
    depth_map      : (N, H, W)   float32        z-depth (DA3)
    extrinsics_w2c : (N, 4, 4)   float32        world->cam, frame-0 = identity
    fx_fy_cx_cy    : (4,)        float64        intrinsics at (H, W)
    frame_indices  : (N,)        original frame indices

These feed the model exactly like the curated training data (Pj is computed from
depth at load time). No GT tracks (these are zero-shot test clips).

Usage:
    python scripts/comparison/preprocess_ood_videos.py \
      --video-dir zero-shot-eval --out-dir zero-shot-eval/processed \
      --num-frames 22 --sample-mode context50 --height 480 --width 832
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

REPO = Path(__file__).resolve().parents[2]
DA3_SRC = REPO / "external/depth-anything-3/src"
if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default="zero-shot-eval")
    ap.add_argument("--out-dir", default="zero-shot-eval/processed")
    ap.add_argument("--num-frames", type=int, default=10, help="number of frames to sample")
    ap.add_argument("--frame-stride", type=int, default=0,
                    help="0 = uniform linspace of num-frames over the whole clip (observed 7 + future "
                         "15 span the full video). >0 = num-frames CONSECUTIVE frames at this stride "
                         "from --start-frame (matches training's stride-1 clips).")
    ap.add_argument("--sample-mode", choices=("uniform", "context50"), default="uniform",
                    help="uniform = sample --num-frames across the full clip, or use --frame-stride. "
                         "context50 = for full videos, sample --obs-frames context frames uniformly "
                         "from the first 50%% and the remaining frames uniformly from the second 50%%.")
    ap.add_argument("--obs-frames", type=int, default=7,
                    help="number of context frames when --sample-mode context50")
    ap.add_argument("--start-frame", type=int, default=0, help="first frame index when --frame-stride>0")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--model-name", default="depth-anything/DA3NESTED-GIANT-LARGE")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def sample_indices(total, n, frame_stride=0, start_frame=0, sample_mode="uniform", obs_frames=7):
    if total <= 0:
        raise RuntimeError("cannot sample frames from an empty video")
    if sample_mode == "context50":
        if frame_stride > 0:
            raise ValueError("--sample-mode context50 cannot be combined with --frame-stride > 0")
        if not (0 < obs_frames < n):
            raise ValueError("--obs-frames must be between 1 and --num-frames - 1")
        split = int(round((total - 1) * 0.5))
        ctx = np.linspace(0, split, obs_frames).round().astype(int)
        fut_start = min(total - 1, split + 1)
        fut = np.linspace(fut_start, total - 1, n - obs_frames).round().astype(int)
        return np.clip(np.maximum.accumulate(np.concatenate([ctx, fut]).astype(int)), 0, total - 1)
    if frame_stride > 0:
        idx = start_frame + np.arange(n) * frame_stride
        if idx[-1] >= total:
            raise RuntimeError(
                f"requested frame index {int(idx[-1])} from {total} frames "
                f"(start={start_frame}, stride={frame_stride}, n={n})"
            )
        return idx.astype(int)
    return np.linspace(0, total - 1, n).round().astype(int)


def sample_frames(path, n, H, W, frame_stride=0, start_frame=0, sample_mode="uniform", obs_frames=7):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA))
    cap.release()
    total = len(frames)
    if total == 0:
        raise RuntimeError(f"no frames decoded from {path}")
    idx = sample_indices(total, n, frame_stride, start_frame, sample_mode, obs_frames)
    rgb_seq = np.stack([frames[int(j)] for j in idx])  # N,H,W,3
    return rgb_seq, idx, total


def main():
    a = parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    from depth_anything_3.api import DepthAnything3
    print(f"loading DA3 {a.model_name} ...", flush=True)
    model = DepthAnything3.from_pretrained(a.model_name).to(a.device).eval()

    vids = sorted(glob.glob(os.path.join(a.video_dir, "*.mp4")))
    print(f"{len(vids)} videos", flush=True)
    manifest = []
    for vp in vids:
        vid = Path(vp).stem
        rgb, idx, raw_total = sample_frames(
            vp, a.num_frames, a.height, a.width,
            frame_stride=a.frame_stride, start_frame=a.start_frame,
            sample_mode=a.sample_mode, obs_frames=a.obs_frames,
        )   # N,H,W,3
        pil = [Image.fromarray(f) for f in rgb]
        with torch.no_grad():
            pred = model.inference(image=pil, process_res=a.process_res,
                                   process_res_method="upper_bound_resize", export_dir=None)
        depth_p = np.asarray(pred.depth, np.float32)                    # N,Hp,Wp z-depth
        extr = np.asarray(pred.extrinsics, np.float64)                 # N,3,4 w2c
        intr = np.asarray(pred.intrinsics, np.float64)                 # N,3,3 @ proc res
        Hp, Wp = depth_p.shape[1:]
        # resize depth -> (H,W)
        depth = np.stack([cv2.resize(depth_p[t], (a.width, a.height), interpolation=cv2.INTER_LINEAR)
                          for t in range(len(pil))]).astype(np.float32)
        # extrinsics (N,3,4)->(N,4,4), frame-0 normalize to identity
        E = np.tile(np.eye(4), (len(pil), 1, 1)); E[:, :3, :4] = extr
        inv0 = np.linalg.inv(E[0]); E = np.stack([E[t] @ inv0 for t in range(len(pil))]).astype(np.float32)
        # intrinsics: scale proc-res -> (H,W), take frame 0
        K = intr[0].copy(); sx = a.width / Wp; sy = a.height / Hp
        fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy
        # JPEG-encode frames so TrackCraft3r's loader (images_jpeg_bytes) can run
        # dense tracking on these clips too (real GT tracks). Keep raw rgb as well.
        jpg = np.array([cv2.imencode(".jpg", cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()
                        for t in range(len(rgb))], dtype=object)
        np.savez_compressed(out / f"{vid}_user.npz",
                            images_jpeg_bytes=jpg,
                            rgb=rgb.astype(np.uint8), depth_map=depth,
                            extrinsics_w2c=E, fx_fy_cx_cy=np.array([fx, fy, cx, cy], np.float64),
                            frame_indices=idx.astype(np.int32))
        manifest.append({
            "video_id": vid,
            "raw_num_frames": int(raw_total),
            "num_sampled_frames": int(len(idx)),
            "sample_mode": a.sample_mode,
            "context_frame_indices": [int(x) for x in idx[:a.obs_frames]],
            "future_frame_indices_not_input": [int(x) for x in idx[a.obs_frames:]],
        })
        print(f"  {vid}: {len(pil)} frames | depth {depth.shape} | "
              f"context {idx[:a.obs_frames].tolist()} | future {idx[a.obs_frames:].tolist()} | "
              f"fx,fy,cx,cy=({fx:.1f},{fy:.1f},{cx:.1f},{cy:.1f}) | "
              f"cam0->camN trans {np.round(E[-1][:3,3],3)}", flush=True)
    (out / "frame_sampling_manifest.json").write_text(json.dumps({"items": manifest}, indent=2))
    print(f"wrote {len(vids)} user npz to {out}")


if __name__ == "__main__":
    main()
