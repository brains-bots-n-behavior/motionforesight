#!/usr/bin/env python3
"""Predict future 3D point tracks with the pretrained-TrackCraft3r model.

Loads the base TrackCraft3r weights + a small trained checkpoint from
``train_future_scene_flow.py`` (which stored only the fine-tuned params), feeds
the first ``obs_frames`` of a dense clip, and writes a dense prediction NPZ in
the same schema TrackCraft3r's ``visualize_dense.py`` / the repo's future-track
viewer understand::

    track_map : (T, H, W, 3)   predicted per-pixel 3D track in frame-0 cam space
    recon_map : (T, H, W, 3)   observed Pj(t) (future frames left as zeros)
    rgb       : (obs, H, W, 3)  observed RGB frames (model resolution)

Example::

    CUDA_VISIBLE_DEVICES=0 python scripts/predict_future_scene_flow.py \
      --train-ckpt data/.../pretrained_tc3r_10f_to_32f/best.pt \
      --dense-npz data/something_something/anchor_tracks32_curated_dense/100368_anchor_dense.npz \
      --output-npz /tmp/100368_future_pred.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models_pretrained  # noqa: E402,F401
from models_pretrained.future_scene_flow import FutureSceneFlowConfig, FutureSceneFlowModel  # noqa: E402
from models_pretrained.future_scene_flow.dataset import (  # noqa: E402
    FutureSceneFlowDataset,
    TrackSampleIndex,
)

DEFAULT_CKPT = REPO_ROOT / "models_pretrained" / "checkpoints" / "trackcraft3r" / "model.safetensors"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-ckpt", type=Path, required=True,
                   help="best.pt/last.pt from train_future_scene_flow.py")
    p.add_argument("--dense-npz", type=Path, required=True,
                   help="A *_dense.npz to read observed rgb/recon_map from.")
    p.add_argument("--output-npz", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_CKPT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda")

    state = torch.load(args.train_ckpt, map_location="cpu", weights_only=False)
    cfg_d = state["config"]
    config = FutureSceneFlowConfig(
        checkpoint_path=str(args.base_checkpoint.expanduser().resolve()),
        lora_rank=cfg_d.get("lora_rank", 1024),
        height=cfg_d["height"], width=cfg_d["width"],
        obs_frames=cfg_d["obs_frames"], total_frames=cfg_d["total_frames"],
        trainable=tuple(cfg_d.get("trainable", ["mask", "io", "head"])),
        predict_vis=cfg_d.get("predict_vis", True),
    )
    model = FutureSceneFlowModel(config)
    missing, unexpected = model.load_state_dict(state["trainable_state"], strict=False)
    loaded = len(state["trainable_state"])
    print(f"loaded {loaded} fine-tuned tensors (unexpected={len(unexpected)})")
    model.eval()

    # Build one sample from the dense clip (reuses dataset normalization).
    item = TrackSampleIndex(
        video_id=args.dense_npz.stem, dense_path=args.dense_npz, mask_paths=(),
    )
    ds = FutureSceneFlowDataset(
        [item], obs_frames=config.obs_frames, total_frames=config.total_frames,
        image_size=(config.height, config.width),
    )
    s = ds[0]
    rgb_obs = s["rgb_obs"].unsqueeze(0).to(device)
    pj_obs = s["pj_obs"].unsqueeze(0).to(device)
    p0 = s["p0_t0_norm"].unsqueeze(0).to(device).float()
    pj_mean = s["pj_mean"].unsqueeze(0).to(device).float()
    pj_scale = s["pj_scale"].unsqueeze(0).to(device).float()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        query = model(rgb_obs, pj_obs)
        xyz, _ = model.split_latent(query)
        delta = model.decode_xyz(xyz).float()
        pred_norm = model.reconstruct(delta, p0)
        pred_m = model.denormalize(pred_norm, pj_mean, pj_scale)  # 1,3,T,H,W

    track_map = pred_m[0].permute(1, 2, 3, 0).cpu().numpy().astype(np.float32)  # T,H,W,3
    recon_obs = pj_obs[0].permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)  # obs,H,W,3
    # Denormalize observed Pj for the viewer's point cloud.
    recon_obs = recon_obs * pj_scale.item() + pj_mean[0].cpu().numpy()
    T = config.total_frames
    recon_full = np.zeros((T, config.height, config.width, 3), dtype=np.float32)
    recon_full[: config.obs_frames] = recon_obs
    rgb_np = ((rgb_obs[0].permute(0, 2, 3, 1).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, track_map=track_map, recon_map=recon_full, rgb=rgb_np)
    print(f"saved {args.output_npz}")
    print(f"  track_map {track_map.shape}  ({config.obs_frames} observed, "
          f"{config.future_frames} predicted future frames)")


if __name__ == "__main__":
    main()
