"""Dataset for future-scene-flow training on TrackCraft3r dense outputs.

Each ``*_dense.npz`` produced by TrackCraft3r stores exactly the tensors the
pretrained pipeline consumes and supervises against:

    rgb        : (T, H, W, 3) uint8   model-resolution RGB frames
    recon_map  : (T, H, W, 3) float32 Pj(t): per-frame depth back-projection in
                                      frame-0 camera space (the geometry input)
    track_map  : (T, H, W, 3) float32 GT P0(t): where each frame-0 point goes

We feed the observed ``rgb`` / ``recon_map`` (frames ``0 .. obs-1``) to the
model and supervise the full-clip ``track_map``.  Pj normalization (z-inlier
percentile -> mean-center -> max-distance scale) mirrors
``WanSceneFlowPredictor.predict`` but is computed from the *observed* frames
only, because at inference the future depth is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Reuse the repo-local index/mask helpers (these do not touch TrackCraft3r).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from models.future_3d_tracks.dataset import (  # noqa: E402
    TrackSampleIndex,
    _load_union_mask,
    build_track_index,
    split_items,
)

__all__ = [
    "FutureSceneFlowDataset",
    "TrackSampleIndex",
    "build_track_index",
    "split_items",
    "future_collate",
]


@dataclass
class PjNorm:
    mean: np.ndarray  # (3,)
    scale: float


def _resize_coord_map(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a (T,H,W,3) coordinate map with nearest interpolation."""
    out = np.empty((arr.shape[0], h, w, 3), dtype=np.float32)
    for t in range(arr.shape[0]):
        out[t] = cv2.resize(arr[t], (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def _resize_rgb(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    """(T,H,W,3) uint8 -> (T,3,h,w) float32 in [-1, 1]."""
    out = np.empty((rgb.shape[0], h, w, 3), dtype=np.float32)
    for t in range(rgb.shape[0]):
        out[t] = cv2.resize(rgb[t], (w, h), interpolation=cv2.INTER_AREA)
    out = out / 127.5 - 1.0
    return np.transpose(out, (0, 3, 1, 2))


def _compute_pj_norm(
    recon_obs: np.ndarray,
    diag_max_depth: float,
    lo: float,
    hi: float,
) -> PjNorm:
    """Replicate TrackCraft3r Pj normalization from observed frames."""
    pts = recon_obs.reshape(-1, 3).copy()
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if diag_max_depth > 0:
        pts[:, 2] = np.clip(pts[:, 2], 1e-6, diag_max_depth)
    z = pts[:, 2]
    z_lo, z_hi = np.percentile(z, lo), np.percentile(z, hi)
    inlier = (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)
    inlier_pts = pts[inlier] if inlier.any() else pts
    mean = inlier_pts.mean(axis=0)
    centered = inlier_pts - mean
    scale = float(np.linalg.norm(centered, axis=1).max())
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return PjNorm(mean=mean.astype(np.float32), scale=scale)


class FutureSceneFlowDataset(Dataset):
    def __init__(
        self,
        items: list[TrackSampleIndex],
        obs_frames: int = 10,
        total_frames: int = 32,
        image_size: tuple[int, int] = (224, 384),
        diag_max_depth: float = 80.0,
        pj_norm_percentile_lo: float = 2.0,
        pj_norm_percentile_hi: float = 98.0,
        samples_per_clip: int = 1,
        seed: int = 0,
    ) -> None:
        if obs_frames >= total_frames:
            raise ValueError("obs_frames must be smaller than total_frames")
        if not items:
            raise ValueError("FutureSceneFlowDataset received an empty item list")
        h, w = image_size
        if h % 16 or w % 16:
            raise ValueError(f"image_size must be multiples of 16, got {image_size}")
        self.items = items
        self.obs_frames = obs_frames
        self.total_frames = total_frames
        self.image_size = image_size
        self.diag_max_depth = diag_max_depth
        self.lo = pj_norm_percentile_lo
        self.hi = pj_norm_percentile_hi
        self.samples_per_clip = max(1, samples_per_clip)
        self.seed = seed

    def __len__(self) -> int:
        return len(self.items) * self.samples_per_clip

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx % len(self.items)]
        h, w = self.image_size
        T = self.total_frames
        obs = self.obs_frames

        with np.load(item.dense_path, allow_pickle=True) as data:
            track_map = data["track_map"][:T].astype(np.float32)   # T,H0,W0,3
            recon_map = data["recon_map"][:T].astype(np.float32)
            rgb = data["rgb"][:obs]
        if track_map.shape[0] < T:
            raise RuntimeError(f"{item.dense_path} has only {track_map.shape[0]} frames")

        obj_mask_full = _load_union_mask(item.mask_paths, track_map.shape[1:3])

        # Resize everything to model resolution.
        rgb_t = _resize_rgb(rgb, h, w)                         # obs,3,h,w
        recon_r = _resize_coord_map(recon_map, h, w)          # T,h,w,3
        track_r = _resize_coord_map(track_map, h, w)          # T,h,w,3
        obj_mask = cv2.resize(
            obj_mask_full.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # Pj normalization from observed frames only.
        norm = _compute_pj_norm(recon_r[:obs], self.diag_max_depth, self.lo, self.hi)
        mean, scale = norm.mean, norm.scale

        recon_finite = np.isfinite(recon_r).all(axis=-1)       # T,h,w
        track_finite = np.isfinite(track_r).all(axis=-1)

        recon_norm = (np.nan_to_num(recon_r, nan=0.0, posinf=0.0, neginf=0.0) - mean) / scale
        track_norm = (np.nan_to_num(track_r, nan=0.0, posinf=0.0, neginf=0.0) - mean) / scale
        recon_norm[~recon_finite] = 0.0
        track_norm[~track_finite] = 0.0

        p0_t0_norm = recon_norm[0]                              # h,w,3
        target_delta = track_norm - p0_t0_norm[None]           # T,h,w,3
        valid = track_finite & recon_finite[0][None]           # need frame-0 anchor too

        to_chw = lambda a: np.transpose(a, (0, 3, 1, 2))       # T,h,w,3 -> T,3,h,w

        return {
            "rgb_obs": torch.from_numpy(rgb_t),                                  # obs,3,h,w
            "pj_obs": torch.from_numpy(to_chw(recon_norm[:obs])),               # obs,3,h,w
            "target_delta": torch.from_numpy(to_chw(target_delta)),            # T,3,h,w
            "p0_t0_norm": torch.from_numpy(np.transpose(p0_t0_norm, (2, 0, 1))),  # 3,h,w
            "track_norm": torch.from_numpy(to_chw(track_norm)),                # T,3,h,w (metric via denorm)
            "valid": torch.from_numpy(valid),                                  # T,h,w bool
            "obj_mask": torch.from_numpy(obj_mask),                            # h,w bool
            "pj_mean": torch.from_numpy(mean),                                 # 3,
            "pj_scale": torch.tensor(scale, dtype=torch.float32),
            "video_id": item.video_id,
            "text": item.text,
        }


def future_collate(batch: list[dict]) -> dict:
    out: dict = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if torch.is_tensor(vals[0]):
            out[key] = torch.stack(vals, dim=0)
        else:
            out[key] = vals
    return out
