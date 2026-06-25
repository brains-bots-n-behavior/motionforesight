"""Dataset for the sparse TrackCraft3r data (`*_sparse.npz`) with camera-motion
subtraction to the last-observed frame.

Unlike the dense `*_dense.npz` (which ships `rgb`/`recon_map`/`track_map`), the
sparse clips ship `rgb` + sparse `tracks_xyz (T,N,3)` (cam-0) + `query_uv` +
`visibility` + `extrinsics_w2c`, and the paired `*_user.npz` ships `depth_map`.
So here we:

  1. build the Pj geometry input by unprojecting observed depth into cam-0,
  2. **subtract camera motion** by expressing Pj, the frame-0 anchor, and the GT
     tracks in the *last observed frame's* camera (apply ``w2c[obs-1]``), so the
     model predicts/supervises future tracks as seen from a camera frozen at the
     last observed pose (ego-motion removed),
  3. supervise sparsely at the `query_uv` points (visibility-masked).

Returns per-clip variable-N tensors -> use ``batch_size=1`` (+ grad-accum).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .dataset import _compute_pj_norm  # reuse z-inlier Pj normalization


@dataclass(frozen=True)
class SparseItem:
    video_id: str
    sparse_path: Path
    text: str = ""


def build_sparse_index(root: Path, tracks_name: str, limit_clips: int = 0) -> list[SparseItem]:
    root = Path(root).expanduser().resolve()
    items = []
    for p in sorted((root / tracks_name).glob("*_sparse.npz")):
        vid = p.name[: -len("_sparse.npz")]
        items.append(SparseItem(video_id=vid, sparse_path=p))
        if limit_clips and len(items) >= limit_clips:
            break
    return items


def split_items(items, val_fraction, seed):
    rng = np.random.default_rng(seed)
    order = np.arange(len(items)); rng.shuffle(order)
    nval = max(1, int(round(len(items) * val_fraction))) if len(items) > 1 else 0
    vs = set(order[:nval].tolist())
    return [it for i, it in enumerate(items) if i not in vs], [it for i, it in enumerate(items) if i in vs]


def _resize_rgb(rgb, h, w):
    out = np.empty((rgb.shape[0], h, w, 3), np.float32)
    for t in range(rgb.shape[0]):
        out[t] = cv2.resize(rgb[t], (w, h), interpolation=cv2.INTER_AREA)
    return np.transpose(out / 127.5 - 1.0, (0, 3, 1, 2))


def _unproject_world(depth_hw, intr, w2c_t):
    """depth (h,w) + intrinsics(at h,w) + w2c_t -> world XYZ (h,w,3) (w2c[0]=I)."""
    h, w = depth_hw.shape
    fx, fy, cx, cy = intr
    vv, uu = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    z = depth_hw.astype(np.float64)
    pts = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z, np.ones_like(z)], -1)  # h,w,4 cam_t
    c2w = np.linalg.inv(w2c_t)
    return (pts @ c2w.T)[..., :3]


def _apply_rigid(T, X):
    """Apply 4x4 rigid T to points X (...,3)."""
    return X @ T[:3, :3].T + T[:3, 3]


class SparseTrackDataset(Dataset):
    def __init__(self, items, obs_frames=10, total_frames=32, image_size=(320, 576),
                 diag_max_depth=80.0, lo=2.0, hi=98.0, samples_per_clip=1, seed=0,
                 subtract_camera_motion=True):
        if not items:
            raise ValueError("empty item list")
        self.items = items
        self.obs, self.total = obs_frames, total_frames
        self.h, self.w = image_size
        self.diag_max_depth, self.lo, self.hi = diag_max_depth, lo, hi
        self.spc = max(1, samples_per_clip)
        self.seed = seed
        self.sub_cam = subtract_camera_motion

    def __len__(self):
        return len(self.items) * self.spc

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        h, w, obs, T = self.h, self.w, self.obs, self.total

        s = np.load(item.sparse_path, allow_pickle=True)
        rgb = s["rgb"][:T]                                   # T,Hr,Wr,3
        tracks = s["tracks_xyz"][:T].astype(np.float64)      # T,N,3 (cam0=world)
        quv = s["query_uv"].astype(np.float64)               # N,2 at (Hi,Wi)
        vis = s["visibility"][:T].astype(np.float32)         # T,N
        w2c = s["extrinsics_w2c"][:T].astype(np.float64)     # T,4,4
        Hi, Wi = [int(x) for x in s["image_size_hw"]]
        fx, fy, cx, cy = s["fx_fy_cx_cy"].astype(np.float64)  # at (Hi,Wi)
        user = np.load(str(s["user_npz"]) if Path(str(s["user_npz"])).exists()
                       else str(item.sparse_path).replace("_sparse.npz", "_user.npz"),
                       allow_pickle=True)
        depth = user["depth_map"][:T].astype(np.float64)     # T,Hd,Wd

        # ---- Pj (geometry) input from observed depth, at model resolution ----
        intr_m = np.array([fx * w / Wi, fy * h / Hi, cx * w / Wi, cy * h / Hi])
        pj_world = np.empty((obs, h, w, 3), np.float64)
        for t in range(obs):
            d = cv2.resize(depth[t], (w, h), interpolation=cv2.INTER_NEAREST)
            pj_world[t] = _unproject_world(d, intr_m, w2c[t])

        # ---- camera-motion subtraction: express in last-observed camera ----
        T_sub = w2c[obs - 1] if self.sub_cam else np.eye(4)
        pj = _apply_rigid(T_sub, pj_world)                   # obs,h,w,3
        tracks_s = _apply_rigid(T_sub, tracks)               # T,N,3

        # ---- normalize with observed-Pj statistics ----
        norm = _compute_pj_norm(pj.reshape(obs, h, w, 3), self.diag_max_depth, self.lo, self.hi)
        mean, scale = norm.mean.astype(np.float64), float(norm.scale)
        pj_n = ((pj - mean) / scale).astype(np.float32)      # obs,h,w,3
        p0 = pj_n[0]                                          # h,w,3
        gt_n = ((tracks_s - mean) / scale).astype(np.float32)  # T,N,3

        # query_uv at model res (for sampling our dense prediction)
        quv_m = np.stack([quv[:, 0] * w / Wi, quv[:, 1] * h / Hi], -1).astype(np.float32)
        finite = np.isfinite(gt_n).all(axis=(0, 2))          # N
        vis = vis * finite[None, :]

        to_chw = lambda a: np.transpose(a, (0, 3, 1, 2))     # T,h,w,3 -> T,3,h,w
        return {
            "rgb_obs": torch.from_numpy(_resize_rgb(rgb[:obs], h, w)),     # obs,3,h,w
            "pj_obs": torch.from_numpy(to_chw(pj_n)),                      # obs,3,h,w
            "p0_t0_norm": torch.from_numpy(np.transpose(p0, (2, 0, 1))),   # 3,h,w
            "gt_tracks_norm": torch.from_numpy(gt_n),                      # T,N,3
            "query_uv_model": torch.from_numpy(quv_m),                     # N,2 (x,y)
            "visibility": torch.from_numpy(vis.astype(np.float32)),        # T,N
            "pj_mean": torch.from_numpy(mean.astype(np.float32)),          # 3
            "pj_scale": torch.tensor(scale, dtype=torch.float32),
            "w2c_lastobs": torch.from_numpy(T_sub.astype(np.float32)),     # 4,4
            "video_id": item.video_id, "text": str(s["text"]),
        }


def sparse_collate(batch):
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        out[k] = torch.stack(v, 0) if torch.is_tensor(v[0]) else v
    return out
