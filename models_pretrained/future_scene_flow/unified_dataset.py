"""Unified dataset over BOTH dense (`*_dense.npz`) and sparse (`*_sparse.npz`)
curated TrackCraft3r clips, emitting one common sparse-style supervision target
(GT tracks at query points) with camera-motion subtraction to the last observed
frame. Lets us train the fresh-LoRA model on all curated data at once.

Per-clip output (same keys for both formats; variable N -> use batch_size=1):
    rgb_obs (obs,3,h,w) | pj_obs (obs,3,h,w) | p0_t0_norm (3,h,w)
    gt_tracks_norm (T,N,3) | query_uv_model (N,2) | visibility (T,N)
    pj_mean (3,) | pj_scale () | w2c_lastobs (4,4) | video_id | text
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .dataset import _compute_pj_norm, _load_union_mask, build_track_index
from .sparse_dataset import (build_sparse_index, _resize_rgb, _unproject_world,
                             _apply_rigid, sparse_collate)


@dataclass(frozen=True)
class UnifiedItem:
    fmt: str            # "dense" | "sparse"
    video_id: str
    path: Path          # dense_path or sparse_path
    mask_paths: tuple = ()
    text: str = ""


def build_unified_index(root, dense_tracks_name, dense_manifest, sparse_tracks_name,
                        limit_dense=0, limit_sparse=0):
    root = Path(root).expanduser().resolve()
    items = []
    if dense_tracks_name:
        for it in build_track_index(root, tracks_name=dense_tracks_name, manifest=dense_manifest,
                                    require_masks=True, limit_clips=limit_dense):
            items.append(UnifiedItem("dense", it.video_id, it.dense_path, tuple(it.mask_paths), it.text))
    if sparse_tracks_name:
        for it in build_sparse_index(root, sparse_tracks_name, limit_clips=limit_sparse):
            items.append(UnifiedItem("sparse", it.video_id, it.sparse_path, (), it.text))
    return items


def split_items(items, val_fraction, seed):
    rng = np.random.default_rng(seed)
    order = np.arange(len(items)); rng.shuffle(order)
    nval = max(1, int(round(len(items) * val_fraction))) if len(items) > 1 else 0
    vs = set(order[:nval].tolist())
    return [it for i, it in enumerate(items) if i not in vs], [it for i, it in enumerate(items) if i in vs]


def _user_path(p: Path, fmt: str) -> Path:
    suf = "_dense.npz" if fmt == "dense" else "_sparse.npz"
    return p.with_name(p.name.replace(suf, "_user.npz"))


def _resize_coord(arr, h, w):
    out = np.empty((arr.shape[0], h, w, 3), np.float64)
    for t in range(arr.shape[0]):
        out[t] = cv2.resize(arr[t].astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return out


class UnifiedTrackDataset(Dataset):
    def __init__(self, items, obs_frames=10, total_frames=32, image_size=(320, 576),
                 num_points=48, diag_max_depth=80.0, lo=2.0, hi=98.0,
                 samples_per_clip=1, seed=0, subtract_camera_motion=True):
        if not items:
            raise ValueError("empty item list")
        self.items = items
        self.obs, self.total = obs_frames, total_frames
        self.h, self.w = image_size
        self.num_points = num_points
        self.diag_max_depth, self.lo, self.hi = diag_max_depth, lo, hi
        self.spc = max(1, samples_per_clip); self.seed = seed
        self.sub_cam = subtract_camera_motion

    def __len__(self):
        return len(self.items) * self.spc

    def _finish(self, pj_obs_world, tracks_world, quv_xy, vis, w2c_lastobs, rgb_obs_chw, video_id, text):
        """Shared tail: camera-subtract, normalize, package."""
        h, w, obs = self.h, self.w, self.obs
        T_sub = w2c_lastobs if self.sub_cam else np.eye(4)
        pj = _apply_rigid(T_sub, pj_obs_world)                  # obs,h,w,3
        tr = _apply_rigid(T_sub, tracks_world)                 # T,N,3
        norm = _compute_pj_norm(pj.reshape(obs, h, w, 3), self.diag_max_depth, self.lo, self.hi)
        mean, scale = norm.mean.astype(np.float64), float(norm.scale)
        pj_n = ((pj - mean) / scale).astype(np.float32)
        gt_n = ((tr - mean) / scale).astype(np.float32)
        to_chw = lambda a: np.transpose(a, (0, 3, 1, 2))
        return {
            "rgb_obs": torch.from_numpy(rgb_obs_chw),
            "pj_obs": torch.from_numpy(to_chw(pj_n)),
            "p0_t0_norm": torch.from_numpy(np.transpose(pj_n[0], (2, 0, 1))),
            "gt_tracks_norm": torch.from_numpy(gt_n),
            "query_uv_model": torch.from_numpy(quv_xy.astype(np.float32)),
            "visibility": torch.from_numpy(vis.astype(np.float32)),
            "pj_mean": torch.from_numpy(mean.astype(np.float32)),
            "pj_scale": torch.tensor(scale, dtype=torch.float32),
            "w2c_lastobs": torch.from_numpy(T_sub.astype(np.float32)),
            "video_id": video_id, "text": text,
        }

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        h, w, obs, T = self.h, self.w, self.obs, self.total
        rng = np.random.default_rng(self.seed + idx * 9973)
        if item.fmt == "sparse":
            return self._get_sparse(item, rng)
        return self._get_dense(item, rng)

    # ------------------------------------------------------------------ sparse
    def _get_sparse(self, item, rng):
        h, w, obs, T = self.h, self.w, self.obs, self.total
        s = np.load(item.path, allow_pickle=True)
        rgb = s["rgb"][:T]
        tracks = s["tracks_xyz"][:T].astype(np.float64)            # T,N,3 world
        quv = s["query_uv"].astype(np.float64)                     # N,2 at image_size
        vis = s["visibility"][:T].astype(np.float32)
        w2c = s["extrinsics_w2c"][:T].astype(np.float64)
        Hi, Wi = [int(x) for x in s["image_size_hw"]]
        fx, fy, cx, cy = s["fx_fy_cx_cy"].astype(np.float64)
        u = np.load(_user_path(item.path, "sparse"), allow_pickle=True)
        depth = u["depth_map"][:T].astype(np.float64)
        intr_m = np.array([fx * w / Wi, fy * h / Hi, cx * w / Wi, cy * h / Hi])
        pj_world = np.stack([
            _unproject_world(cv2.resize(depth[t], (w, h), interpolation=cv2.INTER_NEAREST), intr_m, w2c[t])
            for t in range(obs)])                                  # obs,h,w,3
        quv_m = np.stack([quv[:, 0] * w / Wi, quv[:, 1] * h / Hi], -1)
        finite = np.isfinite(tracks).all(axis=(0, 2))
        vis = vis * finite[None, :]
        return self._finish(pj_world, tracks, quv_m, vis, w2c[obs - 1],
                            _resize_rgb(rgb[:obs], h, w), item.video_id, str(s["text"]))

    # ------------------------------------------------------------------- dense
    def _get_dense(self, item, rng):
        h, w, obs, T, N = self.h, self.w, self.obs, self.total, self.num_points
        d = np.load(item.path)
        track_map = d["track_map"][:T].astype(np.float64)          # T,Hd,Wd,3 world
        recon_map = d["recon_map"][:obs].astype(np.float64)        # obs,Hd,Wd,3 world (Pj)
        rgb = d["rgb"][:obs]
        Hd, Wd = track_map.shape[1:3]
        u = np.load(_user_path(item.path, "dense"), allow_pickle=True)
        w2c = u["extrinsics_w2c"][:T].astype(np.float64)

        mask = _load_union_mask(item.mask_paths, (Hd, Wd))
        valid = np.isfinite(track_map).all(axis=(0, 3))
        cand = mask & valid                      # ONLY points on the SAM object mask
        if cand.sum() == 0:                      # degenerate: mask has no valid pixel
            cand = valid
        ys, xs = np.where(cand)
        sel = rng.choice(ys.size, size=N, replace=ys.size < N)  # N from mask (w/ replacement if few)
        ys, xs = ys[sel], xs[sel]
        tracks = track_map[:, ys, xs, :]                           # T,N,3 world
        vis = np.ones((T, len(xs)), np.float32)

        pj_world = _resize_coord(recon_map, h, w)                  # obs,h,w,3 world
        quv_m = np.stack([xs.astype(np.float64) * w / Wd, ys.astype(np.float64) * h / Hd], -1)
        return self._finish(pj_world, tracks, quv_m, vis, w2c[obs - 1],
                            _resize_rgb(rgb, h, w), item.video_id, item.text)
