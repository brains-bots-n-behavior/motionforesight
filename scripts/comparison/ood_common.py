"""Shared OOD inference: processed user.npz (+ SAM mask) -> model prediction.

Returns predicted future 3D tracks (camera-subtracted, last-observed frame) at
the masked object's points, plus the observed point cloud and intrinsics, for
both the viser viewer and the static HTML renderer.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.cm as cm

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models_pretrained.future_scene_flow.sparse_dataset import (  # noqa: E402
    _unproject_world, _apply_rigid, _compute_pj_norm)


def ood_predict(model, user_npz, mask_png, grid_stride=2):
    cfg = model.config
    obs, total, mh, mw = cfg.obs_frames, cfg.total_frames, cfg.height, cfg.width
    dev = model.device
    d = np.load(user_npz)
    rgb, depth, w2c = d["rgb"], d["depth_map"], d["extrinsics_w2c"].astype(np.float64)
    fx, fy, cx, cy = d["fx_fy_cx_cy"]; Hd, Wd = rgb.shape[1], rgb.shape[2]
    mask = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE) > 0

    rgb_obs = (np.stack([cv2.resize(rgb[t], (mw, mh)) for t in range(obs)]).astype(np.float32) / 127.5 - 1).transpose(0, 3, 1, 2)
    intr_m = np.array([fx * mw / Wd, fy * mh / Hd, cx * mw / Wd, cy * mh / Hd])
    pjw = np.stack([_unproject_world(cv2.resize(depth[t], (mw, mh)).astype(np.float64), intr_m, w2c[t]) for t in range(obs)])
    pj = _apply_rigid(w2c[obs - 1], pjw)                              # obs,mh,mw,3 cam-lastobs (m)
    nm = _compute_pj_norm(pj, 80, 2, 98); mean, scale = nm.mean, float(nm.scale)
    pj_n = ((pj - mean) / scale).astype(np.float32)
    p0 = np.transpose(pj_n[0], (2, 0, 1))

    rgb_t = torch.from_numpy(rgb_obs).unsqueeze(0).to(dev)
    pj_t = torch.from_numpy(np.transpose(pj_n, (0, 3, 1, 2))).unsqueeze(0).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        q = model(rgb_t, pj_t); xyz, _ = model.split_latent(q); delta = model.decode_xyz(xyz).float()
    pred = model.reconstruct(delta, torch.from_numpy(p0).unsqueeze(0).to(dev).float())
    pred_m = model.denormalize(pred, torch.tensor(mean).unsqueeze(0).to(dev).float(),
                               torch.tensor(scale).reshape(1).to(dev).float())[0].cpu().numpy()  # 3,T,mh,mw

    mask_m = cv2.resize(mask.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST) > 0
    gy, gx = np.mgrid[0:mh, 0:mw]
    grid = mask_m & (gy % grid_stride == 0) & (gx % grid_stride == 0)
    ys, xs = np.where(grid)
    if ys.size == 0:
        ys, xs = np.where(mask_m)
    pred_pts = pred_m[:, :, ys, xs].transpose(2, 1, 0)               # N,T,3 cam-lastobs
    xn = (xs - xs.min()) / max(1, xs.max() - xs.min())
    track_col = (np.array([cm.hsv(v)[:3] for v in xn]) * 255).astype(np.uint8)
    rgb8 = ((rgb_obs.transpose(0, 2, 3, 1) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return dict(pred_pts=pred_pts, pred_m=pred_m, xs=xs, ys=ys, track_col=track_col, pj=pj, rgb8=rgb8,
                intr_m=intr_m, obs=obs, total=total, mh=mh, mw=mw, N=int(ys.size))
