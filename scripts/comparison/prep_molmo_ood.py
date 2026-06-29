#!/usr/bin/env python3
"""Build MolmoMotion inputs for OOD clips (3dflow env).

MolmoMotion needs the query points' 3D history. OOD clips have no GT tracks, so
we track the SAM-mask points through the observed frames with CoTracker and lift
them to 3D using the DA3 depth + camera. Saves a per-clip history npz consumed by
run_molmo_ood.py (molmomotion env).
"""
from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path
import numpy as np
import cv2
import torch

REPO = Path(__file__).resolve().parents[2]
COTRACKER = os.path.expanduser("~/.cache/torch/hub/facebookresearch_co-tracker_main")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", default="zero-shot-eval/processed")
    ap.add_argument("--mask-dir", default="zero-shot-eval/masks")
    ap.add_argument("--out-dir", default="zero-shot-eval/molmo/prep")
    ap.add_argument("--num-points", type=int, default=48)
    ap.add_argument("--prompt", action="append", default=[], help="videoid:action caption")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def lift(u, v, depth, fx, fy, cx, cy, c2w):
    z = float(depth[int(round(np.clip(v, 0, depth.shape[0] - 1))),
                    int(round(np.clip(u, 0, depth.shape[1] - 1)))])
    cam = np.array([(u - cx) / fx * z, (v - cy) / fy * z, z, 1.0])
    return (c2w @ cam)[:3]


def main():
    a = parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    actions = {}
    for kv in a.prompt:
        vid, txt = kv.split(":", 1); actions[vid] = txt
    print("loading CoTracker3 ...", flush=True)
    ct = torch.hub.load(COTRACKER, "cotracker3_offline", source="local", pretrained=True).to(a.device).eval()

    for f in sorted(glob.glob(os.path.join(a.proc_dir, "*_user.npz"))):
        vid = os.path.basename(f).replace("_user.npz", "")
        mp = Path(a.mask_dir) / f"{vid}_mask.png"
        if not mp.exists():
            print(f"  {vid}: no mask"); continue
        d = np.load(f); rgb = d["rgb"]; depth = d["depth_map"]; w2c = d["extrinsics_w2c"].astype(np.float64)
        fx, fy, cx, cy = d["fx_fy_cx_cy"]; T, H, W = rgb.shape[:3]
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0
        ys, xs = np.where(mask)
        rng = np.random.default_rng(0)
        sel = rng.choice(ys.size, size=min(a.num_points, ys.size), replace=ys.size < a.num_points)
        qx, qy = xs[sel].astype(np.float32), ys[sel].astype(np.float32)

        video = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().unsqueeze(0).to(a.device)  # 1,T,3,H,W
        queries = torch.from_numpy(np.stack([np.zeros_like(qx), qx, qy], -1)).float().unsqueeze(0).to(a.device)
        with torch.no_grad():
            tracks, vis = ct(video, queries=queries)            # 1,T,P,2 ; 1,T,P
        tracks = tracks[0].cpu().numpy()                         # T,P,2 (px @ HxW)
        P = tracks.shape[1]
        c2w = np.stack([np.linalg.inv(w2c[t]) for t in range(T)])
        pts3d_world = np.stack([[lift(tracks[t, p, 0], tracks[t, p, 1], depth[t], fx, fy, cx, cy, c2w[t])
                                 for p in range(P)] for t in range(T)]).astype(np.float32)  # T,P,3 world

        np.savez(out / f"{vid}.npz", video_id=vid,
                 pts2d_track=tracks.astype(np.float32),          # T,P,2 (px)
                 pts3d_world=pts3d_world,                         # T,P,3 (world=cam0)
                 query_xy0=np.stack([qx, qy], -1).astype(np.float32),  # P,2 at frame 0
                 w2c=w2c.astype(np.float32), fx_fy_cx_cy=d["fx_fy_cx_cy"],
                 rgb=rgb, action=actions.get(vid, "a hand interacting with the object"))
        print(f"  {vid}: P={P} | action='{actions.get(vid,'(default)')}'", flush=True)
    print(f"wrote prep to {out}")


if __name__ == "__main__":
    main()
