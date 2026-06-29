#!/usr/bin/env python3
"""Run MolmoMotion on the OOD clips (molmomotion conda env).

Reads the CoTracker history prep, runs MolmoMotion (chunked to its fixed
num_points), and saves predicted future tracks in the camera-at-t0 frame
(= last-observed camera), for projection/comparison.
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from molmo_motion import MolmoMotion, MolmoMotionProcessor

CKPT = "allenai/MolmoMotion-4B-H3-F30"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", default="zero-shot-eval/molmo/prep")
    ap.add_argument("--out-dir", default="zero-shot-eval/molmo/results")
    ap.add_argument("--obs", type=int, default=10)
    ap.add_argument("--total", type=int, default=32)
    return ap.parse_args()


def main():
    a = parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    proc = MolmoMotionProcessor.from_pretrained(CKPT)
    model = MolmoMotion.from_pretrained(CKPT)
    model._internal = model._internal.to(torch.bfloat16).cuda()
    H, P = proc.config.history_size, proc.config.num_points
    obs, total, F = a.obs, a.total, a.total - a.obs
    print(f"MolmoMotion H={H} P={P} | future={F}", flush=True)

    for f in sorted(glob.glob(os.path.join(a.prep_dir, "*.npz"))):
        d = np.load(f, allow_pickle=True); vid = str(d["video_id"])
        rgb = d["rgb"]; w2c = d["w2c"].astype(np.float64)
        pts2d = d["pts2d_track"].astype(np.float64)      # T,Pn,2
        pts3d_world = d["pts3d_world"].astype(np.float64)  # T,Pn,3
        action = str(d["action"]); Pn = pts2d.shape[1]
        t0 = obs - 1; c2w9 = np.linalg.inv(w2c[t0])
        hist = list(range(t0 - (H - 1), t0 + 1))
        frames = [Image.fromarray(rgb[j]).convert("RGB") for j in hist]
        p2d_t0 = pts2d[t0]                                # Pn,2
        p3d_hist = pts3d_world[hist]                      # H,Pn,3 (world)

        fut_camt0 = np.zeros((Pn, F, 3), np.float64); ok = True
        for g0 in range(0, Pn, P):
            idx = np.arange(g0, min(g0 + P, Pn))
            if len(idx) < P:
                idx = np.concatenate([idx, np.full(P - len(idx), idx[-1])])
            inp = proc(history_frames=frames,
                       points_2d_at_t0=torch.from_numpy(p2d_t0[idx]).float(),
                       points_3d_history=torch.from_numpy(p3d_hist[:, idx]).float(),
                       action=action, future_horizon=F,
                       c2w_at_t0=torch.from_numpy(c2w9).float())
            inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model.predict_trajectory(**inp)
            g = o.future_3d.float().cpu().numpy()         # P,F,3 cam-at-t0
            if (g == 0).all():
                ok = False
            n = min(P, Pn - g0); fut_camt0[g0:g0 + n] = g[:n]

        np.savez(out / f"{vid}.npz", video_id=vid, molmo_camt0=fut_camt0.astype(np.float32),
                 query_xy0=d["query_xy0"], action=action, ok=ok)
        print(f"  {vid}: {Pn} pts, future {fut_camt0.shape} ok={ok}", flush=True)
    print("RUN_MOLMO_OOD_DONE")


if __name__ == "__main__":
    main()
