#!/usr/bin/env python3
"""Run MolmoMotion-4B-H3-F30 on the prepared comparison clips.

RUN IN THE `molmomotion` CONDA ENV (not 3dflow). Reads the prep .npz files
written by prepare_comparison.py, runs MolmoMotion with its native interface
(3 history frames + query-point 3D history + action), and writes per-clip
results (predicted 3D tracks in cam-0/world + ADE/FDE).

MolmoMotion specifics handled here:
  - num_points is fixed (=8); prep already sampled exactly that many.
  - history_size=3 -> use frames [obs-3, obs-2, obs-1] (t0 = last observed).
  - points_2d_at_t0 = query points projected into the t0 (frame obs-1) image.
  - points_3d_history passed in WORLD frame (= cam-0, since w2c[0]=I) with
    c2w_at_t0 so the processor converts to camera-frame-at-t0.
  - output future_3d is in camera-frame-at-t0; converted back to world/cam-0.
"""
import argparse, glob, os, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from molmo_motion import MolmoMotion, MolmoMotionProcessor

CKPT = "allenai/MolmoMotion-4B-H3-F30"


def project(Xw, w2c9, intr):
    """world XYZ (...,3) -> pixel uv (...,2) in the t0 image."""
    fx, fy, cx, cy = intr
    flat = Xw.reshape(-1, 3)
    cam = (w2c9[:3, :3] @ flat.T).T + w2c9[:3, 3]
    z = np.clip(cam[:, 2], 1e-6, None)
    u = fx * cam[:, 0] / z + cx
    v = fy * cam[:, 1] / z + cy
    return np.stack([u, v], -1).reshape(*Xw.shape[:-1], 2).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    proc = MolmoMotionProcessor.from_pretrained(CKPT)
    model = MolmoMotion.from_pretrained(CKPT)
    model._internal = model._internal.to(torch.bfloat16).cuda()
    H, P = proc.config.history_size, proc.config.num_points
    print(f"loaded MolmoMotion (H={H}, P={P})", flush=True)

    rows = []
    files = sorted(glob.glob(os.path.join(args.prep_dir, "*.npz")))
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        vid0 = str(d["video_id"])
        if (out / f"{vid0}.npz").exists():
            print(f"[{i+1}/{len(files)}] {vid0} (skip, exists)", flush=True); continue
        try:
            _run_clip(i, f, d, proc, model, H, P, out, rows, len(files))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[{i+1}/{len(files)}] {vid0} OOM -> skipped (rerun to resume)", flush=True)
    _summary(rows)
    print("RUN_MOLMO_DONE")


def _summary(rows):
    okrows = [r for r in rows if r[2]]
    if okrows:
        print(f"\nMolmoMotion over {len(okrows)}/{len(rows)} parsed clips: "
              f"ADE={np.mean([r[0] for r in okrows])*100:.2f}cm "
              f"FDE={np.mean([r[1] for r in okrows])*100:.2f}cm")


def _run_clip(i, f, d, proc, model, H, P, out, rows, nfiles):
    if True:
        obs, total = int(d["obs_frames"]), int(d["total_frames"])
        gt = d["gt_3d_cam0"].astype(np.float64)          # T,Ptot,3 (world)
        w2c = d["w2c"].astype(np.float64)                 # T,4,4
        intr = d["intr_dense"].astype(np.float64)
        action = str(d["action"])
        Ptot = gt.shape[1]
        t0 = obs - 1
        w2c9 = w2c[t0]; c2w9 = np.linalg.inv(w2c9)
        F = total - obs

        # history frames [t0-(H-1) .. t0] as PIL (shared across point-groups)
        dense = np.load(str(d["dense_path"]))
        rgb = dense["rgb"]
        hist_idx = list(range(t0 - (H - 1), t0 + 1))
        frames = [Image.fromarray(rgb[j].astype("uint8")).convert("RGB") for j in hist_idx]
        pts2d_all = project(gt[t0], w2c9, intr)                      # (Ptot,2) at t0

        tp = time.time()
        # MolmoMotion predicts exactly P points/call -> chunk into groups of P.
        fut_world = np.zeros((Ptot, F, 3), np.float64)
        ok = True
        for g0 in range(0, Ptot, P):
            idx = np.arange(g0, min(g0 + P, Ptot))
            if len(idx) < P:                                        # pad last group
                idx = np.concatenate([idx, np.full(P - len(idx), idx[-1])])
            pts2d = pts2d_all[idx]                                  # (P,2)
            pts3d_hist = gt[hist_idx][:, idx, :]                    # (H,P,3) world
            inp = proc(history_frames=frames,
                       points_2d_at_t0=torch.from_numpy(pts2d).float(),
                       points_3d_history=torch.from_numpy(pts3d_hist).float(),
                       action=action, future_horizon=F,
                       c2w_at_t0=torch.from_numpy(c2w9).float())
            inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                o = model.predict_trajectory(**inp)
            g_cam9 = o.future_3d.float().cpu().numpy()              # (P,F,3) cam-at-t0
            if (g_cam9 == 0).all():
                ok = False
            g_world = (c2w9[:3, :3] @ g_cam9.reshape(-1, 3).T).T + c2w9[:3, 3]
            g_world = g_world.reshape(P, F, 3)
            n = min(P, Ptot - g0)
            fut_world[g0:g0 + n] = g_world[:n]

        gt_fut = gt[obs:total].transpose(1, 0, 2)                    # (Ptot,F,3) world
        dvec = np.linalg.norm(fut_world - gt_fut, axis=-1)          # Ptot,F
        ade, fde = float(dvec.mean()), float(dvec[:, -1].mean())

        # store full-length (T,Ptot,3): obs frames = GT anchor, future = molmo
        molmo_TP3 = np.full((total, Ptot, 3), np.nan, np.float32)
        molmo_TP3[:obs] = gt[:obs].astype(np.float32)
        molmo_TP3[obs:] = fut_world.transpose(1, 0, 2).astype(np.float32)

        np.savez(out / f"{str(d['video_id'])}.npz",
                 video_id=str(d["video_id"]), molmo_3d_cam0=molmo_TP3,
                 ade_m=ade, fde_m=fde, ok=ok, future_text=o.future_text[:2000])
        rows.append((ade, fde, ok))
        print(f"[{i+1}/{nfiles}] {d['video_id']} ade={ade*100:.1f}cm fde={fde*100:.1f}cm "
              f"ok={ok} ({time.time()-tp:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
