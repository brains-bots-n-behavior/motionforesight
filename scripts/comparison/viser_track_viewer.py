#!/usr/bin/env python3
"""Interactive 3D viser viewer for one clip's future-track prediction, as a video.

A master TIME slider (+ play button) animates everything together:
  - the observed RGB point cloud at the current frame (frozen at last-obs in the
    future), and
  - the predicted (and optional GT) 3D track positions at that frame, moving
    through 3D like particles, optionally leaving growing trails.

Tracks are rainbow-colored by spatial position (left->right). Track density is
controlled by a SPACING slider (pixels between sampled track points), not a count.

Run (3dflow env):
    python scripts/comparison/viser_track_viewer.py \
      --checkpoint .../freshlora_unified_camsub_320x576/last.pt \
      --video-id 106182_anchor --grid-stride 2 --port 8129
"""
from __future__ import annotations
import argparse, sys, threading, time
from pathlib import Path
import numpy as np
import torch
import matplotlib.cm as cm

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
import models_pretrained  # noqa: E402,F401
from models_pretrained.future_scene_flow.unified_dataset import UnifiedTrackDataset, UnifiedItem  # noqa: E402
from render_future_scene_flow_viewer import build_model_from_ckpt  # noqa: E402
import viser  # noqa: E402

DEFAULT_BASE = REPO / "models_pretrained/checkpoints/trackcraft3r/model.safetensors"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-checkpoint", default=str(DEFAULT_BASE))
    ap.add_argument("--video-id", default="106182_anchor")
    ap.add_argument("--tracks-name", default="anchor_tracks32_curated_dense")
    ap.add_argument("--mask-dir", default="data/something_something/sam3_anchor_masks/clips")
    ap.add_argument("--grid-stride", type=int, default=2, help="finest grid of candidate track points")
    ap.add_argument("--max-points", type=int, default=4000)
    ap.add_argument("--pc-stride", type=int, default=3, help="point-cloud subsample stride")
    ap.add_argument("--port", type=int, default=8129)
    return ap.parse_args()


def flipY(P):
    out = np.array(P, dtype=np.float32, copy=True); out[..., 1] *= -1.0
    return out


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    model, state = build_model_from_ckpt(Path(a.checkpoint), Path(a.base_checkpoint)); model.eval()
    cfg = model.config; obs, total, h, w = cfg.obs_frames, cfg.total_frames, cfg.height, cfg.width

    mask_paths = tuple(sorted((Path(a.mask_dir) / a.video_id).glob("mask_*.png")))
    item = UnifiedItem("dense", a.video_id,
                       Path("data/something_something") / a.tracks_name / f"{a.video_id}_dense.npz",
                       mask_paths, "")
    ds = UnifiedTrackDataset([item], obs_frames=obs, total_frames=total, image_size=(h, w),
                             samples_per_clip=1, seed=0, subtract_camera_motion=True,
                             dense_grid_stride=a.grid_stride, max_points=a.max_points)
    s = ds[0]
    dev = model.device
    b = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        q = model(b["rgb_obs"], b["pj_obs"]); xyz, _ = model.split_latent(q)
        delta = model.decode_xyz(xyz).float()
    mean = b["pj_mean"].float(); sc = b["pj_scale"].float()
    pred_m = model.denormalize(model.reconstruct(delta, b["p0_t0_norm"].float()), mean, sc)[0].cpu().numpy()

    quv = s["query_uv_model"].numpy(); xs = np.clip(quv[:, 0].round().astype(int), 0, w - 1)
    ys = np.clip(quv[:, 1].round().astype(int), 0, h - 1)
    predV = flipY(pred_m[:, :, ys, xs].transpose(2, 1, 0))                   # N,T,3
    gtV = flipY((s["gt_tracks_norm"].numpy() * float(sc) + s["pj_mean"].numpy()).transpose(1, 0, 2))
    N, T = predV.shape[0], predV.shape[1]
    xn = (xs - xs.min()) / max(1, xs.max() - xs.min())
    track_col = (np.array([cm.hsv(v)[:3] for v in xn]) * 255).astype(np.uint8)  # N,3

    pcs = a.pc_stride
    pj = s["pj_obs"].numpy().transpose(0, 2, 3, 1) * float(sc) + s["pj_mean"].numpy()
    rgbo = ((s["rgb_obs"].numpy().transpose(0, 2, 3, 1) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    pc_pts = [flipY(pj[f, ::pcs, ::pcs].reshape(-1, 3)) for f in range(obs)]
    pc_col = [rgbo[f, ::pcs, ::pcs].reshape(-1, 3) for f in range(obs)]
    print(f"clip {a.video_id} | {N} candidate track pts | epoch {state.get('epoch')}", flush=True)

    server = viser.ViserServer(host="0.0.0.0", port=a.port)
    try:
        server.scene.set_up_direction("+y")
    except Exception:
        pass
    pc_handles = [server.scene.add_point_cloud(f"/pc/f{f}", points=pc_pts[f], colors=pc_col[f],
                                               point_size=0.004, visible=False) for f in range(obs)]

    # --- GUI ---
    server.gui.add_markdown(f"**{a.video_id}**  \nlifting joy stick up (val) · epoch {state.get('epoch')}")
    g_time = server.gui.add_slider("time (frame)", min=0, max=T - 1, step=1, initial_value=0)
    g_play = server.gui.add_checkbox("play", False)
    g_fps = server.gui.add_slider("fps", min=1, max=15, step=1, initial_value=6)
    g_space = server.gui.add_slider("track spacing (px)", min=a.grid_stride, max=48,
                                    step=a.grid_stride, initial_value=max(a.grid_stride, 16))
    g_pred = server.gui.add_checkbox("show prediction", True)
    g_gt = server.gui.add_checkbox("show ground truth", False)
    g_trail = server.gui.add_checkbox("show trails (growing)", True)
    g_psize = server.gui.add_slider("track point size", min=0.004, max=0.03, step=0.002, initial_value=0.012)
    g_obs = server.gui.add_checkbox("show observed cloud", True)
    g_freeze = server.gui.add_checkbox("freeze cloud after obs", True)

    trail_handles = {}

    def select():
        sel = np.where((xs % int(g_space.value) == 0) & (ys % int(g_space.value) == 0))[0]
        return sel if sel.size else np.array([int(np.argmin(xs))])

    def render():
        t = int(g_time.value); sel = select()
        # tracks only appear AFTER the last anchor/observed frame (obs-1)
        show_tracks = t >= obs
        # observed point cloud
        fr = t if t < obs else (obs - 1 if g_freeze.value else -1)
        for f in range(obs):
            pc_handles[f].visible = g_obs.value and (f == fr)
        # moving track points at frame t (only in the future)
        for tag, arr, on in (("pred", predV, g_pred.value), ("gt", gtV, g_gt.value)):
            nm = f"/trk_{tag}"
            if on and show_tracks:
                server.scene.add_point_cloud(nm, points=arr[sel, t], colors=track_col[sel],
                                             point_size=float(g_psize.value))
            else:
                server.scene.add_point_cloud(nm, points=np.zeros((0, 3), np.float32),
                                             colors=np.zeros((0, 3), np.uint8), point_size=0.001)
        # growing trails from the last anchor (obs-1) up to t, future only
        want = set()
        if g_trail.value and show_tracks:
            lo = obs - 1; hi = t + 1
            if hi - lo >= 2:
                for tag, arr, on in (("pred", predV, g_pred.value), ("gt", gtV, g_gt.value)):
                    if not on:
                        continue
                    for i in sel:
                        nm = f"/tr_{tag}/{i}"; want.add(nm)
                        trail_handles[nm] = server.scene.add_spline_catmull_rom(
                            nm, points=arr[i, lo:hi], color=tuple(int(c) for c in track_col[i]),
                            line_width=2.5)
        for nm in [k for k in trail_handles if k not in want]:
            try:
                trail_handles.pop(nm).remove()
            except Exception:
                pass

    for gh in (g_time, g_space, g_pred, g_gt, g_trail, g_psize, g_obs, g_freeze):
        gh.on_update(lambda _=None: render())
    render()

    def play_loop():
        while True:
            if g_play.value:
                g_time.value = (int(g_time.value) + 1) % T
                render()
                time.sleep(1.0 / max(1, int(g_fps.value)))
            else:
                time.sleep(0.1)
    threading.Thread(target=play_loop, daemon=True).start()

    print(f"\nviser running -> http://localhost:{a.port}   (Ctrl-C to stop)\n", flush=True)
    while True:
        time.sleep(2)


if __name__ == "__main__":
    main()
