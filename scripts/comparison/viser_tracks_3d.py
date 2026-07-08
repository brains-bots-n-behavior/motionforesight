#!/usr/bin/env python3
"""Merged interactive viser 3D viewer for 10 random val + 4 OOD clips.

Shows, per clip (dropdown), the MOVING scene point cloud (recon_map animated over
all frames, transformed into the model's last-observed camera frame) with GT
(green) and model-PREDICTED (red) 3D point tracks moving + leaving growing
trails. Master time slider + play animates the point cloud AND the tracks
together. Shareable via viser's public share URL (`--share`).

Run (3dflow env, on the GPU node):
    CUDA_VISIBLE_DEVICES=0 MODELSCOPE_CACHE=<ck>/wan_models MODELSCOPE_OFFLINE=1 \
    python scripts/comparison/viser_tracks_3d.py \
      --checkpoint .../best.pt --port 8130 --share
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
from models_pretrained.future_scene_flow.unified_dataset import (  # noqa: E402
    UnifiedTrackDataset, build_unified_index, split_items)
from models_pretrained.future_scene_flow.sparse_dataset import _apply_rigid  # noqa: E402
from render_future_scene_flow_viewer import build_model_from_ckpt  # noqa: E402
import viser  # noqa: E402

DEFAULT_BASE = REPO / "models_pretrained/checkpoints/trackcraft3r/model.safetensors"
OOD_ROOT = Path("/home/yjangir1/scratchhbharad2/users/yjangir1/future-3d-scene-flow/"
                "data/3d_tracks_data/zero-shot-eval-videos")


def flipY(P):
    o = np.array(P, dtype=np.float32, copy=True); o[..., 1] *= -1.0
    return o


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-checkpoint", default=str(DEFAULT_BASE))
    ap.add_argument("--val-root", default="data/ss_subset100k")
    ap.add_argument("--val-tracks-name", default="anchor_tracks32")
    ap.add_argument("--val-manifest", default="sam3_anchor_masks/manifest_merged.json")
    ap.add_argument("--ood-root", default=str(OOD_ROOT))
    ap.add_argument("--ood-tracks-name", default="processed22")
    ap.add_argument("--ood-manifest", default="ood_manifest.json")
    ap.add_argument("--num-val", type=int, default=10)
    ap.add_argument("--pick-seed", type=int, default=0)
    ap.add_argument("--num-points", type=int, default=64)
    ap.add_argument("--pc-stride", type=int, default=8, help="point-cloud subsample (dense res)")
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--share", action="store_true", help="request a public viser share URL")
    return ap.parse_args()


@torch.no_grad()
def build_clip(model, item, obs, total, h, w, num_points, pc_stride):
    ds = UnifiedTrackDataset([item], obs_frames=obs, total_frames=total, image_size=(h, w),
                             num_points=num_points, samples_per_clip=1, seed=0,
                             subtract_camera_motion=True)
    s = ds[0]
    dev = model.device
    b = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q = model(b["rgb_obs"], b["pj_obs"]); xyz, _ = model.split_latent(q)
        delta = model.decode_xyz(xyz).float()
    mean = b["pj_mean"].float(); scv = b["pj_scale"].float()
    pred_m = model.denormalize(model.reconstruct(delta, b["p0_t0_norm"].float()), mean, scv)[0].cpu().numpy()
    quv = s["query_uv_model"].numpy()
    xs = np.clip(quv[:, 0].round().astype(int), 0, w - 1)
    ys = np.clip(quv[:, 1].round().astype(int), 0, h - 1)
    pred = flipY(pred_m[:, :, ys, xs].transpose(2, 1, 0))                    # N,T,3 last-obs cam
    gt = flipY((s["gt_tracks_norm"].numpy() * float(scv) + s["pj_mean"].numpy()).transpose(1, 0, 2))
    xn = (xs - xs.min()) / max(1, xs.max() - xs.min())
    col = (np.array([cm.hsv(v)[:3] for v in xn]) * 255).astype(np.uint8)
    # animated scene point cloud: recon_map (world, all frames) -> last-obs cam
    dense = np.load(item.path)
    recon = dense["recon_map"].astype(np.float64); rgb = dense["rgb"]
    Tn = min(total, recon.shape[0])
    user = np.load(str(item.path).replace("_dense.npz", "_user.npz"), allow_pickle=True)
    w2c = user["extrinsics_w2c"][:Tn].astype(np.float64)
    pc_all, pc_col = [], []
    for t in range(Tn):
        P = recon[t, ::pc_stride, ::pc_stride].reshape(-1, 3)
        C = rgb[t, ::pc_stride, ::pc_stride].reshape(-1, 3)
        fin = np.isfinite(P).all(1)
        Pl = _apply_rigid(w2c[obs - 1], P[fin])                             # world -> last-obs cam
        pc_all.append(flipY(Pl.astype(np.float32))); pc_col.append(C[fin].astype(np.uint8))
    ade = float(np.linalg.norm(pred[:, obs:] - gt[:, obs:], axis=-1).mean())
    return dict(pred=pred, gt=gt, col=col, pc=pc_all, pccol=pc_col, obs=obs, total=Tn,
                ade=ade, N=int(pred.shape[0]))


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    model, state = build_model_from_ckpt(Path(a.checkpoint), Path(a.base_checkpoint)); model.eval()
    cfg = model.config
    obs, total, h, w = cfg.obs_frames, cfg.total_frames, cfg.height, cfg.width

    val_items = build_unified_index(Path(a.val_root), a.val_tracks_name, Path(a.val_manifest), "")
    _, val = split_items(val_items, 0.05, 7)
    rng = np.random.default_rng(a.pick_seed)
    val_sel = [val[i] for i in sorted(rng.choice(len(val), min(a.num_val, len(val)), replace=False))]
    ood_items = build_unified_index(Path(a.ood_root), a.ood_tracks_name, Path(a.ood_manifest), "")
    clip_items = [("VAL", it) for it in val_sel] + [("OOD", it) for it in ood_items]

    cache, names = {}, []
    for split, it in clip_items:
        try:
            d = build_clip(model, it, obs, total, h, w, a.num_points, a.pc_stride)
        except Exception as e:
            print(f"  skip {it.video_id}: {type(e).__name__}: {e}", flush=True); continue
        nm = f"[{split}] {it.video_id}  (ADE {d['ade']*100:.1f}cm)"
        cache[nm] = d; names.append(nm)
        print(f"  loaded {nm} · frames {d['total']} · {d['N']} tracks · pc~{len(d['pc'][0])}", flush=True)
    if not names:
        raise SystemExit("no clips loaded")

    # Everything the viewer serves is now CPU numpy in `cache`; drop the model so
    # we don't hold ~17 GB on the GPU (avoids contending with a training job that
    # shares the same GPU, which made the server flaky).
    del model
    torch.cuda.empty_cache()
    print("model freed from GPU; serving from CPU cache", flush=True)

    server = viser.ViserServer(host="0.0.0.0", port=a.port)
    try:
        server.scene.set_up_direction("+y")
    except Exception:
        pass

    g_clip = server.gui.add_dropdown("clip", options=names, initial_value=names[0])
    g_time = server.gui.add_slider("time (frame)", min=0, max=total - 1, step=1, initial_value=0)
    g_play = server.gui.add_checkbox("play", True)
    g_fps = server.gui.add_slider("fps", min=1, max=15, step=1, initial_value=6)
    g_pred = server.gui.add_checkbox("show prediction (red)", True)
    g_gt = server.gui.add_checkbox("show ground truth (green)", True)
    g_scene = server.gui.add_checkbox("show scene cloud", True)
    g_trail = server.gui.add_checkbox("trails (growing)", True)
    g_psize = server.gui.add_slider("track size", min=0.004, max=0.03, step=0.002, initial_value=0.013)
    g_csize = server.gui.add_slider("cloud size", min=0.002, max=0.02, step=0.001, initial_value=0.006)
    g_info = server.gui.add_markdown("")
    trail_handles = {}

    def cur():
        return cache[g_clip.value]

    def render():
        d = cur(); T = d["total"]; o = d["obs"]
        t = min(int(g_time.value), T - 1)
        # moving scene point cloud
        if g_scene.value:
            server.scene.add_point_cloud("/scene", points=d["pc"][t], colors=d["pccol"][t],
                                         point_size=float(g_csize.value), point_shape="circle")
        else:
            server.scene.add_point_cloud("/scene", points=np.zeros((0, 3), np.float32),
                                         colors=np.zeros((0, 3), np.uint8), point_size=0.001)
        show_tracks = t >= o - 1
        for tag, arr, on, headcol in (("pred", d["pred"], g_pred.value, None),
                                      ("gt", d["gt"], g_gt.value, np.array([50, 220, 90], np.uint8))):
            nm = f"/{tag}_head"
            if on and show_tracks:
                hc = d["col"] if headcol is None else np.tile(headcol, (d["pred"].shape[0], 1))
                server.scene.add_point_cloud(nm, points=arr[:, t], colors=hc,
                                             point_size=float(g_psize.value), point_shape="circle")
            else:
                server.scene.add_point_cloud(nm, points=np.zeros((0, 3), np.float32),
                                             colors=np.zeros((0, 3), np.uint8), point_size=0.001)
        # growing trails obs-1..t
        want = set()
        if g_trail.value and show_tracks and t >= o:
            for tag, arr, on, linecol in (("pred", d["pred"], g_pred.value, (255, 70, 70)),
                                          ("gt", d["gt"], g_gt.value, (50, 220, 90))):
                if not on:
                    continue
                for i in range(arr.shape[0]):
                    nm = f"/tr_{tag}/{i}"; want.add(nm)
                    trail_handles[nm] = server.scene.add_spline_catmull_rom(
                        nm, points=arr[i, o - 1:t + 1], color=linecol, line_width=2.5)
        for nm in [k for k in trail_handles if k not in want]:
            try:
                trail_handles.pop(nm).remove()
            except Exception:
                pass
        g_info.content = (f"**{g_clip.value}**  \nobserve {o} → predict {T-o} · "
                          f"frame {t+1}/{T} · epoch {state.get('epoch')}  \n"
                          f"green = TrackCraft3r GT · red = model prediction")

    def on_clip(_=None):
        d = cur()
        g_time.max = d["total"] - 1
        if int(g_time.value) > g_time.max:
            g_time.value = 0
        for k in list(trail_handles):
            try:
                trail_handles.pop(k).remove()
            except Exception:
                pass
        render()

    g_clip.on_update(on_clip)
    for gh in (g_time, g_pred, g_gt, g_scene, g_trail, g_psize, g_csize):
        gh.on_update(lambda _=None: render())
    on_clip()

    def play_loop():
        while True:
            if g_play.value:
                d = cur()
                g_time.value = (int(g_time.value) + 1) % d["total"]
                render()
                time.sleep(1.0 / max(1, int(g_fps.value)))
            else:
                time.sleep(0.08)
    threading.Thread(target=play_loop, daemon=True).start()

    print(f"\nviser running -> http://localhost:{a.port}", flush=True)
    if a.share:
        try:
            url = server.request_share_url()
            print(f"SHARE URL (send to Homanga; live while this stays running): {url}", flush=True)
        except Exception as e:
            print(f"share URL failed ({e}); use SSH port-forward: "
                  f"ssh -N -L {a.port}:localhost:{a.port} <this-node>", flush=True)
    while True:
        time.sleep(2)


if __name__ == "__main__":
    main()
