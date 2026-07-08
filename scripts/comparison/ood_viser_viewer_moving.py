#!/usr/bin/env python3
"""Multi-clip viser 3D viewer for zero-shot OOD predictions — MOVING scene cloud.

Like ood_viser_viewer.py, but instead of freezing the background after the 7
observed frames, it animates the model's FULL-FRAME predicted geometry
(`pred_m`, 3,T,mh,mw) for every future frame — so the whole scene plays like a
video (observed motion for frames 0..obs-1, then the predicted scene for
obs..total-1). The masked-object track splines are overlaid on top.

    python scripts/comparison/ood_viser_viewer_moving.py \
      --checkpoint .../best.pt --proc-dir data/new_samples/processed7 \
      --mask-dir data/new_samples/masks --port 8130
"""
from __future__ import annotations
import argparse, glob, os, sys, threading, time
from pathlib import Path
import numpy as np
import cv2

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "scripts"), str(REPO / "scripts/comparison")):
    if p not in sys.path:
        sys.path.insert(0, p)
import models_pretrained  # noqa: E402,F401
from render_future_scene_flow_viewer import build_model_from_ckpt  # noqa: E402
from ood_common import ood_predict  # noqa: E402
import viser  # noqa: E402

DEFAULT_BASE = REPO / "models_pretrained/checkpoints/trackcraft3r/model.safetensors"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-checkpoint", default=str(DEFAULT_BASE))
    ap.add_argument("--proc-dir", default="data/new_samples/processed7")
    ap.add_argument("--mask-dir", default="data/new_samples/masks")
    ap.add_argument("--grid-stride", type=int, default=2, help="track sampling on mask")
    ap.add_argument("--pc-stride", type=int, default=5, help="scene-cloud pixel subsample")
    ap.add_argument("--port", type=int, default=8130)
    return ap.parse_args()


def flipY(P):
    out = np.array(P, np.float32, copy=True); out[..., 1] *= -1.0
    return out


def clean(pts, cols):
    """Drop non-finite / behind-camera / far-outlier points for display."""
    z = pts[:, 2]
    good = np.isfinite(pts).all(1) & (z > 0.02) & (np.abs(pts) < 20).all(1)
    return pts[good], cols[good]


def main():
    a = parse_args()
    model, st = build_model_from_ckpt(Path(a.checkpoint), Path(a.base_checkpoint)); model.eval()
    obs = total = None
    clips = {}
    for f in sorted(glob.glob(os.path.join(a.proc_dir, "*_user.npz"))):
        vid = os.path.basename(f).replace("_user.npz", "")
        mask = Path(a.mask_dir) / f"{vid}_mask.png"
        if not mask.exists():
            continue
        r = ood_predict(model, f, mask, grid_stride=a.grid_stride)
        obs, total = r["obs"], r["total"]
        pcs = a.pc_stride
        mh, mw = r["mh"], r["mw"]
        pred_m = r["pred_m"]                       # 3,T,mh,mw  cam-lastobs (m) — dense, but
        pj = r["pj"]                               # obs,mh,mw,3 observed geom     ONLY the mask
        rgb8 = r["rgb8"]                            # obs,mh,mw,3                    is supervised
        # subsampled object mask (the ONLY region the model is trained to predict)
        mk = cv2.resize(cv2.imread(str(mask), 0), (mw, mh), interpolation=cv2.INTER_NEAREST) > 0
        mk_sub = mk[::pcs, ::pcs].reshape(-1)
        rgb_last = rgb8[obs - 1, ::pcs, ::pcs].reshape(-1, 3)

        # observed frames: full REAL cloud, per-frame RGB (frames 0..obs-1)
        obs_full = []
        for t in range(obs):
            p = pj[t, ::pcs, ::pcs].reshape(-1, 3)
            c = rgb8[t, ::pcs, ::pcs].reshape(-1, 3)
            obs_full.append(clean(np.ascontiguousarray(p, np.float32), np.ascontiguousarray(c, np.uint8)))
        obs_full = [(flipY(p), c) for p, c in obs_full]
        # static REAL background (outside mask), frozen at the last observed frame
        bgp = pj[obs - 1, ::pcs, ::pcs].reshape(-1, 3)[~mk_sub]
        bgc = rgb_last[~mk_sub]
        bgp, bgc = clean(np.ascontiguousarray(bgp, np.float32), np.ascontiguousarray(bgc, np.uint8))
        bg_static = (flipY(bgp), bgc)
        # moving PREDICTED object (inside mask only) per future frame
        obj_fut = {}
        objc = rgb_last[mk_sub]
        for t in range(obs, total):
            op = pred_m[:, t, ::pcs, ::pcs].reshape(3, -1).T[mk_sub]
            p, c = clean(np.ascontiguousarray(op, np.float32), np.ascontiguousarray(objc, np.uint8))
            obj_fut[t] = (flipY(p), c)
        clips[vid] = dict(predV=flipY(r["pred_pts"]), xs=r["xs"], ys=r["ys"],
                          col=r["track_col"], N=r["N"],
                          obs_full=obs_full, bg_static=bg_static, obj_fut=obj_fut)
        print(f"  {vid}: {r['N']} masked pts | bg {bg_static[0].shape[0]} pts | "
              f"obj ~{obj_fut[obs][0].shape[0]} pts/frame", flush=True)
    names = list(clips)
    print(f"loaded {len(names)} clips | epoch {st.get('epoch')}", flush=True)

    server = viser.ViserServer(host="0.0.0.0", port=a.port)
    try:
        server.scene.set_up_direction("+y")
    except Exception:
        pass
    server.gui.add_markdown(f"**Zero-shot OOD (moving cloud)** · epoch {st.get('epoch')}")
    g_clip = server.gui.add_dropdown("clip", options=names, initial_value=names[0])
    g_time = server.gui.add_slider("time (frame)", min=0, max=total - 1, step=1, initial_value=0)
    g_play = server.gui.add_checkbox("play", False)
    g_fps = server.gui.add_slider("fps", min=1, max=15, step=1, initial_value=6)
    g_space = server.gui.add_slider("track spacing (px)", min=a.grid_stride, max=48,
                                    step=a.grid_stride, initial_value=12)
    g_trail = server.gui.add_checkbox("show trails (growing)", True)
    _fut = total - obs
    g_tlen = server.gui.add_slider("trail length (frames)", min=1, max=_fut, step=1,
                                   initial_value=min(5, _fut))
    g_trkpts = server.gui.add_checkbox("show track points", False)
    g_psize = server.gui.add_slider("track point size", min=0.004, max=0.03, step=0.002, initial_value=0.012)
    g_cloud = server.gui.add_checkbox("show background cloud (real, frozen)", True)
    g_objcloud = server.gui.add_checkbox("show predicted object cloud", True)
    g_csize = server.gui.add_slider("cloud point size", min=0.001, max=0.02, step=0.001, initial_value=0.003)

    EMPTY_SEG = np.zeros((0, 2, 3), np.float32)
    EMPTY_SEGC = np.zeros((0, 2, 3), np.uint8)

    EMPTY_PC = np.zeros((0, 3), np.float32)
    EMPTY_PCC = np.zeros((0, 3), np.uint8)

    def set_pc(name, data, size):
        p, col = data if data is not None else (EMPTY_PC, EMPTY_PCC)
        server.scene.add_point_cloud(name, points=p, colors=col, point_size=size)

    def render():
        c = clips[g_clip.value]; t = int(g_time.value); cs = float(g_csize.value)
        # Scene cloud. Observed frames (t<obs): full REAL cloud (moves with the real
        # camera). Future frames: only the object is a genuine prediction, so the
        # background is the REAL cloud FROZEN at the last observed frame, and only the
        # masked object animates (its prediction). Off-object dense output is
        # UNSUPERVISED and deliberately NOT shown.
        if t < obs:
            set_pc("/scene", c["obs_full"][t] if g_cloud.value else None, cs)
            set_pc("/obj", None, cs)
        else:
            set_pc("/scene", c["bg_static"] if g_cloud.value else None, cs)
            set_pc("/obj", c["obj_fut"][t] if g_objcloud.value else None, cs)
        # track points + growing trails on the masked object (future only)
        xs, ys, predV, col = c["xs"], c["ys"], c["predV"], c["col"]
        s = np.where((xs % int(g_space.value) == 0) & (ys % int(g_space.value) == 0))[0]
        if s.size == 0:
            s = np.arange(min(50, c["N"]))
        show = t >= obs
        if show and g_trkpts.value:
            server.scene.add_point_cloud("/trk", points=predV[s, t], colors=col[s],
                                         point_size=float(g_psize.value))
        else:
            server.scene.add_point_cloud("/trk", points=np.zeros((0, 3), np.float32),
                                         colors=np.zeros((0, 3), np.uint8), point_size=0.001)
        # growing trails: ONE batched line-segments node (not hundreds of splines)
        if g_trail.value and show and t > obs - 1:
            start = max(obs - 1, t - int(g_tlen.value))         # sliding window
            seg = predV[s, start:t + 1]                         # (S, L, 3)
            segs = np.stack([seg[:, :-1], seg[:, 1:]], axis=2)  # (S, L-1, 2, 3)
            S, Lm1 = segs.shape[0], segs.shape[1]
            pts = segs.reshape(S * Lm1, 2, 3).astype(np.float32)
            cc = np.repeat(col[s], Lm1, axis=0)                 # (S*(L-1), 3)
            cc = np.repeat(cc[:, None, :], 2, axis=1).astype(np.uint8)  # (.,2,3)
            server.scene.add_line_segments("/trails", points=pts, colors=cc, line_width=2.5)
        else:
            server.scene.add_line_segments("/trails", points=EMPTY_SEG, colors=EMPTY_SEGC, line_width=2.5)

    def on_clip(_=None):
        g_time.value = 0; render()

    g_clip.on_update(on_clip)
    for gh in (g_time, g_space, g_trail, g_tlen, g_trkpts, g_psize, g_cloud, g_objcloud, g_csize):
        gh.on_update(lambda _=None: render())
    render()

    # Default camera to a video-like straight-on view (look down the optical axis
    # of the last-observed camera), so the scene is framed like the original clip.
    _all0 = np.concatenate([clips[n]["obs_full"][0][0] for n in names], 0)
    _ctr = np.median(_all0, axis=0).astype(np.float32)
    _look = (float(_ctr[0]), float(_ctr[1]), float(_ctr[2]))
    _pos = (float(_ctr[0]), float(_ctr[1]), float(_ctr[2]) - 1.3)   # step back along -z

    @server.on_client_connect
    def _on_connect(client):
        try:
            client.camera.up_direction = np.array([0.0, 1.0, 0.0], np.float32)
            client.camera.position = np.array(_pos, np.float32)
            client.camera.look_at = np.array(_look, np.float32)
        except Exception:
            pass

    def loop():
        while True:
            if g_play.value:
                g_time.value = (int(g_time.value) + 1) % total  # triggers render via on_update
                time.sleep(1.0 / max(1, int(g_fps.value)))
            else:
                time.sleep(0.1)
    threading.Thread(target=loop, daemon=True).start()
    print(f"\nviser running -> http://localhost:{a.port}\n", flush=True)
    while True:
        time.sleep(2)


if __name__ == "__main__":
    main()
