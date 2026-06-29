#!/usr/bin/env python3
"""Multi-clip viser 3D viewer for zero-shot OOD predictions (one model load).

Loads the model once, runs it on every processed clip + SAM mask, and serves a
viser scene with a CLIP dropdown to switch between them. Shows the observed point
cloud + predicted future 3D tracks on the masked object, time-animated (play),
tracks start after the last anchor frame, rainbow spatial coloring, density via
spacing. No GT (zero-shot).

    python scripts/comparison/ood_viser_viewer.py \
      --checkpoint .../best.pt --proc-dir zero-shot-eval/processed \
      --mask-dir zero-shot-eval/masks --port 8130
"""
from __future__ import annotations
import argparse, glob, os, sys, threading, time
from pathlib import Path
import numpy as np

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
    ap.add_argument("--proc-dir", default="zero-shot-eval/processed")
    ap.add_argument("--mask-dir", default="zero-shot-eval/masks")
    ap.add_argument("--grid-stride", type=int, default=2)
    ap.add_argument("--pc-stride", type=int, default=3)
    ap.add_argument("--port", type=int, default=8130)
    return ap.parse_args()


def flipY(P):
    out = np.array(P, np.float32, copy=True); out[..., 1] *= -1.0
    return out


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
        clips[vid] = dict(
            predV=flipY(r["pred_pts"]), xs=r["xs"], ys=r["ys"], col=r["track_col"], N=r["N"],
            pc_pts=[flipY(r["pj"][fr, ::pcs, ::pcs].reshape(-1, 3)) for fr in range(obs)],
            pc_col=[r["rgb8"][fr, ::pcs, ::pcs].reshape(-1, 3) for fr in range(obs)])
        print(f"  {vid}: {r['N']} masked pts", flush=True)
    names = list(clips)
    print(f"loaded {len(names)} clips | epoch {st.get('epoch')}", flush=True)

    server = viser.ViserServer(host="0.0.0.0", port=a.port)
    try:
        server.scene.set_up_direction("+y")
    except Exception:
        pass
    server.gui.add_markdown(f"**Zero-shot OOD** · epoch {st.get('epoch')}")
    g_clip = server.gui.add_dropdown("clip", options=names, initial_value=names[0])
    g_time = server.gui.add_slider("time (frame)", min=0, max=total - 1, step=1, initial_value=0)
    g_play = server.gui.add_checkbox("play", False)
    g_fps = server.gui.add_slider("fps", min=1, max=15, step=1, initial_value=6)
    g_space = server.gui.add_slider("track spacing (px)", min=a.grid_stride, max=48, step=a.grid_stride, initial_value=12)
    g_trail = server.gui.add_checkbox("show trails (growing)", True)
    g_psize = server.gui.add_slider("track point size", min=0.004, max=0.03, step=0.002, initial_value=0.012)
    g_obs = server.gui.add_checkbox("show observed cloud", True)
    g_freeze = server.gui.add_checkbox("freeze cloud after obs", True)

    state = {"pc": [], "trails": {}}

    def rebuild_pc():
        for h in state["pc"]:
            try: h.remove()
            except Exception: pass
        c = clips[g_clip.value]
        state["pc"] = [server.scene.add_point_cloud(f"/pc/f{fr}", points=c["pc_pts"][fr],
                       colors=c["pc_col"][fr], point_size=0.004, visible=False) for fr in range(obs)]

    def render():
        c = clips[g_clip.value]; t = int(g_time.value); show = t >= obs
        xs, ys, predV, col = c["xs"], c["ys"], c["predV"], c["col"]
        s = np.where((xs % int(g_space.value) == 0) & (ys % int(g_space.value) == 0))[0]
        if s.size == 0:
            s = np.arange(min(50, c["N"]))
        fr = t if t < obs else (obs - 1 if g_freeze.value else -1)
        for f in range(obs):
            state["pc"][f].visible = g_obs.value and (f == fr)
        if show:
            server.scene.add_point_cloud("/trk", points=predV[s, t], colors=col[s], point_size=float(g_psize.value))
        else:
            server.scene.add_point_cloud("/trk", points=np.zeros((0, 3), np.float32), colors=np.zeros((0, 3), np.uint8), point_size=0.001)
        want = set()
        if g_trail.value and show:
            for i in s:
                nm = f"/tr/{i}"; want.add(nm)
                state["trails"][nm] = server.scene.add_spline_catmull_rom(
                    nm, points=predV[i, obs - 1:t + 1], color=tuple(int(x) for x in col[i]), line_width=2.5)
        for k in [k for k in state["trails"] if k not in want]:
            try: state["trails"].pop(k).remove()
            except Exception: pass

    def on_clip(_=None):
        for k in list(state["trails"]):
            try: state["trails"].pop(k).remove()
            except Exception: pass
        g_time.value = 0
        rebuild_pc(); render()

    g_clip.on_update(on_clip)
    for gh in (g_time, g_space, g_trail, g_psize, g_obs, g_freeze):
        gh.on_update(lambda _=None: render())
    rebuild_pc(); render()

    def loop():
        while True:
            if g_play.value:
                g_time.value = (int(g_time.value) + 1) % total; render()
                time.sleep(1.0 / max(1, int(g_fps.value)))
            else:
                time.sleep(0.1)
    threading.Thread(target=loop, daemon=True).start()
    print(f"\nviser running -> http://localhost:{a.port}\n", flush=True)
    while True:
        time.sleep(2)


if __name__ == "__main__":
    main()
