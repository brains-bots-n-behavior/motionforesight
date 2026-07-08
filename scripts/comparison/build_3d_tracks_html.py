#!/usr/bin/env python3
"""Build ONE self-contained, interactive, animated 3D HTML viewer:
depth-based point cloud + GT (green) and predicted (red) 3D point tracks,
animated over time, for 10 random val clips + the 4 OOD clips. A clip dropdown
lazily renders one plotly figure at a time; plotly.js is inlined so the single
.html opens anywhere (e.g. uploaded to SharePoint).

Reuses the exact inference + point-cloud logic from
scripts/comparison/viser_track_viewer.py (UnifiedTrackDataset, camera-subtracted
last-observed frame). Val and OOD share the code path (OOD processed22 dense
clips have real TrackCraft3r GT).

Run (3dflow env):
    CUDA_VISIBLE_DEVICES=0 MODELSCOPE_CACHE=<ck>/wan_models MODELSCOPE_OFFLINE=1 \
    python scripts/comparison/build_3d_tracks_html.py \
      --checkpoint .../best.pt --out-html .../tracks3d.html
"""
from __future__ import annotations
import argparse, json, sys
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
    UnifiedTrackDataset, UnifiedItem, build_unified_index, split_items)
from render_future_scene_flow_viewer import build_model_from_ckpt  # noqa: E402

DEFAULT_BASE = REPO / "models_pretrained/checkpoints/trackcraft3r/model.safetensors"
OOD_ROOT = Path("/home/yjangir1/scratchhbharad2/users/yjangir1/future-3d-scene-flow/"
                "data/3d_tracks_data/zero-shot-eval-videos")


def flipY(P):
    out = np.array(P, dtype=np.float32, copy=True); out[..., 1] *= -1.0
    return out


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
    ap.add_argument("--pick-seed", type=int, default=0, help="RNG for random val-clip selection")
    ap.add_argument("--num-points", type=int, default=56, help="tracks sampled per clip")
    ap.add_argument("--pc-stride", type=int, default=4, help="point-cloud subsample stride")
    ap.add_argument("--max-pc", type=int, default=4000)
    ap.add_argument("--out-html", required=True)
    return ap.parse_args()


@torch.no_grad()
def clip_data(model, item, obs, total, h, w, num_points, pc_stride, max_pc):
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
    pred = flipY(pred_m[:, :, ys, xs].transpose(2, 1, 0))                     # N,T,3 (last-obs cam)
    gt = flipY((s["gt_tracks_norm"].numpy() * float(scv) + s["pj_mean"].numpy()).transpose(1, 0, 2))
    # last-observed point cloud (subsampled), colored by RGB
    pj = s["pj_obs"].numpy().transpose(0, 2, 3, 1) * float(scv) + s["pj_mean"].numpy()   # obs,h,w,3
    rgbo = ((s["rgb_obs"].numpy().transpose(0, 2, 3, 1) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    pc = flipY(pj[obs - 1, ::pc_stride, ::pc_stride].reshape(-1, 3))
    pccol = rgbo[obs - 1, ::pc_stride, ::pc_stride].reshape(-1, 3)
    fin = np.isfinite(pc).all(1)
    pc, pccol = pc[fin], pccol[fin]
    if pc.shape[0] > max_pc:
        idx = np.random.default_rng(0).choice(pc.shape[0], max_pc, replace=False)
        pc, pccol = pc[idx], pccol[idx]
    ade = float(np.linalg.norm(pred[:, obs:] - gt[:, obs:], axis=-1).mean())
    return dict(pred=pred, gt=gt, pc=pc, pccol=pccol, ade=ade, N=int(pred.shape[0]))


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs

    model, state = build_model_from_ckpt(Path(a.checkpoint), Path(a.base_checkpoint)); model.eval()
    cfg = model.config
    obs, total, h, w = cfg.obs_frames, cfg.total_frames, cfg.height, cfg.width

    # ---- clip list: 10 random val + 4 OOD ----
    val_items = build_unified_index(Path(a.val_root), a.val_tracks_name, Path(a.val_manifest), "")
    _, val = split_items(val_items, 0.05, 7)
    rng = np.random.default_rng(a.pick_seed)
    val_sel = [val[i] for i in sorted(rng.choice(len(val), min(a.num_val, len(val)), replace=False))]
    ood_items = build_unified_index(Path(a.ood_root), a.ood_tracks_name, Path(a.ood_manifest), "")
    clips = [("val", it) for it in val_sel] + [("ood", it) for it in ood_items]
    print(f"clips: {len(val_sel)} val + {len(ood_items)} ood ; epoch {state.get('epoch')}", flush=True)

    def rgbstr(c):  # gl3d markers render reliably with 'rgb(...)' strings (not hex)
        return "rgb(%d,%d,%d)" % (int(c[0]), int(c[1]), int(c[2]))

    def _s(v):  # sanitize: non-finite -> None (gap), else rounded float
        v = float(v)
        return round(v, 3) if np.isfinite(v) else None

    def trail_xyz(pos, sel, lo, hi):
        """None-separated polylines for points `sel`, frames lo..hi -> one line trace."""
        X, Y, Z = [], [], []
        for i in sel:
            seg = pos[i, lo:hi]
            X += [_s(v) for v in seg[:, 0]] + [None]
            Y += [_s(v) for v in seg[:, 1]] + [None]
            Z += [_s(v) for v in seg[:, 2]] + [None]
        return X, Y, Z

    figs = {}
    order = []
    for split, item in clips:
        try:
            d = clip_data(model, item, obs, total, h, w, a.num_points, a.pc_stride, a.max_pc)
        except Exception as e:
            print(f"  skip {item.video_id}: {type(e).__name__}: {e}", flush=True)
            continue
        pred, gt = d["pred"], d["gt"]
        sel = list(range(pred.shape[0]))
        # future frames obs-1 .. total-1 (trails grow); GT green, pred red
        frames = []
        f0 = obs - 1
        anim_ts = list(range(f0, total))
        for t in anim_ts:
            gx, gy, gz = trail_xyz(gt, sel, f0, t + 1)
            px, py, pz = trail_xyz(pred, sel, f0, t + 1)
            frames.append(go.Frame(name=str(t), data=[
                go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", line=dict(color="rgb(40,210,90)", width=4)),
                go.Scatter3d(x=px, y=py, z=pz, mode="lines", line=dict(color="rgb(255,70,70)", width=4)),
            ], traces=[1, 2]))
        # base traces: point cloud (static) + GT/pred at first anim frame
        gx0, gy0, gz0 = trail_xyz(gt, sel, f0, f0 + 1)
        px0, py0, pz0 = trail_xyz(pred, sel, f0, f0 + 1)
        pc_colors = [rgbstr(c) for c in d["pccol"]]
        base = [
            go.Scatter3d(x=[_s(v) for v in d["pc"][:, 0]],
                         y=[_s(v) for v in d["pc"][:, 1]],
                         z=[_s(v) for v in d["pc"][:, 2]],
                         mode="markers", marker=dict(size=1.6, color=pc_colors, opacity=0.85),
                         name="scene", hoverinfo="skip"),
            go.Scatter3d(x=gx0, y=gy0, z=gz0, mode="lines",
                         line=dict(color="rgb(40,210,90)", width=4), name="GT track"),
            go.Scatter3d(x=px0, y=py0, z=pz0, mode="lines",
                         line=dict(color="rgb(255,70,70)", width=4), name="predicted track"),
        ]
        play = dict(type="buttons", showactive=False, y=0, x=0.02, xanchor="left", yanchor="top",
                    buttons=[
                        dict(label="Play", method="animate",
                             args=[None, dict(frame=dict(duration=180, redraw=True),
                                              fromcurrent=True, transition=dict(duration=0))]),
                        dict(label="Pause", method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                    ])
        slider = dict(active=0, y=0, x=0.12, len=0.85, currentvalue=dict(prefix="frame "),
                      steps=[dict(label=str(t), method="animate",
                                  args=[[str(t)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                             for t in anim_ts])
        layout = dict(
            scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                       zaxis=dict(visible=False), bgcolor="rgb(12,14,16)"),
            paper_bgcolor="rgb(12,14,16)", font=dict(color="#eee"),
            margin=dict(l=0, r=0, t=28, b=0), showlegend=True,
            legend=dict(x=0.7, y=0.98, bgcolor="rgba(0,0,0,0.3)"),
            title=dict(text=f"[{split.upper()}] {item.video_id} · future-ADE {d['ade']*100:.1f} cm · "
                            f"observe {obs} → predict {total-obs}", x=0.5, font=dict(size=13)),
            updatemenus=[play], sliders=[slider],
        )
        fig = go.Figure(data=base, frames=frames, layout=layout)
        cid = f"{split}:{item.video_id}"
        figs[cid] = json.loads(fig.to_json())
        order.append((cid, f"[{split.upper()}] {item.video_id}  (ADE {d['ade']*100:.1f}cm, {d['N']}pts)"))
        print(f"  {cid} ade={d['ade']*100:.1f}cm N={d['N']} pc={len(d['pc'])}", flush=True)

    # ---- assemble self-contained HTML ----
    plotlyjs = get_plotlyjs()
    opts = "".join(f'<option value="{cid}">{lbl}</option>' for cid, lbl in order)
    # scrub any residual non-finite (plotly to_json can emit bare NaN) so the
    # embedded JS object is valid + plotly renders; allow_nan=False as a guard.
    def scrub(o):
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, list):
            return [scrub(x) for x in o]
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items()}
        return o
    figs = scrub(figs)
    data_json = json.dumps(figs, allow_nan=False)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Future 3D scene flow — tracks in 3D (point cloud + GT vs predicted)</title>
<style>
body{{margin:0;background:#0c0e10;color:#eee;font-family:system-ui,sans-serif}}
header{{padding:10px 16px;border-bottom:1px solid #2a3037;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}
select{{background:#181b1f;color:#eee;border:1px solid #2a3037;border-radius:6px;padding:6px 10px;font-size:14px}}
#plot{{width:100vw;height:calc(100vh - 92px)}}
.legend{{font-size:12px;color:#a6afb8}} .g{{color:rgb(40,210,90)}} .r{{color:rgb(255,70,70)}}
</style>
<script>{plotlyjs}</script></head><body>
<header>
  <b>Future 3D scene flow</b>
  <label>clip: <select id="clip">{opts}</select></label>
  <span class="legend">scene point cloud · <span class="g">GT track</span> vs <span class="r">predicted track</span> · drag to rotate, scroll to zoom, Play to animate</span>
</header>
<div id="plot"></div>
<script>
const FIGS = {data_json};
const div = document.getElementById('plot');
function show(cid){{
  const f = FIGS[cid];
  Plotly.newPlot(div, f.data, f.layout, {{responsive:true}}).then(()=>{{
    if (f.frames && f.frames.length) Plotly.addFrames(div, f.frames);
  }});
}}
document.getElementById('clip').addEventListener('change', e=>show(e.target.value));
show(document.getElementById('clip').value);
</script></body></html>"""
    out = Path(a.out_html); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    mb = out.stat().st_size / 1e6
    print(f"\nwrote {out} ({mb:.1f} MB, {len(figs)} clips) — self-contained, upload to SharePoint", flush=True)


if __name__ == "__main__":
    main()
