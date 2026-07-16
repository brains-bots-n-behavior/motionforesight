#!/usr/bin/env python3
"""Build a self-contained 3D HTML viewer for one zero-shot OOD clip.

Input is a processed ``*_user.npz`` plus a mask PNG. The output is a single
Plotly HTML showing observed point clouds and predicted future 3D tracks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "scripts"), str(REPO / "scripts/comparison")):
    if p not in sys.path:
        sys.path.insert(0, p)

import models_pretrained  # noqa: E402,F401
from ood_common import ood_predict  # noqa: E402
from plotly.offline import get_plotlyjs  # noqa: E402
from inference_model import build_model_from_ckpt  # noqa: E402

DEFAULT_BASE = REPO / "models_pretrained/checkpoints/trackcraft3r/model.safetensors"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-checkpoint", default=str(DEFAULT_BASE))
    ap.add_argument("--user-npz", default=None)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--out-html", default=None)
    ap.add_argument("--user-dir", default=None, help="batch mode: directory with *_user.npz files")
    ap.add_argument("--mask-dir", default=None, help="batch mode: directory with <videoid>_mask.png files")
    ap.add_argument("--out-dir", default=None, help="batch mode output directory")
    ap.add_argument("--suffix", default="_zero_shot_3d.html", help="batch mode output filename suffix")
    ap.add_argument("--grid-stride", type=int, default=8, help="mask track sampling stride")
    ap.add_argument("--pc-stride", type=int, default=5, help="point-cloud sampling stride")
    ap.add_argument("--max-tracks", type=int, default=220)
    ap.add_argument("--rainbow-tracks", action="store_true", help="draw tracks as individually colored traces")
    ap.add_argument("--track-count-options", default="25,50,100,220", help="comma-separated track-count UI choices")
    return ap.parse_args()


def flip_y(x):
    y = np.array(x, np.float32, copy=True)
    y[..., 1] *= -1
    return y


def finite_points(points, colors):
    good = np.isfinite(points).all(1) & (points[:, 2] > 0.02) & (np.abs(points) < 20).all(1)
    return points[good], colors[good]


def line_segments(tracks, indices, start, end):
    xs, ys, zs = [], [], []
    for i in indices:
        seg = tracks[i, start:end]
        xs.extend([round(float(v), 4) for v in seg[:, 0]])
        ys.extend([round(float(v), 4) for v in seg[:, 1]])
        zs.extend([round(float(v), 4) for v in seg[:, 2]])
        xs.append(None); ys.append(None); zs.append(None)
    return xs, ys, zs


def rgb_strings(colors):
    return [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in colors]


def rounded_points(points):
    return {
        "x": np.round(points[:, 0], 4).tolist(),
        "y": np.round(points[:, 1], 4).tolist(),
        "z": np.round(points[:, 2], 4).tolist(),
    }


def build_rainbow_html(vid, state, obs, total, pcs, pred, idx, track_col, track_options):
    pc_frames = []
    for pc, col in pcs:
        item = rounded_points(pc)
        item["color"] = rgb_strings(col)
        pc_frames.append(item)

    tracks = []
    for i in idx:
        coords = np.round(pred[i], 4)
        color = track_col[i % len(track_col)]
        tracks.append({
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "z": coords[:, 2].tolist(),
            "color": f"rgb({int(color[0])},{int(color[1])},{int(color[2])})",
        })

    options = sorted({int(v) for v in track_options if int(v) > 0 and int(v) <= len(tracks)})
    if len(tracks) not in options:
        options.append(len(tracks))
    payload = json.dumps({
        "vid": vid,
        "epoch": state.get("epoch"),
        "obs": obs,
        "total": total,
        "pcFrames": pc_frames,
        "tracks": tracks,
        "trackOptions": options,
    }, allow_nan=False)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Zero-shot future 3D scene flow</title>
<style>
body{{margin:0;background:#0c0e10;color:#eee;font-family:system-ui,sans-serif}}
#plot{{width:100vw;height:100vh}}
#controls{{position:fixed;left:14px;bottom:14px;z-index:10;background:rgba(12,14,16,.82);padding:10px 12px;border:1px solid #333;border-radius:6px;display:flex;gap:12px;align-items:center}}
button,select,input{{background:#1b1f24;color:#eee;border:1px solid #444;border-radius:4px;padding:5px 7px}}
label{{font-size:13px;color:#ddd;display:flex;gap:6px;align-items:center}}
</style>
<script>{get_plotlyjs()}</script></head><body>
<div id="controls">
  <button id="play">Play</button>
  <button id="pause">Pause</button>
  <label>Frame <input id="frame" type="range" min="0" max="0" value="0"></label>
  <label>Tracks <select id="tracks"></select></label>
</div>
<div id="plot"></div>
<script>
const fig = {payload};
const frameSlider = document.getElementById('frame');
const trackSelect = document.getElementById('tracks');
frameSlider.max = String(fig.total - 1);
for (const n of fig.trackOptions) {{
  const opt = document.createElement('option');
  opt.value = String(n);
  opt.textContent = String(n);
  if (n === Math.min(100, fig.tracks.length)) opt.selected = true;
  trackSelect.appendChild(opt);
}}
if (!trackSelect.value) trackSelect.value = String(fig.tracks.length);
let timer = null;

function selectedTrackIndices(count) {{
  if (count >= fig.tracks.length) return [...Array(fig.tracks.length).keys()];
  const out = [];
  for (let i = 0; i < count; i++) out.push(Math.round(i * (fig.tracks.length - 1) / Math.max(1, count - 1)));
  return [...new Set(out)];
}}

function makeData(t) {{
  const pc = fig.pcFrames[Math.min(t, fig.obs - 1)];
  const data = [{{
    type: 'scatter3d', mode: 'markers',
    x: pc.x, y: pc.y, z: pc.z,
    marker: {{size: 1.25, color: pc.color, opacity: 0.86}},
    name: 'observed scene cloud', hoverinfo: 'skip'
  }}];
  if (t >= fig.obs) {{
    const count = Number(trackSelect.value);
    for (const i of selectedTrackIndices(count)) {{
      const tr = fig.tracks[i];
      data.push({{
        type: 'scatter3d', mode: 'lines',
        x: tr.x.slice(fig.obs - 1, t + 1),
        y: tr.y.slice(fig.obs - 1, t + 1),
        z: tr.z.slice(fig.obs - 1, t + 1),
        line: {{color: tr.color, width: 5}},
        name: 'track', hoverinfo: 'skip', showlegend: false
      }});
    }}
  }}
  return data;
}}

const layout = {{
  title: {{text: `Zero-shot future 3D scene flow: ${{fig.vid}} | observe ${{fig.obs}} -> predict ${{fig.total - fig.obs}} | epoch ${{fig.epoch}}`, font: {{size: 14}}}},
  scene: {{aspectmode: 'data', bgcolor: 'rgb(12,14,16)', xaxis: {{visible: false}}, yaxis: {{visible: false}}, zaxis: {{visible: false}}}},
  paper_bgcolor: 'rgb(12,14,16)', font: {{color: '#eee'}}, margin: {{l: 0, r: 0, t: 42, b: 0}}, showlegend: false
}};

function draw() {{
  const t = Number(frameSlider.value);
  Plotly.react('plot', makeData(t), layout, {{responsive: true}});
}}

document.getElementById('play').onclick = () => {{
  if (timer) return;
  timer = setInterval(() => {{
    let t = Number(frameSlider.value) + 1;
    if (t >= fig.total) t = fig.obs;
    frameSlider.value = String(t);
    draw();
  }}, 220);
}};
document.getElementById('pause').onclick = () => {{ clearInterval(timer); timer = null; }};
frameSlider.oninput = draw;
trackSelect.onchange = draw;
draw();
</script></body></html>"""


def render_one(model, state, user_npz, mask, out_html, grid_stride, pc_stride, max_tracks,
               rainbow_tracks=False, track_count_options="25,50,100,220"):
    result = ood_predict(model, user_npz, mask, grid_stride=grid_stride)
    pred = flip_y(result["pred_pts"])  # N,T,3
    obs, total = int(result["obs"]), int(result["total"])
    rgb8 = result["rgb8"]
    pj = result["pj"]
    n = pred.shape[0]
    if n > max_tracks:
        idx = np.linspace(0, n - 1, max_tracks).round().astype(int)
    else:
        idx = np.arange(n)

    pcs = []
    for t in range(obs):
        pts = flip_y(pj[t, :: pc_stride, :: pc_stride].reshape(-1, 3))
        cols = rgb8[t, :: pc_stride, :: pc_stride].reshape(-1, 3)
        pts, cols = finite_points(pts, cols)
        pcs.append((pts, cols))

    out = Path(out_html)
    if rainbow_tracks:
        track_options = [int(v) for v in track_count_options.split(",") if v.strip()]
        html = build_rainbow_html(
            Path(user_npz).name.replace("_user.npz", ""),
            state, obs, total, pcs, pred, idx, result["track_col"], track_options,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(idx)} rainbow tracks)")
        return

    # Plotly JSON with one point cloud trace and one predicted-track trace.
    frames = []
    for t in range(total):
        pc_t = min(t, obs - 1)
        pc, col = pcs[pc_t]
        if t < obs:
            lx, ly, lz = [], [], []
        else:
            lx, ly, lz = line_segments(pred, idx, obs - 1, t + 1)
        frames.append({
            "name": str(t),
            "data": [
                {
                    "type": "scatter3d",
                    "mode": "markers",
                    "x": np.round(pc[:, 0], 4).tolist(),
                    "y": np.round(pc[:, 1], 4).tolist(),
                    "z": np.round(pc[:, 2], 4).tolist(),
                    "marker": {"size": 1.5, "color": rgb_strings(col), "opacity": 0.85},
                    "name": "observed scene cloud",
                    "hoverinfo": "skip",
                },
                {
                    "type": "scatter3d",
                    "mode": "lines",
                    "x": lx,
                    "y": ly,
                    "z": lz,
                    "line": {"color": "rgb(255,70,70)", "width": 5},
                    "name": "predicted future tracks",
                    "hoverinfo": "skip",
                },
            ],
        })

    init = frames[0]["data"]
    steps = [
        {
            "label": str(t),
            "method": "animate",
            "args": [[str(t)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
        }
        for t in range(total)
    ]
    layout = {
        "title": {
            "text": (
                f"Zero-shot future 3D scene flow: {Path(user_npz).name.replace('_user.npz', '')} "
                f"| observe {obs} -> predict {total - obs} | epoch {state.get('epoch')}"
            ),
            "font": {"size": 14},
        },
        "scene": {
            "aspectmode": "data",
            "bgcolor": "rgb(12,14,16)",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
        },
        "paper_bgcolor": "rgb(12,14,16)",
        "font": {"color": "#eee"},
        "margin": {"l": 0, "r": 0, "t": 42, "b": 0},
        "showlegend": True,
        "updatemenus": [{
            "type": "buttons",
            "showactive": False,
            "x": 0.02,
            "y": 0,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"fromcurrent": True, "frame": {"duration": 220, "redraw": True}}],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                },
            ],
        }],
        "sliders": [{"active": 0, "steps": steps, "x": 0.12, "y": 0, "len": 0.84}],
    }
    fig = {"data": init, "layout": layout, "frames": frames}
    payload = json.dumps(fig, allow_nan=False)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Zero-shot future 3D scene flow</title>
<style>body{{margin:0;background:#0c0e10;color:#eee;font-family:system-ui,sans-serif}}
#plot{{width:100vw;height:100vh}}</style>
<script>{get_plotlyjs()}</script></head><body>
<div id="plot"></div>
<script>
const fig = {payload};
Plotly.newPlot('plot', fig.data, fig.layout, {{responsive:true}}).then(() => {{
  Plotly.addFrames('plot', fig.frames);
}});
</script></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(idx)} tracks)")


def batch_items(a):
    if a.user_dir or a.mask_dir or a.out_dir:
        if not (a.user_dir and a.mask_dir and a.out_dir):
            raise SystemExit("--user-dir, --mask-dir, and --out-dir are required together")
        items = []
        user_dir = Path(a.user_dir)
        mask_dir = Path(a.mask_dir)
        out_dir = Path(a.out_dir)
        for user_npz in sorted(user_dir.glob("*_user.npz")):
            vid = user_npz.name.replace("_user.npz", "")
            mask = mask_dir / f"{vid}_mask.png"
            if not mask.exists():
                print(f"skip {vid}: missing mask {mask}")
                continue
            items.append((str(user_npz), str(mask), str(out_dir / f"{vid}{a.suffix}")))
        return items
    if not (a.user_npz and a.mask and a.out_html):
        raise SystemExit("--user-npz, --mask, and --out-html are required in single-clip mode")
    return [(a.user_npz, a.mask, a.out_html)]


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    items = batch_items(a)
    if not items:
        raise SystemExit("no clips to render")

    model, state = build_model_from_ckpt(Path(a.checkpoint), Path(a.base_checkpoint))
    model.eval()
    try:
        for user_npz, mask, out_html in items:
            render_one(
                model, state, user_npz, mask, out_html, a.grid_stride, a.pc_stride, a.max_tracks,
                rainbow_tracks=a.rainbow_tracks, track_count_options=a.track_count_options,
            )
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
