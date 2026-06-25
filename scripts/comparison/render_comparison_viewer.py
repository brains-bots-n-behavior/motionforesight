#!/usr/bin/env python3
"""3-way comparison viewer: GT | Ours (pretrained-TrackCraft3r) | MolmoMotion.

Run in the 3dflow env. Reads prep .npz (GT + ours) and the MolmoMotion result
.npz, projects all future tracks into the last-observed (t0) camera, renders
side-by-side trail videos on the same query points, computes ADE/FDE for ours
and MolmoMotion, and writes an index.html.
"""
from __future__ import annotations
import argparse, glob, html, json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
from render_future_track_prediction_viewer import (  # noqa: E402
    _add_label, _draw_trails, _point_colors, _project, _write_video)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", required=True)
    ap.add_argument("--molmo-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--line-width", type=int, default=2)
    ap.add_argument("--video-ext", default=".webm")
    ap.add_argument("--video-codec", default="VP80")
    return ap.parse_args()


def ade_fde(pred_TP3, gt_TP3, obs):
    d = np.linalg.norm(pred_TP3[obs:] - gt_TP3[obs:], axis=-1)  # F,P
    return float(d.mean()), float(d[-1].mean())


def main():
    a = parse_args()
    out = Path(a.output_dir).resolve(); (out / "videos").mkdir(parents=True, exist_ok=True)
    items = []
    agg = {"ours": [], "molmo": []}
    for f in sorted(glob.glob(os.path.join(a.prep_dir, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        vid = str(d["video_id"])
        mf = Path(a.molmo_dir) / f"{vid}.npz"
        if not mf.exists():
            print(f"[skip] no molmo result for {vid}"); continue
        m = np.load(mf, allow_pickle=True)
        obs, total = int(d["obs_frames"]), int(d["total_frames"])
        gt = d["gt_3d_cam0"].astype(np.float64)        # T,P,3
        ours = d["ours_3d_cam0"].astype(np.float64)
        molmo = m["molmo_3d_cam0"].astype(np.float64)
        ok = bool(m["ok"])
        w2c = d["w2c"].astype(np.float64)[obs - 1]
        intr = d["intr_dense"].astype(np.float64)
        P = gt.shape[1]

        ours_ade, ours_fde = ade_fde(ours, gt, obs)
        molmo_ade, molmo_fde = float(m["ade_m"]), float(m["fde_m"])
        agg["ours"].append((ours_ade, ours_fde))
        if ok:
            agg["molmo"].append((molmo_ade, molmo_fde))

        # build trails anchored at GT last-observed point
        def traj(arr):  # arr T,P,3 -> P,(1+F),3
            t = np.concatenate([gt[obs - 1:obs], arr[obs:total]], axis=0)
            return np.transpose(t, (1, 0, 2))
        rgb = np.load(str(d["dense_path"]))["rgb"][:total].astype(np.uint8)
        rh, rw = rgb.shape[1:3]
        last_obs = rgb[obs - 1]
        qxy = d["query_xy_dense"].astype(np.float64)
        uvn = np.stack([qxy[:, 0] / max(1, rw - 1) * 2 - 1, qxy[:, 1] / max(1, rh - 1) * 2 - 1], -1)
        colors = _point_colors(uvn.astype(np.float32))
        future = total - obs

        panes = {"gt": traj(gt), "ours": traj(ours), "molmo": traj(molmo)}
        labels = {"gt": "Ground Truth", "ours": f"Ours  ADE {ours_ade*100:.1f}cm",
                  "molmo": (f"MolmoMotion  ADE {molmo_ade*100:.1f}cm" if ok else "MolmoMotion (parse fail)")}
        vids = {}
        for key, xyz in panes.items():
            uv, val = _project(xyz, w2c, intr, rw, rh)
            frames = []
            for t in range(obs):
                fr = rgb[t].copy(); _add_label(fr, f"observed {t+1}/{obs}"); frames.append(fr)
            for k in range(future):
                frames.append(_draw_trails(last_obs, uv, val, colors, k + 1,
                                           f"{labels[key]}  ({k+1}/{future})", a.line_width))
            vp = out / "videos" / f"{vid}_{key}{a.video_ext}"
            _write_video(vp, frames, a.fps, a.video_codec)
            vids[key] = f"videos/{vp.name}"

        items.append({"videoId": vid, "title": str(d["action"]), "numPoints": int(P),
                      "obsFrames": obs, "totalFrames": total, "oursAde": ours_ade,
                      "molmoAde": molmo_ade, "molmoOk": ok, **{f"{k}Video": v for k, v in vids.items()}})
        print(f"{vid}: ours {ours_ade*100:.1f}cm | molmo {molmo_ade*100:.1f}cm ok={ok}")

    _write_html(out, items, agg)
    for k in ("ours", "molmo"):
        if agg[k]:
            ad = np.mean([x[0] for x in agg[k]]) * 100; fd = np.mean([x[1] for x in agg[k]]) * 100
            print(f"{k}: ADE={ad:.2f}cm FDE={fd:.2f}cm over {len(agg[k])} clips")
    print(f"wrote {out/'index.html'}")


def _write_html(out, items, agg):
    def agg_str(k):
        if not agg[k]:
            return "n/a"
        return f"ADE {np.mean([x[0] for x in agg[k]])*100:.2f}cm · FDE {np.mean([x[1] for x in agg[k]])*100:.2f}cm ({len(agg[k])} clips)"
    css = """
    :root{color-scheme:dark;--bg:#101214;--panel:#181b1f;--text:#f3f5f7;--muted:#a6afb8;--line:#2a3037;--accent:#65b8ff}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}
    header{position:sticky;top:0;background:rgba(16,18,20,.95);border-bottom:1px solid var(--line);padding:14px 20px;z-index:5}
    h1{font-size:18px;margin:0 0 6px}header p{margin:0;color:var(--muted);font-size:13px}
    main{padding:18px 20px}.card{border:1px solid var(--line);background:var(--panel);border-radius:8px;margin-bottom:18px;overflow:hidden}
    .meta{padding:10px 14px;border-bottom:1px solid var(--line)}.meta h2{font-size:14px;margin:0}.meta p{font-size:12px;color:var(--muted);margin:4px 0 0}
    .videos{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line)}.pane{background:#090a0b;padding:8px}
    .pane h3{font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin:0 0 6px;color:var(--muted)}video{width:100%;background:#000;border-radius:4px}
    @media(max-width:900px){.videos{grid-template-columns:1fr}}
    """
    cards = []
    for it in sorted(items, key=lambda x: x["videoId"]):
        cards.append(f"""<article class="card"><div class="meta"><h2>{html.escape(it['title'])}</h2>
<p>{html.escape(it['videoId'])} · {it['numPoints']} pts · obs {it['obsFrames']}/{it['totalFrames']} ·
Ours ADE {it['oursAde']*100:.1f}cm · MolmoMotion ADE {it['molmoAde']*100:.1f}cm{'' if it['molmoOk'] else ' (parse fail)'}</p></div>
<div class="videos">
<section class="pane"><h3>Ground Truth</h3><video src="{it['gtVideo']}" controls muted loop></video></section>
<section class="pane"><h3>Ours (TrackCraft3r-based)</h3><video src="{it['oursVideo']}" controls muted loop></video></section>
<section class="pane"><h3>MolmoMotion-4B</h3><video src="{it['molmoVideo']}" controls muted loop></video></section>
</div></article>""")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Future 3D tracks: Ours vs MolmoMotion</title><style>{css}</style></head><body>
<header><h1>Future 3D track prediction — Ours vs MolmoMotion-4B-H3-F30</h1>
<p><b>Ours</b> (10 RGB+depth frames): {agg_str('ours')} &nbsp;|&nbsp; <b>MolmoMotion</b> (3 frames + 3D history + action): {agg_str('molmo')}
&nbsp;|&nbsp; same val clips, same query points, future-frame 3D ADE/FDE in meters</p></header>
<main>{chr(10).join(cards)}</main></body></html>"""
    (out / "index.html").write_text(doc)
    (out / "manifest.json").write_text(json.dumps({"items": items}, indent=2))


if __name__ == "__main__":
    main()
