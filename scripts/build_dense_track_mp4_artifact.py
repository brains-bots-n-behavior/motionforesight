#!/usr/bin/env python3
"""Self-contained HTML grid of N tracking VIDEOS (mp4) from dense results.

For each sampled ``<id>_dense.npz``: overlays the predicted 3D tracks on the
ACTUAL RGB frames (projected per-frame with that frame's camera, so points stick
to the moving video), encodes a small H.264 mp4 (bundled ffmpeg via
imageio-ffmpeg), embeds it as a data URI, and writes ONE self-contained HTML
page of looping <video> players. Publishable as an Artifact.
"""
import argparse, base64, glob, io, json, os, sys, tempfile
import numpy as np
import cv2
import imageio.v2 as imageio


def scale_intr(fxfycxcy, ow, oh, tw, th):
    fx, fy, cx, cy = [float(v) for v in fxfycxcy]
    return fx * tw / ow, fy * th / oh, cx * tw / ow, cy * th / oh


def load_labels(labels_dir):
    lab = {}
    if not labels_dir:
        return lab
    for name in ("train.json", "validation.json"):
        p = os.path.join(labels_dir, name)
        if not os.path.exists(p):
            continue
        try:
            for row in json.load(open(p)):
                vid = str(row.get("id") or row.get("video_id") or "")
                txt = (row.get("label") or row.get("template") or "").strip()
                if vid:
                    lab[vid] = txt
        except Exception as e:  # noqa: BLE001
            print("label warn", name, e, file=sys.stderr)
    return lab


def hue_bgr(c):
    # c in [0,1]: 0 top -> blue(220), 1 bottom -> red(0). HSV(OpenCV H:0-179).
    h = int((220 - 220 * c) / 2)
    hsv = np.uint8([[[h, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def even(x):
    x = int(round(x));  return x - (x % 2)


def _load_union_mask(mask_paths):
    from PIL import Image
    m = None
    for p in mask_paths:
        a = np.asarray(Image.open(p).convert("L")) > 0
        m = a if m is None else (m | a)
    return m


def _sample_mask_grid(mask, max_points, min_step):
    import math
    h, w = mask.shape
    area = int(mask.sum())
    if area == 0:
        return [], (h, w)
    step = max(min_step, int(math.sqrt(max(1, area) / max(1, max_points))))
    pts = []
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            y0, y1 = max(0, y - step // 2), min(h, y + step // 2 + 1)
            x0, x1 = max(0, x - step // 2), min(w, x + step // 2 + 1)
            patch = mask[y0:y1, x0:x1]
            aw = np.argwhere(patch)
            if len(aw) == 0:
                continue
            py, px = aw[len(aw) // 2]
            pts.append((x0 + int(px), y0 + int(py)))
    if len(pts) > max_points:
        keep = np.linspace(0, len(pts) - 1, max_points, dtype=int)
        pts = [pts[i] for i in keep]
    return pts, (h, w)


def seed_points(tm, tw, th, grid, mask_dir, max_points, min_step, include_hand):
    """Return list of (tx, ty, colorval) seeds at track resolution.
    If mask_dir given & has masks -> sample within union(object[, hand]) mask;
    else uniform grid over the frame. Filters to finite track_map points."""
    cand = []  # (tx, ty)
    if mask_dir and os.path.isdir(mask_dir):
        paths = sorted(glob.glob(os.path.join(mask_dir, "mask_*.png")))
        if include_hand:
            paths += sorted(glob.glob(os.path.join(mask_dir, "hand_mask_*.png")))
        if paths:
            mask = _load_union_mask(paths)
            pts, (mh, mw) = _sample_mask_grid(mask, max_points, min_step)
            for x, y in pts:
                tx = min(tw - 1, max(0, round(x / max(1, mw - 1) * (tw - 1))))
                ty = min(th - 1, max(0, round(y / max(1, mh - 1) * (th - 1))))
                cand.append((tx, ty))
    if not cand:  # grid fallback
        gx = grid; gy = max(4, round(gx * th / tw))
        for ty in np.linspace(th * 0.05, th * 0.95, gy).round().astype(int):
            for tx in np.linspace(tw * 0.05, tw * 0.95, gx).round().astype(int):
                cand.append((int(tx), int(ty)))
    seeds = []
    for tx, ty in cand:
        if np.isfinite(tm[:, ty, tx, :]).all():
            seeds.append((tx, ty, ty / max(1, th - 1)))
    return seeds


def build_clip_mp4(f, grid, out_w, fps, crf, trail, labels,
                   mask_root=None, max_points=320, min_step=5, include_hand=True):
    uid = os.path.basename(f)[: -len("_dense.npz")]
    d = np.load(f)
    tm = d["track_map"]; rgb = d["rgb"]
    T, th, tw, _ = tm.shape
    u = np.load(f.replace("_dense.npz", "_user.npz"), allow_pickle=True)
    oh, ow = u["depth_map"].shape[1:3]
    fx, fy, cx, cy = scale_intr(u["fx_fy_cx_cy"], ow, oh, tw, th)
    ext = u["extrinsics_w2c"][:T]
    if ext.shape[0] < T:
        return None

    mask_dir = os.path.join(mask_root, "sam3_anchor_masks", "clips", uid) if mask_root else None
    seeds = seed_points(tm, tw, th, grid, mask_dir, max_points, min_step, include_hand)
    if not seeds:
        return None

    # project every seed for every frame (with that frame's extrinsics)
    xyz_all = np.stack([tm[:, ty, tx, :] for tx, ty, _ in seeds], axis=1).astype(np.float64)  # (T,N,3)
    cols = [hue_bgr(c) for *_, c in seeds]
    proj = np.full((T, len(seeds), 2), np.nan)
    for t in range(T):
        R = ext[t][:3, :3]; tt = ext[t][:3, 3]
        cam = xyz_all[t] @ R.T + tt
        z = cam[:, 2]
        ok = z > 1e-6
        uu = np.where(ok, cam[:, 0] / np.where(ok, z, 1) * fx + cx, np.nan)
        vv = np.where(ok, cam[:, 1] / np.where(ok, z, 1) * fy + cy, np.nan)
        inb = ok & (uu >= 0) & (uu < tw) & (vv >= 0) & (vv < th)
        proj[t, inb, 0] = uu[inb]; proj[t, inb, 1] = vv[inb]

    W = even(out_w); H = even(out_w * th / tw)
    sx, sy = W / tw, H / th
    frames = []
    for t in range(T):
        base = cv2.resize(rgb[t], (W, H), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        s0 = max(0, t - trail)
        for i in range(len(seeds)):
            col = cols[i]
            for tt2 in range(s0 + 1, t + 1):
                p0 = proj[tt2 - 1, i]; p1 = proj[tt2, i]
                if np.isnan(p0).any() or np.isnan(p1).any():
                    continue
                cv2.line(bgr, (int(p0[0] * sx), int(p0[1] * sy)),
                         (int(p1[0] * sx), int(p1[1] * sy)), col, 1, cv2.LINE_AA)
            ph = proj[t, i]
            if not np.isnan(ph).any():
                cv2.circle(bgr, (int(ph[0] * sx), int(ph[1] * sy)), 2, col, -1, cv2.LINE_AA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    imageio.mimwrite(tmp.name, frames, fps=fps, codec="libx264", macro_block_size=1,
                     pixelformat="yuv420p",
                     output_params=["-crf", str(crf), "-movflags", "+faststart"])
    with open(tmp.name, "rb") as fh:
        data = fh.read()
    os.unlink(tmp.name)
    return {
        "id": uid,
        "prompt": labels.get(uid.split("_")[0], ""),
        "ntr": len(seeds), "T": int(T),
        "mp4": "data:video/mp4;base64," + base64.b64encode(data).decode(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tracks-name", default="anchor_tracks32")
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--grid", type=int, default=22)
    ap.add_argument("--out-w", type=int, default=384)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--trail", type=int, default=6)
    ap.add_argument("--mask-root", default=None,
                    help="dir with sam3_anchor_masks/clips/<id>/ ; seed tracks on object(+hand) masks")
    ap.add_argument("--max-points", type=int, default=320)
    ap.add_argument("--min-step", type=int, default=5)
    ap.add_argument("--no-hand", action="store_true", help="object masks only (exclude hands)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, args.tracks_name, "*_dense.npz")))
    if not files:
        sys.exit("no dense npz")
    n = min(args.n, len(files))
    pick = sorted(set(np.linspace(0, len(files) - 1, n).round().astype(int).tolist()))
    sample = [files[i] for i in pick]
    labels = load_labels(args.labels_dir)

    clips = []
    for k, f in enumerate(sample, 1):
        try:
            c = build_clip_mp4(f, args.grid, args.out_w, args.fps, args.crf, args.trail, labels,
                               mask_root=args.mask_root, max_points=args.max_points,
                               min_step=args.min_step, include_hand=not args.no_hand)
            if c:
                clips.append(c)
            if k % 10 == 0:
                print(f"  {k}/{len(sample)} ...", flush=True)
        except Exception as e:  # noqa: BLE001
            print("skip", os.path.basename(f), e, file=sys.stderr)
    total_mb = sum(len(c["mp4"]) for c in clips) / 1e6
    print(f"built {len(clips)} videos (~{total_mb:.1f} MB of mp4 data)")
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(clips, separators=(",", ":")))
    html = html.replace("__COUNT__", str(len(clips)))
    with open(args.out, "w") as fh:
        fh.write(html)
    print("wrote", args.out, f"({os.path.getsize(args.out) / 1e6:.1f} MB)")


TEMPLATE = r"""<title>Something-Something · dense 3D-track videos (100 samples)</title>
<meta name="description" content="100 sampled TrackCraft3R clips with predicted 3D tracks overlaid on the playing video.">
<style>
  :root{--bg:#0b0d10;--panel:#13161c;--line:#242b34;--ink:#e8ecf2;--muted:#828c9a;--accent:#48d6c4}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;-webkit-font-smoothing:antialiased}
  header{position:sticky;top:0;z-index:5;background:rgba(11,13,16,.93);backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:14px 20px}
  .titlerow{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
  h1{font-size:16px;font-weight:600;margin:0}
  .eyebrow{font-family:ui-monospace,monospace;font-size:10.5px;letter-spacing:.18em;text-transform:uppercase;color:var(--accent)}
  .sub{color:var(--muted);font-size:12.5px;margin-left:auto}
  .controls{display:flex;gap:14px;align-items:center;margin-top:11px}
  button{background:#181c23;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 14px;font-size:12px;cursor:pointer;font-family:inherit}
  button:hover{border-color:#2a6f69}
  button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  main{padding:18px 20px 60px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(264px,1fr));gap:16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:9px;overflow:hidden}
  video{display:block;width:100%;height:auto;background:#000}
  .meta{padding:9px 11px 11px}
  .cid{font-family:ui-monospace,monospace;font-size:11px;color:var(--accent)}
  .cprompt{font-size:12.5px;margin:3px 0 0;line-height:1.32;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
  .cprompt.empty{color:var(--muted);font-style:italic}
  .cstat{font-family:ui-monospace,monospace;font-size:10px;color:var(--muted);margin-top:5px}
</style>
<header>
  <div class="titlerow">
    <span class="eyebrow">TrackCraft3R · dense 3D tracks</span>
    <h1>Something-Something — __COUNT__ track videos</h1>
    <span class="sub">predicted 3D tracks projected onto each playing clip · hue = vertical seed position</span>
  </div>
  <div class="controls">
    <button id="toggle" aria-label="Play or pause all videos">❚❚ Pause all</button>
  </div>
</header>
<main><div class="grid" id="grid"></div></main>
<script type="application/json" id="data">/*__DATA__*/</script>
<script>
(function(){
  const clips=JSON.parse(document.getElementById('data').textContent);
  const grid=document.getElementById('grid');
  const vids=[];
  for(const c of clips){
    const card=document.createElement('article'); card.className='card';
    const v=document.createElement('video');
    v.src=c.mp4; v.loop=true; v.muted=true; v.playsInline=true; v.preload='none';
    const pr=(c.prompt||'').trim();
    const meta=document.createElement('div'); meta.className='meta';
    meta.innerHTML='<div class="cid">'+c.id.replace(/_anchor$/,'')+'</div>'+
      '<p class="cprompt'+(pr?'':' empty')+'">'+(pr?esc(pr):'(no label)')+'</p>'+
      '<div class="cstat">'+c.ntr+' tracks · '+c.T+' frames</div>';
    card.appendChild(v); card.appendChild(meta); grid.appendChild(card); vids.push(v);
  }
  function esc(s){return s.replace(/[&<>"]/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[x]));}
  // Only play videos near the viewport (100 concurrent decoders would choke).
  let enabled=true;
  const io=new IntersectionObserver((es)=>{
    if(!enabled) return;
    for(const e of es){ if(e.isIntersecting){ e.target.play().catch(()=>{}); } else { e.target.pause(); } }
  },{rootMargin:'200px'});
  vids.forEach(v=>io.observe(v));
  const btn=document.getElementById('toggle');
  btn.onclick=()=>{enabled=!enabled;btn.textContent=enabled?'❚❚ Pause all':'▶ Play all';
    if(!enabled) vids.forEach(v=>v.pause());
    else vids.forEach(v=>{const r=v.getBoundingClientRect();if(r.top<innerHeight+200&&r.bottom>-200)v.play().catch(()=>{});});};
})();
</script>
"""

if __name__ == "__main__":
    main()
