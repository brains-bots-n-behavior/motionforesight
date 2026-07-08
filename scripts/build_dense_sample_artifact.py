#!/usr/bin/env python3
"""Self-contained HTML viewer for N sampled dense 3D-track results.

Samples N ``<id>_dense.npz``, projects each clip's 3D ``track_map`` into the
ANCHOR camera (scaled intrinsics, w2c=identity -> trails align with the anchor
RGB frame), embeds a downscaled anchor JPEG + the 2D trails as JSON, and writes
ONE self-contained HTML page (grid of animated track overlays). No external
assets, so it can be published as an Artifact.
"""
import argparse, base64, glob, io, json, os, sys
import numpy as np
from PIL import Image


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
            print("label load warn", name, e, file=sys.stderr)
    return lab


def build_clip(f, grid, bg_width, labels):
    uid = os.path.basename(f)[: -len("_dense.npz")]
    d = np.load(f)
    tm = d["track_map"]            # (T,H,W,3) anchor-cam frame
    rgb = d["rgb"]                 # (T,H,W,3) u8
    T, th, tw, _ = tm.shape
    u = np.load(f.replace("_dense.npz", "_user.npz"), allow_pickle=True)
    oh, ow = u["depth_map"].shape[1:3]
    fx, fy, cx, cy = scale_intr(u["fx_fy_cx_cy"], ow, oh, tw, th)

    gx = grid
    gy = max(4, round(gx * th / tw))
    xs = np.linspace(tw * 0.06, tw * 0.94, gx).round().astype(int)
    ys = np.linspace(th * 0.06, th * 0.94, gy).round().astype(int)
    traces, colors = [], []
    for ty in ys:
        for tx in xs:
            xyz = tm[:, ty, tx, :].astype(np.float64)
            if not np.isfinite(xyz).all():
                continue
            z = np.maximum(xyz[:, 2], 1e-6)
            uu = xyz[:, 0] / z * fx + cx
            vv = xyz[:, 1] / z * fy + cy
            vis = (xyz[:, 2] > 1e-6) & (uu >= 0) & (uu < tw) & (vv >= 0) & (vv < th)
            if int(vis.sum()) < 2:
                continue
            tr = [[round(float(uu[t] / tw), 4), round(float(vv[t] / th), 4)] if vis[t] else None
                  for t in range(T)]
            traces.append(tr)
            colors.append(round(float(ty / max(1, th - 1)), 3))
    if not traces:
        return None

    im = Image.fromarray(rgb[0])
    w = bg_width
    h = round(w * th / tw)
    im = im.resize((w, h))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=60)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {
        "id": uid,
        "T": int(T),
        "ar": round(tw / th, 4),
        "prompt": labels.get(uid.split("_")[0], ""),
        "bg": "data:image/jpeg;base64," + b64,
        "traces": traces,
        "colors": colors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tracks-name", default="anchor_tracks32")
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--grid", type=int, default=13)
    ap.add_argument("--bg-width", type=int, default=320)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, args.tracks_name, "*_dense.npz")))
    if not files:
        sys.exit("no dense npz found")
    n = min(args.n, len(files))
    pick = sorted(set(np.linspace(0, len(files) - 1, n).round().astype(int).tolist()))
    sample = [files[i] for i in pick]
    labels = load_labels(args.labels_dir)

    clips = []
    for f in sample:
        try:
            c = build_clip(f, args.grid, args.bg_width, labels)
            if c:
                clips.append(c)
        except Exception as e:  # noqa: BLE001
            print("skip", os.path.basename(f), e, file=sys.stderr)
    print(f"built {len(clips)} clips from {len(files)} available")
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(clips, separators=(",", ":")))
    html = html.replace("__COUNT__", str(len(clips)))
    with open(args.out, "w") as fh:
        fh.write(html)
    print("wrote", args.out, f"({os.path.getsize(args.out) / 1e6:.1f} MB)")


TEMPLATE = r"""<title>Something-Something · dense 3D tracks (100 samples)</title>
<meta name="description" content="Projected 3D track trails over anchor frames for 100 sampled TrackCraft3R dense outputs.">
<style>
  :root{
    --bg:#0b0d10; --panel:#13161c; --panel2:#181c23; --line:#242b34;
    --ink:#e8ecf2; --muted:#828c9a; --accent:#48d6c4; --accent-dim:#2a6f69;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    -webkit-font-smoothing:antialiased}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  header{position:sticky;top:0;z-index:5;background:rgba(11,13,16,.93);
    backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:14px 20px}
  .titlerow{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
  h1{font-size:16px;font-weight:600;margin:0;letter-spacing:.2px}
  .eyebrow{font-family:ui-monospace,monospace;font-size:10.5px;letter-spacing:.18em;
    text-transform:uppercase;color:var(--accent)}
  .sub{color:var(--muted);font-size:12.5px;margin-left:auto}
  .controls{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin-top:12px}
  .ctl{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--muted)}
  .ctl label{text-transform:uppercase;letter-spacing:.1em;font-family:ui-monospace,monospace}
  input[type=range]{accent-color:var(--accent);width:120px}
  button{background:var(--panel2);color:var(--ink);border:1px solid var(--line);
    border-radius:6px;padding:6px 14px;font-size:12px;cursor:pointer;font-family:inherit}
  button:hover{border-color:var(--accent-dim)}
  button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  #frameval{color:var(--ink);font-variant-numeric:tabular-nums;min-width:54px;text-align:right}
  main{padding:18px 20px 60px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(248px,1fr));gap:16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:9px;overflow:hidden}
  .stage{position:relative;width:100%;background:#000;line-height:0}
  canvas{display:block;width:100%;height:auto}
  .meta{padding:9px 11px 11px}
  .cid{font-family:ui-monospace,monospace;font-size:11px;color:var(--accent);letter-spacing:.02em}
  .cprompt{font-size:12.5px;color:var(--ink);margin:3px 0 0;line-height:1.32;
    display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
  .cprompt.empty{color:var(--muted);font-style:italic}
  .cstat{font-family:ui-monospace,monospace;font-size:10px;color:var(--muted);margin-top:5px;
    letter-spacing:.04em}
  @media (prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style>

<header>
  <div class="titlerow">
    <span class="eyebrow">TrackCraft3R · dense 3D tracks</span>
    <h1>Something-Something — __COUNT__ sampled clips</h1>
    <span class="sub">trails = predicted 3D points reprojected into the anchor camera · hue = vertical seed position</span>
  </div>
  <div class="controls">
    <button id="play" aria-label="Play or pause animation">▶ Play</button>
    <div class="ctl"><label for="frame">Frame</label>
      <input id="frame" type="range" min="0" max="1" value="0" step="1">
      <span id="frameval" class="mono">0</span></div>
    <div class="ctl"><label for="trail">Trail</label>
      <input id="trail" type="range" min="2" max="48" value="48" step="1"></div>
    <div class="ctl"><label for="width">Width</label>
      <input id="width" type="range" min="0.5" max="3" value="1.3" step="0.1"></div>
    <div class="ctl"><label for="op">Opacity</label>
      <input id="op" type="range" min="0.1" max="1" value="0.85" step="0.05"></div>
  </div>
</header>

<main><div class="grid" id="grid"></div></main>

<script type="application/json" id="data">/*__DATA__*/</script>
<script>
(function(){
  const clips = JSON.parse(document.getElementById('data').textContent);
  const grid = document.getElementById('grid');
  const maxT = clips.reduce((m,c)=>Math.max(m,c.T),1);
  const fEl=document.getElementById('frame'), fVal=document.getElementById('frameval');
  const trailEl=document.getElementById('trail'), wEl=document.getElementById('width'), opEl=document.getElementById('op');
  const playBtn=document.getElementById('play');
  fEl.max = maxT-1;
  let frame=0, playing=false;
  const cards=[];

  function hue(c){ return 222 - 222*c; } // 0 top -> blue(222), 1 bottom -> red(0)

  function makeCard(clip){
    const card=document.createElement('article'); card.className='card';
    const stage=document.createElement('div'); stage.className='stage';
    const cv=document.createElement('canvas'); stage.appendChild(cv);
    const meta=document.createElement('div'); meta.className='meta';
    const pr=(clip.prompt||'').trim();
    meta.innerHTML='<div class="cid">'+clip.id.replace(/_anchor$/,'')+'</div>'+
      '<p class="cprompt'+(pr?'':' empty')+'">'+(pr?escapeHtml(pr):'(no label)')+'</p>'+
      '<div class="cstat">'+clip.traces.length+' tracks · '+clip.T+' frames</div>';
    card.appendChild(stage); card.appendChild(meta); grid.appendChild(card);
    const img=new Image();
    const o={clip,cv,ctx:cv.getContext('2d'),img,ready:false};
    img.onload=()=>{o.ready=true; sizeCanvas(o); draw(o);};
    img.src=clip.bg;
    cards.push(o);
  }
  function sizeCanvas(o){
    const w=o.cv.clientWidth||248, ar=o.clip.ar||(o.img.width/o.img.height);
    const dpr=Math.min(2,window.devicePixelRatio||1);
    o.cv.width=Math.round(w*dpr); o.cv.height=Math.round(w/ar*dpr);
    o.cv.style.height=(w/ar)+'px'; o.dpr=dpr;
  }
  function draw(o){
    if(!o.ready) return;
    const {ctx,cv,clip}=o, W=cv.width, H=cv.height;
    ctx.clearRect(0,0,W,H);
    ctx.globalAlpha=1; ctx.drawImage(o.img,0,0,W,H);
    const f=Math.min(frame,clip.T-1);
    const trail=+trailEl.value, lw=+wEl.value*o.dpr, baseOp=+opEl.value;
    ctx.lineCap='round'; ctx.lineJoin='round';
    const start=Math.max(0,f-trail);
    for(let i=0;i<clip.traces.length;i++){
      const tr=clip.traces[i], col='hsl('+hue(clip.colors[i]).toFixed(0)+',80%,58%)';
      ctx.strokeStyle=col;
      for(let t=start+1;t<=f;t++){
        const p0=tr[t-1], p1=tr[t]; if(!p0||!p1) continue;
        ctx.globalAlpha=baseOp*(0.18+0.82*((t-start)/Math.max(1,f-start)));
        ctx.lineWidth=lw;
        ctx.beginPath(); ctx.moveTo(p0[0]*W,p0[1]*H); ctx.lineTo(p1[0]*W,p1[1]*H); ctx.stroke();
      }
      const head=tr[f];
      if(head){ ctx.globalAlpha=0.95; ctx.fillStyle=col;
        ctx.beginPath(); ctx.arc(head[0]*W,head[1]*H,1.7*o.dpr,0,7); ctx.fill(); }
    }
    ctx.globalAlpha=1;
  }
  function drawAll(){ for(const o of cards) draw(o); }
  function setFrame(v){ frame=v; fEl.value=v; fVal.textContent=v+' / '+(maxT-1); drawAll(); }

  let last=0;
  function loop(ts){
    if(!playing) return;
    if(ts-last>90){ last=ts; setFrame((frame+1)%maxT); }
    requestAnimationFrame(loop);
  }
  playBtn.onclick=()=>{ playing=!playing; playBtn.textContent=playing?'❚❚ Pause':'▶ Play';
    if(playing) requestAnimationFrame(loop); };
  fEl.oninput=()=>{ playing=false; playBtn.textContent='▶ Play'; setFrame(+fEl.value); };
  [trailEl,wEl,opEl].forEach(el=>el.oninput=drawAll);
  let rz; window.addEventListener('resize',()=>{clearTimeout(rz);rz=setTimeout(()=>{cards.forEach(sizeCanvas);drawAll();},120);});
  function escapeHtml(s){return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}

  clips.forEach(makeCard);
  setFrame(0);
})();
</script>
"""

if __name__ == "__main__":
    main()
