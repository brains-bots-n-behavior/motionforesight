#!/usr/bin/env python3
"""Build a local HTML viewer for SAM3-mask-seeded TrackCraft3r traces."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def _clip_id(clip_path: str) -> str:
    return Path(clip_path).stem.split(".")[0]


def _rel(path: Path, start: Path) -> str:
    return path.resolve().relative_to(start.resolve()).as_posix()


def _load_union_mask(mask_paths: list[Path]) -> np.ndarray:
    masks = []
    for path in mask_paths:
        if path.exists():
            masks.append(np.asarray(Image.open(path).convert("L")) > 0)
    if not masks:
        raise ValueError("no mask images")
    return np.logical_or.reduce(masks)


def _sample_mask_grid(mask: np.ndarray, max_points: int, min_step: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []

    area = len(xs)
    step = max(min_step, int(math.sqrt(max(1, area) / max(1, max_points))))
    points: list[tuple[int, int]] = []
    offset = step // 2
    for y in range(offset, h, step):
        for x in range(offset, w, step):
            y0, y1 = max(0, y - step // 2), min(h, y + step // 2 + 1)
            x0, x1 = max(0, x - step // 2), min(w, x + step // 2 + 1)
            patch = mask[y0:y1, x0:x1]
            if not patch.any():
                continue
            py, px = np.argwhere(patch)[len(np.argwhere(patch)) // 2]
            points.append((x0 + int(px), y0 + int(py)))

    if len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = [points[i] for i in keep]
    return points


def _make_overlay(mask: np.ndarray, out_path: Path) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = [0, 170, 255, 78]
    Image.fromarray(rgba).save(out_path)


def _scale_intrinsics_for_model(
    intrinsics: np.ndarray,
    orig_w: int,
    orig_h: int,
    model_w: int,
    model_h: int,
) -> np.ndarray:
    fx, fy, cx, cy = intrinsics.astype(np.float64)
    return np.array(
        [
            fx * (model_w / orig_w),
            fy * (model_h / orig_h),
            cx * (model_w / orig_w),
            cy * (model_h / orig_h),
        ],
        dtype=np.float64,
    )


def _project_points(points_cam0: np.ndarray, w2c_t: np.ndarray, intr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ones = np.ones((points_cam0.shape[0], 1), dtype=np.float64)
    points_t = (w2c_t @ np.concatenate([points_cam0.astype(np.float64), ones], axis=1).T).T[:, :3]
    z = points_t[:, 2]
    fx, fy, cx, cy = intr
    uv = np.empty((points_t.shape[0], 2), dtype=np.float64)
    uv[:, 0] = points_t[:, 0] / np.maximum(z, 1e-8) * fx + cx
    uv[:, 1] = points_t[:, 1] / np.maximum(z, 1e-8) * fy + cy
    return uv, z > 1e-6


def _write_rgb_frames(rgb: np.ndarray, frames_dir: Path, uid: str) -> list[str]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for t, frame in enumerate(rgb):
        frame_path = frames_dir / f"{uid}_{t:02d}.jpg"
        if not frame_path.exists():
            Image.fromarray(frame).save(frame_path, quality=82)
        out.append(frame_path.name)
    return out


def _extract_traces(
    dense_npz: Path,
    user_npz: Path,
    points: list[tuple[int, int]],
    mask_shape: tuple[int, int],
    frame_stride: int,
    frames_dir: Path,
    uid: str,
) -> tuple[list[list[list[float]]], dict[str, int]]:
    data = np.load(dense_npz)
    track_map = data["track_map"]
    rgb = data["rgb"]
    frame_count, track_h, track_w, _ = track_map.shape
    mask_h, mask_w = mask_shape
    if not points:
        return [], {"w": int(track_w), "h": int(track_h)}

    user = np.load(user_npz, allow_pickle=True)
    orig_h, orig_w = user["depth_map"].shape[1:3]
    intr = _scale_intrinsics_for_model(user["fx_fy_cx_cy"], orig_w, orig_h, track_w, track_h)
    extrinsics = user["extrinsics_w2c"][: frame_count * frame_stride : frame_stride]
    if extrinsics.shape[0] < frame_count:
        raise ValueError(f"{user_npz} has too few sampled extrinsics for {frame_count} frames")

    anchors_xyz = []
    anchors = []
    for x, y in points:
        tx = min(track_w - 1, max(0, round(x / max(1, mask_w - 1) * (track_w - 1))))
        ty = min(track_h - 1, max(0, round(y / max(1, mask_h - 1) * (track_h - 1))))
        xyz = track_map[:, ty, tx, :].astype(np.float32)
        if not np.isfinite(xyz).all():
            continue
        anchors_xyz.append(xyz)
        anchors.append((tx / max(1, track_w - 1), ty / max(1, track_h - 1)))

    if not anchors_xyz:
        return [], {"w": int(track_w), "h": int(track_h)}

    xyz_arr = np.stack(anchors_xyz, axis=1)  # (T, N, 3)
    projected = np.zeros((frame_count, xyz_arr.shape[1], 2), dtype=np.float32)
    visible = np.zeros((frame_count, xyz_arr.shape[1]), dtype=bool)
    for t in range(frame_count):
        uv, valid_z = _project_points(xyz_arr[t], extrinsics[t], intr)
        in_frame = (
            valid_z
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < track_w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < track_h)
        )
        projected[t] = np.stack([uv[:, 0] / track_w, uv[:, 1] / track_h], axis=-1)
        visible[t] = in_frame

    traces: list[list[list[float]]] = []
    for idx, anchor in enumerate(anchors):
        trace = []
        for t in range(frame_count):
            px, py = projected[t, idx]
            if not visible[t, idx]:
                trace.append([None, None])
            else:
                trace.append([round(float(px), 5), round(float(py), 5)])
        if sum(p[0] is not None for p in trace) < 2:
            continue
        traces.append(trace)
    frame_files = _write_rgb_frames(rgb, frames_dir, uid)
    return traces, {"w": int(track_w), "h": int(track_h), "frames": int(frame_count), "rgbFrames": frame_files}


def build(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    manifest = json.loads((root / args.sam_manifest).read_text())
    tracks_dir = root / args.tracks_name
    viewer_dir = root / args.viewer_name
    viewer_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for entry in manifest:
        if not (args.min_masks <= int(entry["num_instances"]) <= args.max_masks):
            continue
        uid = _clip_id(entry["clip_path"])
        dense_npz = tracks_dir / f"{uid}_dense.npz"
        user_npz = tracks_dir / f"{uid}_user.npz"
        if not dense_npz.exists():
            continue
        if not user_npz.exists():
            continue

        clip_mask_dir = root / Path(entry["overlay_path"]).parent
        mask_paths = sorted(clip_mask_dir.glob("mask_*.png"))
        try:
            mask = _load_union_mask(mask_paths)
        except ValueError:
            continue

        points = _sample_mask_grid(mask, max_points=args.max_points, min_step=args.min_step)
        try:
            traces, track_size = _extract_traces(
                dense_npz,
                user_npz,
                points,
                mask.shape,
                args.frame_stride,
                viewer_dir / "frames",
                uid,
            )
        except (OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
            print(f"skipping unreadable/incomplete track {dense_npz}: {exc}")
            continue
        if len(traces) < args.min_points:
            continue

        mask_out = viewer_dir / f"{uid}_mask.png"
        _make_overlay(mask, mask_out)

        items.append(
            {
                "id": uid,
                "title": entry.get("segment_text") or entry.get("prompt") or uid,
                "videoTitle": entry.get("video_title", ""),
                "prompt": entry.get("prompt", ""),
                "numMasks": int(entry["num_instances"]),
                "numTraces": len(traces),
                "frameCount": len(traces[0]),
                "video": "../" + entry["clip_path"],
                "firstFrame": "../" + entry["frame_path"],
                "samOverlay": "../" + entry["overlay_path"],
                "maskOverlay": mask_out.name,
                "rgbFrameDir": "frames",
                "rgbFrames": track_size.pop("rgbFrames"),
                "trackSize": track_size,
                "traces": traces,
            }
        )
        if args.limit and len(items) >= args.limit:
            break

    css = """
    :root{color-scheme:dark;--bg:#101214;--panel:#181b1f;--text:#f2f4f7;--muted:#a8b0ba;--line:#2c3238;--accent:#5fb4ff}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,sans-serif}
    header{position:sticky;top:0;z-index:2;background:rgba(16,18,20,.92);backdrop-filter:blur(12px);border-bottom:1px solid var(--line);padding:14px 20px}
    h1{font-size:18px;line-height:1.2;margin:0 0 10px} .controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center;color:var(--muted);font-size:13px}
    label{display:flex;align-items:center;gap:8px}input[type=range]{width:150px}main{padding:18px 20px 32px}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
    .stage{position:relative;aspect-ratio:832/480;background:#050607}.stage img.frame,.stage canvas,.stage .mask{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}.stage canvas{pointer-events:none}.stage .mask{opacity:.32;mix-blend-mode:screen;pointer-events:none}
    .meta{padding:12px 14px 14px}.meta h2{font-size:15px;line-height:1.25;margin:0 0 7px}.meta p{margin:5px 0;color:var(--muted);font-size:12px;line-height:1.35}a{color:var(--accent)}
    """
    js = """
    const DATA = __DATA__;
    const grid=document.querySelector('.grid'),gain=document.querySelector('#gain'),alpha=document.querySelector('#alpha'),cards=[];
    document.querySelector('#count').textContent=`${DATA.items.length} clips`;
    function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
    function resize(c){const r=c.getBoundingClientRect(),d=devicePixelRatio||1,w=Math.max(1,Math.round(r.width*d)),h=Math.max(1,Math.round(r.height*d));if(c.width!==w||c.height!==h){c.width=w;c.height=h;}}
    function mediaRect(item,c){const cw=c.width,ch=c.height,vw=item.trackSize.w||832,vh=item.trackSize.h||480,cr=cw/ch,vr=vw/vh;if(vr>cr){const h=cw/vr;return{x:0,y:(ch-h)/2,w:cw,h};}const w=ch*vr;return{x:(cw-w)/2,y:0,w,h:ch};}
    function anchorY(trace){for(const p of trace){if(p[1]!==null)return p[1];}return .5;}
    function traceColors(traces){const ys=traces.map(anchorY),lo=Math.min(...ys),hi=Math.max(...ys),span=Math.max(.001,hi-lo);return ys.map(y=>`hsl(${Math.round(235-230*((y-lo)/span))}, 98%, 58%)`);}
    function setFrame(card,idx){card.frameIndex=Math.max(0,Math.min(card.item.frameCount-1,idx));card.frame.src=`${card.item.rgbFrameDir}/${card.item.rgbFrames[card.frameIndex]}`;draw(card);}
    function draw(card){const {item,canvas}=card;resize(canvas);const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);const rect=mediaRect(item,canvas),end=card.frameIndex||0,g=Number(gain.value),a=Number(alpha.value),d=devicePixelRatio||1;ctx.lineCap='round';ctx.lineJoin='round';item.traces.forEach((trace,idx)=>{if(trace.length<2)return;const color=item.traceColors[idx];for(let i=1;i<=end;i++){const p0=trace[i-1],p1=trace[i];if(p0[0]===null||p1[0]===null)continue;const x0=rect.x+p0[0]*rect.w,y0=rect.y+p0[1]*rect.h,x1=rect.x+p1[0]*rect.w,y1=rect.y+p1[1]*rect.h;ctx.globalAlpha=a*(.22+.78*(i/Math.max(1,item.frameCount-1)));ctx.strokeStyle=color;ctx.lineWidth=1.35*d*g;ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.stroke();}const head=trace[end];if(head[0]!==null){const x=rect.x+head[0]*rect.w,y=rect.y+head[1]*rect.h;ctx.globalAlpha=.95;ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,2.1*d,0,Math.PI*2);ctx.fill();}});ctx.globalAlpha=1;ctx.fillStyle='rgba(0,0,0,.45)';ctx.fillRect(rect.x+10,rect.y+10,88,26);ctx.fillStyle='#fff';ctx.font=`${12*d}px system-ui,sans-serif`;ctx.fillText(`${end+1}/${item.frameCount}`,rect.x+20,rect.y+28);}
    function makeCard(item){item.traceColors=traceColors(item.traces);const article=document.createElement('article');article.className='card';article.innerHTML=`<div class="stage"><img class="frame" alt=""><img class="mask" src="${esc(item.maskOverlay)}" alt=""><canvas></canvas></div><div class="meta"><h2>${esc(item.title)}</h2><p>${esc(item.videoTitle)}</p><p>${item.numMasks} SAM3 mask(s) · ${item.numTraces} projected mask-grid traces · ${item.frameCount} tracked frames</p><p><input class="scrub" type="range" min="0" max="${item.frameCount-1}" step="1" value="0"></p><p><a href="${esc(item.video)}" target="_blank" rel="noreferrer">source clip</a> · <a href="${esc(item.samOverlay)}" target="_blank" rel="noreferrer">SAM3 overlay</a> · <a href="${esc(item.firstFrame)}" target="_blank" rel="noreferrer">first frame</a></p></div>`;grid.appendChild(article);const frame=article.querySelector('img.frame'),canvas=article.querySelector('canvas'),scrub=article.querySelector('.scrub'),card={item,frame,canvas,frameIndex:0};cards.push(card);frame.addEventListener('load',()=>draw(card));scrub.addEventListener('input',()=>setFrame(card,Number(scrub.value)));setFrame(card,0);}
    DATA.items.forEach(makeCard);for(const input of [gain,alpha])input.addEventListener('input',()=>cards.forEach(draw));addEventListener('resize',()=>cards.forEach(draw));
    """
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Action100M 32-frame Mask Tracks</title><style>{css}</style></head>
<body>
<header><h1>Action100M 32-frame projected mask-seeded tracks <span id="count"></span></h1><div class="controls"><label>Line width <input id="gain" type="range" min="0.4" max="4" step="0.05" value="1.8"></label><label>Opacity <input id="alpha" type="range" min="0.05" max="1" step="0.05" value="0.78"></label></div></header>
<main><section class="grid"></section></main>
<script>{js.replace("__DATA__", json.dumps({"items": items}, separators=(",", ":")))}</script>
</body></html>
"""
    if args.template and args.template.exists():
        html = args.template.read_text().replace(
            "__DATA__",
            json.dumps({"items": items}, separators=(",", ":")),
        )
    (viewer_dir / "index.html").write_text(html)
    if args.copy_json:
        (viewer_dir / "mask_traces.json").write_text(json.dumps({"items": items}, indent=2))
    print(f"wrote {viewer_dir / 'index.html'} ({len(items)} clips)")
    return len(items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/action100m"))
    parser.add_argument("--sam-manifest", default="sam3_first_frame_masks/manifest.json")
    parser.add_argument("--tracks-name", default="mask_trace32_tracks")
    parser.add_argument("--viewer-name", default="mask_trace32_viewer")
    parser.add_argument("--min-masks", type=int, default=1)
    parser.add_argument("--max-masks", type=int, default=3)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=260)
    parser.add_argument("--min-step", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("viewer/action100m_projected_tracks_template.html"),
        help="HTML template containing a __DATA__ placeholder.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--copy-json", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
