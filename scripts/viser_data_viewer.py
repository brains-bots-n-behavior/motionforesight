#!/usr/bin/env python3
"""Interactive per-phase 3D viewer for the Something-Something data-curation pipeline.

Launches a viser web server that lets you inspect, for any curated clip, the
output of each pipeline phase:

  Phase 1  SAM3 anchor masks   - anchor frame + object/hand mask overlay, and the
                                 object-mask points highlighted in 3D.
  Phase 2  DA3 geometry        - per-frame reconstructed 3D point cloud
                                 (recon_map) colored by RGB, with camera frustum.
  Phase 3  Dense 3D tracks     - the TrackCraft3r track_map animated over time;
                                 this is the pseudo-label the future-track model
                                 trains on.

Usage (on the GPU node):

    conda activate 3dflow
    python scripts/viser_data_viewer.py \
        --root data/_smoke100k \
        --tracks-name anchor_tracks32 \
        --port 8080

Then forward the port from your laptop:

    ssh -N -L 8080:localhost:8080 <this-node>

and open http://localhost:8080
"""

from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
from PIL import Image
import viser


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@dataclass
class ClipData:
    video_id: str
    track_map: np.ndarray | None       # (T, H, W, 3) float32  dense 3D tracks
    recon_map: np.ndarray | None       # (T, H, W, 3) float32  reconstructed geometry
    rgb: np.ndarray | None             # (T, H, W, 3) uint8
    overlay: np.ndarray | None         # (h, w, 3) uint8  SAM3 overlay
    object_mask: np.ndarray | None     # (H, W) bool, resized onto the 3D track grid
    summary: dict
    extrinsics_w2c: np.ndarray | None  # (T, 4, 4)
    intrinsics: np.ndarray | None      # (fx, fy, cx, cy)

    @property
    def num_frames(self) -> int:
        for arr in (self.track_map, self.recon_map, self.rgb):
            if arr is not None:
                return int(arr.shape[0])
        return 0

    @property
    def grid_hw(self) -> tuple[int, int]:
        for arr in (self.track_map, self.recon_map, self.rgb):
            if arr is not None:
                return int(arr.shape[1]), int(arr.shape[2])
        return (0, 0)


def _read_image(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.array(Image.open(path).convert("RGB"))


def _resize_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if mask.shape == (h, w):
        return mask
    if cv2 is not None:
        return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    img = Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
    return np.array(img) > 0


def _rotmat_to_wxyz(rot: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (w, x, y, z) quaternion."""
    m = rot
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def discover_clips(root: Path, tracks_name: str) -> list[str]:
    tracks_dir = root / tracks_name
    ids = {p.name[: -len("_dense.npz")] for p in tracks_dir.glob("*_dense.npz")}
    if not ids:  # useful even before dense finishes
        ids = {p.name[: -len("_user.npz")] for p in tracks_dir.glob("*_user.npz")}
    return sorted(ids)


def load_clip(root: Path, tracks_name: str, video_id: str, label_map: dict | None = None) -> ClipData:
    tracks_dir = root / tracks_name
    dense_path = tracks_dir / f"{video_id}_dense.npz"
    user_path = tracks_dir / f"{video_id}_user.npz"

    track_map = recon_map = rgb = None
    if dense_path.exists():
        with np.load(dense_path, allow_pickle=True) as d:
            track_map = d["track_map"].astype(np.float32) if "track_map" in d else None
            recon_map = d["recon_map"].astype(np.float32) if "recon_map" in d else None
            rgb = d["rgb"] if "rgb" in d else None

    extrinsics = intrinsics = None
    if user_path.exists():
        with np.load(user_path, allow_pickle=True) as u:
            extrinsics = u["extrinsics_w2c"].astype(np.float32) if "extrinsics_w2c" in u else None
            intrinsics = u["fx_fy_cx_cy"].astype(np.float64) if "fx_fy_cx_cy" in u else None
            if rgb is None and "images_jpeg_bytes" in u:
                rgb = np.stack(
                    [np.array(Image.open(io.BytesIO(b)).convert("RGB")) for b in u["images_jpeg_bytes"]],
                    axis=0,
                )

    # video_id already ends in "_anchor" (dense files are "<id>_anchor_dense.npz")
    clip_dir = root / "sam3_anchor_masks" / "clips" / video_id
    overlay = _read_image(clip_dir / "overlay.png")
    summary = {}
    if (clip_dir / "summary.json").exists():
        summary = json.loads((clip_dir / "summary.json").read_text())

    grid_hw = None
    for arr in (track_map, recon_map, rgb):
        if arr is not None:
            grid_hw = (int(arr.shape[1]), int(arr.shape[2]))
            break

    object_mask = None
    # Preferred: the combined interaction mask = union(all hands + all objects)
    # written by run_something_sam3_anchor_masks.py. Falls back to the older
    # per-object mask layout for datasets built before that change.
    interaction_path = clip_dir / "interaction_mask.png"
    if grid_hw is not None and interaction_path.exists():
        union = np.array(Image.open(interaction_path).convert("L")) > 0
        object_mask = _resize_mask(union, grid_hw)
    if object_mask is None:
        # collect mask files across the two known layouts:
        #  (a) subset: sam3_anchor_masks/clips/<id>/mask_*.png (+ hand_mask_*.png)
        #  (b) pilot : <id>_mask.png alongside / above the tracks dir
        mask_files = (
            sorted(clip_dir.glob("mask_*.png")) + sorted(clip_dir.glob("hand_mask_*.png"))
            if clip_dir.exists() else []
        )
        if not mask_files:
            for base in (tracks_dir, root, root.parent, tracks_dir.parent.parent):
                hit = sorted(base.glob(f"{video_id}_mask*.png"))
                if hit:
                    mask_files = hit
                    break
        if grid_hw is not None and mask_files:
            union = None
            for mp in mask_files:
                m = np.array(Image.open(mp).convert("L")) > 0
                union = m if union is None else (union | m)
            if union is not None:
                object_mask = _resize_mask(union, grid_hw)

    # fill label/prompt from the SSv2 label map when there's no summary.json
    if not summary and label_map is not None:
        base_id = video_id[: -len("_anchor")] if video_id.endswith("_anchor") else video_id
        meta = label_map.get(base_id)
        if meta:
            summary = {"label": meta.get("label", base_id),
                       "prompt": ", ".join(meta.get("placeholders", []))}
    # if no overlay image, fall back to the anchor RGB frame so the panel isn't blank
    if overlay is None and rgb is not None:
        overlay = rgb[0]

    return ClipData(
        video_id=video_id, track_map=track_map, recon_map=recon_map, rgb=rgb,
        overlay=overlay, object_mask=object_mask, summary=summary,
        extrinsics_w2c=extrinsics, intrinsics=intrinsics,
    )


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _rainbow(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def sample_frame_points(
    field: np.ndarray,             # (H, W, 3) xyz
    rgb_frame: np.ndarray | None,  # (H, W, 3) uint8
    stride: int,
    mask: np.ndarray | None,
    mask_only: bool,
    color_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    pts = field[::stride, ::stride].reshape(-1, 3)
    cols_src = rgb_frame[::stride, ::stride].reshape(-1, 3) if rgb_frame is not None else None
    sub_mask = mask[::stride, ::stride].reshape(-1) if mask is not None else None

    keep = np.isfinite(pts).all(axis=1)
    if mask_only and sub_mask is not None:
        keep = keep & sub_mask
    pts = pts[keep]

    if color_mode == "height" and len(pts):
        z = pts[:, 1]
        cols = _rainbow((z - z.min()) / (np.ptp(z) + 1e-6))
    elif color_mode == "mask" and sub_mask is not None:
        on = sub_mask[keep]
        cols = np.full((len(pts), 3), 130, dtype=np.uint8)
        cols[on] = np.array([240, 40, 40], dtype=np.uint8)
    elif cols_src is not None:
        cols = cols_src[keep].astype(np.uint8)
    else:
        cols = np.full((len(pts), 3), 200, dtype=np.uint8)
    return pts.astype(np.float32), cols


def sample_tracks(
    track_map: np.ndarray,          # (T, H, W, 3)
    rgb0: np.ndarray | None,        # (H, W, 3) anchor-frame colors
    mask: np.ndarray | None,
    use_mask: bool,
    n: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (tracks (T, M, 3), anchor_rgb (M, 3)) for M<=n points with finite trajectories."""
    T, H, W, _ = track_map.shape
    finite = np.isfinite(track_map).all(axis=(0, 3))  # (H, W) finite across all frames
    cand = finite & mask if (use_mask and mask is not None) else finite
    ys, xs = np.where(cand)
    if len(ys) == 0:
        ys, xs = np.where(finite)
    if len(ys) == 0:
        return np.zeros((T, 0, 3), np.float32), np.zeros((0, 3), np.uint8)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    ys, xs = ys[idx], xs[idx]
    tracks = track_map[:, ys, xs, :].astype(np.float32)            # (T, M, 3)
    cols = rgb0[ys, xs].astype(np.uint8) if rgb0 is not None else np.full((len(ys), 3), 220, np.uint8)
    return tracks, cols


def sample_indices(finite_hw, mask, use_mask, n, seed=0):
    """Pick up to n (y,x) pixels with finite trajectories (intersected with mask)."""
    cand = finite_hw & mask if (use_mask and mask is not None) else finite_hw
    ys, xs = np.where(cand)
    if len(ys) == 0:
        ys, xs = np.where(finite_hw)
    if len(ys) == 0:
        return np.array([], int), np.array([], int)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    return ys[sel], xs[sel]


def build_trail_segments(tracks: np.ndarray, cols_rgb: np.ndarray, mode: str):
    """tracks (T, M, 3) -> line segments (N,2,3) and colors (N,2,3) for add_line_segments."""
    T, M, _ = tracks.shape
    if T < 2 or M == 0:
        return np.zeros((0, 2, 3), np.float32), np.zeros((0, 2, 3), np.uint8)
    seg = np.stack([tracks[:-1], tracks[1:]], axis=2)             # (T-1, M, 2, 3)
    seg = seg.transpose(1, 0, 2, 3).reshape(-1, 2, 3)            # (M*(T-1), 2, 3)
    if mode == "time":
        tc = _rainbow(np.linspace(0, 1, T - 1))                  # (T-1, 3)
        c = np.broadcast_to(tc[None, :, None, :], (M, T - 1, 2, 3))
    elif mode == "track":
        tc = _rainbow(np.linspace(0, 1, M))                      # (M, 3)
        c = np.broadcast_to(tc[:, None, None, :], (M, T - 1, 2, 3))
    else:  # rgb
        c = np.broadcast_to(cols_rgb[:, None, None, :], (M, T - 1, 2, 3))
    return seg.astype(np.float32), c.reshape(-1, 2, 3).astype(np.uint8)


PHASES = ["1. SAM3 masks", "2. DA3 geometry", "3. Dense tracks"]


# --------------------------------------------------------------------------- #
# Viewer
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/_smoke100k"))
    ap.add_argument("--tracks-name", default="anchor_tracks32")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-points", type=int, default=120_000)
    ap.add_argument("--labels", type=Path, default=None,
                    help="SSv2 train.json for action labels (optional; "
                         "falls back to each clip's summary.json).")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    clips = discover_clips(root, args.tracks_name)
    if not clips:
        raise SystemExit(f"no clips with dense/user NPZ under {root / args.tracks_name}")
    print(f"found {len(clips)} clips: {clips[:8]}{' ...' if len(clips) > 8 else ''}")

    label_map = None
    if args.labels and args.labels.expanduser().exists():
        try:
            label_map = {row["id"]: row for row in json.loads(args.labels.expanduser().read_text())}
            print(f"loaded {len(label_map)} SSv2 labels from {args.labels}")
        except Exception as e:
            print(f"could not load labels ({e}); continuing without")

    server = viser.ViserServer(host=args.host, port=args.port, label="3D Track Curation Viewer")
    server.scene.set_up_direction("-y")
    server.scene.add_frame("/world", show_axes=True, axes_length=0.3, axes_radius=0.004)
    state: dict = {"clip": None, "playing": False, "center": np.zeros(3), "radius": 1.0}
    gui = server.gui

    def frame_camera(cam):
        c = state["center"]
        r = max(float(state["radius"]), 0.1)
        cam.look_at = tuple(float(x) for x in c)
        cam.position = tuple(float(x) for x in (c + np.array([1.4 * r, -1.4 * r, -2.0 * r])))

    def set_video_camera(cam, t):
        """Point the viewer camera at the scene from the estimated video camera pose."""
        cd: ClipData = state["clip"]
        if cd is None or cd.extrinsics_w2c is None or cd.intrinsics is None:
            frame_camera(cam)
            return
        w2c = cd.extrinsics_w2c[min(int(t), len(cd.extrinsics_w2c) - 1)]
        c2w = np.linalg.inv(w2c)
        rot, pos = c2w[:3, :3], c2w[:3, 3]
        forward, up = rot[:, 2], -rot[:, 1]   # OpenCV cam: +Z forward, +Y down
        _, fy, _, cy = cd.intrinsics
        cam.fov = float(2 * np.arctan2(cy, fy)) if fy else 1.0
        cam.up_direction = tuple(float(x) for x in up)
        cam.position = tuple(float(x) for x in pos)
        cam.look_at = tuple(float(x) for x in (pos + forward))

    def apply_view(cam, t):
        if state.get("lock_cam"):
            set_video_camera(cam, t)

    gui.add_markdown("## 3D Track Curation Viewer")
    clip_dd = gui.add_dropdown("Clip", options=clips, initial_value=clips[0])
    phase_dd = gui.add_dropdown("Phase", options=PHASES, initial_value=PHASES[2])
    info_md = gui.add_markdown("")

    with gui.add_folder("Playback"):
        frame_sl = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
        play_btn = gui.add_button("Play / Pause")
        fps_sl = gui.add_slider("FPS", min=1, max=24, step=1, initial_value=8)
        rgb_panel = gui.add_image(np.zeros((4, 4, 3), np.uint8), label="current frame")

    with gui.add_folder("SAM3 overlay"):
        overlay_panel = gui.add_image(np.zeros((4, 4, 3), np.uint8), label="anchor + masks")

    with gui.add_folder("Display"):
        stride_sl = gui.add_slider("Subsample stride", min=1, max=12, step=1, initial_value=4)
        psize_sl = gui.add_slider("Point size", min=0.001, max=0.05, step=0.001, initial_value=0.012)
        frame_btn = gui.add_button("Frame camera on points")
        vidcam_btn = gui.add_button("View from video camera")
        lockcam_cb = gui.add_checkbox("Lock view to video camera", initial_value=False)
        color_dd = gui.add_dropdown("Color", options=["rgb", "height", "mask"], initial_value="rgb")
        maskonly_cb = gui.add_checkbox("Object-mask points only", initial_value=False)
        frustum_cb = gui.add_checkbox("Show camera frustum", initial_value=True)

    with gui.add_folder("Tracks (phase 3)"):
        ntracks_sl = gui.add_slider("Num tracks", min=20, max=4000, step=20, initial_value=800)
        trailcol_dd = gui.add_dropdown("Trail color", options=["rgb", "time", "track"], initial_value="rgb")
        tracksrc_dd = gui.add_dropdown("Sample from", options=["object mask", "whole frame"],
                                       initial_value="object mask")
        gtmode_dd = gui.add_dropdown(
            "GT vs pred", options=["off", "overlay", "side-by-side"], initial_value="overlay",
            hint="Compare predicted track_map vs GT recon_map (from inference npz). GT trails are "
                 "gray, GT markers green; predicted use the Trail color above.")
        bgcloud_cb = gui.add_checkbox("Faint background cloud", initial_value=True)

    def phase_idx() -> int:
        return PHASES.index(phase_dd.value)

    def auto_stride(cd: ClipData) -> int:
        h, w = cd.grid_hw
        s = int(stride_sl.value)
        while h and (h // s) * (w // s) > args.max_points and s < 16:
            s += 1
        return s

    def _remove(name):
        try:
            server.scene[name].remove()
        except Exception:
            pass

    def render():
        cd: ClipData = state["clip"]
        if cd is None or cd.num_frames == 0:
            return
        phase = phase_idx()
        t = 0 if phase == 0 else min(int(frame_sl.value), cd.num_frames - 1)

        if cd.rgb is not None:
            rgb_panel.image = cd.rgb[t]
        if cd.overlay is not None:
            overlay_panel.image = cd.overlay

        if phase == 2 and cd.track_map is not None:
            # ----- TRACKS: predicted trails (+ GT trails for comparison) -----
            gt_mode = gtmode_dd.value
            has_gt = gt_mode != "off" and cd.recon_map is not None
            sig = (cd.video_id, int(ntracks_sl.value), tracksrc_dd.value, trailcol_dd.value, gt_mode)
            if state.get("trail_sig") != sig:
                # sample the SAME pixels for pred and GT (finite in both) so the
                # two trajectory sets are directly comparable point-for-point.
                finite = np.isfinite(cd.track_map).all(axis=(0, 3))
                if has_gt:
                    finite = finite & np.isfinite(cd.recon_map).all(axis=(0, 3))
                ys, xs = sample_indices(finite, cd.object_mask,
                                        tracksrc_dd.value == "object mask", int(ntracks_sl.value))
                pred = cd.track_map[:, ys, xs, :].astype(np.float32)        # (T, M, 3)
                tcols = (cd.rgb[0][ys, xs].astype(np.uint8) if cd.rgb is not None
                         else np.full((len(ys), 3), 220, np.uint8))
                gt = cd.recon_map[:, ys, xs, :].astype(np.float32) if has_gt else None
                # side-by-side: shift GT by ~1.3x the predicted x-extent
                dx = 0.0
                if gt is not None and gt_mode == "side-by-side" and pred.shape[1]:
                    dx = 1.3 * float((np.nanmax(pred[..., 0]) - np.nanmin(pred[..., 0])) + 1e-3)
                    gt = gt + np.array([dx, 0, 0], np.float32)

                seg, segc = build_trail_segments(pred, tcols, trailcol_dd.value)
                if len(seg):
                    server.scene.add_line_segments("/tracks", points=seg, colors=segc, line_width=2.0)
                else:
                    _remove("/tracks")
                if gt is not None and gt.shape[1]:
                    gseg, _ = build_trail_segments(gt, tcols, "rgb")
                    gcol = np.full_like(gseg.astype(np.uint8), 190)         # gray GT trails
                    server.scene.add_line_segments("/gt_tracks", points=gseg, colors=gcol, line_width=1.5)
                else:
                    _remove("/gt_tracks")

                state["trail_sig"] = sig
                state["tracks_cache"] = pred
                state["gt_cache"] = gt
                state["track_cols"] = tcols
                allpts = pred.reshape(-1, 3) if gt is None else np.concatenate([pred, gt], 1).reshape(-1, 3)
                if allpts.size:
                    state["center"] = np.nanmean(allpts, axis=0)
                    state["radius"] = float(np.nanmean(np.linalg.norm(allpts - state["center"], axis=1)) * 2.0)

            # moving markers at the current frame: pred in RGB, GT in green
            pred = state.get("tracks_cache")
            tcols = state.get("track_cols")
            gt = state.get("gt_cache")
            if pred is not None and pred.shape[1]:
                hc = tcols if tcols is not None and len(tcols) == pred.shape[1] else \
                    np.full((pred.shape[1], 3), 255, np.uint8)
                server.scene.add_point_cloud("/track_heads", points=pred[t], colors=hc,
                                             point_size=float(psize_sl.value) * 2.6, point_shape="circle")
            if gt is not None and gt.shape[1]:
                server.scene.add_point_cloud(
                    "/gt_heads", points=gt[t],
                    colors=np.tile(np.array([60, 230, 90], np.uint8), (gt.shape[1], 1)),
                    point_size=float(psize_sl.value) * 2.6, point_shape="circle")
            else:
                _remove("/gt_heads")

            # optional faint context cloud
            if bgcloud_cb.value:
                pts, cols = sample_frame_points(
                    cd.track_map[t], cd.rgb[0] if cd.rgb is not None else None,
                    auto_stride(cd), cd.object_mask, mask_only=False, color_mode="rgb")
                server.scene.add_point_cloud("/points", points=pts, colors=(cols * 0.35).astype(np.uint8),
                                             point_size=float(psize_sl.value) * 0.6, point_shape="circle")
            else:
                _remove("/points")
        else:
            # ----- SAM3 / DA3: single per-frame point cloud -----
            _remove("/tracks")
            _remove("/track_heads")
            _remove("/gt_tracks")
            _remove("/gt_heads")
            field = cd.recon_map[t] if (phase == 1 and cd.recon_map is not None) else (
                cd.track_map[t] if cd.track_map is not None else
                (cd.recon_map[t] if cd.recon_map is not None else None))
            if field is None:
                return
            color_mode = "mask" if phase == 0 else color_dd.value
            pts, cols = sample_frame_points(
                field, cd.rgb[t] if cd.rgb is not None else None, auto_stride(cd),
                cd.object_mask, mask_only=maskonly_cb.value, color_mode=color_mode,
            )
            server.scene.add_point_cloud(
                "/points", points=pts, colors=cols,
                point_size=float(psize_sl.value), point_shape="circle",
            )
            if len(pts):
                state["center"] = pts.mean(axis=0)
                state["radius"] = float(np.linalg.norm(pts - state["center"], axis=1).mean() * 2.0)

        # camera frustum
        if frustum_cb.value and cd.extrinsics_w2c is not None and cd.intrinsics is not None:
            try:
                w2c = cd.extrinsics_w2c[min(t, len(cd.extrinsics_w2c) - 1)]
                c2w = np.linalg.inv(w2c)
                _, fy, _, _ = cd.intrinsics
                h, w = cd.grid_hw
                fov = float(2 * np.arctan2(h / 2, fy)) if fy else 1.0
                server.scene.add_camera_frustum(
                    "/cam_frustum", fov=fov, aspect=float(w / max(h, 1)), scale=0.06,
                    wxyz=_rotmat_to_wxyz(c2w[:3, :3]), position=c2w[:3, 3], color=(40, 200, 120),
                )
            except Exception:
                pass
        else:
            try:
                server.scene["/cam_frustum"].remove()
            except Exception:
                pass

    def load_and_render(video_id: str):
        cd = load_clip(root, args.tracks_name, video_id, label_map)
        state["clip"] = cd
        frame_sl.max = max(cd.num_frames - 1, 1)
        if frame_sl.value > frame_sl.max:
            frame_sl.value = 0
        s = cd.summary
        info_md.content = (
            f"**{video_id}** — {s.get('label', '?')}  \n"
            f"prompt: `{s.get('prompt', '?')}` · frames: {cd.num_frames} · "
            f"grid: {cd.grid_hw} · dense: {'yes' if cd.track_map is not None else 'no (user-npz only)'}"
        )
        render()

    @clip_dd.on_update
    def _(_):
        state["trail_sig"] = None  # force trail rebuild for the new clip
        load_and_render(clip_dd.value)

    for ctrl in (phase_dd, frame_sl, stride_sl, psize_sl, color_dd, maskonly_cb, frustum_cb,
                 ntracks_sl, trailcol_dd, tracksrc_dd, gtmode_dd, bgcloud_cb):
        @ctrl.on_update
        def _(_):
            render()

    @play_btn.on_click
    def _(_):
        state["playing"] = not state["playing"]

    @frame_btn.on_click
    def _(event):
        frame_camera(event.client.camera)

    @vidcam_btn.on_click
    def _(event):
        set_video_camera(event.client.camera, frame_sl.value)

    @lockcam_cb.on_update
    def _(_):
        state["lock_cam"] = lockcam_cb.value
        if lockcam_cb.value:
            for c in server.get_clients().values():
                set_video_camera(c.camera, frame_sl.value)

    @server.on_client_connect
    def _(client):
        # default to the estimated video viewpoint so tracks line up with the frame
        set_video_camera(client.camera, frame_sl.value)

    load_and_render(clips[0])
    print(f"\nviewer ready on http://{args.host}:{args.port}  (forward the port, then open in a browser)\n")

    while True:
        if state["playing"] and state["clip"] is not None:
            nf = state["clip"].num_frames
            frame_sl.value = (int(frame_sl.value) + 1) % max(nf, 1)
            render()
            if state.get("lock_cam"):
                for c in server.get_clients().values():
                    set_video_camera(c.camera, frame_sl.value)
            time.sleep(1.0 / max(fps_sl.value, 1))
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
