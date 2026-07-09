# Future-3D-Scene-Flow — Inference & Visualization on new videos (OOD) and val

This guide lets you run the trained future-scene-flow model on **arbitrary new videos**
(zero-shot / "OOD") and on the **validation set**, and produce 2D and 3D
visualizations (including a self-contained shareable HTML and an interactive viser
viewer). Written so another agent can process new clips end-to-end.

---

## 0. What the model does (read first)

- Input: the **first 7 consecutive frames** of a clip (`obs=7`) as RGB + a per-frame
  geometry map `Pj` (depth back-projected into the frame-0 camera).
- Output: **predicted 3D point tracks for frames 7..21** (`total=22`, so 15 future
  frames) — "observe 7, predict 15".
- All predictions are in the **last-observed-frame camera** (ego-motion subtracted).
- It reuses the pretrained TrackCraft3r model (Wan2.1 DiT + a small fresh LoRA). The
  fine-tuned checkpoint (`best.pt`, ~140 MB) stores only the trained tensors and
  **requires the base TrackCraft3r + Wan2.1 weights** at load time.

**CRITICAL — frame sampling must match training.** Training clips are **22 CONSECUTIVE
frames (stride 1)**. If you feed frames that are far apart in time (e.g. uniformly
sampled across a long video), the model sees ~N× the motion it trained on and predicts
garbage. Always sample frames so inter-frame motion is training-like (stride 1, or a
small stride to roughly match ~12 fps). See the OOD preprocessing step.

---

## 1. Environment

```bash
PY=/weka/scratch/hbharad2/users/yjangir1/conda-envs/3dflow/bin/python   # the 3dflow env
REPO=/weka/scratch/hbharad2/users/yjangir1/future-3d-scene-flow-clean
cd $REPO

# base weights (NOT in git). Either of these trees has trackcraft3r/ + wan_models/ :
CK=/weka/scratch/hbharad2/users/yjangir1/future_flow_homaga/future-3d-scene-flow/models_pretrained/checkpoints
# CK=$REPO/external/TrackCraft3r/checkpoints        # alt (symlink), same contents

# ALWAYS export these before any inference script (the pipeline resolves the Wan2.1
# base from MODELSCOPE_CACHE and must not try to download):
export MODELSCOPE_CACHE="$CK/wan_models" MODELSCOPE_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
```

**Trained checkpoint (the model):**
`$REPO/data/ss_subset100k/future_track_training/freshlora_ss100k_7to22_320x576/best.pt`
(epoch 68, best val future-ADE ≈ 3.98 cm). Pass it as `--checkpoint`; pass
`$CK/trackcraft3r/model.safetensors` as `--base-checkpoint`.

If it's not on local disk, it's hosted on the JHU SharePoint (folder
`3d_tracks_data/checkpoint_based_on_trackcraft3r/trainedon_ss100k/`, with
`best.pt` + `last.pt` + `config.json`). Download from the browser, or with rclone
(the `shared3d:` remote resolves to `.../Documents/3d_tracks_data`):
```bash
rclone copy shared3d:checkpoint_based_on_trackcraft3r/trainedon_ss100k \
  "$REPO/checkpoints_future/trainedon_ss100k" -P
CKPT="$REPO/checkpoints_future/trainedon_ss100k/best.pt"
```
To evaluate it on the val split (ADE/FDE), see REPRODUCE.md §5
("Evaluate the `trainedon_ss100k` checkpoint").

Every inference script needs **1 GPU** (`CUDA_VISIBLE_DEVICES=<idx>`). ~18 GB to load;
fits alongside other jobs on an 80 GB card.

---

## 2. Data formats

Per clip, in a "tracks dir", named by `<vid>`:

- `<vid>_user.npz` — geometry source. Keys:
  `images_jpeg_bytes` (T,) JPEG frames, `rgb` (T,H,W,3) uint8, `depth_map` (T,H,W) z-depth,
  `extrinsics_w2c` (T,4,4) world→cam (frame-0 = identity), `fx_fy_cx_cy` (4,).
  **`images_jpeg_bytes` is required by TrackCraft3r's loader** (RGB→BGR, JPEG q95).
- `<vid>_dense.npz` — TrackCraft3r dense output (the pseudo-GT). Keys:
  `rgb` (T,H,W,3), `recon_map` (T,H,W,3) = Pj(t) world geometry, `track_map` (T,H,W,3) =
  GT P0(t) tracks. **This provides the real (GT) tracks** for the GT-vs-pred comparison.
- mask: a `<vid>_mask.png` (SAM3 object mask on frame 0), referenced by a small manifest.

Training/val clips live under `data/ss_subset100k/anchor_tracks32/` with
`sam3_anchor_masks/manifest_merged.json`. They are **22-frame** clips (obs 7 → 15 future).

---

## 3. Process NEW videos end-to-end (the OOD pipeline)

Put `.mp4`s in a directory `VID=/path/to/videos`. Everything writes under it.

### 3a. Frames + depth/camera (DA3) → `<vid>_user.npz`
```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/comparison/preprocess_ood_videos.py \
  --video-dir "$VID" --out-dir "$VID/processed22" \
  --num-frames 22 --frame-stride 0 --height 480 --width 832
```
- `--frame-stride 0` = 22 frames uniformly spanning the clip (observe 7 + future 15 cover
  the whole video). **This is OOD vs training** (which is stride-1 consecutive); use it to
  see motion over a full clip. For results closest to training, use `--frame-stride 1`
  (22 consecutive from `--start-frame`), or a small stride (2–3) to roughly match ~12 fps.
- Writes `images_jpeg_bytes` + `rgb` + depth + extrinsics + intrinsics (see §2).

### 3b. SAM3 object mask on frame 0 → `<vid>_mask.png`
```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/comparison/sam3_ood_masks.py \
  --proc-dir "$VID/processed22" --out-dir "$VID/masks" \
  --prompt <vid>:"the object being manipulated" --prompt <vid2>:"a cup" ...
```
One `--prompt <vid>:"text"` per clip (text = what to segment). Writes `<vid>_mask.png`
(+ overlay). `--frame 0 --conf 0.2 --min-area 200` are the defaults.

### 3c. Real GT tracks (TrackCraft3r dense) → `<vid>_dense.npz`
Needed for GT-vs-pred viz. Reuses the 22-frame `_user.npz`:
```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_trackcraft3r_dense_batch.py \
  --root "$VID" --tracks-name processed22 \
  --video <vid> --video <vid2> ... \
  --trackcraft-root external/TrackCraft3r \
  --num-frames 22 --frame-stride 1 --device cuda --keep-going
```
(`--video X` = the `<vid>` stem; reads `processed22/<vid>_user.npz`, writes
`processed22/<vid>_dense.npz`. It synthesizes dummy tracks internally; only needs
`images_jpeg_bytes` + depth + extrinsics.) Skip this if you only want zero-shot
prediction with no GT overlay (see §4d).

### 3d. Manifest mapping each clip → its mask
```bash
$PY - <<'PYEOF'
import json, glob, os
VID=os.environ["VID"]; items=[]
for f in sorted(glob.glob(f"{VID}/masks/*_mask.png")):
    v=os.path.basename(f).replace("_mask.png","")
    items.append({"clip_path": f"{v}.mp4", "video_uid": v, "label": v,
                  "instances":[{"mask_path": f"masks/{v}_mask.png"}]})
json.dump(items, open(f"{VID}/ood_manifest.json","w"), indent=2); print("wrote", len(items))
PYEOF
```
Now `$VID/processed22/<vid>_dense.npz` + `$VID/ood_manifest.json` + `$VID/masks/` make
the OOD clips **structurally identical to val clips**, so the same viewers work.

---

## 4. Visualizations

Set `CKPT=.../best.pt` and the env from §1. For **val** use
`--root data/ss_subset100k --dense-tracks-name anchor_tracks32 --dense-manifest sam3_anchor_masks/manifest_merged.json`;
for **OOD** use `--root "$VID" --dense-tracks-name processed22 --dense-manifest ood_manifest.json`.
All use `--sparse-tracks-name ""`.

### 4a. 2D GT-vs-pred trail videos + HTML gallery (`render_unified_viewer.py`)
Tracks projected into the last-observed frame (frozen backdrop).
```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/comparison/render_unified_viewer.py \
  --checkpoint "$CKPT" --base-checkpoint "$CK/trackcraft3r/model.safetensors" \
  --root "$VID" --dense-tracks-name processed22 --dense-manifest ood_manifest.json \
  --sparse-tracks-name "" --output-dir "$VID/viewer" \
  --train-count 999 --val-count 999 --num-points 64 --dense-only
# writes videos/*.webm + index.html + per-clip ADE/FDE
```

### 4b. FULL-video GT-vs-pred (tracks glued to the moving video) — recommended 2D
`render_full_video_gt_vs_pred.py` plays all frames and re-projects tracks into each
frame's real camera pose (not frozen). GT left, prediction right.
```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/comparison/render_full_video_gt_vs_pred.py \
  --checkpoint "$CKPT" --base-checkpoint "$CK/trackcraft3r/model.safetensors" \
  --root "$VID" --dense-tracks-name processed22 --dense-manifest ood_manifest.json \
  --sparse-tracks-name "" --output-dir "$VID/viewer_fullvideo" \
  --split all --count 0 --num-points 80
```

### 4c. Self-contained interactive 3D HTML (shareable single file) (`build_3d_tracks_html.py`)
Point cloud + GT (green) & predicted (red) tracks, animated, clip dropdown, plotly.js
inlined → one `.html` you can upload to SharePoint / open anywhere.
```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/comparison/build_3d_tracks_html.py \
  --checkpoint "$CKPT" --base-checkpoint "$CK/trackcraft3r/model.safetensors" \
  --ood-root "$VID" --ood-tracks-name processed22 --ood-manifest ood_manifest.json \
  --num-val 10 --out-html "$VID/tracks3d.html"
# (also pulls 10 random val clips by default; set --num-val 0 to skip val)
```

### 4d. Live interactive viser 3D viewer (moving point cloud) (`viser_tracks_3d.py`)
Best interactive experience: animated scene cloud + GT/pred tracks, dropdown, play.
```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/comparison/viser_tracks_3d.py \
  --checkpoint "$CKPT" --base-checkpoint "$CK/trackcraft3r/model.safetensors" \
  --ood-root "$VID" --ood-tracks-name processed22 --ood-manifest ood_manifest.json \
  --num-val 10 --port 8130          # add --share for a public URL (exposes data publicly)
# view: ssh -N -L 8130:<node>:8130 <login>  then  http://localhost:8130
```
It frees the GPU model after caching the clips, so it can run next to a training job.

### 4e. Zero-shot only (NO GT, skip §3c) — model prediction on the mask (`ood_render_html.py` / `ood_viser_viewer.py`)
If you don't run TrackCraft3r (no GT), use these — they call `ood_common.ood_predict`
(runs the model on the SAM-masked object) and show prediction only:
```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/comparison/ood_render_html.py \
  --checkpoint "$CKPT" --proc-dir "$VID/processed22" --mask-dir "$VID/masks" \
  --out-dir "$VID/viewer_pred" --grid-stride 4
```

**Observe-7-over-the-whole-video variant.** For "sample 7 frames spanning the
full clip, then predict the future tracks," preprocess with `--num-frames 7
--frame-stride 0` (the npz then holds exactly the 7 observed frames; the model
predicts the 15 future internally). Then use the zero-shot scripts as above with
`--proc-dir "$VID/processed7"`.

### 4f. Push side-by-side videos to Weights & Biases (`log_val_videos_wandb.py`)
After a sharded `render_unified_viewer` run, `merge_val_viewers.py` merges shards and
`log_val_videos_wandb.py` logs GT|pred side-by-side as a wandb Table.

### 4g. Live viser 3D for zero-shot (moving predicted OBJECT) (`ood_viser_viewer_moving.py`)
Animates the scene through all 22 frames. Frames 0–6 are the real observed cloud
(moves with the real camera); for the future frames the **background is the real
observed cloud FROZEN at the last observed frame** and **only the masked object
animates** (its supervised prediction). Trails are batched into one line-segments
node (smooth playback); controls for trail length, track spacing, point sizes,
and a default video-like camera pose.
```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/comparison/ood_viser_viewer_moving.py \
  --checkpoint "$CKPT" --base-checkpoint "$CK/trackcraft3r/model.safetensors" \
  --proc-dir "$VID/processed7" --mask-dir "$VID/masks" \
  --grid-stride 2 --pc-stride 5 --port 8130
```
**Why only the object moves:** the network emits a dense full-frame field, but
training supervises ONLY the SAM-object mask pixels
(`unified_dataset.py`: `cand = mask & valid`). Off-object motion is unsupervised
(measured 1–5 cm move, and 7–22 cm offset from the true observed geometry), so it
is deliberately NOT rendered as prediction — the frozen real background is shown
instead. See gotcha 9.

---

## 5. Critical gotchas (things that bit us)

1. **Frame sampling ≠ training → garbage.** The original `preprocess_ood_videos.py`
   used `np.linspace` (10 frames over the whole video); the "observed 7" then spanned
   seconds and the model blew up. Fixed to support `--num-frames 22` + `--frame-stride`.
   Match training's spacing (stride 1, or small stride).
2. **`images_jpeg_bytes` required.** TrackCraft3r's loader reads it; a `_user.npz` with
   only `rgb` fails with `BadZipFile`/KeyError. The preprocessor now writes both.
3. **Base-weights resolution.** Must set `MODELSCOPE_CACHE=$CK/wan_models` (+ offline
   flags) or the pipeline tries to download the Wan2.1 base. `--checkpoint` is the
   fine-tuned `best.pt`; `--base-checkpoint` is `trackcraft3r/model.safetensors`.
4. **Coordinate frame.** Predictions/GT are in the last-observed camera (camera-motion
   subtracted). For "tracks on the moving video," re-project to world then per-frame
   camera (done in `render_full_video_gt_vs_pred.py` / `viser_tracks_3d.py`).
5. **Corrupt `.npz`.** The dataset (`unified_dataset.__getitem__`) skips corrupt/truncated
   clips (interrupted writes) instead of crashing — expect `[dataset] skipping bad clip`.
6. **Self-contained HTML must avoid bare `NaN`** and use `rgb(r,g,b)` marker colors (hex
   arrays don't render in Scatter3d/WebGL) — handled in `build_3d_tracks_html.py`.
7. **wandb "no space"** = the shared filesystem briefly filled; it knocks out wandb
   writes (and checkpoint saves). Use `--save-every 10` to reduce checkpoint churn; the
   dense `anchor_tracks32` tree is ~7.6 TB (prune/convert to sparse for headroom).
8. **viser** is a live server (port-forward to view). Its `--share` public URL exposes
   data to the internet and is normally blocked — use only with explicit approval.
9. **The model predicts a DENSE field but is supervised ONLY on the mask.** The
   Wan head outputs a full (3,T,H,W) geometry field, yet the loss samples only
   SAM-object pixels (`unified_dataset.py` `cand = mask & valid`). So off-object
   ("background") predictions are unsupervised and untrustworthy — do NOT render
   the dense field as if the whole scene were forecast. `ood_viser_viewer_moving.py`
   shows the real observed background frozen and animates only the masked object.
   (Verified: background predicted move 1–5 cm; predicted-vs-true-observed
   background offset 7–22 cm.) The observed frames 0–6 are real reconstruction,
   not prediction — most of the "looks great" impression comes from those + the
   real RGB texture, not from prediction accuracy. No leakage: input tensor is 7
   frames, output 22 (15 generated), deterministic, clip-specific; npz holds only
   7 frames on disk.

---

## 6. Script index (all under `scripts/` or `scripts/comparison/`)

| script | role |
| --- | --- |
| `comparison/preprocess_ood_videos.py` | video → 22 frames → DA3 → `_user.npz` (writes `images_jpeg_bytes`) |
| `comparison/sam3_ood_masks.py` | SAM3 object mask on frame 0 (text-prompted) |
| `run_trackcraft3r_dense_batch.py` | TrackCraft3r dense → `_dense.npz` (real GT tracks) |
| `comparison/render_unified_viewer.py` | 2D GT-vs-pred trail videos + HTML + ADE/FDE |
| `comparison/render_full_video_gt_vs_pred.py` | 2D GT-vs-pred on the MOVING video (per-frame reprojection) |
| `comparison/build_3d_tracks_html.py` | self-contained animated 3D HTML (point cloud + GT/pred), shareable |
| `comparison/viser_tracks_3d.py` | live interactive viser 3D (moving cloud + GT/pred, dropdown) |
| `comparison/ood_render_html.py`, `ood_viser_viewer.py`, `ood_common.py` | zero-shot (no GT) prediction + viz |
| `comparison/ood_viser_viewer_moving.py` | zero-shot live viser 3D: frozen real background + moving predicted OBJECT (masked) |
| `comparison/merge_val_viewers.py`, `log_val_videos_wandb.py` | merge shards / push videos to wandb |
| `render_future_scene_flow_viewer.py` | provides `build_model_from_ckpt(ckpt, base_ckpt)` used everywhere |
| `models_pretrained/future_scene_flow/unified_dataset.py` | `UnifiedTrackDataset` (dense clips → model inputs + GT); val split via `split_items(items, 0.05, 7)` |

**Model build (in any custom script):**
```python
import models_pretrained  # noqa  (import isolation; must precede diffsynth/evaluation)
from render_future_scene_flow_viewer import build_model_from_ckpt
model, state = build_model_from_ckpt(Path(ckpt), Path(base_ckpt)); model.eval()
```
