# Data

This folder documents the local video-data recipes used for future 3D scene-flow experiments. Generated media, model outputs, masks, and local HTML viewers live under `data/` but are ignored by git.

Ignored local data roots:

```text
data/action100m/
data/something_something/
```

The repo stores the scripts and reproducibility notes; the large downloaded videos and model outputs stay local.

## Environment Setup

External model code is tracked as git submodules:

```text
external/sam3
external/TrackCraft3r
external/depth-anything-3
```

Fresh clone:

```bash
git clone --recursive https://github.com/homangab/future-3d-scene-flow.git
cd future-3d-scene-flow
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Create the fresh conda environment used for the local Something-Something tracking runs:

```bash
. /home/homanga/miniconda3/etc/profile.d/conda.sh  # skip if conda is already initialized

conda create -y -n 3dflow python=3.10 pip
conda activate 3dflow

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.11.0 torchvision==0.26.0 xformers==0.0.35

python -m pip install -e external/sam3 -e external/depth-anything-3
python -m pip install --no-build-isolation -e external/TrackCraft3r

python -m pip install \
  datasets pyarrow yt-dlp huggingface_hub \
  pycocotools numba python-rapidjson

python -m pip install --force-reinstall \
  numpy==1.26.4 \
  opencv-python==4.10.0.84 \
  opencv-python-headless==4.10.0.84
```

The NumPy/OpenCV pins avoid binary-ABI mismatches with the current SAM3, DA3, and TrackCraft3r stack. If SAM3 raises PyTorch dtype or `init_state` keyword compatibility errors, apply the local compatibility patch:

```bash
git -C external/sam3 apply ../../patches/sam3_pytorch_compat.patch
```

### Local Model Files

This setup installs packages only; it should reuse model files that are already on disk.

TrackCraft3r expects its checkpoint and Wan model cache under `external/TrackCraft3r/checkpoints`. Symlink that path to the existing local checkpoint directory:

```bash
export TRACKCRAFT_CHECKPOINTS=/path/to/local/TrackCraft3r/checkpoints
test -f "$TRACKCRAFT_CHECKPOINTS/trackcraft3r/model.safetensors"
test -d "$TRACKCRAFT_CHECKPOINTS/wan_models"

if [ ! -e external/TrackCraft3r/checkpoints ]; then
  ln -s "$TRACKCRAFT_CHECKPOINTS" external/TrackCraft3r/checkpoints
fi
```

SAM3 and Depth Anything 3 can use their existing Hugging Face cache entries. To prevent any hidden model downloads during a run, export offline mode after confirming the cache is present:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

If you prefer to pin SAM3 explicitly, pass a local checkpoint path to the SAM3 scripts:

```bash
python scripts/run_something_sam3_anchor_masks.py \
  --checkpoint ~/.cache/huggingface/hub/models--facebook--sam3/snapshots/<snapshot>/sam3.pt \
  --device cuda
```

The SAM3 and TrackCraft3r steps require CUDA. If running inside a restricted shell, check that PyTorch can see the GPUs, not only `nvidia-smi`:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
PY
```

Verify the installed Python stack:

```bash
python -m pip check

python - <<'PY'
import cv2, numpy, sam3, torch, torchvision
from depth_anything_3.api import DepthAnything3

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("cuda available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("numpy", numpy.__version__, "opencv", cv2.__version__)
print("imports ok")
PY

PYTHONPATH=external/TrackCraft3r:$PYTHONPATH python - <<'PY'
from evaluation.wan_scene_flow_predictor import WanSceneFlowPredictor
print("trackcraft imports ok")
PY
```

## Helper Scripts

Relevant helper scripts in this repo:

```text
scripts/build_action100m_viewer.py
scripts/build_action100m_segment_viewer.py
scripts/run_action100m_sam3_first_frame_masks.py
scripts/prepare_action100m_track_lists.py
scripts/run_action100m_trackcraft3r.py
scripts/preprocess_da3_chunked.py
scripts/build_action100m_mask_trace_viewer.py
scripts/run_something_sam3_anchor_masks.py
scripts/prepare_something_track_lists.py
scripts/run_trackcraft3r_dense_batch.py
scripts/run_something_anchor_tracking_test.sh
scripts/train_future_3d_tracks.py
```

`scripts/run_action100m_trackcraft3r.py` is dataset-agnostic despite the historical name: it can run TrackCraft3r on any local video list.
For large runs, use it first with `--skip-dense` to create DA3/user-NPZ files, then use `scripts/run_trackcraft3r_dense_batch.py` to run dense TrackCraft3r while loading the model once per GPU.

## Action100M Workflow

### Local Layout

After running the Action100M workflow, local data should look like:

```text
data/action100m/
├── annotations/              # Cached Action100M rows, one JSON per video UID
├── raw/                      # Downloaded YouTube source videos and metadata
├── segments/
│   ├── clips/                # Short mid-level segment MP4s
│   ├── segments_manifest.json
│   └── viewer/index.html     # Local segment clip grid
├── selected_videos.json      # Selected Action100M video IDs
├── sam3_first_frame_masks/   # SAM3 masks on segment first frames
├── mask_trace32_preproc/     # DA3 depth/camera outputs
├── mask_trace32_tracks/      # TrackCraft user/dense NPZ files
├── mask_trace32_viewer/      # Projected 3D track HTML viewer
└── viewer/index.html         # Local full-video grid
```

The curated segment viewer is:

```text
data/action100m/segments/viewer/index.html
```

Current verified local state:

- 509 segment clips after excluding the PHD2/PHP-looking source video.
- 35 videos represented in the segment viewer.
- Segment clips are mid-level Action100M annotations, usually 5-18 seconds.

### Cache Annotations

Action100M preview stores YouTube video IDs and hierarchical segment annotations in Hugging Face parquet files. Cache selected rows as JSON so the viewer can be rebuilt without re-querying Hugging Face:

```bash
cd /path/to/future-3d-scene-flow

python - <<'PY'
import json
from pathlib import Path
from datasets import load_dataset

root = Path("data/action100m")
root.mkdir(parents=True, exist_ok=True)
(root / "annotations").mkdir(exist_ok=True)

selected_path = root / "selected_videos.json"
selected = json.loads(selected_path.read_text()) if selected_path.exists() else []
existing = {row["video_uid"] for row in selected}

target_new = 30
added = []
ds = load_dataset(
    "parquet",
    data_files="hf://datasets/facebook/Action100M-preview/data/*.parquet",
    streaming=True,
)

for sample in ds["train"]:
    uid = sample.get("video_uid")
    if not uid or uid in existing:
        continue

    metadata = sample.get("metadata") or {}
    try:
        duration = float(metadata.get("duration"))
    except (TypeError, ValueError):
        continue

    if duration < 45 or duration > 420:
        continue

    mid_nodes = 0
    for node in sample.get("nodes") or []:
        try:
            seg_duration = float(node.get("end", 0)) - float(node.get("start", 0))
            level = int(node.get("level", -1))
        except (TypeError, ValueError):
            continue
        text = ((node.get("gpt") or {}).get("action") or {}).get("brief") or node.get("plm_action") or ""
        if level >= 3 and 5 <= seg_duration <= 18 and text and not text.lower().startswith("na - no actions"):
            mid_nodes += 1

    if mid_nodes < 12:
        continue

    row = {
        "video_uid": uid,
        "title": metadata.get("title") or uid,
        "duration_seconds": duration,
        "source": "Action100M-preview",
    }
    selected.append(row)
    existing.add(uid)

    data = dict(sample)
    data["local_selection"] = row
    (root / "annotations" / f"{uid}.json").write_text(json.dumps(data, indent=2))

    added.append(uid)
    print(f"ADD {len(added):02d} {uid} mid={mid_nodes} title={row['title'][:90]}")
    if len(added) >= target_new:
        break

selected_path.write_text(json.dumps(selected, indent=2))
print("total selected", len(selected))
PY
```

### Download Source Videos

Build a batch file for selected videos missing from `raw/`:

```bash
cd /path/to/future-3d-scene-flow

python - <<'PY'
import json
from pathlib import Path

root = Path("data/action100m")
raw = {p.stem for p in (root / "raw").glob("*.mp4") if ".f" not in p.stem}
missing = [
    row["video_uid"]
    for row in json.loads((root / "selected_videos.json").read_text())
    if row["video_uid"] not in raw
]

Path("data/action100m_missing_urls.txt").write_text(
    "\n".join(f"https://www.youtube.com/watch?v={uid}" for uid in missing) + "\n"
)
print("wrote", len(missing), "urls")
PY
```

Download modest-resolution MP4s:

```bash
yt-dlp \
  --ignore-errors \
  --no-playlist \
  --write-info-json \
  --write-thumbnail \
  --merge-output-format mp4 \
  -f "bv*[height<=480][ext=mp4]+ba[ext=m4a]/b[height<=480][ext=mp4]/b[height<=480]" \
  -o "data/action100m/raw/%(id)s.%(ext)s" \
  --batch-file data/action100m_missing_urls.txt
```

### Build Video Viewers

Create the full-video grid:

```bash
python scripts/build_action100m_viewer.py \
  --root data/action100m
```

Open:

```text
data/action100m/viewer/index.html
```

Create mid-level segment clips and the segment-grid UI:

```bash
python scripts/build_action100m_segment_viewer.py \
  --root data/action100m \
  --per-video 15 \
  --exclude-video-uid=-M6cLOV4aW8
```

Open:

```text
data/action100m/segments/viewer/index.html
```

The `--exclude-video-uid=-M6cLOV4aW8` flag removes the PHD2/PHP-looking software tutorial clips.

### Run SAM3 On Segment First Frames

Run SAM3 on the first frame of each segment clip. The text prompt is each segment's action text:

```bash
python scripts/run_action100m_sam3_first_frame_masks.py \
  --root data/action100m \
  --sam3-root external/sam3 \
  --device cuda
```

Outputs:

```text
data/action100m/sam3_first_frame_masks/
├── clips/<clip-id>/
│   ├── first_frame.jpg
│   ├── overlay.png
│   ├── mask_00.png
│   └── summary.json
├── manifest.json
└── viewer/index.html
```

### Prepare Tracking Lists

Select clips with 1-3 SAM3 masks and split them across two GPU workers:

```bash
python scripts/prepare_action100m_track_lists.py \
  --root data/action100m \
  --sam-manifest sam3_first_frame_masks/manifest.json \
  --min-masks 1 \
  --max-masks 3 \
  --num-shards 2 \
  --output-prefix mask_trace32 \
  --require-clip
```

Outputs:

```text
data/action100m/mask_trace32_selected_videos.txt
data/action100m/mask_trace32_gpu0.txt
data/action100m/mask_trace32_gpu1.txt
```

### Run 3D Tracking

Run Depth Anything 3 preprocessing, TrackCraft user-NPZ creation, and TrackCraft dense tracking:

```bash
python scripts/run_action100m_trackcraft3r.py \
  --root data/action100m \
  --preproc-name mask_trace32_preproc \
  --tracks-name mask_trace32_tracks \
  --video-list data/action100m/mask_trace32_selected_videos.txt \
  --trackcraft-root external/TrackCraft3r \
  --da3-root external/depth-anything-3 \
  --process-res 336 \
  --chunk-size 24 \
  --num-frames 32 \
  --frame-stride 2 \
  --device cuda \
  --keep-going
```

For two GPUs, run two shells with explicit CUDA visibility:

```bash
python scripts/run_action100m_trackcraft3r.py \
  --root data/action100m \
  --preproc-name mask_trace32_preproc \
  --tracks-name mask_trace32_tracks \
  --video-list data/action100m/mask_trace32_gpu0.txt \
  --cuda-visible-devices GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  --process-res 336 \
  --chunk-size 24 \
  --num-frames 32 \
  --frame-stride 2 \
  --device cuda \
  --keep-going

python scripts/run_action100m_trackcraft3r.py \
  --root data/action100m \
  --preproc-name mask_trace32_preproc \
  --tracks-name mask_trace32_tracks \
  --video-list data/action100m/mask_trace32_gpu1.txt \
  --cuda-visible-devices GPU-yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy \
  --process-res 336 \
  --chunk-size 24 \
  --num-frames 32 \
  --frame-stride 2 \
  --device cuda \
  --keep-going
```

The runner is resumable: existing `depth.npy`, `*_user.npz`, and `*_dense.npz` files are skipped.

### Build Projected Track Viewer

Build a local HTML viewer for dense tracks:

```bash
python scripts/build_action100m_mask_trace_viewer.py \
  --root data/action100m \
  --sam-manifest sam3_first_frame_masks/manifest.json \
  --tracks-name mask_trace32_tracks \
  --viewer-name mask_trace32_viewer \
  --frame-stride 2 \
  --copy-json
```

Open:

```text
data/action100m/mask_trace32_viewer/index.html
```

The viewer draws mask-grid tracks on TrackCraft's sampled model RGB frames, not the original video element. For each timestep it projects predicted 3D points into the corresponding estimated camera frame, so the displayed 2D trails include camera-induced image motion. The static SAM3 mask overlay is from the anchor/first frame and is not time-varying.

## Something-Something Workflow

### Local Layout

The Something-Something workflow assumes videos and labels were unpacked under `~/Downloads`:

```text
~/Downloads/20bn-something-something-v2/<video-id>.webm
~/Downloads/20bn-something-something-download-package-labels/labels/train.json
```

Local generated outputs are ignored by git:

```text
data/something_something/
├── anchor_clips/             # MP4 clips starting at the selected anchor frame
├── anchor_selected_videos.txt
├── anchor_selected_videos_test4.txt
├── anchor_selected_videos_track32.txt
├── anchor_selected_videos_track32_gpu0.txt
├── anchor_selected_videos_track32_gpu1.txt
├── sam3_anchor_masks/
│   ├── clips/<video-id>_anchor/
│   ├── manifest.json
│   ├── manifest_all.json
│   └── viewer/index.html
├── anchor_preproc/           # DA3 depth/camera outputs
├── anchor_tracks32/          # 32-frame TrackCraft user/dense NPZ files
└── anchor_track32_viewer/    # Projected 32-frame track viewer
```

The key idea is to avoid tracking from an uninformative first frame. The script scans each video for the first frame where SAM3 detects a hand, chooses an anchor frame a few frames later, prompts SAM3 on that anchor frame with the object names from `train.json` `placeholders`, and writes a new MP4 starting at the anchor frame.

### Create Anchor Clips And SAM3 Object Masks

Run SAM3 hand detection and object prompting:

```bash
python scripts/run_something_sam3_anchor_masks.py \
  --root data/something_something \
  --video-root ~/Downloads/20bn-something-something-v2 \
  --labels ~/Downloads/20bn-something-something-download-package-labels/labels/train.json \
  --sam3-root external/sam3 \
  --limit 10 \
  --device cuda \
  --hand-prompt hand \
  --scan-step 3 \
  --anchor-offset 3 \
  --min-hand-area 150 \
  --min-frames-after-anchor 20 \
  --clip-max-frames 48
```

For each candidate video, the script:

1. Reads the object prompts from `placeholders` in `train.json`.
2. Scans every `--scan-step` frames with SAM3 prompt `hand`.
3. Chooses `anchor_frame = first_hand_frame + --anchor-offset`.
4. Writes `data/something_something/anchor_clips/<id>_anchor.mp4`.
5. Runs SAM3 on the anchor frame for each object prompt.
6. Saves object masks as `mask_XX.png` and hand masks as `hand_mask_XX.png`.
7. Writes `sam3_anchor_masks/manifest.json` and `sam3_anchor_masks/viewer/index.html`.
8. Writes `anchor_selected_videos.txt` for clips with at least one object mask.

To force specific videos, pass one or more IDs:

```bash
python scripts/run_something_sam3_anchor_masks.py \
  --root data/something_something \
  --video-id 78687 \
  --video-id 42326 \
  --device cuda
```

### Scale To All Something-Something Videos

For the full dataset, run SAM3 in shards. `--limit 0` means all matching videos in that shard, and `--no-viewer` avoids rebuilding an HTML page from each worker:

```bash
mkdir -p data/something_something/logs

python scripts/run_something_sam3_anchor_masks.py \
  --root data/something_something \
  --video-root ~/Downloads/20bn-something-something-v2 \
  --labels ~/Downloads/20bn-something-something-download-package-labels/labels/train.json \
  --sam3-root external/sam3 \
  --limit 0 \
  --num-shards 2 \
  --shard-index 0 \
  --manifest-name manifest_shard0.json \
  --selected-list-name anchor_selected_videos_shard0.txt \
  --no-viewer \
  --device cuda \
  --hand-prompt hand \
  --scan-step 3 \
  --anchor-offset 3 \
  --min-hand-area 150 \
  --min-frames-after-anchor 20 \
  --clip-max-frames 48 \
  > data/something_something/logs/sam3_shard0.log 2>&1

python scripts/run_something_sam3_anchor_masks.py \
  --root data/something_something \
  --video-root ~/Downloads/20bn-something-something-v2 \
  --labels ~/Downloads/20bn-something-something-download-package-labels/labels/train.json \
  --sam3-root external/sam3 \
  --limit 0 \
  --num-shards 2 \
  --shard-index 1 \
  --manifest-name manifest_shard1.json \
  --selected-list-name anchor_selected_videos_shard1.txt \
  --no-viewer \
  --device cuda \
  --hand-prompt hand \
  --scan-step 3 \
  --anchor-offset 3 \
  --min-hand-area 150 \
  --min-frames-after-anchor 20 \
  --clip-max-frames 48 \
  > data/something_something/logs/sam3_shard1.log 2>&1
```

If using two GPUs, run the two commands in separate shells with `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`, or set the visible device in the shell before launching each worker.

Merge the shard manifests and write 32-frame trackable lists:

```bash
python scripts/prepare_something_track_lists.py \
  --root data/something_something \
  --manifest-glob "sam3_anchor_masks/manifest_shard*.json" \
  --merged-manifest sam3_anchor_masks/manifest_all.json \
  --selected-name anchor_selected_videos_track32.txt \
  --shard-prefix anchor_selected_videos_track32_gpu \
  --num-shards 2 \
  --min-masks 1 \
  --min-anchor-frames 32
```

Add `--max-masks 3` if you want the conservative 1-3 mask subset. Leave it unset for all clips with at least one SAM3 object mask and enough post-anchor frames.

Create DA3 preprocessing outputs and TrackCraft user NPZ files. This pass is resumable and skips existing `depth.npy` and `*_user.npz` files:

```bash
python scripts/run_action100m_trackcraft3r.py \
  --root data/something_something \
  --preproc-name anchor_preproc_all \
  --tracks-name anchor_tracks32_all \
  --video-list data/something_something/anchor_selected_videos_track32_gpu0.txt \
  --trackcraft-root external/TrackCraft3r \
  --da3-root external/depth-anything-3 \
  --process-res 336 \
  --chunk-size 24 \
  --num-frames 32 \
  --frame-stride 1 \
  --cuda-visible-devices 0 \
  --device cuda \
  --skip-dense \
  --keep-going \
  > data/something_something/logs/prep_gpu0.log 2>&1

python scripts/run_action100m_trackcraft3r.py \
  --root data/something_something \
  --preproc-name anchor_preproc_all \
  --tracks-name anchor_tracks32_all \
  --video-list data/something_something/anchor_selected_videos_track32_gpu1.txt \
  --trackcraft-root external/TrackCraft3r \
  --da3-root external/depth-anything-3 \
  --process-res 336 \
  --chunk-size 24 \
  --num-frames 32 \
  --frame-stride 1 \
  --cuda-visible-devices 1 \
  --device cuda \
  --skip-dense \
  --keep-going \
  > data/something_something/logs/prep_gpu1.log 2>&1
```

Run dense TrackCraft3r in batch mode. This avoids reloading the Wan/TrackCraft model for every clip and skips any existing `*_dense.npz` files:

```bash
python scripts/run_trackcraft3r_dense_batch.py \
  --root data/something_something \
  --tracks-name anchor_tracks32_all \
  --video-list data/something_something/anchor_selected_videos_track32_gpu0.txt \
  --trackcraft-root external/TrackCraft3r \
  --num-frames 32 \
  --frame-stride 1 \
  --cuda-visible-devices 0 \
  --device cuda \
  --keep-going \
  > data/something_something/logs/dense_gpu0.log 2>&1

python scripts/run_trackcraft3r_dense_batch.py \
  --root data/something_something \
  --tracks-name anchor_tracks32_all \
  --video-list data/something_something/anchor_selected_videos_track32_gpu1.txt \
  --trackcraft-root external/TrackCraft3r \
  --num-frames 32 \
  --frame-stride 1 \
  --cuda-visible-devices 1 \
  --device cuda \
  --keep-going \
  > data/something_something/logs/dense_gpu1.log 2>&1
```

Monitor progress:

```bash
find data/something_something/anchor_tracks32_all -name '*_dense.npz' | wc -l
rg -n '^failed:' data/something_something/logs/*.log
du -sh data/something_something/anchor_tracks32_all
```

Build the full projected-track viewer:

```bash
python scripts/build_action100m_mask_trace_viewer.py \
  --root data/something_something \
  --sam-manifest sam3_anchor_masks/manifest_all.json \
  --tracks-name anchor_tracks32_all \
  --viewer-name anchor_track32_all_viewer \
  --frame-stride 1 \
  --max-masks 100 \
  --min-points 1 \
  --copy-json
```

Open:

```text
data/something_something/anchor_track32_all_viewer/index.html
```

The 500-candidate pilot produced 201 successful 32-frame dense tracks and about 55 GB of dense NPZ files. A full-dataset run can be multiple TB, so check available disk space before launching dense tracking.

### Train Future 3D Track Prediction

Once `anchor_tracks32_500/` or another dense-track directory exists, train the repo-local future predictor. The model uses TrackCraft3r dense maps as pseudo-labels but only receives the first 10 video frames and first 10 observed 3D positions for sampled SAM3 object-mask points.

```bash
python scripts/train_future_3d_tracks.py \
  --root data/something_something \
  --tracks-name anchor_tracks32_500 \
  --manifest sam3_anchor_masks/manifest_500.json \
  --output-dir data/something_something/future_track_training/initial_10f_to_32f \
  --obs-frames 10 \
  --total-frames 32 \
  --num-points 128 \
  --image-size 96 160 \
  --embed-dim 128 \
  --num-heads 4 \
  --num-layers 2 \
  --batch-size 2 \
  --epochs 2 \
  --steps-per-epoch 50 \
  --val-steps 10 \
  --samples-per-clip 4 \
  --num-workers 2 \
  --device cuda \
  --amp
```

The first local run indexed 201 dense clips, split them into 181 train and 20 validation clips, and wrote `best.pt`, `last.pt`, `epoch_001.pt`, `epoch_002.pt`, and `config.json` under the output directory above. The best validation ADE from this short run was about 5.8 cm over sampled masked points.

### Select A Small Test Set

For a 3-4 video test run, select the first four usable anchor clips:

```bash
head -n 4 \
  data/something_something/anchor_selected_videos.txt \
  > data/something_something/anchor_selected_videos_test4.txt
```

The local test run used:

```text
78687_anchor.mp4   # potato, vicks vaporub bottle
42326_anchor.mp4   # margarine, bread
34899_anchor.mp4   # bulb
112783_anchor.mp4  # mouthwash, roll on
```

### Run 32-Frame 3D Tracking

Run TrackCraft3r from the anchor frame for 32 frames at stride 1:

```bash
python scripts/run_action100m_trackcraft3r.py \
  --root data/something_something \
  --preproc-name anchor_preproc \
  --tracks-name anchor_tracks32 \
  --video-list data/something_something/anchor_selected_videos_test4.txt \
  --trackcraft-root external/TrackCraft3r \
  --da3-root external/depth-anything-3 \
  --process-res 336 \
  --chunk-size 24 \
  --num-frames 32 \
  --frame-stride 1 \
  --device cuda \
  --keep-going
```

This tracks from each anchor clip's frame 0 through frame 31. In original-video coordinates, that means `anchor_frame` through `anchor_frame + 31`.

### Build The 32-Frame Projected Viewer

Build the HTML viewer over the anchor-frame object masks:

```bash
python scripts/build_action100m_mask_trace_viewer.py \
  --root data/something_something \
  --sam-manifest sam3_anchor_masks/manifest.json \
  --tracks-name anchor_tracks32 \
  --viewer-name anchor_track32_viewer \
  --frame-stride 1 \
  --copy-json
```

Open:

```text
data/something_something/anchor_track32_viewer/index.html
```

The current viewer uses TrackCraft's sampled RGB frames and projects 3D points into each timestep's estimated camera frame. It also contrast-stretches point colors per clip from blue at the top of the sampled object mask points to red at the bottom.

### One-Command Test

The full 4-video Something-Something test scans 10 candidates by default, selects the first 4 usable anchor clips, runs 32-frame tracking, and rebuilds the viewer:

```bash
bash scripts/run_something_anchor_tracking_test.sh
```

Optional environment overrides:

```bash
SAM3_LIMIT=10 \
TRACK_LIMIT=4 \
NUM_FRAMES=32 \
FRAME_STRIDE=1 \
TRACKS_NAME=anchor_tracks32 \
VIEWER_NAME=anchor_track32_viewer \
bash scripts/run_something_anchor_tracking_test.sh
```

## Data Notes

- Action100M annotations are hierarchical. The viewer intentionally uses mid-level clips rather than every node, because many nodes are whole-video spans or sub-second micro-actions.
- Action100M YouTube availability changes over time. The exact set of downloadable videos may differ.
- Something-Something videos are local `.webm` files; the recipe writes short anchor-starting MP4s for TrackCraft compatibility.
- The local HTML viewers use relative paths, so they can be opened directly in a browser from the filesystem.
- Generated videos and model outputs remain local and are not committed.
