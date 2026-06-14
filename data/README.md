# Data

This folder documents how to reproduce the local Action100M preview video and segment-clip browser used for future 3D scene flow experiments.

Generated media is kept locally under this folder but ignored by git:

```text
data/action100m/
```

This keeps the repository light while making the local data path explicit and reproducible.

## Resulting Local Layout

After running the workflow, the data directory in the main workspace should look like:

```text
data/action100m/
├── annotations/              # Cached Action100M rows, one JSON per video UID
├── raw/                      # Downloaded YouTube source videos and metadata
├── segments/
│   ├── clips/                # Short mid-level segment MP4s
│   ├── segments_manifest.json
│   └── viewer/index.html     # Local segment clip grid
├── selected_videos.json      # Selected Action100M video IDs
├── tracks/                   # Reserved for 3D tracking outputs
└── viewer/index.html         # Local full-video grid
```

The current curated segment viewer is:

```text
data/action100m/segments/viewer/index.html
```

Current verified state:

- 509 segment clips in the viewer after excluding the PHD2/PHP-looking source video.
- 35 videos represented in the segment viewer.
- Segment clips are mid-level Action100M annotations, usually 5-18 seconds.

## Dependencies

Use any Python environment with the required packages:

```bash
python -m pip install yt-dlp datasets pyarrow huggingface_hub
```

`ffmpeg` must be available on `PATH`, or passed to the segment builder with `--ffmpeg`.

## Helper Scripts

The workflow uses helper scripts included in this repository:

```text
scripts/build_action100m_viewer.py
scripts/build_action100m_segment_viewer.py
scripts/run_action100m_trackcraft3r.py
```

## Reproducing the Annotation Cache

Action100M preview stores YouTube video IDs and hierarchical segment annotations in Hugging Face parquet files. The local annotations are cached as JSON so the viewer can be rebuilt without re-querying Hugging Face.

Example pattern used to fetch and cache selected rows:

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

Note: in this environment, the Hugging Face streaming reader sometimes crashed during Python shutdown after all files were written. The resulting JSON cache was still valid; verify counts after the command.

## Downloading Source Videos

Build a batch file from `selected_videos.json` for selected videos missing from `raw/`:

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

Download modest-resolution source MP4s:

```bash
cd /path/to/future-3d-scene-flow

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

Some Action100M YouTube IDs may be unavailable, private, or return unusably tiny files. The segment builder below skips missing files and MP4s smaller than 1 MB.

## Full-Video HTML UI

Create a grid UI for the downloaded full videos:

```bash
cd /path/to/future-3d-scene-flow

python \
  scripts/build_action100m_viewer.py \
  --root data/action100m
```

Open:

```text
data/action100m/viewer/index.html
```

## Segmented Clip HTML UI

Create mid-level segment clips and a segment-grid UI:

```bash
cd /path/to/future-3d-scene-flow

python \
  scripts/build_action100m_segment_viewer.py \
  --root data/action100m \
  --per-video 15 \
  --exclude-video-uid=-M6cLOV4aW8
```

Open:

```text
data/action100m/segments/viewer/index.html
```

The `--exclude-video-uid=-M6cLOV4aW8` flag removes the PHD2/PHP-looking software tutorial clips from the viewer.

The segment builder writes:

```text
data/action100m/segments/clips/*.mp4
data/action100m/segments/segments_manifest.json
data/action100m/segments/viewer/index.html
```

## Preparing for 3D Tracking

Once GPU access is available, run the TrackCraft3r pipeline from the project root:

```bash
cd /path/to/future-3d-scene-flow

python \
  scripts/run_action100m_trackcraft3r.py \
  --root data/action100m
```

This runner performs:

1. Depth/camera preprocessing using Depth Anything 3.
2. TrackCraft3r-format NPZ creation.
3. Dense 3D tracking inference.

Outputs are written to:

```text
data/action100m/preproc/
data/action100m/tracks/
```

The current machine session could not access the NVIDIA driver via `nvidia-smi`, so this step should be run in a GPU-visible environment.

## Data Notes

- Action100M annotations are hierarchical. The viewer intentionally uses mid-level clips rather than every node, because many nodes are whole-video spans or sub-second micro-actions.
- YouTube availability changes over time. The exact set of downloadable videos may differ.
- The local HTML viewers use relative paths, so they can be opened directly in a browser from the filesystem.
- Downloaded videos remain subject to the original source availability and licensing constraints.
