# Future 3D Scene Flow

This project explores re-purposing large video models for future 3D scene flow prediction: given an observed monocular video segment, predict how the visible 3D scene will move in the near future.

The working hypothesis is that video foundation models already encode useful priors about object permanence, contact, manipulation, articulation, and scene dynamics. Instead of training a future 3D tracker from scratch, we can build a pipeline that pairs short action-centric video clips with depth, camera, and dense 3D trajectory estimates, then use those signals to study and train future scene flow predictors.

## Repository Layout

```text
future-3d-scene-flow/
├── data/
│   ├── README.md
│   └── action100m/       # ignored local generated data
├── scripts/
│   ├── build_action100m_segment_viewer.py
│   ├── build_action100m_viewer.py
│   └── run_action100m_trackcraft3r.py
└── models/
    └── README.md
```

## Data Direction

The first data source is the Action100M preview split. Action100M provides YouTube video IDs plus a hierarchy of temporal action segments. The current local workflow:

1. Fetch Action100M preview annotations for selected videos.
2. Download the corresponding source videos from YouTube.
3. Cut mid-level action segments into short MP4 clips.
4. Build a local HTML viewer for browsing clips before running 3D tracking.

The local working copy produced so far lives under:

```text
data/action100m/
```

The segment viewer currently contains hundreds of short mid-level clips:

```text
data/action100m/segments/viewer/index.html
```

See [data/README.md](data/README.md) for reproduction details.

## Model Direction

The near-term model pipeline is:

1. Run depth and camera estimation on each short segment.
2. Run dense 3D point tracking / scene flow estimation.
3. Package observed-frame inputs and future-frame 3D motion targets.
4. Fine-tune or probe video models for future 3D scene flow prediction.

External model dependencies expected by the tracking runner:

- `../../external/TrackCraft3r`: dense 3D tracking from monocular video plus depth/camera.
- `../../external/depth-anything-3`: depth and camera preprocessing.
- `scripts/run_action100m_trackcraft3r.py`: initial runner for Action100M clips/videos.

The current machine session did not expose a working NVIDIA driver through `nvidia-smi`, so the download and HTML data curation steps are complete, while large-scale 3D tracking should be run once GPU access is available.
