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
│   ├── build_action100m_mask_trace_viewer.py
│   ├── prepare_action100m_track_lists.py
│   ├── run_action100m_sam3_first_frame_masks.py
│   └── run_action100m_trackcraft3r.py
├── viewer/
│   └── action100m_projected_tracks_template.html
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

## SAM3 And 3D Tracking

The near-term local pipeline is:

1. Run SAM3 on the first frame of each Action100M segment, using the segment text as the text prompt.
2. Select clips with a small number of SAM3 masks, typically 1-3 object masks.
3. Run Depth Anything 3 and TrackCraft3r on the selected segment clips.
4. Build a projected 2D HTML viewer from the dense 3D tracks.
5. Package observed-frame inputs and future-frame 3D motion targets for future scene flow experiments.

This repo assumes the model code is installed separately. The scripts expect these local checkouts by default:

- `../../external/sam3`: SAM3 image model package/checkpoint assets.
- `../../external/TrackCraft3r`: dense 3D tracking from monocular video plus depth/camera.
- `../../external/depth-anything-3`: depth and camera preprocessing.

Key repo scripts:

- `scripts/run_action100m_sam3_first_frame_masks.py`: runs SAM3 text-prompt masking on segment first frames.
- `scripts/prepare_action100m_track_lists.py`: creates resumable TrackCraft video-list shards from the SAM3 manifest.
- `scripts/run_action100m_trackcraft3r.py`: runs DA3 preprocessing, TrackCraft user NPZ creation, and dense tracking.
- `scripts/build_action100m_mask_trace_viewer.py`: builds the projected HTML track viewer.
- `viewer/action100m_projected_tracks_template.html`: tracked HTML template used by the projected viewer.

See [data/README.md](data/README.md) for exact commands.
