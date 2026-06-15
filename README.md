# Future 3D Scene Flow

This project explores re-purposing large video models for future 3D scene flow prediction: given an observed monocular video segment, predict how the visible 3D scene will move in the near future.

The working hypothesis is that video foundation models already encode useful priors about object permanence, contact, manipulation, articulation, and scene dynamics. Instead of training a future 3D tracker from scratch, we can build a pipeline that pairs short action-centric video clips with depth, camera, and dense 3D trajectory estimates, then use those signals to study and train future scene flow predictors.

## Problem Statement

Given an image or a short video, plus a text instruction such as "knock down the cup", predict future 3D tracks describing how points on the relevant object should evolve over time. The goal is a general model that can produce these future 3D object tracks from arbitrary human videos.

## TODO

Done by Homanga:

- [x] Segment Action100M clips into action segments.
- [x] Generate SAM3 masks of the object in the 10th frame after the first frame where a hand is detected.
- [x] Run 3D tracking on these clips.
- [x] Run this pipeline for 200 clips. The demo video is embedded below. A 20-clip sample is available here: [Something-Something 20-clip track viewer package](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/hbharad2_jh_edu/IgCk0uiG1nuHSZ459dxUs_FsAXdK5Q2Y6GD6ulX_oY1FZYo?email=yjangir1%40jh.edu&e=wptieq). View it with the included HTML viewer.

TODO for Yash:

- [ ] Data curation: scale the steps set up by Homanga to all Something-Something videos.
- [ ] Model wiring: test the future track prediction model on the 20 clips above for debugging.
- [ ] Visualization: the target is 3D prediction, not just 2D tracks, so use Viser to visualize predicted 3D tracks as well.

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
│   ├── prepare_something_track_lists.py
│   ├── run_action100m_sam3_first_frame_masks.py
│   ├── run_action100m_trackcraft3r.py
│   ├── run_something_sam3_anchor_masks.py
│   └── run_trackcraft3r_dense_batch.py
├── viewer/
│   └── action100m_projected_tracks_template.html
├── media/
│   └── 200videos_3dtracks.webm
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

## 200-Clip Demo Video

A WebM screencast of the 200-video 3D track viewer:

<video src="media/200videos_3dtracks.webm" controls width="100%"></video>

[Open the demo video directly](media/200videos_3dtracks.webm).

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
- `scripts/run_something_sam3_anchor_masks.py`: creates Something-Something hand-anchored clips and object masks from `train.json` placeholders.
- `scripts/prepare_something_track_lists.py`: merges sharded Something-Something SAM3 manifests and writes trackable 32-frame GPU lists.
- `scripts/run_trackcraft3r_dense_batch.py`: runs TrackCraft3r dense inference over many prepared user NPZs with one model load.
- `scripts/build_action100m_mask_trace_viewer.py`: builds the projected HTML track viewer.
- `viewer/action100m_projected_tracks_template.html`: tracked HTML template used by the projected viewer.

See [data/README.md](data/README.md) for exact commands.
