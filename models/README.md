# Models

This folder holds model definitions, training configs, and evaluation notes for future 3D scene flow prediction.

## Current Baseline

`future_3d_tracks/` is the first repo-local baseline for future 3D point-track prediction. It uses TrackCraft3r dense outputs as pseudo-ground-truth supervision, but it does not feed the model the full video.

Input:

- RGB frames 0-9.
- Observed 3D point positions at frames 0-9.
- Frame-0 UV coordinates for sampled SAM3 object-mask points.
- A hashed text vector from the Something-Something label.

Target:

- Future 3D point positions for frames 10-31.

Train:

```bash
python scripts/train_future_3d_tracks.py \
  --root data/something_something \
  --tracks-name anchor_tracks32_500 \
  --manifest sam3_anchor_masks/manifest_500.json \
  --obs-frames 10 \
  --total-frames 32 \
  --device cuda \
  --amp
```

Training outputs are written under `data/something_something/future_track_training/`, which is ignored by git.
