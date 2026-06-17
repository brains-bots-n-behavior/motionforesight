# Future 3D Track Predictor

This package trains a repo-local future track predictor from the curated
Something-Something outputs.  TrackCraft3r is used to produce dense 3D tracks,
but the trainable model here does not see the full video and does not modify the
TrackCraft3r submodule.

For each clip, the dataset samples points inside the SAM3 object mask on the
anchor frame.  The model input is:

- RGB frames `0..9`.
- Observed 3D positions for the sampled points at frames `0..9`.
- Normalized frame-0 UV coordinates for those sampled points.
- A hashed text vector from the Something-Something label.

The target is the future 3D position sequence for frames `10..31`.

Default training command:

```bash
conda activate 3dflow

python scripts/train_future_3d_tracks.py \
  --root data/something_something \
  --tracks-name anchor_tracks32_500 \
  --manifest sam3_anchor_masks/manifest_500.json \
  --obs-frames 10 \
  --total-frames 32 \
  --num-points 256 \
  --batch-size 2 \
  --epochs 20 \
  --steps-per-epoch 200 \
  --device cuda
```

Outputs are written under `data/something_something/future_track_training/`,
which is ignored by git.

