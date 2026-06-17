# Models

This folder holds model definitions, training configs, and evaluation notes for future 3D scene flow prediction.

## Future 3D Track Models

`future_3d_tracks/` contains the repo-local models for future 3D point-track prediction. They use TrackCraft3r dense outputs as pseudo-ground-truth supervision, but they do not feed the predictor the full video.

Input:

- RGB frames 0-9.
- Observed 3D point positions at frames 0-9.
- Frame-0 UV coordinates for sampled SAM3 object-mask points.
- Text conditioning from the Something-Something label.

Target:

- Future 3D point positions for frames 10-31.

## Variants

`baseline`

- File: `models/future_3d_tracks/model.py`
- Text conditioning: hashed bag-of-words vector from the Something-Something label.
- Conditioning mechanism: adds video and text tokens to each point token before a point-token Transformer.
- Use this as the lightweight sanity baseline.

`text-adaln`

- File: `models/future_3d_tracks/text_adaln_model.py`
- Text conditioning: trainable token vocabulary plus a small Transformer text encoder.
- Conditioning mechanism: video/text condition vector drives adaLN-Zero shift, scale, and residual gates inside the point-token Transformer blocks.
- Use this when testing whether the natural-language action label helps future motion prediction.

## Training

Baseline:

```bash
python scripts/train_future_3d_tracks.py \
  --model-variant baseline \
  --root data/something_something \
  --tracks-name anchor_tracks32_500 \
  --manifest sam3_anchor_masks/manifest_500.json \
  --output-dir data/something_something/future_track_training/baseline_full_10f_to_32f \
  --obs-frames 10 \
  --total-frames 32 \
  --num-points 256 \
  --image-size 128 224 \
  --embed-dim 256 \
  --num-heads 4 \
  --num-layers 3 \
  --batch-size 2 \
  --epochs 20 \
  --steps-per-epoch 362 \
  --val-steps 10 \
  --samples-per-clip 4 \
  --num-workers 4 \
  --device cuda \
  --amp
```

Text-conditioned AdaLN:

```bash
python scripts/train_future_3d_tracks.py \
  --model-variant text-adaln \
  --root data/something_something \
  --tracks-name anchor_tracks32_500 \
  --manifest sam3_anchor_masks/manifest_500.json \
  --output-dir data/something_something/future_track_training/text_adaln_full_10f_to_32f \
  --obs-frames 10 \
  --total-frames 32 \
  --num-points 256 \
  --image-size 128 224 \
  --embed-dim 256 \
  --num-heads 4 \
  --num-layers 3 \
  --text-layers 2 \
  --batch-size 2 \
  --epochs 20 \
  --steps-per-epoch 362 \
  --val-steps 10 \
  --samples-per-clip 4 \
  --num-workers 4 \
  --device cuda \
  --amp
```

Training outputs are written under `data/something_something/future_track_training/`, which is ignored by git.

## Inference Viewer

Render side-by-side ground-truth and prediction overlays for a trained checkpoint:

```bash
python scripts/render_future_track_prediction_viewer.py \
  --checkpoint data/something_something/future_track_training/text_adaln_full_10f_to_32f/best.pt \
  --output-dir data/something_something/future_track_prediction_viewer_text_adaln \
  --train-count 20 \
  --val-count 0 \
  --device cuda \
  --force
```

The script reads the model variant from the checkpoint, so the same command works for both `baseline` and `text-adaln` checkpoints. The generated HTML shows the first 10 observed RGB frames, then future trajectories overlaid on the last observed frame, with ground truth and prediction videos side by side.

Viewer outputs are written under `data/something_something/`, which is also ignored by git.
