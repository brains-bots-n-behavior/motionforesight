# Future 3D Track Prediction Run Notes

This note records the current data recipe and the exact video-conditioned
training command for the curated Something-Something TrackCraft3r data.

## Working Directory

Run commands from the repository root:

```bash
cd /home/homanga/opentouch/cogsci/future-3d-scene-flow
```

## Data Locations

Curated dense TrackCraft3r outputs used for the current training run:

```bash
data/something_something/anchor_tracks32_500
data/something_something/anchor_tracks32_next1000
```

Merged symlink dataset used by training:

```bash
data/something_something/anchor_tracks32_curated_dense
data/something_something/sam3_anchor_masks/manifest_curated_dense.json
```

The merged dataset currently indexes 629 dense clips:

```text
201 clips from anchor_tracks32_500
428 clips from anchor_tracks32_next1000
```

Training output:

```bash
data/something_something/future_track_training/video_only_curated_dense_10f_to_32f
```

Training log:

```bash
data/something_something/logs/train_video_only_curated_gpu1.log
```

## Recipe Summary

1. Run SAM3 on Something-Something videos using hand/object prompts from the
   Something-Something labels.
2. Find the first frame where a hand is detected.
3. Choose the anchor frame roughly 10 frames after that first hand frame.
4. Segment the hand/object masks on the anchor frame.
5. Run TrackCraft3r for 32 frames starting from the anchor frame.
6. Save dense 3D tracks as `*_dense.npz` and selected/user tracks as
   `*_user.npz`.
7. Merge curated dense folders into `anchor_tracks32_curated_dense` using
   symlinks so the large track files are not copied.
8. Train the video-conditioned future track model using only the first 10 RGB
   frames and observed 3D tracks.
9. Predict future 3D point tracks for frames 10-31.

The current training run is video-only: `--text-dim 0`, so no language
conditioning is used.

## Training Command

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
/home/homanga/miniconda3/envs/3dflow/bin/python scripts/train_future_3d_tracks.py \
  --model-variant baseline \
  --root data/something_something \
  --tracks-name anchor_tracks32_curated_dense \
  --manifest sam3_anchor_masks/manifest_curated_dense.json \
  --output-dir data/something_something/future_track_training/video_only_curated_dense_10f_to_32f \
  --obs-frames 10 \
  --total-frames 32 \
  --num-points 256 \
  --image-size 128 224 \
  --embed-dim 256 \
  --num-heads 4 \
  --num-layers 3 \
  --text-dim 0 \
  --batch-size 2 \
  --epochs 20 \
  --steps-per-epoch 1132 \
  --val-steps 32 \
  --samples-per-clip 4 \
  --num-workers 4 \
  --device cuda \
  --amp \
  --save-every 5
```

## Detached Launch

The current detached launcher lives here:

```bash
data/something_something/logs/train_video_only_curated_gpu1_wrapper.sh
```

Launch with:

```bash
setsid -f data/something_something/logs/train_video_only_curated_gpu1_wrapper.sh \
  > data/something_something/logs/train_video_only_curated_gpu1.log 2>&1 < /dev/null
```

Monitor progress with:

```bash
tail -f data/something_something/logs/train_video_only_curated_gpu1.log
```

Check the main process:

```bash
ps -p "$(cat data/something_something/logs/train_video_only_curated_gpu1.pid)" \
  -o pid=,etime=,pcpu=,pmem=,cmd=
```

Check GPU usage:

```bash
nvidia-smi
```

## Alternate Architecture: Pretrained TrackCraft3r (`models_pretrained`)

The recipe above trains a from-scratch model (`models/future_3d_tracks`). There
is now an alternate architecture that **reuses the pretrained TrackCraft3r model**
instead of training from random init. It lives in `models_pretrained/` (a
self-contained vendored copy of TrackCraft3r + the pretrained weights; the
original `external/TrackCraft3r` submodule is untouched).

Idea: TrackCraft3r maps a fully observed clip to dense 3D tracks via its Wan2.1
DiT. For future prediction we feed only the observed frames (`0..obs-1`) and
replace the unobserved future diagonal latents with two small **learnable mask
latents**; the frozen DiT extrapolates the future tracks. The supervision and
inputs come straight from the existing `*_dense.npz` files (`rgb`, `recon_map`
= Pj, `track_map` = GT), so no depth/camera recomputation is needed.

By default only the future mask latents + I/O projections + head are trained
(~0.40 M params, ~18 GB peak at 160×288, fits a single 24 GB GPU). See
`models_pretrained/README.md` for the full design and knobs.

Train:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
/home/homanga/miniconda3/envs/3dflow/bin/python scripts/train_future_scene_flow.py \
  --root data/something_something \
  --tracks-name anchor_tracks32_curated_dense \
  --manifest sam3_anchor_masks/manifest_curated_dense.json \
  --output-dir data/something_something/future_track_training/pretrained_tc3r_10f_to_32f \
  --obs-frames 10 --total-frames 32 \
  --image-size 224 384 \
  --trainable mask io head \
  --batch-size 1 --grad-accum 8 --grad-checkpoint \
  --epochs 20 --steps-per-epoch 600 --val-steps 32 --amp --save-every 5
```

Predict future tracks for a clip:

```bash
CUDA_VISIBLE_DEVICES=0 /home/homanga/miniconda3/envs/3dflow/bin/python \
  scripts/predict_future_scene_flow.py \
  --train-ckpt data/.../pretrained_tc3r_10f_to_32f/best.pt \
  --dense-npz data/something_something/anchor_tracks32_curated_dense/100368_anchor_dense.npz \
  --output-npz /tmp/100368_future_pred.npz
```

## Notes

The SAM3-only 5k scan is not part of this training set yet because it does not
have dense TrackCraft3r labels. It can be added after DA3/TrackCraft3r tracking
has been run and dense `*_dense.npz` files are available.
