# `models_pretrained` — Future 3D scene flow on top of pretrained TrackCraft3r

This folder is a **self-contained snapshot** of TrackCraft3r plus an alternate
future-track-prediction architecture that *reuses the pretrained model* instead
of training from scratch. Nothing in `external/TrackCraft3r` is modified.

## Why this exists

The from-scratch model (`models/future_3d_tracks`, trained by
`scripts/train_future_3d_tracks.py`) learns future 3D tracks on ~629 curated
clips from random init. TrackCraft3r already solved the hard half of that
problem — mapping video + geometry to dense 3D tracks — so this architecture
keeps the entire pretrained TrackCraft3r pipeline and only learns the small
delta needed to *extrapolate into the future*.

## What is vendored here

```
models_pretrained/
  _bootstrap.py            import isolation (see below)
  diffsynth/               copy of TrackCraft3r's diffsynth package
  evaluation/              copy of TrackCraft3r's evaluation predictor + utils
  checkpoints/
    trackcraft3r/model.safetensors      copied release checkpoint (17 GB)
    wan_models/Wan-AI/Wan2.1-T2V-1.3B/  copied Wan2.1 base (DiT/VAE/T5, 17 GB)
  future_scene_flow/       the NEW architecture (model / dataset / losses)
```

**Import isolation.** The original submodule is `pip install -e .` (editable),
which registers a finder that redirects `import diffsynth` / `import evaluation`
back to `external/TrackCraft3r`. `import models_pretrained` runs
`_bootstrap.activate()`, which removes that finder and puts this folder first on
`sys.path`, so the *vendored* copies win. Always `import models_pretrained`
before importing `diffsynth`/`evaluation`.

## The architecture (minimal change to TrackCraft3r)

TrackCraft3r's `model_fn_wan_video` runs the Wan2.1 DiT over `T` latent frames
with two streams and predicts the "row" (the per-frame 3D track):

```
diagonal[t] = [ RGB_latent(t) | Pj_latent(t) ]   # appearance + geometry at frame t
row[t]      = [ RGB_latent(0) | Pj_latent(0) ]   # frame-0 anchor (the query)
track(t)    = DiT( concat_T(diagonal, row) )     # where the frame-0 point lands at t
```

For **future prediction** we observe only frames `0 .. obs-1`. We keep the whole
pretrained pipeline and make exactly one structural change:

> the diagonal entries for the unobserved future frames `obs .. T-1` are replaced
> by two small **learnable mask latents** (one for the RGB half, one for the Pj
> half), broadcast over space and over all future frames.

The frozen DiT then attends from the future query rows (still the frame-0 anchor)
to the observed diagonal entries + mask tokens and regresses the future tracks.
RoPE already gives each future frame a distinct temporal position, so one shared
mask latent per stream suffices. This leverages both halves of the prior:
TrackCraft3r's learned RGB+geometry→track mapping *and* the Wan video model's
temporal extrapolation.

See `future_scene_flow/model.py` for the implementation.

### Data reuse (no recomputation)

TrackCraft3r's dense outputs (`*_dense.npz`) already store exactly the tensors
the pipeline consumes and supervises against, so no depth/camera recomputation
is needed:

| key in `*_dense.npz` | role |
| --- | --- |
| `rgb` (T,H,W,3) | observed RGB → RGB latents (frames `0..obs-1`) |
| `recon_map` (T,H,W,3) | `Pj(t)`: depth back-projection in frame-0 cam space → Pj latents |
| `track_map` (T,H,W,3) | GT `P0(t)`: supervision target for all `T` frames |

Pj normalization (z-inlier percentile → mean-center → max-distance scale)
mirrors `WanSceneFlowPredictor.predict`, but is computed from the **observed
frames only** because future depth is unavailable at inference.

### Training objective

Latent-space regression of the predicted xyz query latent toward the VAE
encoding of the GT normalized residual track maps (the exact signal TrackCraft3r
itself regresses). This avoids back-prop through the frozen VAE decoder and
keeps a 1.3B DiT over 32 frames trainable on a single 24 GB GPU. Frames are
weighted (`obs_weight` for the observed reconstruction, `future_weight` for the
future). Eval ADE/FDE are computed in meters by decoding, reconstructing,
sampling at the object-mask points, and comparing to GT — matching the
from-scratch baseline.

### What is trained

Everything is frozen by default except the selected `--trainable` groups
(`mask` is always trained):

| group | params | notes |
| --- | --- | --- |
| `mask` | ~tiny | the two future mask latents (always on) |
| `io` | ~0.4 M | + `patch_embedding` and `head` (the I/O projections) |
| `head` | small | just the output head |
| `lora` | ~1 B | the rank-1024 LoRA — full adaptation, needs a big GPU |
| `vae` | large | unfreeze the VAE encoder/decoders |

Default `mask io head` = **0.40 M trainable params**, peak **~18 GB** at
160×288, bs=1, grad-checkpoint. Add `lora` for full adaptation on H100/H200
(8-bit AdamW recommended — rank-1024 LoRA is ~1 B params).

## Commands

Train (single 24 GB GPU):

```bash
cd /home/homanga/opentouch/cogsci/future-3d-scene-flow
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

Checkpoints store only the fine-tuned params (~5 MB), referencing the base
checkpoint by path.

Predict future tracks for a clip (writes a viewer-compatible NPZ):

```bash
CUDA_VISIBLE_DEVICES=0 /home/homanga/miniconda3/envs/3dflow/bin/python \
  scripts/predict_future_scene_flow.py \
  --train-ckpt data/.../pretrained_tc3r_10f_to_32f/best.pt \
  --dense-npz data/something_something/anchor_tracks32_curated_dense/100368_anchor_dense.npz \
  --output-npz /tmp/100368_future_pred.npz
```

## Notes / knobs

- **Resolution drives memory** (full TrackCraft3r res is 480×832; at 32 frames
  that is ~100k attention tokens and will OOM a 4090). Defaults use 224×384.
  Increase if you have an H100/H200.
- `--image-size` must be multiples of 16.
- The visibility head is kept (to load the checkpoint cleanly) and predicted but
  not supervised by default (no future GT visibility); pass `--no-vis` to drop it.
- To swap latent-space loss for decoded metric-space loss, supervise
  `model.decode_xyz(...)` against the GT in `train_future_scene_flow.py`; the
  decode path is already used for eval metrics.
