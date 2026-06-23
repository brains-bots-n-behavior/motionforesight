# Reproducing the pretrained-TrackCraft3r future 3D scene-flow model

End-to-end guide to **train**, **evaluate**, **visualize**, and **predict** with the
future-track model that reuses the frozen pretrained TrackCraft3r (Wan2.1 DiT +
LoRA + dual VAEs). See [README.md](README.md) for the architecture/design; this
file is the reproduction recipe.

All commands run from the repo root:

```bash
cd <repo>/cogsci/future-3d-scene-flow      # or wherever this repo is checked out
```

`<PY>` below is the project Python (a conda env with torch + peft + the vendored
diffsynth deps). On the dev machine that is:

```bash
PY=/home/homanga/miniconda3/envs/3dflow/bin/python
```

---

## 0. Results (what this recipe produces)

Future 3D point-track prediction from the first **10 observed frames** → frames
**10–31**. Metric is future-frame **ADE/FDE** in cam-0 meters (lower is better).

| Variant | Res | Loss | Trainable | Best val ADE |
| --- | --- | --- | --- | --- |
| fresh-LoRA | 320×576 | **decoded** (TC3r-faithful) | 12.2 M | **5.27 cm** (epoch 13) |
| fresh-LoRA | 160×288 | latent | 12.2 M | 5.31 cm (epoch 3) |
| attn-LoRA | 160×288 | latent | 377.9 M | 5.77 cm (epoch 2) |

Takeaway: a **small fresh rank-32 LoRA** (stacked on the frozen rank-1024 release
adapter) + I/O/head + future mask latents matches or beats a much larger
adapter, and 320×576 ≈ 160×288 on this 629-clip set (resolution gave little
gain here). The **decoded** loss reproduces TrackCraft3r's actual training
objective (VAE-decode → coordinate-space MSE ×10).

---

## 1. Environment

```bash
conda create -n 3dflow python=3.10 -y && conda activate 3dflow
pip install torch torchvision                     # CUDA build matching your driver
pip install peft safetensors opencv-python numpy pillow imageio[ffmpeg]
pip install -r models_pretrained/requirements.txt # vendored diffsynth deps (einops, etc.)
```

No `pip install -e` is needed: `models_pretrained/_bootstrap.py` puts the
**vendored** `diffsynth`/`evaluation` copies first on `sys.path` (and removes the
editable-install finder), so the original `external/TrackCraft3r` is never used
or modified. Every script does `import models_pretrained` before importing
`diffsynth`, which activates this isolation.

Tested on a single **NVIDIA RTX 4090 (24 GB)**.

---

## 2. Pretrained weights (NOT in git — download to `models_pretrained/checkpoints/`)

The two frozen base models are gitignored (34 GB). Place them as:

```
models_pretrained/checkpoints/
  trackcraft3r/model.safetensors                              # 17 GB, TrackCraft3r release
  wan_models/Wan-AI/Wan2.1-T2V-1.3B/
    diffusion_pytorch_model*.safetensors                      # Wan2.1-T2V-1.3B base
    models_t5_umt5-xxl-enc-bf16.pth
    Wan2.1_VAE.pth
```

Sources:

```bash
# TrackCraft3r stage-2 release checkpoint
huggingface-cli download trackcraft3r/checkpoint \
  --local-dir models_pretrained/checkpoints/trackcraft3r

# Wan2.1-T2V-1.3B base (DiT + T5 + VAE). Either the HF hub, or TrackCraft3r's helper:
python external/TrackCraft3r/scripts/download_wan_1.3B.py \
  --target models_pretrained/checkpoints/wan_models
```

> Mirror: both base files are also hosted at `<WEB_FOLDER_URL>/` — drop them at
> the paths above. (Fill in the web-folder URL.)

### Trained future-prediction checkpoint

The trained checkpoint stores **only the fine-tuned tensors** (~146 MB: the fresh
LoRA adapter + I/O/head + mask latents), so it is small and is hosted in the web
folder rather than git:

```bash
# download the trained future-prediction checkpoint
mkdir -p checkpoints_future
curl -L -o checkpoints_future/freshlora_decoded_320x576_best_epoch13.pt \
  <WEB_FOLDER_URL>/freshlora_decoded_320x576_best_epoch13.pt
```

Loading it reconstructs the model from its embedded config and applies the
fine-tuned tensors on top of the frozen base checkpoint (above) — see §4/§5.

---

## 3. Data

Training/eval read TrackCraft3r dense outputs under `data/something_something/`
(gitignored). The generation recipe (SAM3 hand/object masks → anchor frame →
TrackCraft3r 32-frame dense tracking) is in [../CODEX.md](../CODEX.md). Layout:

```
data/something_something/
  anchor_tracks32_curated_dense/
    <clip>_dense.npz     # rgb (T,H,W,3 uint8) | recon_map (=Pj, T,H,W,3) | track_map (GT P0(t), T,H,W,3)
    <clip>_user.npz      # depth_map | extrinsics_w2c | fx_fy_cx_cy   (only needed for the viewer)
  sam3_anchor_masks/
    manifest_curated_dense.json
    clips/<clip>/mask_*.png
```

The model consumes observed `rgb`/`recon_map` (frames 0–9) and supervises the
full-clip `track_map`; the viewer additionally uses `_user.npz` camera/intrinsics
to project tracks. The curated set is 629 clips.

---

## 4. Train

### Best run — fresh-LoRA, decoded loss, 320×576 (reproduces the 5.27 cm result)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
$PY scripts/train_future_scene_flow_freshlora.py \
  --root data/something_something \
  --tracks-name anchor_tracks32_curated_dense \
  --manifest sam3_anchor_masks/manifest_curated_dense.json \
  --output-dir data/something_something/future_track_training/freshlora_decoded_320x576_10f_to_32f \
  --obs-frames 10 --total-frames 32 --image-size 320 576 \
  --fresh-lora-rank 32 --fresh-lora-scope attn --extra-trainable io head \
  --loss-space decoded --loss-scale 10.0 \
  --batch-size 1 --grad-accum 8 --grad-checkpoint \
  --epochs 15 --steps-per-epoch 400 --val-steps 24 --samples-per-clip 2 \
  --num-workers 4 --amp --save-every 2
```

Outputs to `--output-dir`: `last.pt` (every epoch), `best.pt` (on val-ADE
improvement), `epoch_NNN.pt` (every `--save-every`), and `config.json`. Each
checkpoint holds only the trainable tensors. ~2.8 h/epoch at 320×576 on a 4090;
the val curve plateaus by ~epoch 5, best was epoch 13.

### Faster baseline — fresh-LoRA, decoded loss, 160×288

Same command with `--image-size 160 288` (and you can raise `--steps-per-epoch
600 --val-steps 32`). ~3–4× faster per step.

### Other variants (alternate trainer `train_future_scene_flow.py`)

- **latent loss** (cheap proxy; default): drop `--loss-space decoded`.
- **attn-LoRA** (train the release rank-1024 LoRA's self-attn subset, ~378 M):
  `$PY scripts/train_future_scene_flow.py ... --trainable mask io head lora --lora-scope attn`
- **I/O+head only** (0.40 M params): `--trainable mask io head`.

Key flags: `--loss-space {latent,decoded}`, `--fresh-lora-rank`,
`--fresh-lora-scope {attn,all}`, `--image-size H W` (multiples of 16),
`--grad-checkpoint` (recommended), `--obs-frames/--total-frames`.

---

## 5. Evaluate + visualize (HTML viewer)

Renders GT-vs-prediction future-track trail videos, computes ADE/FDE, writes an
`index.html`. Auto-detects the checkpoint variant (fresh vs attn LoRA).

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
$PY scripts/render_future_scene_flow_viewer.py \
  --checkpoint checkpoints_future/freshlora_decoded_320x576_best_epoch13.pt \
  --base-checkpoint models_pretrained/checkpoints/trackcraft3r/model.safetensors \
  --output-dir data/something_something/future_scene_flow_viewer \
  --train-count 10 --val-count 0 --num-points 48      # val-count 0 = all val clips
```

Outputs `index.html` + `manifest.json` + `videos/*.webm` to `--output-dir`. The
header shows aggregate train/val ADE/FDE; cards are sorted by ADE.

**Viewing:** browsers often block local `.webm` over `file://`, so serve over HTTP:

```bash
cd data/something_something/future_scene_flow_viewer && $PY -m http.server 8009
# open http://localhost:8009/index.html  (VS Code will offer to forward the port)
```

The checkpoint is copied to a temp file before loading, so the viewer is safe to
run against a checkpoint dir that a training job is still writing.

---

## 6. Predict on a single clip (viewer-compatible NPZ)

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/predict_future_scene_flow.py \
  --train-ckpt checkpoints_future/freshlora_decoded_320x576_best_epoch13.pt \
  --base-checkpoint models_pretrained/checkpoints/trackcraft3r/model.safetensors \
  --dense-npz data/something_something/anchor_tracks32_curated_dense/100368_anchor_dense.npz \
  --output-npz /tmp/100368_future_pred.npz
```

Writes `track_map (T,H,W,3)` (10 observed + 22 predicted future frames),
`recon_map`, and `rgb` — the schema TrackCraft3r's `visualize_dense.py` reads.

---

## 7. Memory / resolution reference (single 24 GB GPU)

Measured peaks for fresh-LoRA training (decoded loss, bs 1, grad-checkpoint,
`expandable_segments:True`):

| Resolution | ~tokens | decoded loss | peak | fits? |
| --- | --- | --- | --- | --- |
| 160×288 | 12k | ✅ | ~12–18 GB | yes |
| 224×384 | 21k | ✅ | 13.4 GB | yes |
| **320×576** | 46k | ✅ | **21.3 GB** | **yes (max for decoded)** |
| 384×672 | 64k | ❌ | 23.8 GB | OOM |
| 480×832 | 100k | ❌ | — | OOM (latent+CPU-offload fits at 23.6 GB but ~82 s/step — impractical) |

Native 480×832 (TrackCraft3r's resolution) needs an 80 GB+ GPU for 32-frame
training. The bottleneck is one DiT block's self-attention over the stacked
(diagonal+row = 2T) latent-frame sequence.

---

## 8. What is / isn't trained

Trained (`fresh-LoRA` variant, 12.19 M): a fresh rank-32 LoRA on self-attn
q,k,v,o (11.8 M) + `patch_embedding` + `head` (0.40 M) + two future mask latents.
Frozen: the entire base Wan2.1 DiT, the release rank-1024 LoRA, all three VAEs,
the T5 text encoder. The decoded loss back-props **through** the frozen VAE
decoder but does not update it. See [README.md](README.md) for details.
