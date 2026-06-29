# Full reproduction — pretrained-TrackCraft3r future 3D scene-flow

Master guide to reproduce **everything** from scratch on a new machine: data
generation, processing, training the best model, evaluation, the 2D + 3D
visualizers, the zero-shot OOD pipeline, and the MolmoMotion baseline.

Companion docs (read alongside): [README.md](README.md) (architecture),
[MODIFICATIONS.md](MODIFICATIONS.md) (what changed from TrackCraft3r + figures),
[REPRODUCE.md](REPRODUCE.md) (concise train/eval recipe).

Repo root used below: `cogsci/future-3d-scene-flow/`. `$PY` = the 3dflow env
python; `$PYM` = the molmomotion env python.

---

## 0. Final results

Best model: **fresh-LoRA (rank-32, self-attn) + TrackCraft3r-faithful decoded
loss @ 320×576, camera-motion subtracted, trained on all curated data
(629 dense + 1990 sparse).**

| Model | Data | Val future-ADE | Notes |
| --- | --- | --- | --- |
| **unified (best)** | 629 dense + 1990 sparse | **7.45 cm** (epoch 25) | camera-subtracted, mask-point supervision |
| dense-only (earlier) | 629 dense | 5.27 cm (epoch 13) | cam-0 frame, easier dense-only val |

> The unified val ADE is higher only because its 131-clip val pool includes the
> harder **sparse** clips (large motions, variable mask sizes); the dense-only
> number is on an easier pool. See [README.md](README.md).

**Best checkpoint:** `data/something_something/future_track_training/freshlora_unified_camsub_320x576/best.pt`
(epoch 25, ~140 MB — stores only the 12.19 M fine-tuned tensors; needs the base
TrackCraft3r checkpoint to run). Host it; see §2.

---

## 1. Environments & external repos

Three external repos (git submodules under `external/`):
`TrackCraft3r`, `sam3`, `depth-anything-3`.

### 1a. Main env (`3dflow`) — data gen, training, eval, viz
```bash
conda create -n 3dflow python=3.10 -y && conda activate 3dflow
pip install torch torchvision           # CUDA build for your driver
pip install peft safetensors opencv-python numpy pillow imageio[ffmpeg] matplotlib scipy viser
pip install -r models_pretrained/requirements.txt        # vendored diffsynth deps
pip install -e external/sam3                              # SAM3 (object/hand masks)
pip install -e external/depth-anything-3                  # DA3 (depth + camera)
# CoTracker is loaded via torch.hub (facebookresearch/co-tracker) on first use.
```
No `pip install` needed for the vendored `diffsynth`/`evaluation` — `models_pretrained/_bootstrap.py` shadows them so `external/TrackCraft3r` stays untouched.

### 1b. MolmoMotion baseline env (`molmomotion`)
```bash
conda create -n molmomotion python=3.11 -y && conda activate molmomotion
git clone https://github.com/allenai/molmo-motion.git ~/molmo-motion
cd ~/molmo-motion && pip install -e ".[viz]"
```

Tested on 2× RTX 4090 (24 GB).

---

## 2. Pretrained weights (NOT in git — `models_pretrained/checkpoints/`)
```
trackcraft3r/model.safetensors                     # 17 GB, HF: trackcraft3r/checkpoint
wan_models/Wan-AI/Wan2.1-T2V-1.3B/{diffusion_pytorch_model*.safetensors,
                                   models_t5_umt5-xxl-enc-bf16.pth, Wan2.1_VAE.pth}
```
```bash
huggingface-cli download trackcraft3r/checkpoint --local-dir models_pretrained/checkpoints/trackcraft3r
python external/TrackCraft3r/scripts/download_wan_1.3B.py --target models_pretrained/checkpoints/wan_models
```
DA3 (`depth-anything/DA3NESTED-GIANT-LARGE`), SAM3, MolmoMotion-4B and CoTracker3
download from their hubs on first use.

Hosted checkpoints (JHU SharePoint): the trained future-prediction checkpoints
(incl. `best.pt`) are at
`https://livejohnshopkins-my.sharepoint.com/:f:/r/personal/hbharad2_jh_edu/Documents/3d_tracks_data/checkpoint_based_on_trackcraft3r`
(browse + download; not a direct curl target).

---

## 3. Data generation from scratch (Something-Something → tracks)

Pipeline per video: **SAM3 anchor masks → DA3 depth/camera → TrackCraft3r
tracking**. Two output formats are produced (we train on both, unified):

| format | file | contents | used for |
| --- | --- | --- | --- |
| dense | `*_dense.npz` | `rgb (T,H,W,3)`, `recon_map`=Pj `(T,H,W,3)`, `track_map`=GT P0(t) `(T,H,W,3)` | full-image tracks; sample any points |
| sparse | `*_sparse.npz` | `rgb`, `tracks_xyz (T,M,3)`, `query_uv (M,2)`, `visibility (T,M)`, `extrinsics_w2c`, `fx_fy_cx_cy` | mask-point tracks only (disk-light) |
| both pair with | `*_user.npz` | `depth_map (T,h,w)`, `extrinsics_w2c (T,4,4)` (frame-0=I), `fx_fy_cx_cy` | geometry source |

T = 32 (`--num-frames 32`); object points come from SAM masks; cameras frame-0
normalized so cam-0 = world.

**Step 1 — SAM3 anchor masks.** Scan for first hand frame → anchor ≈ +10 frames →
segment hand (`--hand-prompt hand`) + objects (placeholder text prompts from the
SSv2 label) on the anchor frame. Writes `anchor_clips/`, `sam3_anchor_masks/clips/<id>/mask_*.png`,
and a manifest.
```bash
$PY scripts/run_something_sam3_anchor_masks.py --labels <ssv2_labels.json> \
    --root data/something_something --limit 0 --manifest-name manifest.json
$PY scripts/prepare_something_track_lists.py   # merge shard manifests -> tracking lists
```

**Step 2 — depth + camera (DA3).** DA3 gives z-depth + W2C extrinsics + intrinsics
in one pass (no ViPE needed); `build_user_npz.py` packs them into `*_user.npz`.
```bash
$PY external/TrackCraft3r/scripts/preprocess_da3.py --video_path <clip>.mp4 \
    --output_dir preproc/da3 --da3_root external/depth-anything-3
$PY external/TrackCraft3r/scripts/build_user_npz.py --video_path <clip>.mp4 \
    --depth_npy preproc/da3/depth.npy --extrinsics_npy preproc/da3/extrinsics.npy \
    --intrinsics_npy preproc/da3/intrinsics.npy --extrinsics_convention w2c \
    --depth_convention z --output_npz <clip>_user.npz
```
(`scripts/preprocess_da3_chunked.py` batches this for many clips.)

**Step 3 — TrackCraft3r tracking** (one model load over many `*_user.npz`):
```bash
# dense (full-image recon_map + track_map):
$PY scripts/run_trackcraft3r_dense_batch.py --video <ids...> \
    --checkpoint models_pretrained/checkpoints/trackcraft3r/model.safetensors \
    --num-frames 32 --out-dir data/something_something/anchor_tracks32_curated_dense
# sparse (mask-point tracks only, much smaller on disk):
$PY scripts/run_trackcraft3r_mask_sparse_batch.py ... --save-rgb \
    --out-dir data/something_something/anchor_tracks32_next2000_sparse
```

**The two curated sets actually used to train the released model:**
- `anchor_tracks32_curated_dense/` — **629** dense clips (`manifest_curated_dense.json`).
- `anchor_tracks32_next2000_sparse/` — **1990** sparse clips (`manifest_next2000_sparse.json`).

All data lives under `data/something_something/` (gitignored).

---

## 4. Train the best model (unified, camera-subtracted, 2-GPU DDP)

How the data feeds the model (see [MODIFICATIONS.md](MODIFICATIONS.md)):
- both formats are unified to **query-point supervision** (dense: 256 points
  sampled from the SAM mask; sparse: its native mask points);
- the geometry input Pj is built from depth at load time;
- everything (Pj, anchor, GT) is expressed in the **last-observed-frame camera**
  (`subtract_camera_motion`) so predictions have ego-motion removed;
- only a fresh rank-32 LoRA (self-attn) + I/O/head + mask latents train (12.19 M);
  the base DiT, the release rank-1024 LoRA, the VAEs and T5 stay frozen;
- loss = TrackCraft3r-faithful **decoded** MSE×10 at the query points; obs 10 →
  predict 22 future (total 32).

**2-GPU DDP command (the released run):**
```bash
NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 \
$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29521 \
  scripts/train_future_scene_flow_freshlora_unified.py \
  --output-dir data/something_something/future_track_training/freshlora_unified_camsub_320x576 \
  --image-size 320 576 --num-points 256 \
  --batch-size 1 --grad-accum 8 --grad-checkpoint \
  --epochs 30 --steps-per-epoch 300 --val-steps 24 --samples-per-clip 1 \
  --num-workers 4 --amp --save-every 1
```
- Single-GPU: drop `torch.distributed.run`/`--nproc_per_node` and set one GPU.
- **Auto-resumes** from `last.pt`; checkpoint every epoch (`best.pt` tracks val
  future-ADE, `epoch_NNN.pt` snapshots). ~2 h/epoch/GPU at 320×576.
- Best was **epoch 25 (7.45 cm)**; it overfits slightly after, so use `best.pt`.

Variants: `train_future_scene_flow.py` (dense-only, latent or decoded loss,
attn/all-LoRA), `train_future_scene_flow_freshlora.py` (single-set fresh-LoRA),
`train_future_scene_flow_freshlora_sparse.py` (sparse-only).

---

## 5. Evaluate + 2D HTML viewer
```bash
$PY scripts/comparison/render_unified_viewer.py \
  --checkpoint .../freshlora_unified_camsub_320x576/best.pt \
  --output-dir .../viewer_best --train-count 20 --val-count 44 --num-points 64
# dense grid of mask points (flow field): add --dense-only --grid-stride 6
```
Writes `index.html` + `.webm` trail videos (GT vs camera-subtracted prediction,
projected into the last-observed camera) + per-split ADE/FDE. Serve over HTTP for
playback: `cd <dir> && $PY -m http.server 8009`.

Training curves: `$PY scripts/plot_training_curves.py --logs <train>.log --out curves.png`.

---

## 6. 3D viser viewer (point cloud + animated tracks)
```bash
$PY scripts/comparison/viser_track_viewer.py \
  --checkpoint .../best.pt --video-id 106182_anchor --grid-stride 2 --port 8129
# open http://localhost:8129
```
Observed RGB point cloud + predicted 3D tracks, **time-animated** (play = video),
tracks start after the last anchor frame, rainbow spatial coloring, density via a
spacing slider. Up = +Y.

---

## 7. Zero-shot OOD on arbitrary videos

Put `.mp4`s in `zero-shot-eval/`. Pipeline: uniform-sample 10 frames → DA3
depth/camera → SAM3 object mask → model → viz. **Same processing as training.**
```bash
# 1. video -> 10 frames -> DA3 -> _user.npz (rgb+depth+camera)
$PY scripts/comparison/preprocess_ood_videos.py \
   --video-dir zero-shot-eval --out-dir zero-shot-eval/processed --num-frames 10 --height 480 --width 832
# 2. SAM3 mask of the hand-interacted object (one text prompt per clip)
$PY scripts/comparison/sam3_ood_masks.py --proc-dir zero-shot-eval/processed \
   --out-dir zero-shot-eval/masks --prompt <id>:"cup" --prompt <id>:"dumbbell" ...
# 3a. static 2D HTML for all clips
$PY scripts/comparison/ood_render_html.py --checkpoint .../best.pt \
   --proc-dir zero-shot-eval/processed --mask-dir zero-shot-eval/masks --out-dir zero-shot-eval/viewer
# 3b. interactive 3D viser (all clips, dropdown)
$PY scripts/comparison/ood_viser_viewer.py --checkpoint .../best.pt \
   --proc-dir zero-shot-eval/processed --mask-dir zero-shot-eval/masks --port 8130
```
`ood_common.py` holds the shared OOD inference (build Pj → run model → tracks on
the masked object, camera-subtracted frame).

---

## 8. MolmoMotion-4B baseline (comparison)

MolmoMotion needs each query point's **3D history**; we provide it from GT tracks
(val) or CoTracker+depth (OOD). It also gets a language action caption. **This is
not a controlled match** — MolmoMotion is fed explicit 3D tracks + language; ours
sees only RGB+depth frames. Run the model parts in the `molmomotion` env.

**Val set** (3-way GT | Ours | MolmoMotion):
```bash
$PY  scripts/comparison/prepare_comparison.py --checkpoint .../best.pt --output-dir .../cmp/prep --val-count 20 --num-points 8
$PYM scripts/comparison/run_molmomotion.py --prep-dir .../cmp/prep --out-dir .../cmp/molmo
$PY  scripts/comparison/render_comparison_viewer.py --prep-dir .../cmp/prep --molmo-dir .../cmp/molmo --output-dir .../cmp/viewer
```
Result (20 val clips, 8 pts): ours mean ADE 5.79 cm vs MolmoMotion 7.55 cm — but
MolmoMotion wins the median/per-clip; its mean is inflated by a few tail failures.

**OOD** (Ours vs MolmoMotion, side by side):
```bash
$PY  scripts/comparison/prep_molmo_ood.py --proc-dir zero-shot-eval/processed --mask-dir zero-shot-eval/masks \
     --out-dir zero-shot-eval/molmo/prep --num-points 48 --prompt <id>:"picking up a cup" ...
$PYM scripts/comparison/run_molmo_ood.py --prep-dir zero-shot-eval/molmo/prep --out-dir zero-shot-eval/molmo/results
$PY  scripts/comparison/ood_compare_html.py --checkpoint .../best.pt --out-dir zero-shot-eval/compare
```
(`prep_molmo_ood.py` uses CoTracker3 to track the SAM-mask points over the
observed frames and lifts them to 3D with the DA3 depth.)

---

## 9. Code map

Model / data (`models_pretrained/future_scene_flow/`):
- `model.py` — `FutureSceneFlowModel` (mask-latent future design, grad-ckpt decode).
- `model_fresh_lora.py` — fresh rank-32 LoRA stacked on the frozen rank-1024 adapter.
- `dataset.py` / `sparse_dataset.py` / `unified_dataset.py` — dense / sparse / combined loaders (+ camera subtraction, intrinsics, grid sampling).
- `losses.py` — `decoded_loss` (TC3r-faithful), `sparse_decoded_loss`, `sparse_metrics`, `latent_loss`.

Scripts (`scripts/`):
- data gen: `run_something_sam3_anchor_masks.py`, `preprocess_da3_chunked.py`, `run_trackcraft3r_dense_batch.py`, `run_trackcraft3r_mask_sparse_batch.py`, `prepare_something_track_lists.py`.
- train: `train_future_scene_flow*.py` (base / freshlora / _sparse / _unified-DDP).
- eval/viz: `comparison/render_unified_viewer.py`, `comparison/viser_track_viewer.py`, `plot_training_curves.py`.
- OOD: `comparison/{preprocess_ood_videos,sam3_ood_masks,ood_common,ood_render_html,ood_viser_viewer}.py`.
- MolmoMotion: `comparison/{prepare_comparison,run_molmomotion,render_comparison_viewer,prep_molmo_ood,run_molmo_ood,ood_compare_html}.py`.

Gitignored (regenerate or download): `data/`, `models_pretrained/checkpoints/`, `zero-shot-eval/processed|masks|viewer|compare|molmo`.
