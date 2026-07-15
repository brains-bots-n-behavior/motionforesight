# OOD Full-Video Context-50 Eval/Viz Protocol

This protocol is for arbitrary full videos where the model should observe more
context before predicting. It differs from `--sample-mode uniform` over the full
video:

- sample 22 frames total from each full video;
- frames 0..6 are context frames sampled uniformly from the first 50% of the raw
  video;
- frames 7..21 are future frames sampled uniformly from the second 50%;
- only frames 0..6 are fed to the model;
- frames 7..21 are recorded as the future timeline and are not model input.

The preprocessor writes `frame_sampling_manifest.json` with
`context_frame_indices` and `future_frame_indices_not_input` for every video.

## Inputs

Put `.mp4` files in one directory and create a prompt TSV:

```text
video_stem<TAB>object prompt
20260712_163103<TAB>telephone handset
20260712_163122<TAB>coffee cup
```

Use specific object names. Avoid generic prompts such as `the object being
manipulated`; they often fail or mask the wrong object.

## One-Command Run

```bash
cd /home/hbharad2/future-3d-scene-flow

VIDEO_DIR=/path/to/full_videos \
OUT_ROOT=/path/to/eval_outputs \
PROMPTS_TSV=/path/to/object_prompts.tsv \
CUDA_VISIBLE_DEVICES=0 \
PC_STRIDE=1 \
bash scripts/comparison/run_oodfull_context50_eval.sh
```

Outputs:

- `$OUT_ROOT/processed22_context50/*_user.npz`
- `$OUT_ROOT/processed22_context50/frame_sampling_manifest.json`
- `$OUT_ROOT/masks_sam3_object_context50/*_mask.png`
- `$OUT_ROOT/masks_sam3_object_context50/*_overlay.png`
- `$OUT_ROOT/masks_sam3_object_context50/contact_sheet_prompts.jpg`
- `$OUT_ROOT/viewer_pred_context50/index.html`
- `$OUT_ROOT/3d_html_sam3_object_context50_dense_pc/*_context50_dense_pc_3d.html`

`PC_STRIDE=1` gives the densest RGB point cloud in the 3D HTML. The HTML files
are large, often about 100 MB per clip.

## Staged Runs

The runner supports resumable stages:

```bash
STAGE=preprocess VIDEO_DIR=... OUT_ROOT=... PROMPTS_TSV=... \
  bash scripts/comparison/run_oodfull_context50_eval.sh

STAGE=masks VIDEO_DIR=... OUT_ROOT=... PROMPTS_TSV=... \
  bash scripts/comparison/run_oodfull_context50_eval.sh

STAGE=viz VIDEO_DIR=... OUT_ROOT=... PROMPTS_TSV=... PC_STRIDE=1 \
  bash scripts/comparison/run_oodfull_context50_eval.sh
```

Inspect `contact_sheet_prompts.jpg` after the masks stage. If a mask is wrong,
edit the prompt TSV and rerun `STAGE=masks` before running `STAGE=viz`.

## Manual Equivalent

```bash
$PY scripts/comparison/preprocess_ood_videos.py \
  --video-dir "$VIDEO_DIR" \
  --out-dir "$OUT_ROOT/processed22_context50" \
  --num-frames 22 \
  --sample-mode context50 \
  --obs-frames 7 \
  --height 480 \
  --width 832

$PY scripts/comparison/sam3_ood_masks.py \
  --proc-dir "$OUT_ROOT/processed22_context50" \
  --out-dir "$OUT_ROOT/masks_sam3_object_context50" \
  --checkpoint "$SAM3_CHECKPOINT" \
  --prompt 20260712_163103:"telephone handset" \
  --prompt 20260712_163122:"coffee cup"

$PY scripts/comparison/build_ood_3d_html_one.py \
  --checkpoint "$F3DSF_CKPT" \
  --base-checkpoint "$F3DSF_BASE" \
  --user-dir "$OUT_ROOT/processed22_context50" \
  --mask-dir "$OUT_ROOT/masks_sam3_object_context50" \
  --out-dir "$OUT_ROOT/3d_html_sam3_object_context50_dense_pc" \
  --suffix "_context50_dense_pc_3d.html" \
  --grid-stride 2 \
  --pc-stride 1 \
  --max-tracks 220 \
  --rainbow-tracks
```
