# MotionForesight

![MotionForesight qualitative examples](media/motionforesight-qualitative-grid.gif)

MotionForesight predicts future 3D object motion from RGB video. Given a short video
context and an object mask, it forecasts 3D object tracks in the last-observed camera
frame and can export self-contained interactive 3D HTML visualizations.

## Repository Contents

```text
data/processed7_uniform/   example *_user.npz model inputs, 7 observed frames each
scripts/                   preprocessing, masking, model loading, and HTML inference scripts
models_pretrained/         minimal vendored runtime needed to load the model architecture
checkpoints/               ignored; place downloaded fine-tuned checkpoint files here
assets/                    ignored; recommended location for base model assets and caches
outputs/                   ignored; generated visualizations go here
```

The packaged `.npz` examples correspond to four videos used in the paper's OOD
visualization experiments. Raw videos are not tracked; the checked-in `.npz` files
are the actual model inputs.

## Model Assets

Download the fine-tuned MotionForesight checkpoint from the paper release assets
and place it here:

```text
checkpoints/trainedon_ss100k/best.pt
checkpoints/trainedon_ss100k/config.json
```

Checkpoint download link: [model](https://drive.google.com/file/d/1isAB_TnNW8gbMlgs5yVnPBKZSR85-JyZ/view?usp=sharing)

The fine-tuned checkpoint is trained on top of TrackCraft3r/Wan2.1, so inference
also needs the base TrackCraft3r checkpoint and Wan2.1 files. Put those assets
outside git, for example:

```text
assets/models_pretrained/trackcraft3r/model.safetensors
assets/models_pretrained/wan_models/Wan-AI/Wan2.1-T2V-1.3B/...
```

Then either use `scripts/setup_inference_env.sh` defaults or set:

```bash
export MOTIONFORESIGHT_CKPT=$PWD/checkpoints/trainedon_ss100k/best.pt
export MOTIONFORESIGHT_BASE=$PWD/assets/models_pretrained/trackcraft3r/model.safetensors
export MODELSCOPE_CACHE=$PWD/assets/models_pretrained/wan_models
```

SAM3 is only needed when generating masks for new videos. Place the SAM3 checkpoint
under `assets/` and pass its path to `scripts/comparison/sam3_ood_masks.py`.

## Installation

```bash
git clone https://github.com/brains-bots-n-behavior/motionforesight.git
cd motionforesight

git submodule update --init --recursive

python -m venv assets/venv
source assets/venv/bin/activate
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r models_pretrained/requirements.txt
pip install -e external/sam3
pip install -e external/depth-anything-3
pip install opencv-python pillow plotly peft safetensors huggingface_hub hf_transfer viser
```

Configure paths before running inference:

```bash
export MOTIONFORESIGHT_ENV=$PWD/assets/venv
source scripts/setup_inference_env.sh
```

## Run Inference On The Packaged `.npz` Examples

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/comparison/build_ood_3d_html_one.py \
  --checkpoint "$MOTIONFORESIGHT_CKPT" \
  --base-checkpoint "$MOTIONFORESIGHT_BASE" \
  --user-dir data/processed7_uniform \
  --out-dir outputs/viser_3d_html_sam3_broad_dense_pc \
  --suffix _sam3_broad_dense_pc_3d.html \
  --grid-stride 2 \
  --pc-stride 1 \
  --max-tracks 220 \
  --rainbow-tracks \
  --track-count-options 25,50,100,160,220
```

Open the generated HTML files in `outputs/viser_3d_html_sam3_broad_dense_pc/` in a
browser. Increase `--pc-stride` to `2` or `5` for smaller HTML files.

## Preprocess A New Video: Full-Video 7-Frame Context

Use this mode when the entire video should be used as context. The preprocessor
uniformly samples 7 observed frames from the full video.

```bash
VIDEO_DIR=/path/to/mp4s
OUT_ROOT=/path/to/output

CUDA_VISIBLE_DEVICES=0 $PY scripts/comparison/preprocess_ood_videos.py \
  --video-dir "$VIDEO_DIR" \
  --out-dir "$OUT_ROOT/processed7_uniform" \
  --num-frames 7 \
  --sample-mode uniform \
  --frame-stride 0 \
  --height 480 \
  --width 832
```

Each `<video_stem>_user.npz` contains:

```text
rgb                 sampled RGB frames, shape (7,H,W,3)
depth_map           DA3 z-depth for each sampled frame
extrinsics_w2c      camera poses normalized to frame 0
fx_fy_cx_cy         intrinsics at the processed resolution
frame_indices       raw video frame indices selected uniformly
images_jpeg_bytes   JPEG frames for compatibility with TrackCraft3r loaders
```

Generate object masks with SAM3. Use broad but concrete prompts such as `cabinet`,
`box`, `chair`, `trash can`, `pair of shoes`, `paper bag`, or `pans`.

```bash
SAM3_CHECKPOINT=$PWD/assets/sam3/sam3.pt

CUDA_VISIBLE_DEVICES=0 $PY scripts/comparison/sam3_ood_masks.py \
  --proc-dir "$OUT_ROOT/processed7_uniform" \
  --out-dir "$OUT_ROOT/masks_sam3_broad" \
  --checkpoint "$SAM3_CHECKPOINT" \
  --prompt video_stem:"box" \
  --prompt another_video:"cabinet"
```

Then run `build_ood_3d_html_one.py` with `--user-dir` and `--mask-dir` pointed at
the new processed clips and masks.

## Optional Full-Video Split Recipe

For long videos where you want explicit context and future segments, use the
provided runner:

- sample 22 frames total;
- frames 0..6 are context frames sampled uniformly from the first half of the video;
- frames 7..21 are future timeline frames sampled uniformly from the second half;
- only frames 0..6 are fed to the model.

```bash
VIDEO_DIR=/path/to/full_videos \
OUT_ROOT=/path/to/eval_outputs \
PROMPTS_TSV=/path/to/object_prompts.tsv \
CUDA_VISIBLE_DEVICES=0 \
PC_STRIDE=1 \
bash scripts/comparison/run_oodfull_context50_eval.sh
```

`PROMPTS_TSV` format:

```text
video_stem<TAB>object prompt
example_0001<TAB>oven
example_0002<TAB>pair of shoes
```

## Coordinate Frame

The predicted 3D tracks are expressed in the last observed camera frame. The HTML
viewer freezes the background at the last observed frame and draws future object
trajectories in that coordinate system.

## Notes

- Keep downloaded checkpoints, base weights, raw videos, and generated HTML files
  out of git. `.gitignore` is configured for this.
- The model predicts a dense field, but the released visualizations sample tracks
  only from the object mask. Off-object motion is not rendered as a prediction.

## BibTeX

If you find MotionForesight useful, please cite:

```bibtex
@article{bharadhwaj2026motionforesight,
  title={MotionForesight: Re-purposing Video Models for Future 3D Scene-Flow Prediction},
  author={Homanga Bharadhwaj and Yash Jangir},
  year={2026}
}
```

## Acknowledgements

MotionForesight builds on several open-source models and codebases, including
**Wan2.1**, **TrackCraft3r**, **SAM 3**, and **Depth Anything 3 (DA3)**. We thank
their authors and maintainers for making their work publicly available. Please
consult and cite the corresponding projects when using these components.
