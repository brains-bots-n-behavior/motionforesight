#!/usr/bin/env bash
# Source this file from the repo root before running pretrained TrackCraft3r
# future-scene-flow inference/eval scripts.

export F3DSF_SCRATCH=/scratch/hbharad2/users/hbharad2/future-3d-scene-flow
export F3DSF_ENV="$F3DSF_SCRATCH/conda_envs/f3dsf-viser"
export PY="$F3DSF_ENV/bin/python"

export HF_HOME="$F3DSF_SCRATCH/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$F3DSF_SCRATCH/xdg_cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PIP_CACHE_DIR="$F3DSF_SCRATCH/pip_cache"

export F3DSF_CK="$F3DSF_SCRATCH/checkpoints/models_pretrained"
export F3DSF_CKPT="$F3DSF_SCRATCH/checkpoints/checkpoints_future/trainedon_ss100k/best.pt"
export F3DSF_BASE="$F3DSF_CK/trackcraft3r/model.safetensors"
export MODELSCOPE_CACHE="$F3DSF_CK/wan_models"

export MODELSCOPE_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "$MPLCONFIGDIR"

cat <<EOF
Future 3D scene flow env configured.
PY=$PY
F3DSF_CKPT=$F3DSF_CKPT
F3DSF_BASE=$F3DSF_BASE

OOD RGB video preprocessing:
  CUDA_VISIBLE_DEVICES=0 \$PY scripts/comparison/preprocess_ood_videos.py \\
    --video-dir "\$VID" --out-dir "\$VID/processed22" \\
    --num-frames 22 --frame-stride 1 --height 480 --width 832

Prediction-only HTML after masks exist:
  CUDA_VISIBLE_DEVICES=0 \$PY scripts/comparison/ood_render_html.py \\
    --checkpoint "\$F3DSF_CKPT" --proc-dir "\$VID/processed22" \\
    --mask-dir "\$VID/masks" --out-dir "\$VID/viewer_pred"

GT-vs-pred unified viewer after dense tracks and ood_manifest.json exist:
  CUDA_VISIBLE_DEVICES=0 \$PY scripts/comparison/render_unified_viewer.py \\
    --checkpoint "\$F3DSF_CKPT" --base-checkpoint "\$F3DSF_BASE" \\
    --root "\$VID" --dense-tracks-name processed22 --dense-manifest ood_manifest.json \\
    --sparse-tracks-name "" --output-dir "\$VID/viewer" \\
    --train-count 999 --val-count 999 --num-points 64 --dense-only
EOF
