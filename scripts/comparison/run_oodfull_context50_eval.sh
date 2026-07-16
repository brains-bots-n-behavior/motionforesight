#!/usr/bin/env bash
set -euo pipefail

# End-to-end OOD full-video eval/viz protocol.
#
# Required env vars:
#   VIDEO_DIR=/path/to/mp4s
#   OUT_ROOT=/path/to/output/root
#   PROMPTS_TSV=/path/to/prompts.tsv
#
# Optional:
#   STAGE=all|preprocess|masks|viz
#   PC_STRIDE=1
#   GRID_STRIDE=2
#   CUDA_VISIBLE_DEVICES=0
#
# prompts.tsv format:
#   <video_stem><TAB><object prompt>

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

source scripts/setup_inference_env.sh >/dev/null

: "${VIDEO_DIR:?set VIDEO_DIR to a directory of .mp4 files}"
: "${OUT_ROOT:?set OUT_ROOT to an output directory}"
: "${PROMPTS_TSV:?set PROMPTS_TSV to a tab-separated video_stem/object prompt file}"

STAGE="${STAGE:-all}"
PC_STRIDE="${PC_STRIDE:-1}"
GRID_STRIDE="${GRID_STRIDE:-2}"
FPS="${FPS:-8.0}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-$MOTIONFORESIGHT_ROOT/assets/sam3/sam3.pt}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

PROC="$OUT_ROOT/processed22_context50"
MASK="$OUT_ROOT/masks_sam3_object_context50"
VIEW2D="$OUT_ROOT/viewer_pred_context50"
HTML3D="$OUT_ROOT/3d_html_sam3_object_context50_dense_pc"

case "$STAGE" in
  all|preprocess|masks|viz) ;;
  *) echo "STAGE must be all, preprocess, masks, or viz" >&2; exit 2 ;;
esac

if [ "$STAGE" = "all" ] || [ "$STAGE" = "preprocess" ]; then
  "$PY" scripts/comparison/preprocess_ood_videos.py \
    --video-dir "$VIDEO_DIR" \
    --out-dir "$PROC" \
    --num-frames 22 \
    --sample-mode context50 \
    --obs-frames 7 \
    --height 480 \
    --width 832
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "masks" ]; then
  mapfile -t prompt_args < <("$PY" - "$PROMPTS_TSV" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
for line in path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if "\t" in line:
        vid, prompt = line.split("\t", 1)
    else:
        vid, prompt = line.split(None, 1)
    print("--prompt")
    print(f"{vid}:{prompt}")
PY
)
  "$PY" scripts/comparison/sam3_ood_masks.py \
    --proc-dir "$PROC" \
    --out-dir "$MASK" \
    --checkpoint "$SAM3_CHECKPOINT" \
    "${prompt_args[@]}"

  "$PY" - "$MASK" <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
import sys

mask_dir = Path(sys.argv[1])
paths = sorted(mask_dir.glob("*_overlay.png"))
if not paths:
    raise SystemExit("no overlay images found")

thumb_w, thumb_h, label_h, cols = 320, 190, 24, 4
rows = math.ceil(len(paths) / cols)
sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (18, 20, 22))
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 14)
except Exception:
    font = ImageFont.load_default()
for i, path in enumerate(paths):
    img = Image.open(path).convert("RGB")
    img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    x = (i % cols) * thumb_w
    y = (i // cols) * (thumb_h + label_h)
    sheet.paste(img, (x + (thumb_w - img.width) // 2, y))
    draw.text((x + 8, y + thumb_h + 4), path.name.replace("_overlay.png", ""), fill=(235, 235, 235), font=font)
out = mask_dir / "contact_sheet_prompts.jpg"
sheet.save(out, quality=92)
print(out)
PY
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "viz" ]; then
  "$PY" scripts/comparison/ood_render_html.py \
    --checkpoint "$MOTIONFORESIGHT_CKPT" \
    --base-checkpoint "$MOTIONFORESIGHT_BASE" \
    --proc-dir "$PROC" \
    --mask-dir "$MASK" \
    --out-dir "$VIEW2D" \
    --grid-stride 4 \
    --fps "$FPS"

  "$PY" scripts/comparison/build_ood_3d_html_one.py \
    --checkpoint "$MOTIONFORESIGHT_CKPT" \
    --base-checkpoint "$MOTIONFORESIGHT_BASE" \
    --user-dir "$PROC" \
    --mask-dir "$MASK" \
    --out-dir "$HTML3D" \
    --suffix "_context50_dense_pc_3d.html" \
    --grid-stride "$GRID_STRIDE" \
    --pc-stride "$PC_STRIDE" \
    --max-tracks 220 \
    --rainbow-tracks
fi
