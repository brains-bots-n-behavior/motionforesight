#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-data/something_something}"
SAM3_ROOT="${SAM3_ROOT:-../../external/sam3}"
TRACKCRAFT_ROOT="${TRACKCRAFT_ROOT:-../../external/TrackCraft3r}"
DA3_ROOT="${DA3_ROOT:-../../external/depth-anything-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SAM3_LIMIT="${SAM3_LIMIT:-10}"
TRACK_LIMIT="${TRACK_LIMIT:-4}"
DEFAULT_VIDEO_LIST="${ROOT}/anchor_selected_videos_test${TRACK_LIMIT}.txt"
VIDEO_LIST="${VIDEO_LIST:-${DEFAULT_VIDEO_LIST}}"

"${PYTHON_BIN}" scripts/run_something_sam3_anchor_masks.py \
  --root "${ROOT}" \
  --sam3-root "${SAM3_ROOT}" \
  --limit "${SAM3_LIMIT}" \
  --device "${DEVICE}" \
  --scan-step "${SCAN_STEP:-3}" \
  --anchor-offset "${ANCHOR_OFFSET:-3}" \
  --min-frames-after-anchor "${MIN_FRAMES_AFTER_ANCHOR:-20}" \
  --clip-max-frames "${CLIP_MAX_FRAMES:-48}"

if [[ "${VIDEO_LIST}" == "${DEFAULT_VIDEO_LIST}" ]]; then
  head -n "${TRACK_LIMIT}" "${ROOT}/anchor_selected_videos.txt" > "${VIDEO_LIST}"
fi

"${PYTHON_BIN}" scripts/run_action100m_trackcraft3r.py \
  --root "${ROOT}" \
  --trackcraft-root "${TRACKCRAFT_ROOT}" \
  --da3-root "${DA3_ROOT}" \
  --preproc-name anchor_preproc \
  --tracks-name "${TRACKS_NAME:-anchor_tracks32}" \
  --video-list "${VIDEO_LIST}" \
  --process-res "${PROCESS_RES:-336}" \
  --chunk-size "${CHUNK_SIZE:-24}" \
  --num-frames "${NUM_FRAMES:-32}" \
  --frame-stride "${FRAME_STRIDE:-1}" \
  --device "${DEVICE}" \
  --keep-going

"${PYTHON_BIN}" scripts/build_action100m_mask_trace_viewer.py \
  --root "${ROOT}" \
  --sam-manifest sam3_anchor_masks/manifest.json \
  --tracks-name "${TRACKS_NAME:-anchor_tracks32}" \
  --viewer-name "${VIEWER_NAME:-anchor_track32_viewer}" \
  --frame-stride "${FRAME_STRIDE:-1}" \
  --copy-json

echo "Open ${ROOT}/sam3_anchor_masks/viewer/index.html for anchor masks."
echo "Open ${ROOT}/${VIEWER_NAME:-anchor_track32_viewer}/index.html for projected tracks."
