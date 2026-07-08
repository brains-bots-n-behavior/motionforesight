#!/usr/bin/env bash
# =============================================================================
# Scalable, resumable DA3 stage-3 (depth + user-NPZ) using the BATCHED worker
# (scripts/run_da3_preproc_batch.py) which loads the DA3 model ONCE per worker.
#
# Distributes ONLY the remaining clips (those without <uid>_user.npz) across
# GPUS x WORKERS_PER_GPU disjoint round-robin sublists, then launches one
# batched worker per sublist. Fully resumable: re-running recomputes "remaining"
# and skips finished clips, so a killed/preempted run continues with no rework.
#
#   tmux new -s da3batch
#   GPUS="0 1 2 3 4 5 6 7" WORKERS_PER_GPU=2 bash scripts/run_da3_preproc_launch.sh
# =============================================================================
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /weka/scratch/hbharad2/users/yjangir1/miniconda3/etc/profile.d/conda.sh
conda activate 3dflow
export HF_HOME=/weka/scratch/hbharad2/users/yjangir1/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID

# Cap CPU threads PER WORKER so N workers don't oversubscribe the cores (the DA3
# prep is CPU-bound: cv2 decode + scipy depth-resize + jpeg encode). Keep
# WORKERS_PER_GPU*NGPU*THREADS_PER_WORKER <= physical cores to avoid thrashing.
THREADS_PER_WORKER=${THREADS_PER_WORKER:-4}
export OMP_NUM_THREADS=$THREADS_PER_WORKER
export OPENBLAS_NUM_THREADS=$THREADS_PER_WORKER
export MKL_NUM_THREADS=$THREADS_PER_WORKER
export NUMEXPR_NUM_THREADS=$THREADS_PER_WORKER
export OPENCV_NUM_THREADS=$THREADS_PER_WORKER

ROOT=${ROOT:-data/ss_subset100k}
PREPROC=${PREPROC:-anchor_preproc}
TRACKS=${TRACKS:-anchor_tracks32}
TRACKLIST=${TRACKLIST:-$ROOT/anchor_selected_videos_track.txt}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}
PROC_RES=${PROC_RES:-336}
CHUNK=${CHUNK:-24}
LOG=$ROOT/logs; mkdir -p "$LOG"
SPLITDIR=$ROOT/da3_batch_splits; mkdir -p "$SPLITDIR"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
NW=$(( NGPU * WORKERS_PER_GPU ))

echo "[$(ts)] [da3-launch] computing remaining clips (no user.npz) and splitting into $NW sublists ..."
python - "$ROOT" "$TRACKS" "$TRACKLIST" "$SPLITDIR" "$NW" <<'PY'
import sys, pathlib
root, tracks, tracklist, splitdir, nw = sys.argv[1:6]; nw = int(nw)
root = pathlib.Path(root); td = root / tracks
done = set(p.name[:-len('_user.npz')] for p in td.glob('*_user.npz'))
rem = [l.strip() for l in open(tracklist)
       if l.strip() and not l.lstrip().startswith('#')
       and pathlib.Path(l.strip()).stem.split('.')[0] not in done]
buckets = [[] for _ in range(nw)]
for i, l in enumerate(rem):
    buckets[i % nw].append(l)
sd = pathlib.Path(splitdir)
for j, b in enumerate(buckets):
    (sd / f'rem_{j:02d}.txt').write_text("\n".join(b) + ("\n" if b else ""))
print(f"remaining={len(rem)} -> {nw} sublists (~{len(rem)//nw if nw else 0} each); done already={len(done)}")
PY

echo "[$(ts)] [da3-launch] launching GPUS=(${GPU_ARR[*]}) x ${WORKERS_PER_GPU}/gpu = $NW workers"
pids=(); w=0
for g in "${GPU_ARR[@]}"; do
  for ((k=0; k<WORKERS_PER_GPU; k++)); do
    sub="$SPLITDIR/rem_$(printf '%02d' $w).txt"
    if [ ! -s "$sub" ]; then echo "  (empty $sub — skip)"; w=$((w+1)); continue; fi
    CUDA_VISIBLE_DEVICES=$g python -u scripts/run_da3_preproc_batch.py \
      --root "$ROOT" --preproc-name "$PREPROC" --tracks-name "$TRACKS" \
      --video-list "$sub" --da3-root external/depth-anything-3 \
      --process-res "$PROC_RES" --chunk-size "$CHUNK" --device cuda --keep-going \
      > "$LOG/da3batch_g${g}_w${k}.log" 2>&1 &
    pids+=($!)
    echo "  worker $w -> GPU $g  ($sub, $(wc -l < "$sub") clips)"
    w=$((w+1))
  done
done

echo "[$(ts)] [da3-launch] $w workers running; waiting ..."
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[$(ts)] [da3-launch] DONE (worker fail=$fail). user.npz now: $(find "$ROOT/$TRACKS" -name '*_user.npz' | wc -l)"
