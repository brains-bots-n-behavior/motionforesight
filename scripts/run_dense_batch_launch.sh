#!/usr/bin/env bash
# =============================================================================
# Scalable, resumable DENSE TrackCraft3r (stage 4) across all GPUs.
#
# Unlike DA3 (stage 3), the dense script ALREADY loads the Wan model once and
# loops (scripts/run_trackcraft3r_dense_batch.py), and dense is GPU-bound, so we
# run ONE worker per GPU (2/GPU would just contend + risk GPU OOM on the big Wan
# stack). Launches are STAGGERED to smooth the host-RAM spike when each worker
# loads Wan2.1 DiT + umt5-xxl T5 + VAEs.
#
# Distributes ONLY remaining clips (user.npz present, dense.npz missing) across
# the GPUs as disjoint sublists. Fully resumable (dense script skips existing
# dense.npz).
#
#   tmux new -s densebatch
#   GPUS="0 1 2 3 4 5 6 7" bash scripts/run_dense_batch_launch.sh
# =============================================================================
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /weka/scratch/hbharad2/users/yjangir1/miniconda3/etc/profile.d/conda.sh
conda activate 3dflow
export HF_HOME=/weka/scratch/hbharad2/users/yjangir1/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export MODELSCOPE_CACHE="$(pwd)/external/TrackCraft3r/checkpoints/wan_models"
THREADS_PER_WORKER=${THREADS_PER_WORKER:-8}
export OMP_NUM_THREADS=$THREADS_PER_WORKER OPENBLAS_NUM_THREADS=$THREADS_PER_WORKER \
       MKL_NUM_THREADS=$THREADS_PER_WORKER NUMEXPR_NUM_THREADS=$THREADS_PER_WORKER

ROOT=${ROOT:-data/ss_subset100k}
TRACKS=${TRACKS:-anchor_tracks32}
TRACKLIST=${TRACKLIST:-$ROOT/anchor_selected_videos_track.txt}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
NUM_FRAMES=${NUM_FRAMES:-22}
FRAME_STRIDE=${FRAME_STRIDE:-1}
DENSE_STAGGER=${DENSE_STAGGER:-90}
LOG=$ROOT/logs; mkdir -p "$LOG"
SPLITDIR=$ROOT/dense_batch_splits; mkdir -p "$SPLITDIR"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Multi-node: each node takes a DISJOINT modulo slice of remaining clips before
# splitting across its local GPUs. Run on node A: NODE_SHARDS=2 NODE_SHARD=0 ...
# on node B: NODE_SHARDS=2 NODE_SHARD=1 ...  (shared weka fs; race-safe).
NODE_SHARDS=${NODE_SHARDS:-1}
NODE_SHARD=${NODE_SHARD:-0}
read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

echo "[$(ts)] [dense-launch] node-shard $NODE_SHARD/$NODE_SHARDS; splitting remaining across $NGPU local GPUs ..."
python - "$ROOT" "$TRACKS" "$TRACKLIST" "$SPLITDIR" "$NGPU" "$NODE_SHARDS" "$NODE_SHARD" <<'PY'
import sys, pathlib
root, tracks, tracklist, splitdir, ng, nshards, nshard = sys.argv[1:8]
ng, nshards, nshard = int(ng), int(nshards), int(nshard)
root = pathlib.Path(root); td = root / tracks
have_user = set(p.name[:-len('_user.npz')] for p in td.glob('*_user.npz'))
have_dense = set(p.name[:-len('_dense.npz')] for p in td.glob('*_dense.npz'))
# Shard by STABLE index over the full track-list (not the dynamic remaining
# list), so nodes launched at different times stay perfectly disjoint.
mine = []; rem_total = 0; gi = -1
for l in open(tracklist):
    l = l.strip()
    if not l or l.lstrip().startswith('#'): continue
    gi += 1
    uid = pathlib.Path(l).stem.split('.')[0]
    remaining = (uid in have_user) and (uid not in have_dense)
    if remaining: rem_total += 1
    if gi % nshards != nshard: continue        # this node's stable slice
    if remaining: mine.append(l)
buckets = [[] for _ in range(ng)]
for i, l in enumerate(mine): buckets[i % ng].append(l)
sd = pathlib.Path(splitdir)
for j, b in enumerate(buckets):
    (sd / f'drem_n{nshard}_{j:02d}.txt').write_text("\n".join(b) + ("\n" if b else ""))
print(f"remaining_dense_total={rem_total} (user.npz={len(have_user)}, dense done={len(have_dense)}); this_node[shard {nshard}/{nshards}]={len(mine)} -> {ng} local sublists (~{len(mine)//ng if ng else 0} each)")
PY

echo "[$(ts)] [dense-launch] launching 1 worker/GPU on (${GPU_ARR[*]}), stagger=${DENSE_STAGGER}s"
pids=()
for i in "${!GPU_ARR[@]}"; do
  g=${GPU_ARR[$i]}
  sub="$SPLITDIR/drem_n${NODE_SHARD}_$(printf '%02d' $i).txt"
  if [ ! -s "$sub" ]; then echo "  (empty $sub — skip GPU $g)"; continue; fi
  CUDA_VISIBLE_DEVICES=$g python -u scripts/run_trackcraft3r_dense_batch.py \
    --root "$ROOT" --tracks-name "$TRACKS" --video-list "$sub" \
    --trackcraft-root external/TrackCraft3r \
    --num-frames "$NUM_FRAMES" --frame-stride "$FRAME_STRIDE" \
    --cuda-visible-devices "$g" --device cuda --keep-going \
    > "$LOG/densebatch_n${NODE_SHARD}_g${g}.log" 2>&1 &
  pids+=($!)
  echo "  GPU $g -> $sub ($(wc -l < "$sub") clips)"
  if (( i < NGPU - 1 )); then sleep "$DENSE_STAGGER"; fi
done

echo "[$(ts)] [dense-launch] ${#pids[@]} workers running; waiting ..."
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[$(ts)] [dense-launch] DONE (worker fail=$fail). dense.npz now: $(find "$ROOT/$TRACKS" -name '*_dense.npz' | wc -l)"
