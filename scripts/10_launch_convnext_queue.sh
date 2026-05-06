#!/usr/bin/env bash
# Launch one ConvNeXt shard queue in the current tmux/PBS allocation.
set -euo pipefail

label="${1:?label required}"
cuda="${2:?CUDA_VISIBLE_DEVICES value required}"
shift 2
queue="${*:?queue items required}"

cd "$(dirname "$0")/.."
mkdir -p logs/convnext_shards results/crossmodel/convnext_shards

if [[ -n "${CAPSULE_PYTHON_BIN:-}" ]]; then
  export PATH="$(dirname "$CAPSULE_PYTHON_BIN"):$PATH"
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CAPSULE_ROOT="${CAPSULE_ROOT:-$PWD}"
export CAPSULE_ARTIFACT_ROOT="${CAPSULE_ARTIFACT_ROOT:-$CAPSULE_ROOT}"
export CAPSULE_TTA_ROOT="${CAPSULE_TTA_ROOT:-$CAPSULE_ROOT}"
export CV2024_ROOT="${CV2024_ROOT:?Set CV2024_ROOT to the CV2024 Dataset directory or parent}"
export KVASIR_ROOT="${KVASIR_ROOT:-$CAPSULE_ROOT/data/kvasir_capsule/labelled_images}"
export KVASIR_SPLITS_DIR="${KVASIR_SPLITS_DIR:-$CAPSULE_ROOT/data/official_splits}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

log="logs/convnext_shards/${label}_$(date +%Y%m%d_%H%M%S).log"
echo "START_CONVNEXT_SHARD label=$label cuda=$cuda queue=$queue date=$(date)" > "$log"
CUDA_VISIBLE_DEVICES="$cuda" \
  PY="${PY:-${CAPSULE_PYTHON_BIN:-python3}}" \
  QUEUE_SPEC="$queue" \
  EPOCHS="${EPOCHS:-10}" \
  BATCH="${BATCH:-128}" \
  bash scripts/10_run_convnext_shard_queue.sh >> "$log" 2>&1 &
pid=$!
echo "$pid" > "logs/convnext_shards/${label}.pid"
echo "LAUNCHED_CONVNEXT_SHARD label=$label pid=$pid log=$log queue=$queue"
