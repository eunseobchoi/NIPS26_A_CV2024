#!/usr/bin/env bash
# Run ConvNeXt-Tiny cross-model shards sequentially on one visible GPU.
#
# QUEUE_SPEC format: whitespace-separated pool:seed items, e.g.
#   QUEUE_SPEC="le6:0 random:0 baseline:10"
# Each shard writes a distinct JSON under results/crossmodel/convnext_shards/
# so multiple GPU sessions can run without output collisions.
set -euo pipefail

PY="${PY:-python3}"
cd "$(dirname "$0")/.."

: "${CV2024_ROOT:?Set CV2024_ROOT to the CV2024 Dataset directory containing training/ and validation/}"
: "${KVASIR_ROOT:?Set KVASIR_ROOT to Kvasir-Capsule labelled_images/}"
: "${QUEUE_SPEC:?Set QUEUE_SPEC to whitespace-separated pool:seed items}"

source scripts/_setup_data_links.sh
setup_capsule_data_links

cv_dataset="$CV2024_ROOT"
if [[ -d "$CV2024_ROOT/Dataset" ]]; then
  cv_dataset="$CV2024_ROOT/Dataset"
fi
export CV2024_ROOT="$cv_dataset"
export KVASIR_ROOT="${KVASIR_IMAGES_ROOT:-$KVASIR_ROOT}"
export KVASIR_SPLITS_DIR="${KVASIR_SPLITS_DIR:-$PWD/data/official_splits}"

mkdir -p results/crossmodel/convnext_shards

for item in $QUEUE_SPEC; do
  pool="${item%%:*}"
  seed="${item##*:}"
  case "$pool" in
    baseline) train_csv="$cv_dataset/training/training_data.xlsx" ;;
    le6) train_csv="artifacts/csvs/cv2024_training_dedup_le6.csv" ;;
    random) train_csv="artifacts/csvs/cv2024_training_random10596_s0.csv" ;;
    *) echo "ERROR: unknown pool '$pool' in QUEUE_SPEC item '$item'" >&2; exit 2 ;;
  esac
  case "$seed" in
    ''|*[!0-9]*) echo "ERROR: seed must be a non-negative integer in '$item'" >&2; exit 2 ;;
  esac

  out="crossmodel/convnext_shards/phase5_crossmodel_convnextT_${pool}_s${seed}.json"
  if [[ -s "results/$out" ]]; then
    echo "SKIP existing $out"
    continue
  fi
  echo "START convnext_tiny pool=$pool seed=$seed $(date)"
  "$PY" src/counterfactual/03_cross_model.py \
    --backbone convnext_tiny \
    --train_csv "$train_csv" \
    --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
    --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
    --epochs "${EPOCHS:-10}" --batch "${BATCH:-128}" --seeds "$seed" \
    --output "$out"
  echo "DONE convnext_tiny pool=$pool seed=$seed $(date)"
done

echo "ConvNeXt-Tiny shard queue complete."
