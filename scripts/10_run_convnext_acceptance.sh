#!/usr/bin/env bash
# Optional GPU robustness: ConvNeXt-Tiny baseline/le6/random public-pool arms.
set -euo pipefail

PY="${PY:-python3}"
cd "$(dirname "$0")/.."

: "${CV2024_ROOT:?Set CV2024_ROOT to the CV2024 Dataset directory containing training/ and validation/}"
: "${KVASIR_ROOT:?Set KVASIR_ROOT to Kvasir-Capsule labelled_images/}"

source scripts/_setup_data_links.sh
setup_capsule_data_links

cv_dataset="$CV2024_ROOT"
if [[ -d "$CV2024_ROOT/Dataset" ]]; then
  cv_dataset="$CV2024_ROOT/Dataset"
fi
export CV2024_ROOT="$cv_dataset"
export KVASIR_ROOT="${KVASIR_IMAGES_ROOT:-$KVASIR_ROOT}"
export KVASIR_SPLITS_DIR="${KVASIR_SPLITS_DIR:-$PWD/data/official_splits}"

mkdir -p results/crossmodel
read -r -a SEED_ARGS <<< "${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-128}"

for pool in baseline le6 random; do
  train_csv="$(case "$pool" in
    baseline) echo "$cv_dataset/training/training_data.xlsx" ;;
    le6) echo artifacts/csvs/cv2024_training_dedup_le6.csv ;;
    random) echo artifacts/csvs/cv2024_training_random10596_s0.csv ;;
  esac)"
  "$PY" src/counterfactual/03_cross_model.py \
    --backbone convnext_tiny \
    --train_csv "$train_csv" \
    --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
    --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
    --epochs "$EPOCHS" --batch "$BATCH" --seeds "${SEED_ARGS[@]}" \
    --output "crossmodel/phase5_crossmodel_convnextT_${pool}_n${#SEED_ARGS[@]}.json"
done

echo "ConvNeXt-Tiny robustness jobs complete."
