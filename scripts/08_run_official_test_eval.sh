#!/usr/bin/env bash
# Optional GPU rerun of the direct CV2024 official AIIMS-test scope check.
#
# Required:
#   CV2024_ROOT       directory containing Dataset/training and Dataset/validation,
#                     or the Dataset directory itself
#   OFFICIAL_TEST_DIR class-separated official CV2024 test image directory
#
# The runner writes official-format prediction XLSX files and invokes the
# pinned organizer script (`external/cv2024_repo/Results/gen_metrics_test.py`)
# through src/counterfactual/04_official_test_eval.py.
set -euo pipefail
PY="${PY:-python3}"

cd "$(dirname "$0")/.."

: "${CV2024_ROOT:?Set CV2024_ROOT}"
: "${OFFICIAL_TEST_DIR:?Set OFFICIAL_TEST_DIR}"

if [[ -d "$CV2024_ROOT/Dataset" ]]; then
  CV_ROOT="$CV2024_ROOT/Dataset"
else
  CV_ROOT="$CV2024_ROOT"
fi

orig_val="$CV_ROOT/validation/validation_data.xlsx"
dedup_val="artifacts/csvs/cv2024_validation_dedup_le6.csv"
pred_dir="results/official_test/official_prediction_xlsx"

mkdir -p "$pred_dir"

run_arm() {
  local arm="$1"
  local train_csv="$2"
  for seed in {0..9}; do
    echo "== official-test arm=$arm seed=$seed =="
    "$PY" src/counterfactual/04_official_test_eval.py \
      --train_csv "$train_csv" \
      --orig_val_csv "$orig_val" \
      --dedup_val_csv "$dedup_val" \
      --official_test_dir "$OFFICIAL_TEST_DIR" \
      --seeds "$seed" \
      --epochs 10 \
      --batch 128 \
      --workers "${WORKERS:-6}" \
      --output "official_test/${arm}_official_test_s${seed}.json" \
      --save_official_predictions_dir "$pred_dir"
  done
}

run_arm "baseline" "$CV_ROOT/training/training_data.xlsx"
run_arm "random10596_s0" "artifacts/csvs/cv2024_training_random10596_s0.csv"
run_arm "le6" "artifacts/csvs/cv2024_training_dedup_le6.csv"

"$PY" scripts/merge_official_test.py
"$PY" scripts/08_verify_official_test_metrics.py \
  --root . \
  --write-json results/official_test/official_metric_replay.json
