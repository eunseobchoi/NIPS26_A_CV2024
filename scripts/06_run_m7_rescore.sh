#!/usr/bin/env bash
# Optional Stage 6a: reproduce the challenge-level public-validation re-score.
# Requires the CV2024 organizer Results directory with validation prediction
# sheets, so it is intentionally opt-in from run_all.sh.
set -euo pipefail

cd "$(dirname "$0")/.."

export CAPSULE_ARTIFACT_ROOT="${CAPSULE_ARTIFACT_ROOT:-$PWD}"
if [[ -z "${CV2024_RESULTS_DIR:-}" ]]; then
  for candidate in \
    "$PWD/external/cv2024_repo/Results" \
    "$PWD/../external/cv2024_repo/Results"; do
    if [[ -e "$candidate/gen_metrics_report_val_train.py" ]] \
      && [[ -e "$candidate/training_data.xlsx" ]] \
      && [[ -e "$candidate/validation_data.xlsx" ]] \
      && [[ -d "$candidate/submitted_excel_files/validation" ]] \
      && [[ -d "$candidate/metrics_reports/metrics_reports_val" ]]; then
      CV2024_RESULTS_DIR="$candidate"
      break
    fi
  done
fi
export CV2024_RESULTS_DIR="${CV2024_RESULTS_DIR:-$PWD/external/cv2024_repo/Results}"
PY="${PY:-python3}"

required_paths=(
  "$CV2024_RESULTS_DIR/gen_metrics_report_val_train.py"
  "$CV2024_RESULTS_DIR/training_data.xlsx"
  "$CV2024_RESULTS_DIR/validation_data.xlsx"
  "$CV2024_RESULTS_DIR/submitted_excel_files/validation"
  "$CV2024_RESULTS_DIR/metrics_reports/metrics_reports_val"
)
missing=0
for p in "${required_paths[@]}"; do
  if [[ ! -e "$p" ]]; then
    echo "ERROR: missing CV2024 validation asset: $p" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "CV2024_RESULTS_DIR must point to the organizer Results directory with validation_data.xlsx, submitted prediction sheets, metric JSONs, and gen_metrics_report_val_train.py." >&2
  exit 2
fi

"$PY" scripts/09_verify_m7_inputs.py \
  --results-dir "$CV2024_RESULTS_DIR" \
  --write-json results/m7_input_preflight.json

"$PY" src/m7_rescore_orig.py
"$PY" src/m7_rescore_subset.py
"$PY" src/m7_robustness.py
"$PY" src/m7_source_balanced.py
"$PY" src/m7_inference.py
