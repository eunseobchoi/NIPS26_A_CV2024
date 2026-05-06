#!/usr/bin/env bash
# Optional acceptance-strengthening analyses. CPU-only except any separate
# cross-model GPU jobs launched through Stage 2 / 10_run_convnext_acceptance.sh.
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-python3}"
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

"$PY" scripts/09_verify_m7_inputs.py \
  --results-dir "$CV2024_RESULTS_DIR" \
  --write-json results/m7_input_preflight.json

"$PY" src/m7_trivial_leakage_baseline.py \
  --root "$PWD" \
  --cv2024-results-dir "$CV2024_RESULTS_DIR"

"$PY" src/m7_rank_uncertainty.py \
  --root "$PWD" \
  --cv2024-results-dir "$CV2024_RESULTS_DIR" \
  --n-bootstrap "${RANK_BOOTSTRAP_N:-1000}"

"$PY" scripts/10_verify_claim_provenance.py \
  --root "$PWD" \
  --write-json results/claim_provenance_check.json

"$PY" scripts/09_verify_croissant_manifest.py \
  --root "$PWD" \
  --write-json /tmp/capsule_tta_croissant_manifest_check.json

echo "Acceptance-strengthening analyses complete."
