#!/usr/bin/env bash
# Run the full pipeline end-to-end.
# Expected total wall time: stage 02 (canonical) dominates at ~30-50 A100-hours
# (18 arms × n=10 paired seeds); the rest (00, 01, 03, 04, 05) adds ~6 hours.
# Set RUN_CROSSMODEL=0 / RUN_COUNTERFACTUAL_CONTROLS=0 for ~10-hour partial mode
# that still recomputes the headline Δ_le6 contrast.
set -euo pipefail
PY="${PY:-python3}"

: "${CV2024_ROOT:?Set CV2024_ROOT}"
: "${KVASIR_ROOT:?Set KVASIR_ROOT}"

bash scripts/00_run_full_audit.sh
bash scripts/01_run_dedup.sh
bash scripts/02_run_counterfactual.sh
bash scripts/03_run_video_split.sh
bash scripts/04_run_tta.sh
if [[ "${RUN_OFFICIAL_TEST:-0}" == "1" ]]; then
  bash scripts/08_run_official_test_eval.sh
else
  echo "Skipping optional official AIIMS-test scope check because RUN_OFFICIAL_TEST=0"
  "$PY" scripts/08_verify_official_test_metrics.py \
    --root . \
    --write-json results/official_test/official_metric_replay.json
fi
if [[ "${RUN_M7:-0}" == "1" ]]; then
  bash scripts/06_run_m7_rescore.sh
else
  echo "Skipping optional M7 re-score because RUN_M7=0"
fi
if [[ "${RUN_ACCEPTANCE_EXPERIMENTS:-0}" == "1" ]]; then
  bash scripts/09_run_acceptance_experiments.sh
else
  echo "Skipping optional acceptance-strengthening CPU analyses because RUN_ACCEPTANCE_EXPERIMENTS=0"
fi
if [[ "${RUN_CONVNEXT:-0}" == "1" ]]; then
  bash scripts/10_run_convnext_acceptance.sh
else
  echo "Skipping optional ConvNeXt GPU robustness because RUN_CONVNEXT=0"
fi
if [[ "${RUN_ACCEPTANCE_LIFT_MERGE:-0}" == "1" ]]; then
  bash scripts/06_merge_acceptance_lift.sh
else
  echo "Skipping optional acceptance-lift merge because RUN_ACCEPTANCE_LIFT_MERGE=0"
fi
if [[ "${RUN_STRENGTHENING_MERGE:-0}" == "1" ]]; then
  bash scripts/07_merge_strengthening_results.sh
else
  echo "Skipping optional strengthening merge because RUN_STRENGTHENING_MERGE=0"
fi
bash scripts/05_make_figures.sh

echo ""
echo "==== Full pipeline done. See results/ and figures/ for outputs. ===="
