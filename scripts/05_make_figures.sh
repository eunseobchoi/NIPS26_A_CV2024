#!/usr/bin/env bash
# Stage 5: regenerate the two JSON-only shipped paper figures.
# CPU-only, < 1 min.
set -euo pipefail
PY="${PY:-python3}"

OUT=figures
mkdir -p "$OUT"

"$PY" src/utils/make_figures.py \
  --audit-summary artifacts/summaries/phash_audit.json \
  --ncc-summary   artifacts/ncc/cv2024_KVASIR_ncc_full_summary.json \
  --per-patient   artifacts/summaries/per_patient_leakage.json \
  --counterfactual-dir results/counterfactual \
  --multibackbone-dir  results/multibackbone \
  --tta-json       results/tta/tta_bench_official.json \
  --filter-pass    artifacts/summaries/filter_pass_summary.json \
  --scaling        results/split_robustness/side_data_scaling.json \
  --videocount     results/split_robustness/videocount_ablation.json \
  --out "$OUT"

echo ""
echo "Figures written to $OUT/:"
echo "  fig_counterfactual.pdf      (counterfactual Delta bars)"
echo "  fig_counterfactual.png"
echo "  fig_ncc.pdf                 (NCC distribution diagnostic)"
echo "  fig_ncc.png"
