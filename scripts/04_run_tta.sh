#!/usr/bin/env bash
# Stage 4: 8-method TTA benchmark on Kvasir-Capsule official frame-list split.
# Expected wall time: ~2 h on A100, ~3 h on A6000.
set -euo pipefail
PY="${PY:-python3}"

: "${KVASIR_ROOT:?Set KVASIR_ROOT to Kvasir-Capsule labelled_images/}"

cd "$(dirname "$0")/.."
export CAPSULE_ROOT="${CAPSULE_ROOT:-$PWD}"
export CAPSULE_ARTIFACT_ROOT="${CAPSULE_ARTIFACT_ROOT:-$PWD}"

OUT=results/tta
mkdir -p "$OUT"

kvasir_images="$KVASIR_ROOT"
if [[ -d "$KVASIR_ROOT/labelled_images" ]]; then
  kvasir_images="$KVASIR_ROOT/labelled_images"
fi
export KVASIR_ROOT="$kvasir_images"
export KVASIR_SPLITS_DIR="${KVASIR_SPLITS_DIR:-$PWD/data/official_splits}"

if [[ ! -f results/dinov2_lora_official_fold0.pth || ! -f results/dinov2_lora_official_fold1.pth ]]; then
  echo "[0/2] TTA checkpoints missing; training final-epoch official LoRA checkpoints"
  "$PY" src/video_split/06_train_official_lora.py
fi

echo "[1/2] 8-method TTA benchmark: 2 folds x 3 seeds x 6 severities x 8 methods = 288 runs"
"$PY" src/tta/tta_benchmark_full.py \
  --data-root "$kvasir_images" \
  --splits-dir data/official_splits \
  --src-dir src \
  --folds 0,1 --seeds 0,1,2 --severities 0,1,2,3,4,5 \
  --methods no_adapt,ln_adapt,head_ttt,lora_ttt,hybrid_tta,od_tta,sar_official,sar_naive \
  --lora-r 8 --batch 128 \
  --output-dir "$OUT" \
  --output tta_bench_official.json

echo "[2/2] Filter-pass rate measurement (diagnostic support, Fig. 6)"
filter_log="${TTA_FILTER_LOG:-$OUT/stage6_corruption.txt}"
if [[ -f "$filter_log" ]]; then
  export TTA_FILTER_LOG="$filter_log"
  "$PY" src/tta/03_filter_pass_rate.py
else
  echo "  WARN: $filter_log not found; using shipped artifacts/summaries/filter_pass_summary.json"
fi

echo ""
echo "Stage 4 done.  Expected:"
echo "  Best improvement over no_adapt:  +0.016 (at severity 4 or 5)"
echo "  head_ttt / lora_ttt / hybrid_tta: paired Wilcoxon p <= 0.03 vs no_adapt"
echo "  sar_official:                      p >= 0.28 (not significant)"
echo "  sar_naive:                         catastrophic (<= 0.065)"
echo "  Filter-pass at severity 5 (motion_blur): 0.0"
