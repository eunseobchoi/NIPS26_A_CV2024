#!/usr/bin/env bash
# Stage 3: protocol-gap experiments on Kvasir-Capsule's official
# frame-list split (not CV2024; not fully video-disjoint).  4 DINOv2
# backbones, LoRA r-sweep, full FT, split-rule robustness, data scaling.
# Expected wall time: ~3 h on A100 40 GB.
set -euo pipefail
PY="${PY:-python3}"

: "${KVASIR_ROOT:?Set KVASIR_ROOT (Kvasir-Capsule with labelled_images/ and the official 2-fold CSV)}"

cd "$(dirname "$0")/.."
export CAPSULE_ROOT="${CAPSULE_ROOT:-$PWD}"
export CAPSULE_ARTIFACT_ROOT="${CAPSULE_ARTIFACT_ROOT:-$PWD}"

mkdir -p data/kvasir_capsule results/multibackbone results/split_robustness results/auxiliary

kvasir_images="$KVASIR_ROOT"
if [[ -d "$KVASIR_ROOT/labelled_images" ]]; then
  kvasir_images="$KVASIR_ROOT/labelled_images"
fi
if [[ -L data/kvasir_capsule/labelled_images ]]; then
  rm -f data/kvasir_capsule/labelled_images
fi
if [[ ! -e data/kvasir_capsule/labelled_images ]]; then
  ln -s "$kvasir_images" data/kvasir_capsule/labelled_images
fi

OUT=results

echo "[1/4] 4-backbone protocol gap (frame vs video, linear probe)"
"$PY" src/video_split/01_multibackbone_protocol_gap.py \
  --backbones dinov2_vits14 dinov2_vitb14 dinov2_vitl14 dinov2_vitg14 \
  --seeds 42 1 2 \
  --output "multibackbone/phase6_multibackbone.json"

echo "[2/4] Split-rule robustness (60/40, 70/30, 80/20, 90/10, LOVO)"
"$PY" src/video_split/02_split_rule_robustness.py \
  --seeds 0 1 --epochs 12 --n_lovo 10 \
  --output "split_robustness/phase7_split_robustness.json"

echo "[3/4] Matched 11-class frame split (for Table 8 reconciliation)"
"$PY" src/video_split/03_matched_11class_frame_split.py \
  --output "multibackbone/phase4_matched_11class_frame_split.json"

echo "[4/4] Data-scaling summary from shipped ablation JSON"
if [[ -f "$OUT/split_robustness/side_data_scaling.json" ]]; then
  cp -f "$OUT/split_robustness/side_data_scaling.json" "$OUT/side_data_scaling.json"
  "$PY" src/video_split/07_data_scaling_ablation.py
else
  echo "  WARN: $OUT/split_robustness/side_data_scaling.json not found; skipping scaling summary"
fi

echo ""
echo "Stage 3 done.  Expected numbers:"
echo "  Frame split (14-class linear probe), mean over 4 DINOv2 backbones: 0.932"
echo "  Official released split (11-class linear probe), mean over 4 DINOv2 backbones: 0.260"
echo "  Protocol gap:                                                       0.672"
echo "  70/30 random frame split (11-class, LoRA r=8):               0.964 +- 0.014"
echo "  Official 2-fold released split (11-class, LoRA r=8):         0.250 +- 0.030"
