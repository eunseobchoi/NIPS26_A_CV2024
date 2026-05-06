#!/usr/bin/env bash
# Stage 1: generate dedup training/validation CSVs at three thresholds,
# plus the le6_plus_internal variant and the size-matched random controls.
set -euo pipefail
PY="${PY:-python3}"

: "${CV2024_ROOT:?Set CV2024_ROOT}"

cd "$(dirname "$0")/.."
source scripts/_setup_data_links.sh
setup_cv2024_data_link

OUT=artifacts/csvs
mkdir -p "$OUT" results

# The research scripts read intermediate audit files from results/.
# Seed that cache from the released artifact tree so this stage can run either
# after stage 0 or directly from the shipped metadata bundle.
cp -f artifacts/annotations/cv2024_*_phash_annotated.csv results/ 2>/dev/null || true
cp -f artifacts/hashes/hashes_cv2024.json results/hashes_cv2024.json 2>/dev/null || true
cp -f artifacts/hashes/hashes_kvasir.json results/hashes_kvasir.json 2>/dev/null || true
cp -f artifacts/summaries/cv2024_internal_cross_source.json results/cv2024_internal_cross_source.json 2>/dev/null || true

echo "[1/3] le0 / le2 / le6 canonical dedup splits"
"$PY" src/dedup/01_generate_dedup_splits.py
cp -f results/cv2024_*_dedup_*.csv "$OUT"/

echo "[2/3] le6_plus_internal (additionally remove intra-source near-dups)"
"$PY" src/dedup/02_generate_le6_plus_internal.py
cp -f results/cv2024_validation_le6_plus_internal.csv "$OUT"/
cp -f results/le6_plus_internal_summary.json artifacts/summaries/le6_plus_internal_summary.json

echo "[3/6] Size-matched random control (seeds 0 and 42)"
"$PY" src/dedup/03_generate_random_subset.py \
  --n 10596 --seed 0 --out_csv cv2024_training_random10596_s0.csv
"$PY" src/dedup/03_generate_random_subset.py \
  --n 10596 --seed 42 --out_csv cv2024_training_random10596_s42.csv
cp -f results/cv2024_training_random10596_s*.csv "$OUT"/

echo "[4/6] Composition-matched and Ulcer-control CSVs"
"$PY" src/dedup/04_generate_composition_matched.py \
  --train_xlsx "$CV2024_DATASET_ROOT/training/training_data.xlsx" \
  --le6_csv "$OUT/cv2024_training_dedup_le6.csv" \
  --seed 0 \
  --out "$OUT/cv2024_training_compmatched_strict_s0.csv"
"$PY" src/dedup/04_generate_composition_matched.py \
  --train_xlsx "$CV2024_DATASET_ROOT/training/training_data.xlsx" \
  --le6_csv "$OUT/cv2024_training_dedup_le6.csv" \
  --seed 42 \
  --out "$OUT/cv2024_training_compmatched_strict_s42.csv"
"$PY" src/dedup/06_generate_compA_compB_csv.py \
  --train_xlsx "$CV2024_DATASET_ROOT/training/training_data.xlsx" \
  --compmatched_csv "$OUT/cv2024_training_compmatched_strict_s0.csv" \
  --out_dir "$OUT" \
  --arm both

echo "[5/6] Comp-C AIIMS-Ulcer no-recovery control CSV"
"$PY" src/dedup/07_generate_compC_aiims_ulcer_oversampled.py \
  --comp-a-csv "$OUT/cv2024_training_compA_ulcer_aligned_s0.csv" \
  --out-dir "$OUT"

echo "[6/6] Comp-D KVASIR-Ulcer symmetric controls"
"$PY" src/dedup/08_generate_compD_kvasir_ulcer_duplicate.py \
  --compmatched-csv "$OUT/cv2024_training_compmatched_strict_s0.csv" \
  --out-dir "$OUT"
"$PY" src/dedup/09_generate_compD_kvasir_ulcer_oversampled.py \
  --compmatched-csv "$OUT/cv2024_training_compmatched_strict_s0.csv" \
  --train-xlsx "$CV2024_DATASET_ROOT/training/training_data.xlsx" \
  --out-dir "$OUT"

echo ""
echo "Stage 1 done.  Expected file row counts:"
echo "  le0 train 21,241, le0 val 9,054"
echo "  le2 train 10,825, le2 val 4,655"
echo "  le6 train 10,596, le6 val 4,551"
echo "  le6_plus_internal val 4,326"
echo "  random10596 train 10,596 (x2 seeds)"
echo "  compmatched/Comp-A/Comp-B/Comp-C/Comp-D train 10,596"
