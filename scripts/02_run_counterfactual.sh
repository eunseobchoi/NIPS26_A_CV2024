#!/usr/bin/env bash
# Stage 2: counterfactual retraining of DINOv2-ViT-L/14 + LoRA-r=8 from
# scratch on each dedup pool + size control.  Also runs the default
# cross-model backbones (DINOv2-ViT-B/14, DINOv2-ViT-S/14, ResNet-50)
# for Table 5; ConvNeXt-Tiny is available through 10_run_convnext_acceptance.sh.
#
# WALL-CLOCK: canonical mode is large (~30-50 GPU-hours on A100 for 18 arms
# x n=10 paired seeds; partial reproduction is supported via
# RUN_CROSSMODEL=0 and RUN_COUNTERFACTUAL_CONTROLS=0). Set
# COUNTERFACTUAL_MODE=legacy for the historical n=4 sanity sweep instead.
#
# OVERWRITE WARNING: by default this script writes baseline/le6 retraining
# outputs to results/baseline/ — the same path that ships in
# checksums.txt. Re-running will overwrite the shipped JSONs and break
# verify_checksums.sh until the manifest is refreshed. Set
# BASELINE_OUT_DIR to a parallel directory (e.g. baseline_rerun)
# if you want to preserve the shipped headline JSONs for diff comparison.
set -euo pipefail
PY="${PY:-python3}"

: "${CV2024_ROOT:?Set CV2024_ROOT to the CV2024 Dataset directory containing training/ and validation/}"
: "${KVASIR_ROOT:?Set KVASIR_ROOT to Kvasir-Capsule labelled_images/}"

cd "$(dirname "$0")/.."
source scripts/_setup_data_links.sh
setup_capsule_data_links

export KVASIR_SPLITS_DIR="${KVASIR_SPLITS_DIR:-$PWD/data/official_splits}"

cv_dataset="$CV2024_ROOT"
if [[ -d "$CV2024_ROOT/Dataset" ]]; then
  cv_dataset="$CV2024_ROOT/Dataset"
fi
export CV2024_ROOT="$cv_dataset"
export KVASIR_ROOT="${KVASIR_IMAGES_ROOT:-$KVASIR_ROOT}"

OUT=results
CONTROL_OUT_DIR="${CONTROL_OUT_DIR:-counterfactual_n10_rerun}"
mkdir -p "$OUT/counterfactual" "$OUT/crossmodel" "$OUT/baseline" "$OUT/counterfactual_n10" "$OUT/$CONTROL_OUT_DIR"

EPOCHS=10
BATCH="${BATCH:-128}"
MODE="${COUNTERFACTUAL_MODE:-canonical}"
read -r -a SEED_ARGS <<< "${SEEDS:-0 1 2 3 4 5 6 7 8 9}"

if [[ "$MODE" == "canonical" ]]; then
  if [[ "${#SEED_ARGS[@]}" -ne 10 ]]; then
    echo "ERROR: canonical mode writes n10 outputs and requires exactly 10 seeds; got ${#SEED_ARGS[@]}: ${SEED_ARGS[*]}" >&2
    echo "Use COUNTERFACTUAL_MODE=legacy or call src/counterfactual/phase5_counterfactual_v5.py directly for a smaller sweep." >&2
    exit 2
  fi

  echo "[1] Canonical v5 baseline (n=${#SEED_ARGS[@]})"
  "$PY" src/counterfactual/phase5_counterfactual_v5.py \
    --train_csv "$cv_dataset/training/training_data.xlsx" \
    --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
    --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
    --epochs "$EPOCHS" --batch "$BATCH" --seeds "${SEED_ARGS[@]}" \
    --output "baseline/phase5_v5_baseline_n10.json"

  echo "[2] Canonical v5 le6 Kvasir-origin-removed pool (n=${#SEED_ARGS[@]})"
  "$PY" src/counterfactual/phase5_counterfactual_v5.py \
    --train_csv artifacts/csvs/cv2024_training_dedup_le6.csv \
    --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
    --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
    --epochs "$EPOCHS" --batch "$BATCH" --seeds "${SEED_ARGS[@]}" \
    --output "baseline/phase5_v5_le6_n10.json"

  if [[ "${RUN_COUNTERFACTUAL_CONTROLS:-1}" == "1" ]]; then
    echo "[3] Size-matched random control from released CSVs (n=${#SEED_ARGS[@]})"
    "$PY" src/counterfactual/phase5_counterfactual_v5.py \
      --train_csv artifacts/csvs/cv2024_training_random10596_s0.csv \
      --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
      --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
      --epochs "$EPOCHS" --batch "$BATCH" --seeds "${SEED_ARGS[@]}" \
      --output "${CONTROL_OUT_DIR}/phase5_v5_random_s0_n10.json"

    echo "[4] Composition-matched control from released CSVs (n=${#SEED_ARGS[@]})"
    "$PY" src/counterfactual/phase5_counterfactual_v5.py \
      --train_csv artifacts/csvs/cv2024_training_compmatched_strict_s0.csv \
      --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
      --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
      --epochs "$EPOCHS" --batch "$BATCH" --seeds "${SEED_ARGS[@]}" \
      --output "${CONTROL_OUT_DIR}/phase5_v5_compmatched_s0_n10.json"

    echo "[5] Comp-A Ulcer source-aligned control from released CSVs (n=${#SEED_ARGS[@]})"
    "$PY" src/counterfactual/phase5_counterfactual_v5.py \
      --train_csv artifacts/csvs/cv2024_training_compA_ulcer_aligned_s0.csv \
      --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
      --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
      --epochs "$EPOCHS" --batch "$BATCH" --seeds "${SEED_ARGS[@]}" \
      --output "${CONTROL_OUT_DIR}/phase5_v5_compA_s0_n10.json"

    echo "[6] Comp-B Ulcer doubled control from released CSVs (n=${#SEED_ARGS[@]})"
    "$PY" src/counterfactual/phase5_counterfactual_v5.py \
      --train_csv artifacts/csvs/cv2024_training_compB_ulcer_balanced_s0.csv \
      --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
      --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
      --epochs "$EPOCHS" --batch "$BATCH" --seeds "${SEED_ARGS[@]}" \
      --output "${CONTROL_OUT_DIR}/phase5_v5_compB_s0_n10.json"

    echo "[7] Comp-C AIIMS-only duplicate-66 Ulcer control from released CSVs (n=${#SEED_ARGS[@]})"
    "$PY" src/counterfactual/phase5_counterfactual_v5.py \
      --train_csv artifacts/csvs/cv2024_training_compC_aiims_ulcer_oversampled_s0.csv \
      --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
      --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
      --epochs "$EPOCHS" --batch "$BATCH" --seeds "${SEED_ARGS[@]}" \
      --output "${CONTROL_OUT_DIR}/phase5_v5_compC_aiims_ulcer_s0_n10.json"

    echo "[8] Comp-D duplicate KVASIR-Ulcer control from released CSVs (n=${#SEED_ARGS[@]})"
    "$PY" src/counterfactual/phase5_counterfactual_v5.py \
      --train_csv artifacts/csvs/cv2024_training_compD_kvasir_ulcer_duplicate_s0.csv \
      --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
      --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
      --epochs "$EPOCHS" --batch "$BATCH" --seeds "${SEED_ARGS[@]}" \
      --output "${CONTROL_OUT_DIR}/phase5_v5_compD_kvasir_ulcer_duplicate_s0_n10.json"

    echo "[9] Comp-D unique-added KVASIR-Ulcer control from released CSVs (n=${#SEED_ARGS[@]})"
    "$PY" src/counterfactual/phase5_counterfactual_v5.py \
      --train_csv artifacts/csvs/cv2024_training_compD_kvasir_ulcer_oversampled_s0.csv \
      --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
      --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
      --epochs "$EPOCHS" --batch "$BATCH" --seeds "${SEED_ARGS[@]}" \
      --output "${CONTROL_OUT_DIR}/phase5_v5_compD_kvasir_ulcer_s0_n10.json"
  else
    echo "[3] Skipping matched control arms because RUN_COUNTERFACTUAL_CONTROLS=0"
  fi

  if [[ "${RUN_CROSSMODEL:-1}" == "1" ]]; then
    echo "[10] Cross-model Table 5: DINOv2-ViT-B/14 and DINOv2-ViT-S/14 (n=${#SEED_ARGS[@]})"
    for backbone in dinov2_vitb14 dinov2_vits14; do
      short="$(case "$backbone" in dinov2_vitb14) echo dinov2B ;; dinov2_vits14) echo dinov2S ;; esac)"
      for pool in baseline le6 random; do
        "$PY" src/counterfactual/03_cross_model.py \
          --backbone "$backbone" \
          --train_csv "$(case "$pool" in baseline) echo "$cv_dataset/training/training_data.xlsx" ;; le6) echo artifacts/csvs/cv2024_training_dedup_le6.csv ;; random) echo artifacts/csvs/cv2024_training_random10596_s0.csv ;; esac)" \
          --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
          --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
          --epochs "$EPOCHS" --batch "$BATCH" --seeds "${SEED_ARGS[@]}" \
          --output "crossmodel/phase5_crossmodel_${short}_${pool}_n10.json"
      done
    done

    echo "[11] Cross-model Table 5: ResNet-50 (25M, linear probe, n=${#SEED_ARGS[@]})"
    for pool in baseline le6 random; do
      "$PY" src/counterfactual/03_cross_model.py \
        --backbone resnet50 \
        --train_csv "$(case "$pool" in baseline) echo "$cv_dataset/training/training_data.xlsx" ;; le6) echo artifacts/csvs/cv2024_training_dedup_le6.csv ;; random) echo artifacts/csvs/cv2024_training_random10596_s0.csv ;; esac)" \
        --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
        --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
        --epochs "$EPOCHS" --batch "$BATCH" --seeds "${SEED_ARGS[@]}" \
        --output "crossmodel/phase5_crossmodel_resnet50_${pool}_n10.json"
    done
  else
    echo "[10] Skipping Table 5 cross-model arms because RUN_CROSSMODEL=0"
  fi

  echo ""
  echo "Stage 2 canonical done. Expected headline:"
  echo "  baseline n=10 ~= 0.825, le6 n=10 ~= 0.612, Delta ~= -0.213"
  echo "  Fresh control reruns were written to results/${CONTROL_OUT_DIR}/."
  echo "  Shipped archival Figure 3 controls remain in results/counterfactual_n10/."
  exit 0
fi

if [[ "$MODE" != "legacy" ]]; then
  echo "ERROR: COUNTERFACTUAL_MODE must be canonical or legacy, got '$MODE'." >&2
  exit 2
fi

BATCH=64

echo "[1/7] Baseline (full contaminated pool, 2 seeds)"
"$PY" src/counterfactual/01_baseline_pooled.py \
  --experiment pooled \
  --epochs "$EPOCHS" --seeds 0 1 \
  --output "counterfactual/baseline_contaminated_seeds01.json"

for pool in le0 le2 le6; do
  echo "[2/7+] Counterfactual $pool (DINOv2-L, LoRA r=8)"
  "$PY" src/counterfactual/02_counterfactual_dinov2L.py \
    --train_csv artifacts/csvs/cv2024_training_dedup_${pool}.csv \
    --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
    --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_${pool}.csv \
    --epochs "$EPOCHS" --batch "$BATCH" --seeds 0 1 \
    --output "counterfactual/phase5_counterfactual_${pool}_seeds01.json"
done

echo "[+] Extra legacy le6 seeds 2,3 (legacy sanity aggregate only; final tables use packaged n=10 artifacts)"
"$PY" src/counterfactual/02_counterfactual_dinov2L.py \
  --train_csv artifacts/csvs/cv2024_training_dedup_le6.csv \
  --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
  --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
  --epochs "$EPOCHS" --batch "$BATCH" --seeds 2 3 \
  --output "counterfactual/phase5_counterfactual_le6_seeds23.json"

echo "[5/7] Size-matched random control (n=10,596)"
"$PY" src/counterfactual/02_counterfactual_dinov2L.py \
  --train_csv artifacts/csvs/cv2024_training_random10596_s0.csv \
  --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
  --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
  --epochs "$EPOCHS" --batch "$BATCH" --seeds 0 1 \
  --output "counterfactual/phase5_random10596_seeds01.json"

echo "[6/7] Cross-model: DINOv2-ViT-B/14 (86M, LoRA r=8)"
for pool in baseline le6 random; do
  "$PY" src/counterfactual/03_cross_model.py \
    --backbone dinov2_vitb14 \
    --train_csv "$(case "$pool" in baseline) echo "$cv_dataset/training/training_data.xlsx" ;; le6) echo artifacts/csvs/cv2024_training_dedup_le6.csv ;; random) echo artifacts/csvs/cv2024_training_random10596_s0.csv ;; esac)" \
    --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
    --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
    --epochs "$EPOCHS" --batch "$BATCH" --seeds 0 1 2 3 \
    --output "crossmodel/phase5_crossmodel_dinov2B_${pool}_n4.json"
done

echo "[7/7] Cross-model: ResNet-50 (25M, linear probe)"
for pool in baseline le6 random; do
  "$PY" src/counterfactual/03_cross_model.py \
    --backbone resnet50 \
    --train_csv "$(case "$pool" in baseline) echo "$cv_dataset/training/training_data.xlsx" ;; le6) echo artifacts/csvs/cv2024_training_dedup_le6.csv ;; random) echo artifacts/csvs/cv2024_training_random10596_s0.csv ;; esac)" \
    --orig_val_csv "$cv_dataset/validation/validation_data.xlsx" \
    --dedup_val_csv artifacts/csvs/cv2024_validation_dedup_le6.csv \
    --epochs "$EPOCHS" --batch "$BATCH" --seeds 0 1 2 3 \
    --output "crossmodel/phase5_crossmodel_resnet50_${pool}_n4.json"
done

echo ""
echo "[*] Consolidate & render tables"
"$PY" src/counterfactual/04_consolidate_results.py \
  --results-dir "$OUT/counterfactual" --out "$OUT/counterfactual/summary.json"
"$PY" src/counterfactual/05_consolidate_crossmodel.py \
  --results-dir "$OUT/crossmodel"     --out "$OUT/crossmodel/summary.json"

echo ""
echo "Stage 2 done.  Numbers above are LEGACY (n=2/n=4) sanity values."
echo "Canonical paper numbers (DINOv2-L v4/v5 on original CV2024 val, paired-Delta on shared seeds):"
echo "  baseline   n=10:  0.825 +- 0.008   (results/baseline/phase5_v5_baseline_n10.json)"
echo "  le6        n=10:  0.612 +- 0.007   -> Delta_le6  = -0.213, t(9)=71.7, p~1.0e-13"
echo "  Delta_size / Delta_class / Delta_content: see EXPECTED.md and Table 4 provenance"
echo "  for the packaged n=10 shared-seed accounting artifacts."
echo "Path-B Exp.1 (le6_kvfree_s1, n=10 paired aggregate if Stage 7 shards exist): orig-val +0.091 +- 0.003 SE"
echo "  -> source/domain familiarity recovers ~43% of headline drop without pixel overlap."
