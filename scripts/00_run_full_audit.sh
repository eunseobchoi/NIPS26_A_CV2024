#!/usr/bin/env bash
# Stage 0: full-corpus perceptual hash audit + pixel NCC verification.
# Expected wall time: ~3 min (pHash+dHash, CPU) + ~15 min (full NCC, CPU) + ~8 min (PDQ).
# Requires: $CV2024_ROOT, $KVASIR_ROOT.
set -euo pipefail
PY="${PY:-python3}"

: "${CV2024_ROOT:?Set CV2024_ROOT to the CV2024 Dataset/ path}"
: "${KVASIR_ROOT:?Set KVASIR_ROOT to the Kvasir-Capsule root (contains labelled_images/)}"

cd "$(dirname "$0")/.."
source scripts/_setup_data_links.sh
setup_capsule_data_links

mkdir -p artifacts/hashes artifacts/annotations artifacts/ncc artifacts/summaries results

echo "[1/6] pHash + dHash extraction + per-source annotation"
"$PY" src/audit/01_phash_dhash_audit.py \
  --out_json results/phash_audit.json
cp -f results/hashes_kvasir.json artifacts/hashes/hashes_kvasir.json
cp -f results/hashes_cv2024.json artifacts/hashes/hashes_cv2024.json
cp -f results/cv2024_*_phash_annotated.csv artifacts/annotations/

echo "[1b/6] Same-frame joint attribution audit"
"$PY" src/audit/11_same_frame_attribution.py
cp -f results/same_frame_audit.json artifacts/summaries/same_frame_audit.json

echo "[2/6] PDQ 256-bit corroboration"
"$PY" src/audit/02_pdq_audit.py \
  --out_json results/pdq_audit.json
cp -f results/pdq_hashes_kvasir.json artifacts/hashes/pdq_hashes_kvasir.json
cp -f results/pdq_hashes_cv2024.json artifacts/hashes/pdq_hashes_cv2024.json
cp -f results/cv2024_*_pdq_annotated.csv artifacts/annotations/

echo "[3/6] Full pixel-NCC on 38,592 flagged KVASIR pairs"
"$PY" src/audit/03_ncc_verify.py
cp -f results/cv2024_KVASIR_ncc_full.csv artifacts/ncc/cv2024_KVASIR_ncc_full.csv
cp -f results/cv2024_KVASIR_ncc_full_summary.json artifacts/ncc/cv2024_KVASIR_ncc_full_summary.json

echo "[4/6] pHash geometric-robustness calibration (threshold validation, lower bound)"
"$PY" src/audit/04_phash_geometric_robustness.py
cp -f results/phash_geometric_robustness.json artifacts/summaries/phash_geometric_robustness.json

echo "[5/6] Label-inheritance audit (Kvasir 14-class -> CV2024 10-class)"
"$PY" src/audit/05_label_mapping_audit.py
cp -f results/label_mapping_audit.json artifacts/summaries/label_mapping_audit.json
cp -f results/label_mapping_details.csv artifacts/summaries/label_mapping_details.csv

echo "[6/6] CV2024 internal train->val leakage (per-source + video-prefix + cross-source)"
"$PY" src/audit/07_internal_leak.py
"$PY" src/audit/08_internal_per_patient.py
"$PY" src/audit/09_cross_source_internal.py
"$PY" src/audit/10_per_patient_leakage.py
cp -f results/cv2024_internal_leak.json artifacts/summaries/cv2024_internal_leak.json
cp -f results/cv2024_internal_per_patient.json artifacts/summaries/cv2024_internal_per_patient.json
cp -f results/cv2024_internal_cross_source.json artifacts/summaries/cv2024_internal_cross_source.json
cp -f results/per_patient_leakage.json artifacts/summaries/per_patient_leakage.json
cp -f results/phash_audit.json artifacts/summaries/phash_audit.json
cp -f results/pdq_audit.json artifacts/summaries/pdq_audit.json

echo ""
echo "Stage 0 done.  Expected top-line numbers:"
echo "  pHash+dHash: 38,592/38,592 (100%) KVASIR flagged at <=6; non-KVASIR <=0.31% exactly."
echo "  PDQ:        38,357/38,592 (99.4%) KVASIR at <=50; non-KVASIR 0%."
echo "  NCC:        mean 0.998 on flagged pairs; 92.3% >=0.99; 99.6% >=0.95."
echo "  Internal:   1,381/11,581 (11.9%) KVASIR val pHash-exact to train; 41/41 videos shared."
echo "  Label:      100% inheritance across 38,584 mapped entries."
