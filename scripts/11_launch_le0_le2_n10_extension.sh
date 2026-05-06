#!/usr/bin/env bash
set -euo pipefail

# GPU launcher for the le0/le2 seed-count extension in Appendix
# Table le0_le2_n10. Run one copy per visible GPU inside an existing
# allocation, for example:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/11_launch_le0_le2_n10_extension.sh gpu0 &
#   CUDA_VISIBLE_DEVICES=1 bash scripts/11_launch_le0_le2_n10_extension.sh gpu1 &
#   wait
#
# The script writes one JSON per arm/seed under results/le0_le2_extension/.

ROOT="${CAPSULE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${CAPSULE_PYTHON:-python3}"
RUNNER="${RUNNER:-$ROOT/src/provenance/phase5_counterfactual_n10_72047d35_exact.py}"
STATE_DIR="${STATE_DIR:-$ROOT/.le0_le2_n10_state}"
TASK_FILE="$STATE_DIR/tasks.tsv"
LOG_DIR="${LOG_DIR:-$ROOT/logs/le0_le2_n10}"
OUT_SUBDIR="le0_le2_extension"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-128}"
WORKER_ID="${1:-worker}"
ORIG_VAL_CSV="${ORIG_VAL_CSV:-$ROOT/data/cv2024/Dataset/validation/validation_data.xlsx}"
DEDUP_VAL_CSV="${DEDUP_VAL_CSV:-artifacts/csvs/cv2024_validation_dedup_le6.csv}"

cd "$ROOT"
export CAPSULE_ROOT="$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p "$STATE_DIR"/{claims,done,failed} "$LOG_DIR" "$ROOT/results/$OUT_SUBDIR"

if [[ ! -f "$RUNNER" ]]; then
  echo "ERROR: runner not found: $RUNNER" >&2
  exit 2
fi
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: python not found or not executable: $PY" >&2
  exit 2
fi
if [[ ! -f "$ORIG_VAL_CSV" ]]; then
  echo "ERROR: original CV2024 validation spreadsheet not found: $ORIG_VAL_CSV" >&2
  echo "       Link or copy the raw CV2024 Dataset/ tree under data/cv2024/Dataset." >&2
  exit 2
fi
if [[ ! -d "$ROOT/data/cv2024/Dataset/training" ]]; then
  echo "ERROR: raw CV2024 training images not found under data/cv2024/Dataset/training" >&2
  exit 2
fi
if [[ ! -d "$ROOT/data/kvasir_capsule/labelled_images" ]]; then
  echo "ERROR: Kvasir-Capsule labelled images not found under data/kvasir_capsule/labelled_images" >&2
  exit 2
fi

TASK_TMP="$TASK_FILE.tmp.${WORKER_ID}.$$"
cat > "$TASK_TMP" <<'TASKS'
le0_s0	artifacts/csvs/cv2024_training_dedup_le0.csv	0	le0_le2_extension/phase5_v4_le0_s0_n1.json
le0_s1	artifacts/csvs/cv2024_training_dedup_le0.csv	1	le0_le2_extension/phase5_v4_le0_s1_n1.json
le0_s2	artifacts/csvs/cv2024_training_dedup_le0.csv	2	le0_le2_extension/phase5_v4_le0_s2_n1.json
le0_s3	artifacts/csvs/cv2024_training_dedup_le0.csv	3	le0_le2_extension/phase5_v4_le0_s3_n1.json
le0_s4	artifacts/csvs/cv2024_training_dedup_le0.csv	4	le0_le2_extension/phase5_v4_le0_s4_n1.json
le0_s5	artifacts/csvs/cv2024_training_dedup_le0.csv	5	le0_le2_extension/phase5_v4_le0_s5_n1.json
le0_s6	artifacts/csvs/cv2024_training_dedup_le0.csv	6	le0_le2_extension/phase5_v4_le0_s6_n1.json
le0_s7	artifacts/csvs/cv2024_training_dedup_le0.csv	7	le0_le2_extension/phase5_v4_le0_s7_n1.json
le0_s8	artifacts/csvs/cv2024_training_dedup_le0.csv	8	le0_le2_extension/phase5_v4_le0_s8_n1.json
le0_s9	artifacts/csvs/cv2024_training_dedup_le0.csv	9	le0_le2_extension/phase5_v4_le0_s9_n1.json
le2_s0	artifacts/csvs/cv2024_training_dedup_le2.csv	0	le0_le2_extension/phase5_v4_le2_s0_n1.json
le2_s1	artifacts/csvs/cv2024_training_dedup_le2.csv	1	le0_le2_extension/phase5_v4_le2_s1_n1.json
le2_s2	artifacts/csvs/cv2024_training_dedup_le2.csv	2	le0_le2_extension/phase5_v4_le2_s2_n1.json
le2_s3	artifacts/csvs/cv2024_training_dedup_le2.csv	3	le0_le2_extension/phase5_v4_le2_s3_n1.json
le2_s4	artifacts/csvs/cv2024_training_dedup_le2.csv	4	le0_le2_extension/phase5_v4_le2_s4_n1.json
le2_s5	artifacts/csvs/cv2024_training_dedup_le2.csv	5	le0_le2_extension/phase5_v4_le2_s5_n1.json
le2_s6	artifacts/csvs/cv2024_training_dedup_le2.csv	6	le0_le2_extension/phase5_v4_le2_s6_n1.json
le2_s7	artifacts/csvs/cv2024_training_dedup_le2.csv	7	le0_le2_extension/phase5_v4_le2_s7_n1.json
le2_s8	artifacts/csvs/cv2024_training_dedup_le2.csv	8	le0_le2_extension/phase5_v4_le2_s8_n1.json
le2_s9	artifacts/csvs/cv2024_training_dedup_le2.csv	9	le0_le2_extension/phase5_v4_le2_s9_n1.json
TASKS
mv "$TASK_TMP" "$TASK_FILE"

validate_json() {
  local path="$1"
  local seed="$2"
  local train_csv="$3"
  "$PY" -B - "$path" "$seed" "$train_csv" "$DEDUP_VAL_CSV" <<'PY'
import json
import sys

path, seed, train_csv, dedup_val_csv = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
with open(path) as f:
    data = json.load(f)
assert data["args"]["train_csv"] == train_csv, (data["args"]["train_csv"], train_csv)
assert data["args"]["dedup_val_csv"] == dedup_val_csv, (data["args"]["dedup_val_csv"], dedup_val_csv)
runs = data.get("runs", [])
assert len(runs) == 1, len(runs)
assert int(runs[0]["seed"]) == seed, runs[0].get("seed")
last = runs[0]["last"]
for split in ("orig_val", "dedup_val", "kvasir_s1"):
    assert "bal_acc" in last[split], split
PY
}

run_task() {
  local task_id="$1"
  local train_csv="$2"
  local seed="$3"
  local output_json="$4"
  local out_path="$ROOT/results/$output_json"
  local log_path="$LOG_DIR/${task_id}.${WORKER_ID}.log"

  if [[ -s "$out_path" ]] && validate_json "$out_path" "$seed" "$train_csv"; then
    echo "[$WORKER_ID] SKIP valid $task_id"
    touch "$STATE_DIR/done/$task_id"
    return 0
  fi

  echo "[$WORKER_ID] START $task_id seed=$seed gpu=${CUDA_VISIBLE_DEVICES:-unset} $(date)"
  "$PY" -B "$RUNNER" \
    --train_csv "$train_csv" \
    --orig_val_csv "$ORIG_VAL_CSV" \
    --dedup_val_csv "$DEDUP_VAL_CSV" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --seeds "$seed" \
    --output "$output_json" \
    > "$log_path" 2>&1
  validate_json "$out_path" "$seed" "$train_csv"
  touch "$STATE_DIR/done/$task_id"
  echo "[$WORKER_ID] DONE  $task_id $(date)"
}

claim_one() {
  while IFS=$'\t' read -r task_id train_csv seed output_json; do
    [[ -n "${task_id:-}" ]] || continue
    [[ ! -e "$STATE_DIR/done/$task_id" ]] || continue
    [[ ! -e "$STATE_DIR/failed/$task_id" ]] || continue
    if mkdir "$STATE_DIR/claims/${task_id}.lock" 2>/dev/null; then
      echo "$WORKER_ID host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} $(date)" \
        > "$STATE_DIR/claims/${task_id}.lock/owner"
      if run_task "$task_id" "$train_csv" "$seed" "$output_json"; then
        return 0
      fi
      touch "$STATE_DIR/failed/$task_id"
      echo "[$WORKER_ID] FAILED $task_id; see $LOG_DIR/${task_id}.${WORKER_ID}.log" >&2
      return 3
    fi
  done < "$TASK_FILE"
  return 1
}

echo "[$WORKER_ID] le0/le2 n10 extension worker starting host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
"$PY" -B - <<'PY'
import torch
print("[torch]", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), flush=True)
if torch.cuda.is_available():
    print("[gpu]", torch.cuda.get_device_name(0), flush=True)
PY

while claim_one; do :; done

done_count=$(find "$STATE_DIR/done" -maxdepth 1 -type f | wc -l)
failed_count=$(find "$STATE_DIR/failed" -maxdepth 1 -type f | wc -l)
echo "[$WORKER_ID] worker finished done=$done_count/20 failed=$failed_count $(date)"
if [[ "$failed_count" != "0" ]]; then
  exit 4
fi
