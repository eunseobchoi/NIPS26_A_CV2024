#!/usr/bin/env bash
set -euo pipefail

WORKER_ID="${1:?worker id required}"
TASK_FILE="${2:?task file required}"
ROOT="${CAPSULE_ROOT:-$(pwd)}"
PY="${CAPSULE_PYTHON:-python3}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-128}"

cd "$ROOT"
export CAPSULE_ROOT="$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

ORIG_VAL_CSV="${CV2024_VAL_XLSX:-$ROOT/data/cv2024/Dataset/validation/validation_data.xlsx}"

if [[ -f "$ROOT/src/counterfactual/phase5_counterfactual_v5.py" ]]; then
  RUNNER="$ROOT/src/counterfactual/phase5_counterfactual_v5.py"
elif [[ -f "$ROOT/phase5_counterfactual_v5.py" ]]; then
  RUNNER="$ROOT/phase5_counterfactual_v5.py"
else
  echo "ERROR: cannot find phase5_counterfactual_v5.py under $ROOT" >&2
  exit 2
fi

mkdir -p logs/acceptance_lift results/acceptance_lift \
  .acceptance_lift_state/{claims,done,failed}

log() {
  echo "[$WORKER_ID] $*"
}

validate_json() {
  local path="$1"
  local seed="$2"
  local expected_train="$3"
  local expected_eval="$4"
  "$PY" - "$path" "$seed" "$expected_train" "$expected_eval" <<'PY'
import json
import sys

path = sys.argv[1]
seed = int(sys.argv[2])
expected_train = sys.argv[3]
expected_eval = sys.argv[4]
try:
    data = json.load(open(path))
    args = data.get("args", {})
    assert args.get("train_csv") == expected_train, (args.get("train_csv"), expected_train)
    assert args.get("dedup_val_csv") == expected_eval, (args.get("dedup_val_csv"), expected_eval)
    runs = data.get("runs", [])
    assert len(runs) == 1, len(runs)
    assert int(runs[0]["seed"]) == seed, runs[0].get("seed")
    last = runs[0]["last"]
    for key in ("orig_val", "dedup_val", "kvasir_s1"):
        assert "bal_acc" in last[key], key
except Exception as exc:
    print(f"invalid {path}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

run_task() {
  local task_id="$1"
  local arm="$2"
  local seed="$3"
  local train_csv="$4"
  local eval_csv="$5"
  local output_json="$6"
  local out_path="results/${output_json}"
  local out_log="logs/acceptance_lift/${task_id}.${WORKER_ID}.log"

  mkdir -p "$(dirname "$out_path")" "$(dirname "$out_log")"

  if [[ -s "$out_path" ]] && validate_json "$out_path" "$seed" "$train_csv" "$eval_csv"; then
    log "SKIP existing valid $task_id -> $out_path"
    touch ".acceptance_lift_state/done/${task_id}"
    return 0
  fi

  log "START $task_id arm=$arm seed=$seed gpu=${CUDA_VISIBLE_DEVICES:-unset} $(date)"
  log "  train=$train_csv"
  log "  eval =$eval_csv"
  "$PY" -B "$RUNNER" \
    --train_csv "$train_csv" \
    --orig_val_csv "$ORIG_VAL_CSV" \
    --dedup_val_csv "$eval_csv" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --seeds "$seed" \
    --output "$output_json" \
    > "$out_log" 2>&1

  validate_json "$out_path" "$seed" "$train_csv" "$eval_csv"
  touch ".acceptance_lift_state/done/${task_id}"
  log "DONE  $task_id $(date)"
}

claim_and_run_one() {
  while IFS=$'\t' read -r task_id arm seed train_csv eval_csv output_json; do
    [[ -n "${task_id:-}" ]] || continue
    [[ "${task_id:0:1}" != "#" ]] || continue
    [[ ! -e ".acceptance_lift_state/done/${task_id}" ]] || continue
    [[ ! -e ".acceptance_lift_state/failed/${task_id}" ]] || continue
    if mkdir ".acceptance_lift_state/claims/${task_id}.lock" 2>/dev/null; then
      echo "$WORKER_ID $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} $(date)" \
        > ".acceptance_lift_state/claims/${task_id}.lock/owner"
      if run_task "$task_id" "$arm" "$seed" "$train_csv" "$eval_csv" "$output_json"; then
        return 0
      fi
      touch ".acceptance_lift_state/failed/${task_id}"
      log "FAILED $task_id; see logs/acceptance_lift/${task_id}.${WORKER_ID}.log"
      return 3
    fi
  done < "$TASK_FILE"
  return 1
}

task_count() {
  grep -cv '^[[:space:]]*#' "$TASK_FILE"
}

log "worker starting host=$(hostname) root=$ROOT cuda=${CUDA_VISIBLE_DEVICES:-unset}"
log "runner=$RUNNER"
"$PY" -B - <<'PY'
import torch
print("[torch]", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), flush=True)
if torch.cuda.is_available():
    print("[gpu]", torch.cuda.get_device_name(0), flush=True)
PY

while claim_and_run_one; do :; done

done_count=0
failed_count=0
while IFS=$'\t' read -r task_id _; do
  [[ -n "${task_id:-}" ]] || continue
  [[ "${task_id:0:1}" != "#" ]] || continue
  [[ -e ".acceptance_lift_state/done/${task_id}" ]] && done_count=$((done_count + 1))
  [[ -e ".acceptance_lift_state/failed/${task_id}" ]] && failed_count=$((failed_count + 1))
done < "$TASK_FILE"

log "worker finished done=$done_count/$(task_count) failed=$failed_count $(date)"
if [[ "$failed_count" != "0" ]]; then
  exit 4
fi
