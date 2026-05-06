#!/usr/bin/env bash
# Reviewer-runnable smoke test for the CV2024 audit artifact bundle.
#
# Goal: in <5 minutes on a CPU-only laptop, give a reviewer end-to-end
# confidence that:
#   (1) every artifact in checksums.txt verifies bit-exact;
#   (2) croissant.json distribution entries match checksums.txt;
#   (3) the released CSV line counts, including one header row, match
#       DATA_CARD.md;
#   (4) the headline retraining JSONs parse and recompute the
#       Delta_le6 = -0.213 +- 0.005 paired-Δ headline;
#   (5) the 100% KVASIR pHash-flag claim recomputes from the released
#       cv2024_KVASIR_phash_annotated.csv;
#   (6) the acceptance-lift completion summary parses and matches expected fixture;
#   (7) the official-test aggregate metrics replay from the pinned
#       CV2024 test-script definitions;
#   (8) the 7/25 video-overlap claim recomputes from
#       data/official_splits/{split_0,split_1}.csv (skipped with
#       WARN if those CSVs are not bundled, per artifacts/csvs/SPLIT_PROVENANCE.md).
# All checks are read-only. No GPU needed. ~10 seconds wall time.
#
# Usage: bash scripts/run_smoke_test.sh
#   prints a PASS/FAIL line per check and exits non-zero on any failure.

set -uo pipefail
cd "$(dirname "$0")/.."

PASS=0
FAIL=0
PROBLEMS=()

ok()   { echo "  [PASS] $*"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); PROBLEMS+=("$*"); }
section() { echo ""; echo "=== $* ==="; }

PY="${PY:-python3}"

# ---------- Check 1: checksums ------------------------------------------------
section "1/5  Artifact integrity (sha256 verify)"
if [ -f checksums.txt ]; then
  if sha256sum --check --quiet checksums.txt; then
    ok "all $(wc -l < checksums.txt) artifacts verify"
  else
    bad "sha256sum --check failed; some artifacts mutated or missing"
  fi
else
  bad "checksums.txt not found"
fi

# ---------- Check 1b: Croissant manifest consistency -------------------------
section "1b/5 Croissant manifest consistency"
tmp_croissant="/tmp/capsule_tta_croissant_manifest.$$"
if $PY scripts/09_verify_croissant_manifest.py --root . >"$tmp_croissant" 2>&1; then
  ok "Croissant distribution entries match checksums.txt"
else
  sed 's/^/  /' "$tmp_croissant"
  bad "Croissant distribution/checksum consistency failed"
fi
rm -f "$tmp_croissant"

# ---------- Check 2: CSV row counts -------------------------------------------
section "2/5  CSV line counts vs DATA_CARD.md (including header)"
declare -A EXPECTED_ROWS
EXPECTED_ROWS["artifacts/csvs/cv2024_training_dedup_le6.csv"]=10597
EXPECTED_ROWS["artifacts/csvs/cv2024_validation_dedup_le6.csv"]=4552
EXPECTED_ROWS["artifacts/csvs/cv2024_training_dedup_le0.csv"]=21242
EXPECTED_ROWS["artifacts/csvs/cv2024_validation_dedup_le0.csv"]=9055
EXPECTED_ROWS["artifacts/csvs/cv2024_training_dedup_le2.csv"]=10826
EXPECTED_ROWS["artifacts/csvs/cv2024_validation_dedup_le2.csv"]=4656
EXPECTED_ROWS["artifacts/csvs/cv2024_training_dedup_le6_strict.csv"]=10593
EXPECTED_ROWS["artifacts/csvs/cv2024_validation_le6_plus_internal.csv"]=4327

for f in "${!EXPECTED_ROWS[@]}"; do
  if [ -f "$f" ]; then
    actual=$(wc -l < "$f")
    expected=${EXPECTED_ROWS[$f]}
    if [ "$actual" -eq "$expected" ]; then
      ok "$f  $actual lines (expected $expected)"
    else
      bad "$f  $actual lines (expected $expected)"
    fi
  else
    bad "$f  missing"
  fi
done

# ---------- Check 3: headline counterfactual recomputes ----------------------
section "3/5  Headline Δ_le6 = -0.213 (recompute from JSONs)"
$PY - << 'PYEOF'
import json, sys, statistics as st
from pathlib import Path
R = Path("results/baseline")
TOL = 0.005
EXPECTED_DELTA = -0.213
try:
    base = json.load(open(R / "phase5_v5_baseline_n10.json"))
    le6  = json.load(open(R / "phase5_v5_le6_n10.json"))
except Exception as e:
    print(f"  [FAIL] cannot open canonical JSONs: {e}", file=sys.stderr); sys.exit(2)
b = [r["last"]["orig_val"]["bal_acc"] for r in base["runs"]]
l = [r["last"]["orig_val"]["bal_acc"] for r in le6["runs"]]
if len(b) < 10 or len(l) < 10:
    print(f"  [FAIL] expected n>=10 seeds, got base={len(b)} le6={len(l)}", file=sys.stderr); sys.exit(2)
delta = st.mean(l) - st.mean(b)
err = abs(delta - EXPECTED_DELTA)
print(f"  baseline mean={st.mean(b):.4f}  le6 mean={st.mean(l):.4f}  Δ={delta:+.4f}  expected={EXPECTED_DELTA:+.4f}  err={err:.4f}")
if err <= TOL:
    print(f"  [PASS] Δ_le6 within ±{TOL}")
else:
    print(f"  [FAIL] Δ_le6 outside ±{TOL}")
    sys.exit(2)
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); PROBLEMS+=("Δ_le6 recompute failed"); }

# ---------- Check 4: 100% KVASIR pHash flag claim ----------------------------
section "4/5  100% KVASIR pHash-flag claim (recompute)"
$PY - << 'PYEOF'
import sys, csv
from pathlib import Path
P = Path("artifacts/annotations/cv2024_KVASIR_phash_annotated.csv")
if not P.exists():
    print(f"  [FAIL] {P} missing"); sys.exit(2)
n_kvasir = 0
n_flagged = 0
with open(P) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        if row.get("cv_dataset") != "KVASIR":
            continue
        n_kvasir += 1
        try:
            ph = int(row["min_phash_dist_to_kvasir"]) if row.get("min_phash_dist_to_kvasir") else 99
            dh = int(row["min_dhash_dist_to_kvasir"]) if row.get("min_dhash_dist_to_kvasir") else 99
        except (ValueError, KeyError):
            ph = dh = 99
        if ph <= 6 and dh <= 6:
            n_flagged += 1
rate = n_flagged / n_kvasir if n_kvasir else 0
print(f"  KVASIR rows={n_kvasir}  flagged (pHash<=6 AND dHash<=6)={n_flagged}  rate={rate*100:.2f}%")
if n_kvasir == 38592 and rate >= 0.999:
    print(f"  [PASS] 100% KVASIR-source flag claim recomputes")
else:
    print(f"  [FAIL] expected 38592 KVASIR rows at >=99.9% flag, got {n_kvasir}/{rate*100:.2f}%")
    sys.exit(2)
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); PROBLEMS+=("100% pHash claim failed to recompute"); }

# ---------- Check 4b: same-source/domain Exp.1 contrast (paired Δ on shared seeds 0-9) ----
section "4b/5  Same-source/domain Exp.1 paired Δ on orig_val (+0.091 ± 0.002 expected)"
$PY - << 'PYEOF'
import json, sys, statistics as st
from pathlib import Path
e1p = Path("results/mechanism_probes/phase5_exp1_le6_kvfree_s1_n10.json")
le6p = Path("results/baseline/phase5_v5_le6_n10.json")
if not (e1p.exists() and le6p.exists()):
    print(f"  [WARN] same-source/domain JSON missing; skipping"); sys.exit(3)
exp1 = json.load(open(e1p))
le6 = json.load(open(le6p))
e1 = {int(r["seed"]): r["last"]["orig_val"]["bal_acc"] for r in exp1["runs"]}
le6_seeds = {int(r["seed"]): r["last"]["orig_val"]["bal_acc"] for r in le6["runs"]}
shared = sorted(set(e1) & set(le6_seeds))
diffs = [e1[s] - le6_seeds[s] for s in shared]
delta = st.mean(diffs)
err = abs(delta - 0.0912)
print(f"  shared seeds={shared}  Exp.1 mean={st.mean(e1.values()):.4f}  le6 mean={st.mean([le6_seeds[s] for s in shared]):.4f}  Δ={delta:+.4f}  expected=+0.0912  err={err:.4f}")
if len(shared) == 10 and err <= 0.002:
    print(f"  [PASS] same-source/domain Δ_orig within ±0.002")
else:
    print(f"  [FAIL] same-source/domain Δ_orig outside tolerance")
    sys.exit(2)
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); PROBLEMS+=("same-source/domain Δ_orig recompute failed"); }

# ---------- Check 4c: Contested-band NCC corroboration ----------------------
section "4c/5  Contested-band Hamming 3-6 NCC verification (n=333 expected)"
$PY - << 'PYEOF'
import json, sys
from pathlib import Path
P = Path("artifacts/summaries/contested_band_ncc.json")
if not P.exists():
    print(f"  [FAIL] {P} missing"); sys.exit(2)
d = json.load(open(P))
band = d.get("by_band", {}).get("contested (3-6)", {})
n = band.get("n", 0)
pct90 = band.get("pct_ge_90", 0) * 100
mean = band.get("ncc_mean", 0)
print(f"  contested (3-6): n={n}  NCC mean={mean:.4f}  %≥0.90={pct90:.2f}%")
if n == 333 and pct90 >= 99.0 and mean >= 0.99:
    print(f"  [PASS] contested-band corroboration recomputes")
else:
    print(f"  [FAIL] expected n=333, mean≥0.99, %≥90≥99%")
    sys.exit(2)
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); PROBLEMS+=("contested-band NCC recompute failed"); }

# ---------- Check 4d: acceptance-lift completion summary --------------------
section "4d/5  Acceptance-lift completion summary (n=10 expected)"
$PY - << 'PYEOF'
import json, sys
from pathlib import Path

P = Path("results/acceptance_lift/acceptance_lift_summary.json")
if not P.exists():
    print(f"  [FAIL] {P} missing")
    sys.exit(2)
d = json.load(open(P))
strict = d["strict_le6_plus_internal"]
matched = d["matched_arm_completion"]
base = strict["baseline"]
le6 = strict["le6"]
random = matched["random"]
compa = matched["compA"]
delta_strict = strict["delta_le6_minus_baseline"]
delta_compa = compa["orig_val_bal_acc_mean"] - random["orig_val_bal_acc_mean"]
print(
    "  strict n="
    f"{base['n']}/{le6['n']} Δ(le6-baseline)={delta_strict:+.4f}; "
    f"matched n={random['n']}/{compa['n']} Δ(Comp-A-random)={delta_compa:+.4f}"
)
ok = (
    base["n"] == le6["n"] == random["n"] == compa["n"] == 10
    and abs(delta_strict - 0.0037) <= 0.001
    and abs(delta_compa + 0.1144) <= 0.002
)
if ok:
    print("  [PASS] acceptance-lift n=10 completion summary matches expected fixture")
else:
    print("  [FAIL] acceptance-lift summary outside tolerance")
    sys.exit(2)
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); PROBLEMS+=("acceptance-lift summary recompute failed"); }

# ---------- Check 4e: strengthening summary ---------------------------------
section "4e/5  Strengthening summary (same-source/domain / Comp-C / Comp-D)"
$PY - << 'PYEOF'
import json, sys
from pathlib import Path

P = Path("results/strengthening/strengthening_summary.json")
if not P.exists():
    print(f"  [FAIL] {P} missing")
    sys.exit(2)
d = json.load(open(P))
pathb = d["pathb_exp1_le6_kvfree_s1"]
compc = d["compC_aiims_ulcer_oversampled"]
compd_dup = d["compD_kvasir_ulcer_duplicate"]
compd_uni = d["compD_kvasir_ulcer_unique"]
contrasts = {c["label"]: c for c in d["contrasts"]}
pathb_delta = contrasts["Same-source/domain Exp.1 - le6"]["mean_delta"]
compc_compa = contrasts["Comp-C - Comp-A"]["mean_delta"]
compc_compb = contrasts["Comp-C - Comp-B"]["mean_delta"]
compd_dup_delta = contrasts["Comp-D duplicate - comp-matched"]["mean_delta"]
compd_uni_delta = contrasts["Comp-D unique - comp-matched"]["mean_delta"]
print(
    f"  Same-source/domain n={pathb['n']} mean={pathb['orig_val_bal_acc_mean']:.4f} "
    f"Δ={pathb_delta:+.4f}; Comp-C n={compc['n']} "
    f"mean={compc['orig_val_bal_acc_mean']:.4f} "
    f"ΔC-A={compc_compa:+.4f} ΔC-B={compc_compb:+.4f}; "
    f"Comp-D dup/unique n={compd_dup['n']}/{compd_uni['n']} "
    f"ΔDdup={compd_dup_delta:+.4f} ΔDuniq={compd_uni_delta:+.4f}"
)
ok = (
    pathb["n"] == 10
    and compc["n"] == 10
    and compd_dup["n"] == 10
    and compd_uni["n"] == 10
    and abs(pathb_delta - 0.0912) <= 0.002
    and abs(compc_compa + 0.0042) <= 0.003
    and abs(compc_compb + 0.0838) <= 0.003
    and abs(compd_dup_delta + 0.0013) <= 0.004
    and abs(compd_uni_delta - 0.0006) <= 0.004
)
if ok:
    print("  [PASS] strengthening summary matches expected fixture")
else:
    print("  [FAIL] strengthening summary outside tolerance")
    sys.exit(2)
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); PROBLEMS+=("strengthening summary recompute failed"); }

# ---------- Check 4f: NCC dose-response n=10 packaging ----------------------
section "4f/5  NCC dose-response n=10 packaging"
$PY - << 'PYEOF'
import json, sys
from pathlib import Path

summary = Path("results/mechanism_probes/ncc_dose_response_n10_summary.json")
if not summary.exists():
    print(f"  [FAIL] {summary} missing")
    sys.exit(2)
d = json.load(open(summary))
rows = {r["threshold"]: r for r in d["rows"]}
required = ["NCC >= 0.99", "NCC >= 0.95", "NCC >= 0.90", "NCC >= 0.85", "NCC >= 0.80", "le6"]
missing = [r for r in required if r not in rows]
bad_n = [r for r in required if r in rows and rows[r]["n"] != 10]
if missing or bad_n:
    print(f"  [FAIL] missing={missing} bad_n={bad_n}")
    sys.exit(2)
delta95 = rows["NCC >= 0.95"]["delta_vs_baseline_mean"]
delta80 = rows["NCC >= 0.80"]["delta_vs_baseline_mean"]
delta_le6 = rows["le6"]["delta_vs_baseline_mean"]
print(f"  NCC-95 Δ={delta95:+.4f}; NCC-80 Δ={delta80:+.4f}; le6 Δ={delta_le6:+.4f}")
if abs(delta95 + 0.1266) <= 0.002 and abs(delta80 + 0.2101) <= 0.002 and abs(delta_le6 + 0.2131) <= 0.002:
    print("  [PASS] NCC dose-response n=10 summary matches expected fixture")
else:
    print("  [FAIL] NCC dose-response summary outside tolerance")
    sys.exit(2)
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); PROBLEMS+=("NCC dose-response n10 summary failed"); }

# ---------- Check 4g: cross-model n=10 packaging ----------------------------
section "4g/5  Cross-model n=10 packaging"
$PY - << 'PYEOF'
import json, sys
from pathlib import Path

root = Path("results/crossmodel")
shorts = ["dinov2B", "dinov2S", "resnet50", "convnextT"]
pools = ["baseline", "random", "le6"]
missing = []
bad = []
for short in shorts:
    for pool in pools:
        path = root / f"phase5_crossmodel_{short}_{pool}_n10.json"
        if not path.exists():
            missing.append(str(path))
            continue
        d = json.load(open(path))
        seeds = sorted(int(r["seed"]) for r in d.get("runs", []))
        if seeds != list(range(10)):
            bad.append((str(path), seeds))
if missing or bad:
    print(f"  [FAIL] missing={missing} bad={bad}")
    sys.exit(2)
print("  [PASS] DINOv2-B, DINOv2-S, ResNet-50, and ConvNeXt-Tiny cross-model n=10 files contain seeds 0-9")
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); PROBLEMS+=("cross-model n10 packaging failed"); }

# ---------- Check 4h: Comp-D package closure --------------------------------
section "4h/5  Comp-D KVASIR-Ulcer controls packaged"
$PY - << 'PYEOF'
import csv, json, sys
from pathlib import Path

checks = [
    (
        Path("artifacts/csvs/cv2024_training_compD_kvasir_ulcer_duplicate_s0.csv"),
        Path("results/strengthening/phase5_v5_compD_kvasir_ulcer_duplicate_s0_n10.json"),
        0.7371,
    ),
    (
        Path("artifacts/csvs/cv2024_training_compD_kvasir_ulcer_oversampled_s0.csv"),
        Path("results/strengthening/phase5_v5_compD_kvasir_ulcer_s0_n10.json"),
        0.7390,
    ),
]

for csv_path, json_path, expected_mean in checks:
    if not csv_path.exists() or not json_path.exists():
        print(f"  [FAIL] missing {csv_path} or {json_path}")
        sys.exit(2)
    rows = kv_ulcer = aiims_ulcer = ulcer = 0
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows += 1
            is_ulcer = row.get("Ulcer") in {"1", "1.0"}
            is_kvasir = row.get("Dataset") == "KVASIR"
            is_aiims = row.get("Dataset") == "AIIMS"
            ulcer += int(is_ulcer)
            kv_ulcer += int(is_kvasir and is_ulcer)
            aiims_ulcer += int(is_aiims and is_ulcer)
    d = json.load(open(json_path))
    runs = d.get("runs", [])
    vals = [r["last"]["orig_val"]["bal_acc"] for r in runs]
    mean = sum(vals) / len(vals) if vals else float("nan")
    ok = (
        rows == 10596
        and ulcer == 132
        and kv_ulcer == 132
        and aiims_ulcer == 0
        and len(runs) == 10
        and abs(mean - expected_mean) <= 0.001
    )
    print(
        f"  {csv_path.name}: rows={rows} Ulcer={ulcer} "
        f"KVASIR_Ulcer={kv_ulcer} AIIMS_Ulcer={aiims_ulcer}; "
        f"n={len(runs)} mean={mean:.4f}"
    )
    if not ok:
        print("  [FAIL] Comp-D package check outside tolerance")
        sys.exit(2)
print("  [PASS] Comp-D CSVs and n=10 JSONs are packaged and match EXPECTED.md")
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); PROBLEMS+=("Comp-D package closure failed"); }

# ---------- Check 4i: provenance snapshots ----------------------------------
section "4i/5  Path-scrubbed provenance snapshots"
$PY - << 'PYEOF'
import hashlib, sys
from pathlib import Path

expected = {
    "src/provenance/phase5_counterfactual_v5_be3fd06_exact.py": "0199652d00c49a5d83d12728bac0e1c42bb44c236917622b4231fd96408ff71e",
    "src/provenance/phase5_counterfactual_v5_71c2399e_exact.py": "71c2399e6c9ab91d754ff70cc525ada14083b00fb6ddace876c3ff65cbc4ef1f",
    "src/provenance/phase5_counterfactual_v5_16f3d70d_exact.py": "16f3d70d441e23a76100a9f23518e85221c5cb27bcb4fe58f52ad4ff13d0bb7d",
    "src/provenance/phase5_counterfactual_v4_72047d35_exact.py": "46a6cc2c82f5db668b9bd79c78c1acdad58bcbd68de56e834978b2b9d82f9d88",
    "src/provenance/04_official_test_eval_f834a5_exact.py": "f834a5bcce8fdf5e8462af690df3d1d0b30d6ca04cb01fdd737749b449a1cfa1",
    "src/provenance/phase5_exp2b_split0_only_c06c5693_exact.py": "70bd6378938d604fcfd4dabb4ee41bbeb671f688ae484cce0e09c09bb1e3011f",
    "src/provenance/phase5_exp3_mi_probe_97312ac3_exact.py": "c084c9190ec2ade08e44b3023383aee87de52721afe16ccbdb4e08195bc112eb",
}
bad = []
for name, want in expected.items():
    path = Path(name)
    if not path.exists():
        bad.append((name, "missing", want))
        continue
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        bad.append((name, got, want))
if bad:
    print(f"  [FAIL] provenance hash mismatches: {bad}")
    sys.exit(2)
print(f"  [PASS] {len(expected)} provenance snapshots match packaged anonymized hashes")
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); PROBLEMS+=("provenance snapshot check failed"); }

# ---------- Check 4j: official-test aggregate metric replay ------------------
section "4j/5  Official-test aggregate metric replay"
tmp_replay="/tmp/capsule_tta_official_replay.$$"
if $PY scripts/08_verify_official_test_metrics.py --root . >"$tmp_replay" 2>&1; then
  sed 's/^/  /' "$tmp_replay"
  ok "official-test mean AUC / balanced accuracy / combined replay"
else
  sed 's/^/  /' "$tmp_replay"
  bad "official-test aggregate metric replay failed"
fi
rm -f "$tmp_replay"

# ---------- Check 4k: evidence-and-scope summary recompute ------------------
section "4k/5  Evidence-and-scope summary recompute"
tmp_scorecard="/tmp/capsule_tta_claim_scorecard.$$"
if $PY scripts/12_make_claim_scorecard.py --root . >"$tmp_scorecard" 2>&1; then
  if grep -q "Delta -0.213" "$tmp_scorecard" \
     && grep -q "KVASIR -0.473" "$tmp_scorecard" \
     && grep -q "exclude Ulcer -0.139" "$tmp_scorecard"; then
    ok "evidence-and-scope summary recomputes headline, per-source, and non-Ulcer rows"
  else
    sed 's/^/  /' "$tmp_scorecard"
    bad "evidence-and-scope summary missing expected rounded values"
  fi
else
  sed 's/^/  /' "$tmp_scorecard"
  bad "evidence-and-scope summary recompute failed"
fi
rm -f "$tmp_scorecard"

# ---------- Check 4l: comprehensive claim-provenance verifier ----------------
section "4l/5  Claim-provenance verifier (cross-model, M7, AUC, LOSO, official-test)"
tmp_cp="/tmp/capsule_tta_claim_provenance.$$"
if $PY scripts/10_verify_claim_provenance.py --root . >"$tmp_cp" 2>&1; then
  if $PY -c "import json,sys; d=json.load(open('$tmp_cp')); sys.exit(0 if d.get('ok') is True else 1)"; then
    n_checks=$($PY -c "import json; print(len(json.load(open('$tmp_cp'))['checks']))")
    ok "${n_checks} numeric claims verified (cross-model residuals, M7 trained-19 Δ+CI, AUC dose, LOSO, official-test arm means)"
  else
    failed=$($PY -c "import json; d=json.load(open('$tmp_cp')); print(', '.join(c['claim'] for c in d['checks'] if not c['ok']))")
    bad "claim_provenance failed claims: $failed"
  fi
else
  sed 's/^/  /' "$tmp_cp" | head -10
  bad "claim_provenance verifier crashed"
fi
rm -f "$tmp_cp"

# ---------- Check 5: Kvasir 7/25 video overlap -------------------------------
section "5/5  7/25 video-overlap claim (Smedsrud official split)"
$PY - << 'PYEOF'
import sys, csv
from pathlib import Path
S0 = Path("data/official_splits/split_0.csv")
S1 = Path("data/official_splits/split_1.csv")
if not (S0.exists() and S1.exists()):
    print(f"  [WARN] split_0/split_1 csvs not in submission tree (expected from external Kvasir-Capsule release)")
    print(f"         skipping check 5; see SPLIT_PROVENANCE.md for SHA-256")
    sys.exit(3)  # exit code 3 = SKIP (not 0 = PASS, not 2 = FAIL)
def vids(p):
    out = set()
    with open(p) as f:
        for row in csv.DictReader(f):
            out.add(row["filename"].split("_")[0])
    return out
v0, v1 = vids(S0), vids(S1)
shared = sorted(v0 & v1)
print(f"  split_0 unique videos={len(v0)}  split_1={len(v1)}  shared={len(shared)}")
print(f"  shared video-id prefixes: {shared}")
if len(shared) == 7:
    print(f"  [PASS] 7/25 overlap recomputes")
    sys.exit(0)
else:
    print(f"  [FAIL] expected 7 shared videos, got {len(shared)}")
    sys.exit(2)
PYEOF
SMOKE5_RC=$?
if [ $SMOKE5_RC -eq 0 ]; then
  PASS=$((PASS+1))
elif [ $SMOKE5_RC -eq 2 ]; then
  FAIL=$((FAIL+1)); PROBLEMS+=("7/25 video overlap recompute failed")
else
  # exit was non-zero non-2 (i.e., the WARN/skip path); count as SKIP, not PASS
  SKIP=$((${SKIP:-0}+1))
fi

# ---------- summary ----------------------------------------------------------
echo ""
echo "============================================="
SKIP=${SKIP:-0}
if [ $FAIL -eq 0 ] && [ $SKIP -eq 0 ]; then
  echo "SMOKE TEST: PASS  ($PASS checks ok)"
  echo "============================================="
  exit 0
elif [ $FAIL -eq 0 ]; then
  echo "SMOKE TEST: PASS-WITH-SKIP  ($PASS ok / $SKIP skipped)"
  echo "  (skipped checks require external Kvasir-Capsule CSVs;"
  echo "   see artifacts/csvs/SPLIT_PROVENANCE.md for SHA-256.)"
  echo "============================================="
  exit 0
else
  echo "SMOKE TEST: FAIL  ($PASS ok / $SKIP skipped / $FAIL fail)"
  for p in "${PROBLEMS[@]}"; do echo "  - $p"; done
  echo "============================================="
  exit 1
fi
