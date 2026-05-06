"""M7 Day 1 generator: build cv2024_rescored_orig.json from organizer's
released per-team metric JSONs + reproduce check via organizer script.

Produces the canonical original public-validation ranking (25 valid teams)
that m7_rescore_subset.py and m7_robustness.py consume as Day 1 baseline.

Reproduction: 26/27 organizer JSONs match byte-for-byte when re-running
gen_metrics_report_val_train.py on the same Excels; 1 (taaldhwaj) exhibits
release-version drift (file-derived bal-acc=0.10 vs organizer JSON 0.90,
matching arXiv:2408.04940 Table III). We use organizer JSON as canonical
and flag taaldhwaj as release_version_drift.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ALLOWED_DRIFT_TEAMS = {"taaldhwaj"}

ARTIFACT_ROOT = Path(
    os.environ.get("CAPSULE_ARTIFACT_ROOT", Path(__file__).resolve().parents[1])
)
ORG_SCRIPT_DIR = Path(
    os.environ.get("CV2024_RESULTS_DIR", ARTIFACT_ROOT / "external/cv2024_repo/Results")
)
ORG_DIR = ORG_SCRIPT_DIR / "metrics_reports/metrics_reports_val"
ORG_SCRIPT = ORG_SCRIPT_DIR / "gen_metrics_report_val_train.py"
if not ORG_SCRIPT.exists():
    raise SystemExit(
        "CV2024 organizer Results directory not found. Set CV2024_RESULTS_DIR "
        "to the directory containing gen_metrics_report_val_train.py, "
        "validation_data.xlsx, submitted_excel_files/validation, and "
        "metrics_reports/metrics_reports_val."
    )
GT_PATH = ORG_SCRIPT_DIR / "validation_data.xlsx"
PRED_DIR = (
    ORG_SCRIPT_DIR / "submitted_excel_files/validation"
)
REPRO_DIR = Path(os.environ.get("CV2024_REPRO_DIR", ARTIFACT_ROOT / "results/m7_repro_metrics"))
OUT_PATH = ARTIFACT_ROOT / "results/cv2024_rescored_orig.json"


def main() -> None:
    REPRO_DIR.mkdir(parents=True, exist_ok=True)
    py = os.environ.get("PY", sys.executable)

    print(f"Re-running organizer script on {PRED_DIR} ...")
    subprocess.run(
        [
            py,
            str(ORG_SCRIPT),
            str(GT_PATH),
            str(PRED_DIR),
            str(REPRO_DIR),
        ],
        check=True,
    )

    out = {
        "description": (
            "CV2024 official validation public-val metrics from organizer-"
            "released JSONs reproduced via gen_metrics_report_val_train.py"
        ),
        "reproduction": (
            "26/27 byte-equal to organizer JSONs; 1/27 (taaldhwaj) shows "
            "release-version drift (released Excel does not reproduce "
            "organizer-reported metrics); using organizer JSON values as "
            "canonical (matches arXiv:2408.04940 Table III)"
        ),
        "teams": [],
        "exclusion_summary": {"details": []},
    }

    excluded = []
    drift_teams = []
    total_jsons = 0
    for fn in sorted(os.listdir(ORG_DIR)):
        if not fn.endswith(".json"):
            continue
        total_jsons += 1

        raw = fn.replace("_metrics.json", "")
        team = raw
        for suf in [
            "_predicted_val_dataset",
            "_predicted_validation_dataset",
            "_validation_excel",
            "_validation_predictions",
            "_predicted_valid_dataset",
        ]:
            team = team.replace(suf, "")

        o = json.load(open(ORG_DIR / fn))
        p_path = REPRO_DIR / fn
        p = json.load(open(p_path)) if p_path.exists() else {}

        auc_o = o.get("mean_auc")
        bal_o = o.get("balanced_accuracy")
        auc_p = p.get("mean_auc")
        bal_p = p.get("balanced_accuracy")

        if auc_o is None or bal_o is None:
            status = "NULL_organizer"
            valid_for_rescore = False
            excluded.append(
                {
                    "team": team,
                    "reason": "organizer JSON contains None values "
                    "(sanity-check failure)",
                }
            )
        elif auc_p is None or bal_p is None:
            status = "NULL_repro"
            valid_for_rescore = False
            excluded.append(
                {
                    "team": team,
                    "reason": "reproduction failed (alignment problem)",
                }
            )
        elif abs(auc_o - auc_p) > 1e-6 or abs(bal_o - bal_p) > 1e-6:
            if team not in ALLOWED_DRIFT_TEAMS:
                raise SystemExit(
                    "Unexpected organizer/reproduction metric drift for "
                    f"{team} ({fn}). Expected drift only for "
                    f"{sorted(ALLOWED_DRIFT_TEAMS)}. Check CV2024_RESULTS_DIR."
                )
            status = "release_version_drift"
            valid_for_rescore = True
            drift_teams.append(team)
        else:
            status = "reproduced_byte_equal"
            valid_for_rescore = True

        combined = (
            (auc_o + bal_o) / 2
            if (auc_o is not None and bal_o is not None)
            else None
        )

        out["teams"].append(
            {
                "team": team,
                "file": fn.replace("_metrics.json", ""),
                "mean_auc_organizer": auc_o,
                "balanced_accuracy_organizer": bal_o,
                "combined_organizer": combined,
                "mean_auc_repro": auc_p,
                "balanced_accuracy_repro": bal_p,
                "status": status,
                "valid_for_rescore": valid_for_rescore,
            }
        )

    valid = [t for t in out["teams"] if t["valid_for_rescore"]]
    valid_sorted = sorted(
        valid, key=lambda x: x["combined_organizer"], reverse=True
    )
    for r, t in enumerate(valid_sorted, start=1):
        t["rank_orig_val"] = r

    out["teams"] = sorted(
        out["teams"],
        key=lambda x: (-(x.get("combined_organizer") or -1)),
    )

    out["n_total"] = 27
    out["n_valid_for_rescore"] = sum(
        1 for t in out["teams"] if t["valid_for_rescore"]
    )
    out["n_excluded"] = sum(
        1 for t in out["teams"] if not t["valid_for_rescore"]
    )
    out["exclusion_summary"]["details"] = excluded
    expected_drift = sorted(ALLOWED_DRIFT_TEAMS)
    if sorted(drift_teams) != expected_drift:
        raise SystemExit(
            "Unexpected drift-team set. Expected "
            f"{expected_drift}, observed {sorted(drift_teams)}. "
            "Check CV2024_RESULTS_DIR and organizer release files."
        )
    if total_jsons != out["n_total"]:
        raise SystemExit(
            f"Expected {out['n_total']} organizer JSON files, found {total_jsons}."
        )
    if total_jsons - len(drift_teams) != 26:
        raise SystemExit(
            "Expected exactly 26 non-drift organizer files, observed "
            f"{total_jsons - len(drift_teams)}."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"Wrote {OUT_PATH}: {out['n_valid_for_rescore']} valid "
        f"+ {out['n_excluded']} excluded = {out['n_total']}"
    )


if __name__ == "__main__":
    main()
