"""M7 Day 2-3: Re-score 25 valid CV2024 validation submissions on
Kvasir-origin-removed public-pool subsets (le6, le6_plus_internal)
using the organizers' own generate_metrics_report function unchanged.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ARTIFACT_ROOT = Path(
    os.environ.get("CAPSULE_ARTIFACT_ROOT", Path(__file__).resolve().parents[1])
)
ORG_SCRIPT_DIR = Path(
    os.environ.get("CV2024_RESULTS_DIR", ARTIFACT_ROOT / "external/cv2024_repo/Results")
)
ORG_SCRIPT = ORG_SCRIPT_DIR / "gen_metrics_report_val_train.py"
if not ORG_SCRIPT.exists():
    raise SystemExit(
        "CV2024 organizer Results directory not found. Set CV2024_RESULTS_DIR "
        "to the directory containing gen_metrics_report_val_train.py, "
        "validation_data.xlsx, and submitted_excel_files/validation."
    )
sys.path.insert(0, str(ORG_SCRIPT_DIR))
from gen_metrics_report_val_train import (  # noqa: E402
    VALID_CLASSES,
    generate_metrics_report,
    sanity_check,
)

GT_PATH = ORG_SCRIPT_DIR / "validation_data.xlsx"
PRED_DIR = ORG_SCRIPT_DIR / "submitted_excel_files/validation"
DAY1_PATH = ARTIFACT_ROOT / "results/cv2024_rescored_orig.json"
SUBSET_CSVS = {
    "le6": ARTIFACT_ROOT / "artifacts/csvs/cv2024_validation_dedup_le6.csv",
    "le6_plus_internal": ARTIFACT_ROOT
    / "artifacts/csvs/cv2024_validation_le6_plus_internal.csv",
}
EXPECTED_ROWS = {"le6": 4551, "le6_plus_internal": 4326}


def main() -> None:
    gt_full = pd.read_excel(GT_PATH)
    assert len(gt_full) == 16132, f"Expected 16132 gt rows, got {len(gt_full)}"

    day1 = json.load(open(DAY1_PATH))
    valid_teams = [t for t in day1["teams"] if t["valid_for_rescore"]]
    assert len(valid_teams) == 25, f"Expected 25 valid, got {len(valid_teams)}"

    for subset, csv_path in SUBSET_CSVS.items():
        print(f"\n=== Re-scoring on {subset} ===")
        sub_df = pd.read_csv(csv_path)
        assert len(sub_df) == EXPECTED_ROWS[subset]
        sub_paths = set(sub_df["image_path"])

        gt_sub = (
            gt_full[gt_full["image_path"].isin(sub_paths)]
            .copy()
            .reset_index(drop=True)
        )
        assert len(gt_sub) == EXPECTED_ROWS[subset], (
            f"gt filter to {subset}: got {len(gt_sub)} != {EXPECTED_ROWS[subset]}"
        )

        team_results = []
        for team_obj in valid_teams:
            team = team_obj["team"]
            file = team_obj["file"]
            pred_path = PRED_DIR / f"{file}.xlsx"
            if not pred_path.exists():
                print(f"  WARN: {pred_path} missing, skip")
                continue

            pred_df_full = pd.read_excel(pred_path)
            # Filter pred to subset paths first (sanity_check requires
            # exact set equality between gt and pred image_paths).
            pred_df = pred_df_full[
                pred_df_full["image_path"].isin(sub_paths)
            ].copy().reset_index(drop=True)

            ok, aligned_pred = sanity_check(gt_sub, pred_df)
            if not ok:
                print(f"  {team}: sanity_check FAILED")
                continue

            y_true = gt_sub[VALID_CLASSES].to_numpy()
            y_pred = aligned_pred[VALID_CLASSES].to_numpy()
            metrics = generate_metrics_report(y_true, y_pred)

            mean_auc = metrics["mean_auc"]
            bal = metrics["balanced_accuracy"]
            combined = (mean_auc + bal) / 2

            team_results.append(
                {
                    "team": team,
                    "file": file,
                    "mean_auc_subset": float(mean_auc),
                    "balanced_accuracy_subset": float(bal),
                    "combined_subset": float(combined),
                    "mean_auc_orig": float(team_obj["mean_auc_organizer"]),
                    "balanced_accuracy_orig": float(
                        team_obj["balanced_accuracy_organizer"]
                    ),
                    "combined_orig": float(team_obj["combined_organizer"]),
                    "rank_orig_val": int(team_obj["rank_orig_val"]),
                    "delta_combined": float(combined - team_obj["combined_organizer"]),
                    "delta_mean_auc": float(mean_auc - team_obj["mean_auc_organizer"]),
                    "delta_balanced_accuracy": float(
                        bal - team_obj["balanced_accuracy_organizer"]
                    ),
                    "status": team_obj["status"],
                }
            )
            print(
                f"  {team:30s} AUC={mean_auc:.4f} BA={bal:.4f} "
                f"comb={combined:.4f} d={combined - team_obj['combined_organizer']:+.4f}"
            )

        sorted_by_subset = sorted(
            team_results, key=lambda x: x["combined_subset"], reverse=True
        )
        for r, t in enumerate(sorted_by_subset, start=1):
            t["rank_subset"] = r
            t["rank_shift"] = t["rank_orig_val"] - r

        deltas = [t["delta_combined"] for t in team_results]
        rank_shifts = [t["rank_shift"] for t in team_results]
        ranks_orig = [t["rank_orig_val"] for t in team_results]
        ranks_sub = [t["rank_subset"] for t in team_results]
        rho, _ = spearmanr(ranks_orig, ranks_sub)
        tau, _ = kendalltau(ranks_orig, ranks_sub)

        agg = {
            "mean_delta_combined": float(np.mean(deltas)),
            "median_delta_combined": float(np.median(deltas)),
            "max_delta_combined_drop": float(min(deltas)),
            "max_delta_combined_gain": float(max(deltas)),
            "n_rank_changed": int(sum(1 for s in rank_shifts if s != 0)),
            "n_rank_changed_by_at_least_3": int(
                sum(1 for s in rank_shifts if abs(s) >= 3)
            ),
            "max_rank_increase": int(max(rank_shifts)),
            "max_rank_decrease": int(min(rank_shifts)),
            "spearman_rho_orig_vs_subset": float(rho),
            "kendall_tau_orig_vs_subset": float(tau),
        }

        out = {
            "subset": subset,
            "n_validation_rows": int(len(gt_sub)),
            "n_teams": len(team_results),
            "teams": sorted(team_results, key=lambda x: x["rank_subset"]),
            "aggregate_stats": agg,
        }

        out_path = ARTIFACT_ROOT / f"results/cv2024_rescored_{subset}.json"
        tmp_path = out_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(out, f, indent=2)
        tmp_path.replace(out_path)

        print(f"  saved {out_path}")
        print(
            f"  agg: mean_d={agg['mean_delta_combined']:.4f} "
            f"median_d={agg['median_delta_combined']:.4f} "
            f"rank_changed={agg['n_rank_changed']}/{len(team_results)} "
            f"spearman={agg['spearman_rho_orig_vs_subset']:.3f}"
        )


if __name__ == "__main__":
    main()
