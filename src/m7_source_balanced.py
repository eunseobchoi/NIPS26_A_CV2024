"""Source-balanced sensitivity analysis for CV2024 public-val re-scoring.

The organizer metric is row-weighted over the public validation pool, which is
72% KVASIR.  This script computes a source-balanced sensitivity check: source-
balanced balanced accuracy and a present-class AUC variant for each team's
released validation predictions on the original pool and the Kvasir-free public
subsets.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ARTIFACT_ROOT = Path(os.environ.get("CAPSULE_ARTIFACT_ROOT", Path(__file__).resolve().parents[1]))
ORG_SCRIPT_DIR = Path(
    os.environ.get("CV2024_RESULTS_DIR", ARTIFACT_ROOT / "external/cv2024_repo/Results")
)
if not (ORG_SCRIPT_DIR / "gen_metrics_report_val_train.py").exists():
    raise SystemExit(
        "CV2024_RESULTS_DIR must point to the organizer Results directory "
        f"(missing {ORG_SCRIPT_DIR / 'gen_metrics_report_val_train.py'})"
    )
sys.path.insert(0, str(ORG_SCRIPT_DIR))
from gen_metrics_report_val_train import VALID_CLASSES, generate_metrics_report, sanity_check  # noqa: E402

GT_PATH = ORG_SCRIPT_DIR / "validation_data.xlsx"
PRED_DIR = ORG_SCRIPT_DIR / "submitted_excel_files/validation"
DAY1_PATH = ARTIFACT_ROOT / "results/cv2024_rescored_orig.json"
SUBSETS = {
    "le6": ARTIFACT_ROOT / "artifacts/csvs/cv2024_validation_dedup_le6.csv",
    "le6_plus_internal": ARTIFACT_ROOT / "artifacts/csvs/cv2024_validation_le6_plus_internal.csv",
}

TAALDHWAJ = "taaldhwaj"
UNTRAINED = {
    "DeepScope_Innovators",
    "Deep_Learners",
    "EndoAI",
    "ViFo Tech",
    "BotBotBot",
}
STEM = "STEM sisters"
GROUPS = {
    "all_25": lambda team: True,
    "trained_19": lambda team: team != TAALDHWAJ and team not in UNTRAINED,
    "trained_no_stem_18": lambda team: (
        team != TAALDHWAJ and team not in UNTRAINED and team != STEM
    ),
}


def robust_present_class_auc(y_true_onehot, y_score):
    vals = []
    for i in range(y_true_onehot.shape[1]):
        y = y_true_onehot[:, i]
        if y.min() == y.max():
            continue
        vals.append(float(roc_auc_score(y, y_score[:, i])))
    return float(np.mean(vals)) if vals else None


def score_rows(gt_rows, pred_full):
    pred_rows = pred_full[pred_full["image_path"].isin(set(gt_rows["image_path"]))].copy()
    pred_rows = pred_rows.reset_index(drop=True)
    ok, aligned = sanity_check(gt_rows.reset_index(drop=True), pred_rows)
    if not ok:
        return None

    y_true = gt_rows[VALID_CLASSES].to_numpy()
    y_pred = aligned[VALID_CLASSES].to_numpy()
    official_like = generate_metrics_report(y_true, y_pred)

    per_source = {}
    for source, src_gt in gt_rows.groupby("Dataset", sort=True):
        src_pred = aligned.loc[src_gt.index]
        y_s = src_gt[VALID_CLASSES].to_numpy()
        p_s = src_pred[VALID_CLASSES].to_numpy()
        true_label = np.argmax(y_s, axis=1)
        pred_label = np.argmax(p_s, axis=1)
        bal = float(balanced_accuracy_score(true_label, pred_label))
        auc = robust_present_class_auc(y_s, p_s)
        per_source[source] = {
            "n": int(len(src_gt)),
            "balanced_accuracy": bal,
            "mean_auc_present_classes": auc,
            "combined_present_auc": None if auc is None else float((bal + auc) / 2),
            "n_present_classes": int(len(set(true_label.tolist()))),
        }

    sources = list(per_source.values())
    source_balanced_bal = float(np.mean([s["balanced_accuracy"] for s in sources]))
    source_aucs = [s["mean_auc_present_classes"] for s in sources if s["mean_auc_present_classes"] is not None]
    source_combined = [s["combined_present_auc"] for s in sources if s["combined_present_auc"] is not None]

    return {
        "row_weighted_mean_auc": float(official_like["mean_auc"]),
        "row_weighted_balanced_accuracy": float(official_like["balanced_accuracy"]),
        "row_weighted_combined": float(
            (official_like["mean_auc"] + official_like["balanced_accuracy"]) / 2
        ),
        "source_balanced_balanced_accuracy": source_balanced_bal,
        "source_balanced_mean_auc_present_classes": float(np.mean(source_aucs)) if source_aucs else None,
        "source_balanced_combined_present_auc": float(np.mean(source_combined)) if source_combined else None,
        "per_source": per_source,
    }


def bootstrap_mean_ci(delta, rng, n_iter=10000):
    if len(delta) == 0:
        return None
    samples = rng.choice(delta, size=(n_iter, len(delta)), replace=True).mean(axis=1)
    return {
        "n_iter": int(n_iter),
        "method": "paired percentile bootstrap over teams",
        "mean_delta_bootstrap": float(np.mean(samples)),
        "ci95_lo": float(np.quantile(samples, 0.025)),
        "ci95_hi": float(np.quantile(samples, 0.975)),
    }


def aggregate(teams, metric, rng):
    orig = np.array([t[f"orig_{metric}"] for t in teams], dtype=float)
    sub = np.array([t[f"subset_{metric}"] for t in teams], dtype=float)
    delta = sub - orig
    ranks_orig = pd.Series(orig).rank(ascending=False, method="average").to_numpy()
    ranks_sub = pd.Series(sub).rank(ascending=False, method="average").to_numpy()
    rho, _ = spearmanr(ranks_orig, ranks_sub)
    tau, _ = kendalltau(ranks_orig, ranks_sub)
    try:
        _, p_less = wilcoxon(delta, alternative="less")
    except ValueError:
        p_less = None
    boot = bootstrap_mean_ci(delta, rng)
    out = {
        "n": int(len(teams)),
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "sd_delta": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
        "wilcoxon_p_less": None if p_less is None else float(p_less),
        "spearman_rho_ranks": float(rho),
        "kendall_tau_ranks": float(tau),
        "n_rank_changed": int(np.sum(np.abs(ranks_sub - ranks_orig) > 1e-9)),
        "median_abs_rank_shift": float(np.median(np.abs(ranks_sub - ranks_orig))),
    }
    if boot is not None:
        out.update(boot)
    return out


def main():
    rng = np.random.default_rng(seed=42)
    gt_full = pd.read_excel(GT_PATH)
    day1 = json.load(open(DAY1_PATH))
    valid = [t for t in day1["teams"] if t["valid_for_rescore"]]

    pred_cache = {
        t["file"]: pd.read_excel(PRED_DIR / f"{t['file']}.xlsx")
        for t in valid
    }

    output = {
        "description": (
            "Source-balanced sensitivity analysis. The official challenge metric is "
            "row-weighted; source-balanced metrics average per-source scores and use "
            "AUC only over classes present in each source. This is a stress-test "
            "estimand, not a replacement official ranking."
        ),
        "sources_full": gt_full["Dataset"].value_counts().to_dict(),
        "subsets": {},
    }

    for subset_name, subset_csv in SUBSETS.items():
        sub_paths = set(pd.read_csv(subset_csv)["image_path"])
        gt_subset = gt_full[gt_full["image_path"].isin(sub_paths)].copy()
        gt_subset = gt_subset.reset_index(drop=True)
        teams = []
        for t in valid:
            pred_full = pred_cache[t["file"]]
            orig_scores = score_rows(gt_full.copy(), pred_full)
            subset_scores = score_rows(gt_subset.copy(), pred_full)
            if orig_scores is None or subset_scores is None:
                continue
            rec = {
                "team": t["team"],
                "file": t["file"],
                "status": t["status"],
            }
            for metric in [
                "row_weighted_combined",
                "source_balanced_balanced_accuracy",
                "source_balanced_combined_present_auc",
            ]:
                rec[f"orig_{metric}"] = orig_scores[metric]
                rec[f"subset_{metric}"] = subset_scores[metric]
                rec[f"delta_{metric}"] = subset_scores[metric] - orig_scores[metric]
            teams.append(rec)

        metrics = [
            "row_weighted_combined",
            "source_balanced_balanced_accuracy",
            "source_balanced_combined_present_auc",
        ]
        aggregates = {}
        for metric in metrics:
            aggregates[metric] = {}
            for group_name, filt in GROUPS.items():
                group = [t for t in teams if filt(t["team"])]
                aggregates[metric][group_name] = aggregate(group, metric, rng)

        output["subsets"][subset_name] = {
            "n_rows": int(len(gt_subset)),
            "sources": gt_subset["Dataset"].value_counts().to_dict(),
            "teams": teams,
            "aggregates": aggregates,
        }

    out_path = ARTIFACT_ROOT / "results/cv2024_source_balanced_rescore.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {out_path}")
    for subset_name, ss in output["subsets"].items():
        print(f"\n{subset_name}")
        for metric, groups in ss["aggregates"].items():
            r = groups["trained_19"]
            print(
                f"  {metric:42s} trained_19 "
                f"mean_delta={r['mean_delta']:+.4f} "
                f"rho={r['spearman_rho_ranks']:.3f} "
                f"changed={r['n_rank_changed']}/{r['n']}"
            )


if __name__ == "__main__":
    main()
