"""M7 robustness analysis: random null distribution + class-stratified null +
trained-only headline.

Three controls used in the paper:
1. Unstratified random subset null (1000 trials at n=4551 and n=4326)
2. Class-stratified random subset null (preserving CV2024 val class distribution)
3. Trained-only headline (n=19 = 25 - taaldhwaj - 5 untrained controls;
   STEM sisters reported separately as overfit-positive)

Leakage-attributable Delta = observed Delta - mean random null Delta.
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

SUBSET_SIZES = {"le6_size": 4551, "le6_plus_size": 4326}
N_TRIALS = 1000

TAALDHWAJ = "taaldhwaj"
UNTRAINED = {
    "DeepScope_Innovators",
    "Deep_Learners",
    "EndoAI",
    "ViFo Tech",
    "BotBotBot",
}
STEM = "STEM sisters"


def score_team_on_subset(
    pred_df_full: pd.DataFrame,
    gt_sub: pd.DataFrame,
    sub_paths: set,
):
    pred_df = (
        pred_df_full[pred_df_full["image_path"].isin(sub_paths)]
        .copy()
        .reset_index(drop=True)
    )
    ok, aligned = sanity_check(gt_sub, pred_df)
    if not ok:
        return None
    y_t = gt_sub[VALID_CLASSES].to_numpy()
    y_p = aligned[VALID_CLASSES].to_numpy()
    metrics = generate_metrics_report(y_t, y_p)
    return metrics["mean_auc"], metrics["balanced_accuracy"]


def trial_one_subset(
    rng,
    gt_full,
    pred_cache,
    valid_teams,
    subset_size,
    stratified,
    class_freqs,
):
    if stratified:
        gt_sub = (
            gt_full.groupby("predicted_class", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=int(class_freqs.loc[g.name]),
                    random_state=int(rng.integers(0, 2**31)),
                )
            )
            .reset_index(drop=True)
        )
    else:
        idx = rng.choice(len(gt_full), size=subset_size, replace=False)
        gt_sub = gt_full.iloc[idx].copy().reset_index(drop=True)
    sub_paths = set(gt_sub["image_path"])

    per_team = []
    for team_obj in valid_teams:
        pred_full = pred_cache[team_obj["file"]]
        result = score_team_on_subset(pred_full, gt_sub, sub_paths)
        if result is None:
            continue
        auc, bal = result
        combined = (auc + bal) / 2
        per_team.append(
            {
                "team": team_obj["team"],
                "combined_orig": team_obj["combined_organizer"],
                "combined_sub": float(combined),
                "delta": float(combined - team_obj["combined_organizer"]),
            }
        )
    return per_team


def aggregate_group(per_team_subset):
    """Compute mean delta + Spearman rho with tied-rank handling for a sub-group."""
    if not per_team_subset:
        return None
    deltas = np.array([t["delta"] for t in per_team_subset])
    orig = pd.Series([t["combined_orig"] for t in per_team_subset])
    sub = pd.Series([t["combined_sub"] for t in per_team_subset])
    ranks_orig = orig.rank(ascending=False, method="average").to_numpy()
    ranks_sub = sub.rank(ascending=False, method="average").to_numpy()
    rho, _ = spearmanr(ranks_orig, ranks_sub)
    shifts = ranks_orig - ranks_sub
    return {
        "n_teams": len(per_team_subset),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "rho": float(rho),
        "n_changed": int(sum(1 for s in shifts if abs(s) > 1e-9)),
        "max_drop": float(min(shifts)),
        "max_rise": float(max(shifts)),
        "median_abs_shift": float(np.median(np.abs(shifts))),
    }


def main():
    print("Loading validation data + 25 team predictions ...")
    gt_full = pd.read_excel(GT_PATH)
    assert len(gt_full) == 16132

    gt_full["predicted_class"] = gt_full[VALID_CLASSES].idxmax(axis=1)
    class_counts_full = gt_full["predicted_class"].value_counts()
    print(f"  validation class counts: {dict(class_counts_full)}")

    day1 = json.load(open(DAY1_PATH))
    valid_teams = [t for t in day1["teams"] if t["valid_for_rescore"]]
    assert len(valid_teams) == 25

    pred_cache = {}
    for t in valid_teams:
        pred_path = PRED_DIR / f"{t['file']}.xlsx"
        pred_cache[t["file"]] = pd.read_excel(pred_path)
    print(f"  cached {len(pred_cache)} prediction files")

    rng = np.random.default_rng(seed=42)

    null_results = {}
    for label, n_sub in SUBSET_SIZES.items():
        for stratified in [False, True]:
            mode = "stratified" if stratified else "unstratified"
            key = f"{label}__{mode}"
            print(f"\nRunning {N_TRIALS} {mode} trials @ n={n_sub} ({label}) ...")

            if stratified:
                ratio = n_sub / len(gt_full)
                stratified_n = (class_counts_full * ratio).round().astype(int)
                diff = n_sub - stratified_n.sum()
                if diff != 0:
                    largest = stratified_n.idxmax()
                    stratified_n[largest] += diff
                assert stratified_n.sum() == n_sub
                class_freqs = stratified_n
            else:
                class_freqs = None

            trial_per_team = []  # list of per_team list per trial
            for i in range(N_TRIALS):
                if (i + 1) % 100 == 0:
                    print(f"    trial {i + 1}/{N_TRIALS}")
                per_team = trial_one_subset(
                    rng=rng,
                    gt_full=gt_full,
                    pred_cache=pred_cache,
                    valid_teams=valid_teams,
                    subset_size=n_sub,
                    stratified=stratified,
                    class_freqs=class_freqs,
                )
                trial_per_team.append(per_team)

            # Group-matched null aggregation
            groups = {
                "all_25": lambda pt: pt,
                "trained_19": lambda pt: [
                    t for t in pt if t["team"] not in ({TAALDHWAJ} | UNTRAINED)
                ],
                "trained_no_stem_18": lambda pt: [
                    t for t in pt
                    if t["team"] not in ({TAALDHWAJ} | UNTRAINED | {STEM})
                ],
            }
            agg_per_group = {}
            for gname, gfilter in groups.items():
                agg_trials = [aggregate_group(gfilter(pt)) for pt in trial_per_team]
                agg_trials = [a for a in agg_trials if a is not None]
                mean_deltas = [a["mean_delta"] for a in agg_trials]
                rhos = [a["rho"] for a in agg_trials]
                n_changeds = [a["n_changed"] for a in agg_trials]
                agg_per_group[gname] = {
                    "n_teams": agg_trials[0]["n_teams"] if agg_trials else 0,
                    "mean_delta_null_mean": float(np.mean(mean_deltas)),
                    "mean_delta_null_p2_5": float(np.percentile(mean_deltas, 2.5)),
                    "mean_delta_null_p97_5": float(np.percentile(mean_deltas, 97.5)),
                    "rho_null_mean": float(np.mean(rhos)),
                    "rho_null_p2_5": float(np.percentile(rhos, 2.5)),
                    "rho_null_p97_5": float(np.percentile(rhos, 97.5)),
                    "n_changed_null_mean": float(np.mean(n_changeds)),
                    "n_changed_null_p2_5": float(np.percentile(n_changeds, 2.5)),
                    "n_changed_null_p97_5": float(np.percentile(n_changeds, 97.5)),
                }

            agg = {
                "label": label,
                "mode": mode,
                "subset_size": n_sub,
                "n_trials": N_TRIALS,
                "by_group": agg_per_group,
            }
            null_results[key] = {"aggregate": agg}
            for gname, ga in agg_per_group.items():
                print(
                    f"  -> {gname:18s} mean_d={ga['mean_delta_null_mean']:+.4f} "
                    f"[{ga['mean_delta_null_p2_5']:+.4f}, {ga['mean_delta_null_p97_5']:+.4f}] | "
                    f"rho={ga['rho_null_mean']:.3f} "
                    f"[{ga['rho_null_p2_5']:.3f}, {ga['rho_null_p97_5']:.3f}] | "
                    f"n_chg={ga['n_changed_null_mean']:.1f}"
                )

    out_null = ARTIFACT_ROOT / "results/cv2024_robust_null.json"
    with open(out_null, "w") as f:
        json.dump(null_results, f, indent=2)
    print(f"\nSaved {out_null}")

    obs_le6 = json.load(open(ARTIFACT_ROOT / "results/cv2024_rescored_le6.json"))
    obs_le6p = json.load(
        open(ARTIFACT_ROOT / "results/cv2024_rescored_le6_plus_internal.json")
    )

    def filter_trained(teams, exclude):
        return [t for t in teams if t["team"] not in exclude]

    robust = {
        "description": "Trained-only headline analysis used in the paper"
    }
    for subset_label, obs_data in [
        ("le6", obs_le6),
        ("le6_plus_internal", obs_le6p),
    ]:
        teams_25 = obs_data["teams"]
        excl_taal = filter_trained(teams_25, {TAALDHWAJ})
        excl_taal_untr = filter_trained(teams_25, {TAALDHWAJ} | UNTRAINED)
        excl_taal_untr_stem = filter_trained(
            teams_25, {TAALDHWAJ} | UNTRAINED | {STEM}
        )

        for label, group in [
            ("all_25", teams_25),
            ("excl_taaldhwaj_24", excl_taal),
            ("trained_19", excl_taal_untr),
            ("trained_no_stem_18", excl_taal_untr_stem),
        ]:
            deltas = [t["delta_combined"] for t in group]
            sorted_g = sorted(group, key=lambda x: x["combined_subset"], reverse=True)
            new_ranks = {t["team"]: r for r, t in enumerate(sorted_g, start=1)}
            ranks_sub = [new_ranks[t["team"]] for t in group]
            # combined_orig backcomputed from observed schema
            for t in group:
                t["_combined_orig"] = t["combined_subset"] - t["delta_combined"]
            sorted_orig = sorted(group, key=lambda x: x["_combined_orig"], reverse=True)
            new_orig_ranks = {
                t["team"]: r for r, t in enumerate(sorted_orig, start=1)
            }
            ranks_orig_g = [new_orig_ranks[t["team"]] for t in group]
            rho, _ = spearmanr(ranks_orig_g, ranks_sub)
            tau, _ = kendalltau(ranks_orig_g, ranks_sub)
            shifts = [
                new_orig_ranks[t["team"]] - new_ranks[t["team"]] for t in group
            ]
            robust[f"{subset_label}__{label}"] = {
                "n": len(group),
                "mean_delta_combined": float(np.mean(deltas)),
                "median_delta_combined": float(np.median(deltas)),
                "spearman_rho_within": float(rho),
                "kendall_tau_within": float(tau),
                "n_rank_changed_within": int(sum(1 for s in shifts if s != 0)),
                "median_abs_rank_shift": float(np.median(np.abs(shifts))),
            }

    for subset_label in ["le6", "le6_plus_internal"]:
        size_label = "le6_size" if subset_label == "le6" else "le6_plus_size"
        for null_mode in ["unstratified", "stratified"]:
            null_key = f"{size_label}__{null_mode}"
            null_by_group = null_results[null_key]["aggregate"]["by_group"]
            for headline_label in ["all_25", "trained_19", "trained_no_stem_18"]:
                # GROUP-MATCHED null: use the same team subset for null as for observed
                null_mean = null_by_group[headline_label]["mean_delta_null_mean"]
                null_p2_5 = null_by_group[headline_label]["mean_delta_null_p2_5"]
                null_p97_5 = null_by_group[headline_label]["mean_delta_null_p97_5"]
                obs_mean = robust[f"{subset_label}__{headline_label}"][
                    "mean_delta_combined"
                ]
                attr_mean = obs_mean - null_mean
                attr_lo = obs_mean - null_p97_5
                attr_hi = obs_mean - null_p2_5
                key = f"attributable__{subset_label}__{headline_label}__{null_mode}"
                robust[key] = {
                    "observed_mean_delta": obs_mean,
                    "null_mean_delta_group_matched": null_mean,
                    "leakage_attributable_mean_delta": attr_mean,
                    "attributable_null_only_interval_lo": attr_lo,
                    "attributable_null_only_interval_hi": attr_hi,
                    "note": "Interval bounds derived from null-distribution percentiles only; observed-side variance not propagated.",
                }

    out_robust = ARTIFACT_ROOT / "results/cv2024_rescored_robust.json"
    with open(out_robust, "w") as f:
        json.dump(robust, f, indent=2)
    print(f"\nSaved {out_robust}")

    print("\n=== TRAINED-ONLY HEADLINE TABLE ===")
    for subset in ["le6", "le6_plus_internal"]:
        print(f"\n{subset}:")
        for label in [
            "all_25",
            "excl_taaldhwaj_24",
            "trained_19",
            "trained_no_stem_18",
        ]:
            r = robust[f"{subset}__{label}"]
            print(
                f"  {label:25s} n={r['n']:2d}  mean_d={r['mean_delta_combined']:+.4f}  "
                f"med_d={r['median_delta_combined']:+.4f}  rho={r['spearman_rho_within']:.3f}  "
                f"med|shift|={r['median_abs_rank_shift']:.1f}  "
                f"changed={r['n_rank_changed_within']}/{r['n']}"
            )
        for null_mode in ["unstratified", "stratified"]:
            r = robust[f"attributable__{subset}__trained_19__{null_mode}"]
            print(
                f"    leakage-attrib trained_19 ({null_mode}, group-matched): "
                f"{r['leakage_attributable_mean_delta']:+.4f} "
                f"null-only [{r['attributable_null_only_interval_lo']:+.4f}, "
                f"{r['attributable_null_only_interval_hi']:+.4f}]"
            )


if __name__ == "__main__":
    main()
