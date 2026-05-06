"""M7 inference: paired Wilcoxon + paired bootstrap CI on per-team deltas.

Replaces the earlier null-only interval with paired statistical inference.
n=19 (and 24, 25, 18) teams give paired data
(combined_orig vs combined_subset). Standard practice:

- Wilcoxon signed-rank test (non-parametric, no normality assumption)
- Paired bootstrap of mean (10000 iter) for proper 95% CI

Output supersedes attributable_null_only_interval as the headline CI.
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap
from scipy.stats import wilcoxon

ARTIFACT_ROOT = Path(
    os.environ.get("CAPSULE_ARTIFACT_ROOT", Path(__file__).resolve().parents[1])
)
RESULTS_DIR = ARTIFACT_ROOT / "results"

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
    "all_25": lambda team_name: True,
    "excl_taaldhwaj_24": lambda team_name: team_name != TAALDHWAJ,
    "trained_19": lambda team_name: team_name != TAALDHWAJ
    and team_name not in UNTRAINED,
    "trained_no_stem_18": lambda team_name: team_name != TAALDHWAJ
    and team_name not in UNTRAINED
    and team_name != STEM,
}


def paired_bootstrap_ci(deltas: np.ndarray, n_iter: int, alpha: float, rng) -> dict:
    """BCa bootstrap CI for the mean of paired deltas (scipy.stats.bootstrap)."""
    deltas = np.asarray(deltas, dtype=float)
    res = scipy_bootstrap(
        (deltas,),
        np.mean,
        n_resamples=n_iter,
        method="BCa",
        confidence_level=1 - alpha,
        random_state=rng,
    )
    return {
        "mean": float(deltas.mean()),
        "ci_lo": float(res.confidence_interval.low),
        "ci_hi": float(res.confidence_interval.high),
        "n_iter": n_iter,
        "se_bootstrap": float(res.standard_error),
        "method": "BCa",
    }


def main() -> None:
    rng = np.random.default_rng(seed=42)
    n_iter = 10000

    out: dict = {
        "description": (
            "Paired Wilcoxon signed-rank + paired bootstrap CI on per-team "
            "delta_combined. Supersedes attributable_null_only_interval as "
            "primary inference."
        ),
        "n_bootstrap_iter": n_iter,
    }

    for subset in ["le6", "le6_plus_internal"]:
        rescored = json.load(open(RESULTS_DIR / f"cv2024_rescored_{subset}.json"))
        teams = rescored["teams"]

        for gname, gfilter in GROUPS.items():
            group_deltas = np.array(
                [t["delta_combined"] for t in teams if gfilter(t["team"])]
            )
            n = len(group_deltas)
            if n < 5:
                continue

            wstat, p_two = wilcoxon(group_deltas, alternative="two-sided")
            _, p_less = wilcoxon(group_deltas, alternative="less")

            sd = float(group_deltas.std(ddof=1))
            d_z = float(group_deltas.mean() / sd) if sd > 0 else float("nan")

            boot = paired_bootstrap_ci(group_deltas, n_iter=n_iter, alpha=0.05, rng=rng)

            out[f"{subset}__{gname}"] = {
                "n": n,
                "observed_mean": float(group_deltas.mean()),
                "observed_median": float(np.median(group_deltas)),
                "sd_of_deltas": sd,
                "cohens_d_z": d_z,
                "wilcoxon_W": float(wstat),
                "wilcoxon_p_two_sided": float(p_two),
                "wilcoxon_p_less": float(p_less),
                "paired_bootstrap_mean": boot["mean"],
                "paired_bootstrap_se": boot["se_bootstrap"],
                "paired_bootstrap_ci95_lo": boot["ci_lo"],
                "paired_bootstrap_ci95_hi": boot["ci_hi"],
            }

    out_path = RESULTS_DIR / "cv2024_m7_inference.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")

    print("\n=== M7 INFERENCE TABLE ===")
    print(
        f"{'subset':22s} {'group':22s} {'n':>3s} "
        f"{'mean':>8s} {'95%CI_lo':>9s} {'95%CI_hi':>9s} "
        f"{'p_less':>9s} {'d_z':>6s}"
    )
    for subset in ["le6", "le6_plus_internal"]:
        for gname in GROUPS:
            r = out.get(f"{subset}__{gname}")
            if r is None:
                continue
            print(
                f"{subset:22s} {gname:22s} {r['n']:>3d} "
                f"{r['paired_bootstrap_mean']:+.4f} "
                f"{r['paired_bootstrap_ci95_lo']:+.4f} "
                f"{r['paired_bootstrap_ci95_hi']:+.4f} "
                f"{r['wilcoxon_p_less']:.2e} "
                f"{r['cohens_d_z']:+.3f}"
            )


if __name__ == "__main__":
    main()
