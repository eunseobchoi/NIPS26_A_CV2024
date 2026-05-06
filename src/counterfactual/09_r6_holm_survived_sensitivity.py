"""Compute the six-class Holm-survived sensitivity.

This checks whether the headline le6 drop remains when balanced accuracy is
restricted to only the six classes that survive the paper's Holm-Bonferroni
per-class family.  The script reads the packaged per-seed JSONs and recomputes
the paired fixed-list contrast without requiring NumPy/SciPy.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


HOLM_SURVIVED_CLASSES = [
    "Ulcer",
    "Lymphangiectasia",
    "Foreign Body",
    "Normal",
    "Angioectasia",
    "Erythema",
]

# Two-sided 95% critical value for t with df=9.
T_CRIT_95_DF9 = 2.262


def load_runs(path: Path) -> dict[int, dict]:
    with open(path) as f:
        data = json.load(f)
    return {int(run["seed"]): run for run in data["runs"]}


def restricted_bal_acc(run: dict, classes: list[str]) -> float:
    per_class = run["last"]["orig_val"]["per_class"]
    return statistics.fmean(float(per_class[c]["recall"]) for c in classes)


def paired_summary(baseline: dict[int, dict], le6: dict[int, dict]) -> dict:
    seeds = sorted(set(baseline) & set(le6))
    if not seeds:
        raise RuntimeError("No shared seeds between baseline and le6 runs")

    rows = []
    deltas = []
    for seed in seeds:
        base_val = restricted_bal_acc(baseline[seed], HOLM_SURVIVED_CLASSES)
        le6_val = restricted_bal_acc(le6[seed], HOLM_SURVIVED_CLASSES)
        delta = le6_val - base_val
        rows.append(
            {
                "seed": seed,
                "baseline_restricted_bal_acc": base_val,
                "le6_restricted_bal_acc": le6_val,
                "delta_le6_minus_baseline": delta,
            }
        )
        deltas.append(delta)

    mean_delta = statistics.fmean(deltas)
    sd_delta = statistics.stdev(deltas) if len(deltas) >= 2 else 0.0
    se_delta = sd_delta / math.sqrt(len(deltas))
    t_stat = mean_delta / se_delta if se_delta else math.inf
    ci95 = [
        mean_delta - T_CRIT_95_DF9 * se_delta,
        mean_delta + T_CRIT_95_DF9 * se_delta,
    ]

    return {
        "classes": HOLM_SURVIVED_CLASSES,
        "n_paired_seeds": len(seeds),
        "baseline_mean_restricted_bal_acc": statistics.fmean(
            row["baseline_restricted_bal_acc"] for row in rows
        ),
        "le6_mean_restricted_bal_acc": statistics.fmean(
            row["le6_restricted_bal_acc"] for row in rows
        ),
        "delta_le6_minus_baseline_mean": mean_delta,
        "delta_le6_minus_baseline_sd": sd_delta,
        "delta_le6_minus_baseline_se": se_delta,
        "paired_t_df9": t_stat,
        "delta_le6_minus_baseline_95ci_t_df9": ci95,
        "drop_baseline_minus_le6_mean": -mean_delta,
        "drop_baseline_minus_le6_95ci_t_df9": [-ci95[1], -ci95[0]],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=Path("results/baseline/phase5_v4_baseline_n10.json"),
    )
    parser.add_argument(
        "--le6-json",
        type=Path,
        default=Path("results/counterfactual_n10/phase5_v4_le6_n10.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/r6_holm_survived_sensitivity.json"),
    )
    args = parser.parse_args()

    summary = paired_summary(load_runs(args.baseline_json), load_runs(args.le6_json))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(
        "six-class sensitivity: "
        f"baseline={summary['baseline_mean_restricted_bal_acc']:.3f}, "
        f"le6={summary['le6_mean_restricted_bal_acc']:.3f}, "
        f"delta={summary['delta_le6_minus_baseline_mean']:+.3f}, "
        f"95% CI [{summary['delta_le6_minus_baseline_95ci_t_df9'][0]:+.3f}, "
        f"{summary['delta_le6_minus_baseline_95ci_t_df9'][1]:+.3f}], "
        f"t(9)={summary['paired_t_df9']:.1f}"
    )


if __name__ == "__main__":
    main()
