#!/usr/bin/env python3
"""Recompute the paper-facing evidence-and-scope table.

This summary is intentionally not a new experiment. It consolidates already
released audit annotations, n=10 retraining JSONs, and public re-scoring
fixtures into one paper-facing table.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


SOURCES = ("KVASIR", "SEE-AI", "KID", "AIIMS")
CLASSES = (
    "Angioectasia",
    "Bleeding",
    "Erosion",
    "Erythema",
    "Foreign Body",
    "Lymphangiectasia",
    "Normal",
    "Polyp",
    "Ulcer",
    "Worms",
)


def load_json(root: Path, rel: str) -> dict:
    with (root / rel).open() as f:
        return json.load(f)


def mean_sd(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def read_phash_rates(root: Path) -> dict:
    out = {}
    for source in SOURCES:
        rel = f"artifacts/annotations/cv2024_{source}_phash_annotated.csv"
        path = root / rel
        n = flagged = exact_both = 0
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                n += 1
                ph = int(row["min_phash_dist_to_kvasir"])
                dh = int(row["min_dhash_dist_to_kvasir"])
                if ph <= 6 and dh <= 6:
                    flagged += 1
                if ph == 0 and dh == 0:
                    exact_both += 1
        out[source] = {
            "source_file": rel,
            "n": n,
            "joint_phash_dhash_le6": flagged,
            "joint_phash_dhash_le6_rate": flagged / n,
            "joint_exact_rate": exact_both / n,
        }
    return out


def read_retraining(root: Path) -> dict:
    base_rel = "results/baseline/phase5_v5_baseline_n10.json"
    le6_rel = "results/baseline/phase5_v5_le6_n10.json"
    base = load_json(root, base_rel)
    le6 = load_json(root, le6_rel)
    base_runs = {int(r["seed"]): r for r in base["runs"]}
    le6_runs = {int(r["seed"]): r for r in le6["runs"]}
    seeds = sorted(set(base_runs) & set(le6_runs))
    if seeds != list(range(10)):
        raise RuntimeError(f"expected shared seeds 0-9, got {seeds}")

    def metric(run: dict, split: str = "orig_val") -> float:
        return float(run["last"][split]["bal_acc"])

    base_vals = [metric(base_runs[s]) for s in seeds]
    le6_vals = [metric(le6_runs[s]) for s in seeds]
    deltas = [l - b for b, l in zip(base_vals, le6_vals)]

    source_rows = {}
    for source in SOURCES:
        source_deltas = [
            float(le6_runs[s]["last"]["orig_val"]["per_source"][source]["bal_acc"])
            - float(base_runs[s]["last"]["orig_val"]["per_source"][source]["bal_acc"])
            for s in seeds
        ]
        source_rows[source] = {
            "delta_mean": mean_sd(source_deltas)[0],
            "delta_sd": mean_sd(source_deltas)[1],
            "n_val": int(base_runs[seeds[0]]["last"]["orig_val"]["per_source"][source]["n"]),
        }

    def class_subset_delta(exclude: set[str]) -> dict:
        keep = [c for c in CLASSES if c not in exclude]
        vals = []
        for seed in seeds:
            b = base_runs[seed]["last"]["orig_val"]["per_class"]
            l = le6_runs[seed]["last"]["orig_val"]["per_class"]
            vals.append(
                statistics.mean(float(l[c]["recall"]) - float(b[c]["recall"]) for c in keep)
            )
        m, sd = mean_sd(vals)
        return {"classes": keep, "delta_mean": m, "delta_sd": sd}

    return {
        "source_files": [base_rel, le6_rel],
        "n": len(seeds),
        "seeds": seeds,
        "baseline_orig_val_mean": mean_sd(base_vals)[0],
        "baseline_orig_val_sd": mean_sd(base_vals)[1],
        "le6_orig_val_mean": mean_sd(le6_vals)[0],
        "le6_orig_val_sd": mean_sd(le6_vals)[1],
        "delta_le6_minus_baseline_mean": mean_sd(deltas)[0],
        "delta_le6_minus_baseline_sd": mean_sd(deltas)[1],
        "per_source_delta": source_rows,
        "non_ulcer_delta": class_subset_delta({"Ulcer"}),
        "non_ulcer_worms_normal_delta": class_subset_delta({"Ulcer", "Worms", "Normal"}),
    }


def read_public_rescore(root: Path) -> dict:
    m7_rel = "results/cv2024_m7_inference.json"
    bridge_rel = "results/cv2024_public_official_bridge.json"
    m7 = load_json(root, m7_rel)
    bridge = load_json(root, bridge_rel)
    trained = m7["le6__trained_19"]
    groups = bridge["variants"]
    return {
        "source_files": [m7_rel, bridge_rel],
        "trained_19_delta_combined_mean": trained["observed_mean"],
        "trained_19_ci95": [
            trained["paired_bootstrap_ci95_lo"],
            trained["paired_bootstrap_ci95_hi"],
        ],
        "orig_public_spearman_all25": groups["orig_public_val"]["groups"]["all_common_25"][
            "spearman_rho_score"
        ],
        "le6_spearman_all25": groups["le6_public_val"]["groups"]["all_common_25"][
            "spearman_rho_score"
        ],
        "le6_plus_internal_spearman_all25": groups["le6_plus_internal_public_val"]["groups"][
            "all_common_25"
        ]["spearman_rho_score"],
    }


def make_markdown(summary: dict) -> str:
    ph = summary["external_overlap"]["KVASIR"]
    ncc = summary["ncc"]
    ret = summary["retraining"]
    public = summary["public_rescore"]
    source_bits = ", ".join(
        f"{s} {v['delta_mean']:+.3f}"
        for s, v in ret["per_source_delta"].items()
    )
    rows = [
        "# Evidence-and-scope summary",
        "",
        "| Axis | Recomputed value | Source |",
        "| --- | --- | --- |",
        (
            "| External Kvasir-origin overlap | "
            f"{ph['joint_phash_dhash_le6']:,}/{ph['n']:,} KVASIR rows "
            f"({ph['joint_phash_dhash_le6_rate']:.1%}); "
            f"{ph['joint_exact_rate']:.1%} exact on both hashes; "
            f"NCC mean {ncc['ncc_mean']:.3f} | annotations + NCC summary |"
        ),
        (
            "| Fixed-list sensitivity | "
            f"baseline {ret['baseline_orig_val_mean']:.3f}, "
            f"le6 {ret['le6_orig_val_mean']:.3f}, "
            f"Delta {ret['delta_le6_minus_baseline_mean']:+.3f} | "
            f"{', '.join(ret['source_files'])} |"
        ),
        (
            "| Per-source decomposition | "
            f"{source_bits} | baseline/le6 JSON per_source fields |"
        ),
        (
            "| Non-Ulcer sensitivity | "
            f"exclude Ulcer {ret['non_ulcer_delta']['delta_mean']:+.3f}; "
            f"exclude Ulcer/Worms/Normal "
            f"{ret['non_ulcer_worms_normal_delta']['delta_mean']:+.3f} | "
            "baseline/le6 JSON per_class fields |"
        ),
        (
            "| Public re-score boundary | "
            f"trained-team Delta combined {public['trained_19_delta_combined_mean']:+.3f}; "
            f"public-official Spearman orig/le6/le6+internal = "
            f"{public['orig_public_spearman_all25']:.3f}/"
            f"{public['le6_spearman_all25']:.3f}/"
            f"{public['le6_plus_internal_spearman_all25']:.3f} | "
            f"{', '.join(public['source_files'])} |"
        ),
    ]
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    summary = {
        "external_overlap": read_phash_rates(root),
        "pdq": load_json(root, "artifacts/summaries/pdq_audit.json"),
        "internal_reuse": {
            "kvasir": load_json(root, "artifacts/summaries/cv2024_internal_leak.json"),
            "all_sources": load_json(root, "artifacts/summaries/cv2024_internal_cross_source.json"),
        },
        "ncc": load_json(root, "artifacts/ncc/cv2024_KVASIR_ncc_full_summary.json"),
        "retraining": read_retraining(root),
        "public_rescore": read_public_rescore(root),
    }

    out_json = root / "results" / "claim_scorecard_summary.json"
    out_md = root / "results" / "claim_scorecard_summary.md"
    if args.write:
        out_json.write_text(json.dumps(summary, indent=2) + "\n")
        out_md.write_text(make_markdown(summary))
        print(f"wrote {out_json.relative_to(root)}")
        print(f"wrote {out_md.relative_to(root)}")
    else:
        print(make_markdown(summary), end="")


if __name__ == "__main__":
    main()
