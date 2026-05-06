"""Bridge public-validation rescoring to official-test outcomes.

This script uses only organizer-released official-test metric JSONs and
public-validation rescoring outputs. It does not evaluate new models on
official-test images; the direct model-evaluation scope check lives in
`src/counterfactual/04_official_test_eval.py` and requires the separate
organizer-released class-separated test archive.
"""

from __future__ import annotations

import json
import math
import re
import argparse
from pathlib import Path
from statistics import mean, median


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
TEST_METRICS_DIR_CANDIDATES = [
    REPO_ROOT / "data/cv2024_repo/Results/metrics_reports/metrics_reports_test",
    REPO_ROOT / "external/cv2024_repo/Results/metrics_reports/metrics_reports_test",
]
TEST_METRICS_DIR = next((p for p in TEST_METRICS_DIR_CANDIDATES if p.exists()), TEST_METRICS_DIR_CANDIDATES[0])

PUBLIC_VARIANTS = {
    "orig_public_val": {
        "path": RESULTS_DIR / "cv2024_rescored_orig.json",
        "combined_key": "combined_organizer",
        "auc_key": "mean_auc_organizer",
        "bal_key": "balanced_accuracy_organizer",
    },
    "le6_public_val": {
        "path": RESULTS_DIR / "cv2024_rescored_le6.json",
        "combined_key": "combined_subset",
        "auc_key": "mean_auc_subset",
        "bal_key": "balanced_accuracy_subset",
    },
    "le6_plus_internal_public_val": {
        "path": RESULTS_DIR / "cv2024_rescored_le6_plus_internal.json",
        "combined_key": "combined_subset",
        "auc_key": "mean_auc_subset",
        "bal_key": "balanced_accuracy_subset",
    },
}

UNTRAINED = {
    "BotBotBot",
    "DeepScope_Innovators",
    "Deep_Learners",
    "EndoAI",
    "ViFo Tech",
}
TAALDHWAJ = "taaldhwaj"
STEM = "STEM sisters"


def normalize_team(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def team_from_metric_filename(path: Path) -> str:
    name = path.name.replace("_metrics.json", "")
    for suffix in [
        "_predicted_test_dataset",
        "_testing_excel",
        "_test_predictions-",
        "_predicted_testing_dataset",
        "_test_predictions",
    ]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    # The public validation release uses an underscore in this one team name.
    if name == "DeepScope Innovators":
        return "DeepScope_Innovators"
    return name


def ranks_desc(values: dict[str, float]) -> dict[str, float]:
    """Average descending ranks, 1 = best."""
    ordered = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k, _ in ordered[i:j]:
            ranks[k] = avg_rank
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    x_ranks = ranks_desc({str(i): x for i, x in enumerate(xs)})
    y_ranks = ranks_desc({str(i): y for i, y in enumerate(ys)})
    return pearson(
        [x_ranks[str(i)] for i in range(len(xs))],
        [y_ranks[str(i)] for i in range(len(xs))],
    )


def kendall_tau_b(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )
    if denom == 0:
        return None
    return (concordant - discordant) / denom


def load_public_variant(name: str, spec: dict[str, str | Path]) -> dict[str, dict]:
    obj = json.loads(Path(spec["path"]).read_text())
    teams = {}
    for row in obj["teams"]:
        if row.get("valid_for_rescore", True) is False:
            continue
        score = row.get(spec["combined_key"])
        if score is None:
            continue
        team = row["team"]
        teams[normalize_team(team)] = {
            "team": team,
            "score": float(score),
            "mean_auc": float(row[spec["auc_key"]]),
            "balanced_accuracy": float(row[spec["bal_key"]]),
            "source_file": row.get("file"),
            "variant": name,
        }
    return teams


def load_official_test() -> dict[str, dict]:
    out = {}
    for path in sorted(TEST_METRICS_DIR.glob("*.json")):
        row = json.loads(path.read_text())
        auc = row.get("mean_auc")
        bal = row.get("balanced_accuracy")
        if auc is None or bal is None:
            continue
        team = team_from_metric_filename(path)
        out[normalize_team(team)] = {
            "team": team,
            "score": float((auc + bal) / 2),
            "mean_auc": float(auc),
            "balanced_accuracy": float(bal),
            "source_file": path.name,
        }
    return out


def summarize_group(
    group_name: str,
    rows: list[dict],
    public_field: str,
    official_field: str = "official_test_score",
) -> dict:
    xs = [r[public_field] for r in rows]
    ys = [r[official_field] for r in rows]
    public_ranks = ranks_desc({r["team"]: r[public_field] for r in rows})
    official_ranks = ranks_desc({r["team"]: r[official_field] for r in rows})
    rank_abs = [abs(public_ranks[r["team"]] - official_ranks[r["team"]]) for r in rows]
    errors = [r[public_field] - r[official_field] for r in rows]
    abs_errors = [abs(e) for e in errors]
    top5_public = {r["team"] for r in sorted(rows, key=lambda x: -x[public_field])[:5]}
    top5_official = {r["team"] for r in sorted(rows, key=lambda x: -x[official_field])[:5]}
    return {
        "group": group_name,
        "n": len(rows),
        "pearson_score": pearson(xs, ys),
        "spearman_rho_score": spearman(xs, ys),
        "kendall_tau_b_score": kendall_tau_b(xs, ys),
        "rank_mae": mean(rank_abs),
        "rank_median_abs_error": median(rank_abs),
        "score_mae": mean(abs_errors),
        "score_bias_public_minus_official": mean(errors),
        "score_rmse": math.sqrt(mean([e * e for e in errors])),
        "top5_overlap": len(top5_public & top5_official),
        "top5_public": sorted(top5_public),
        "top5_official": sorted(top5_official),
    }


def group_filter(group: str, row: dict) -> bool:
    if group == "all_common_25":
        return True
    if group == "trained_19":
        return row["team"] != TAALDHWAJ and row["team"] not in UNTRAINED
    if group == "trained_no_stem_18":
        return (
            row["team"] != TAALDHWAJ
            and row["team"] not in UNTRAINED
            and row["team"] != STEM
        )
    raise KeyError(group)


def markdown_summary(output: dict) -> str:
    lines = [
        "# Public-validation to official-test bridge",
        "",
        "This analysis uses organizer-released official-test metric JSONs and",
        "public-validation rescoring outputs. It does not evaluate new models.",
        "Direct model evaluation is only possible when the separate",
        "organizer-released class-separated CV2024 test archive is available.",
        "",
        f"- Common public/test teams: {output['n_common_public_official']}",
        f"- Official-test-only teams: {', '.join(output['official_test_only_teams']) or 'none'}",
        f"- Public-only teams: {', '.join(output['public_only_teams']) or 'none'}",
        "",
        "## Proxy quality against official-test combined score",
        "",
        "| Variant | Group | n | Pearson | Spearman | Kendall tau-b | rank MAE | score MAE | bias | top-5 overlap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, result in output["variants"].items():
        for group_name, stats in result["groups"].items():
            lines.append(
                "| {variant} | {group} | {n} | {pearson:.3f} | {rho:.3f} | {tau:.3f} | {rank_mae:.2f} | {score_mae:.3f} | {bias:+.3f} | {top5}/5 |".format(
                    variant=variant,
                    group=group_name,
                    n=stats["n"],
                    pearson=stats["pearson_score"],
                    rho=stats["spearman_rho_score"],
                    tau=stats["kendall_tau_b_score"],
                    rank_mae=stats["rank_mae"],
                    score_mae=stats["score_mae"],
                    bias=stats["score_bias_public_minus_official"],
                    top5=stats["top5_overlap"],
                )
            )
    lines.extend(
        [
            "",
            "## Main readout",
            "",
            output["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-json", default=str(RESULTS_DIR / "cv2024_public_official_bridge.json"))
    parser.add_argument("--write-md", default=str(RESULTS_DIR / "cv2024_public_official_bridge.md"))
    args = parser.parse_args(argv)

    official = load_official_test()
    public_by_variant = {
        name: load_public_variant(name, spec)
        for name, spec in PUBLIC_VARIANTS.items()
    }
    public_keys = set(public_by_variant["orig_public_val"])
    official_keys = set(official)
    common_keys = sorted(public_keys & official_keys)

    output = {
        "description": (
            "Compares public-validation scores (original/le6/le6+internal) "
            "against organizer-released official-test metrics for common teams."
        ),
        "n_official_test_teams": len(official),
        "n_public_valid_teams": len(public_keys),
        "n_common_public_official": len(common_keys),
        "official_test_only_teams": sorted(
            official[k]["team"] for k in official_keys - public_keys
        ),
        "public_only_teams": sorted(
            public_by_variant["orig_public_val"][k]["team"]
            for k in public_keys - official_keys
        ),
        "variants": {},
    }

    for variant, public in public_by_variant.items():
        rows = []
        for key in common_keys:
            p = public[key]
            o = official[key]
            rows.append(
                {
                    "team": p["team"],
                    "public_score": p["score"],
                    "public_mean_auc": p["mean_auc"],
                    "public_balanced_accuracy": p["balanced_accuracy"],
                    "official_test_score": o["score"],
                    "official_test_mean_auc": o["mean_auc"],
                    "official_test_balanced_accuracy": o["balanced_accuracy"],
                }
            )

        public_rank = ranks_desc({r["team"]: r["public_score"] for r in rows})
        official_rank = ranks_desc({r["team"]: r["official_test_score"] for r in rows})
        for row in rows:
            row["public_rank"] = public_rank[row["team"]]
            row["official_test_rank"] = official_rank[row["team"]]
            row["rank_shift_public_minus_official"] = (
                row["public_rank"] - row["official_test_rank"]
            )

        groups = {}
        for group_name in ["all_common_25", "trained_19", "trained_no_stem_18"]:
            group_rows = [r for r in rows if group_filter(group_name, r)]
            groups[group_name] = summarize_group(
                group_name, group_rows, "public_score"
            )

        output["variants"][variant] = {
            "rows": sorted(rows, key=lambda x: x["official_test_rank"]),
            "groups": groups,
        }

    orig_rho = output["variants"]["orig_public_val"]["groups"]["trained_19"][
        "spearman_rho_score"
    ]
    le6_rho = output["variants"]["le6_public_val"]["groups"]["trained_19"][
        "spearman_rho_score"
    ]
    orig_mae = output["variants"]["orig_public_val"]["groups"]["trained_19"][
        "rank_mae"
    ]
    le6_mae = output["variants"]["le6_public_val"]["groups"]["trained_19"][
        "rank_mae"
    ]
    if le6_rho > orig_rho and le6_mae < orig_mae:
        interpretation = (
            "On trained teams, the le6 public subset is a better proxy for the "
            "official-test ranking than the original public validation pool "
            f"(Spearman {le6_rho:.3f} vs {orig_rho:.3f}; rank MAE "
            f"{le6_mae:.2f} vs {orig_mae:.2f}). This supports adding a "
            "bridge-result claim, while still noting that direct official-test "
            "model evaluation requires hidden image bytes."
        )
    else:
        interpretation = (
            "The bridge analysis does not show that le6 is a uniformly better "
            "proxy for the official-test ranking on trained teams "
            f"(Spearman {le6_rho:.3f} vs original {orig_rho:.3f}; rank MAE "
            f"{le6_mae:.2f} vs original {orig_mae:.2f}). It should be used as "
            "a scope-clarifying limitation rather than a positive proxy claim."
        )
    output["interpretation"] = interpretation

    out_json = Path(args.write_json)
    out_md = Path(args.write_md)
    out_json.write_text(json.dumps(output, indent=2) + "\n")
    out_md.write_text(markdown_summary(output) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(interpretation)


if __name__ == "__main__":
    main()
