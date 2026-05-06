"""Bootstrap rank uncertainty for CV2024 public-validation re-scoring.

The deterministic public-validation re-score uses the organizer metric script.
This script adds a sensitivity analysis by row-bootstrap resampling each public
validation estimand and recomputing the same combined metric arithmetic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


VALID_CLASSES = [
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
]

TAALDHWAJ = "taaldhwaj"
UNTRAINED = {"DeepScope_Innovators", "Deep_Learners", "EndoAI", "ViFo Tech", "BotBotBot"}
STEM = "STEM sisters"
GROUPS = {
    "all_25": lambda team: True,
    "trained_19": lambda team: team != TAALDHWAJ and team not in UNTRAINED,
    "trained_no_stem_18": lambda team: team != TAALDHWAJ and team not in UNTRAINED and team != STEM,
}

_BOOT_Y_TRUE = None
_BOOT_ARRAYS = None
_BOOT_TEAMS = None
_BOOT_N = None


def _init_bootstrap_worker(y_true, arrays, teams):
    global _BOOT_Y_TRUE, _BOOT_ARRAYS, _BOOT_TEAMS, _BOOT_N
    _BOOT_Y_TRUE = y_true
    _BOOT_ARRAYS = arrays
    _BOOT_TEAMS = teams
    _BOOT_N = len(y_true)


def _bootstrap_once(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, _BOOT_N, size=_BOOT_N)
    scores = {
        team: official_like_combined(_BOOT_Y_TRUE[idx], _BOOT_ARRAYS[team][idx])[2]
        for team in _BOOT_TEAMS
    }
    ranks = descending_ranks(scores)
    return [float(ranks[team]) for team in _BOOT_TEAMS]


def official_like_combined(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    aucs = []
    for i in range(y_true.shape[1]):
        try:
            aucs.append(float(roc_auc_score(y_true[:, i], y_pred[:, i])))
        except ValueError:
            aucs.append(0.0)
    mean_auc = float(np.mean(aucs))
    bal = float(balanced_accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)))
    return mean_auc, bal, float((mean_auc + bal) / 2)


def load_official_sanity(results_dir: Path):
    script = results_dir / "gen_metrics_report_val_train.py"
    if not script.exists():
        raise SystemExit(f"Missing organizer metric script: {script}")
    sys.path.insert(0, str(results_dir))
    from gen_metrics_report_val_train import sanity_check  # noqa: PLC0415

    return sanity_check


def descending_ranks(scores: dict[str, float]) -> dict[str, int]:
    return {team: rank for rank, (team, _) in enumerate(sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])), start=1)}


def rank_summary(rank_samples: np.ndarray, teams: list[str], top_k: int = 5) -> dict:
    out = {}
    for idx, team in enumerate(teams):
        vals = rank_samples[:, idx]
        out[team] = {
            "rank_median": float(np.median(vals)),
            "rank_ci95_lo": float(np.quantile(vals, 0.025)),
            "rank_ci95_hi": float(np.quantile(vals, 0.975)),
            "top5_probability": float(np.mean(vals <= top_k)),
        }
    return out


def pairwise_win_probability(rank_samples: np.ndarray, teams: list[str]) -> list[dict]:
    rows = []
    for i, a in enumerate(teams):
        for j, b in enumerate(teams):
            if i >= j:
                continue
            p = float(np.mean(rank_samples[:, i] < rank_samples[:, j]))
            if p >= 0.95 or p <= 0.05:
                rows.append({"team_a": a, "team_b": b, "p_a_ranks_above_b": p})
    return rows


def load_team_arrays(root: Path, results_dir: Path, subset_name: str, valid_teams: list[dict], sanity_check):
    gt_full = pd.read_excel(results_dir / "validation_data.xlsx")
    if subset_name == "orig_public_val":
        gt = gt_full.copy()
    else:
        csv_name = {
            "le6": "artifacts/csvs/cv2024_validation_dedup_le6.csv",
            "le6_plus_internal": "artifacts/csvs/cv2024_validation_le6_plus_internal.csv",
        }[subset_name]
        paths = set(pd.read_csv(root / csv_name)["image_path"])
        gt = gt_full[gt_full["image_path"].isin(paths)].copy()
    gt = gt.reset_index(drop=True)
    y_true = gt[VALID_CLASSES].to_numpy()

    pred_dir = results_dir / "submitted_excel_files" / "validation"
    arrays = {}
    for team in valid_teams:
        pred_path = pred_dir / f"{team['file']}.xlsx"
        pred_full = pd.read_excel(pred_path)
        pred_sub = pred_full[pred_full["image_path"].isin(set(gt["image_path"]))].copy()
        ok, aligned = sanity_check(gt, pred_sub.reset_index(drop=True))
        if not ok:
            continue
        arrays[team["team"]] = aligned[VALID_CLASSES].to_numpy()
    return gt, y_true, arrays


def analyze_subset(root: Path, results_dir: Path, subset_name: str, valid_teams: list[dict], sanity_check, n_boot: int, rng, jobs: int) -> dict:
    gt, y_true, arrays = load_team_arrays(root, results_dir, subset_name, valid_teams, sanity_check)
    teams = [t["team"] for t in valid_teams if t["team"] in arrays]
    observed_scores = {
        team: official_like_combined(y_true, arrays[team])[2]
        for team in teams
    }
    observed_ranks = descending_ranks(observed_scores)

    n = len(gt)
    seeds = rng.integers(0, 2**32 - 1, size=n_boot, dtype=np.uint32).astype(int).tolist()
    if jobs > 1:
        with mp.Pool(processes=jobs, initializer=_init_bootstrap_worker, initargs=(y_true, arrays, teams)) as pool:
            rank_samples = np.asarray(pool.map(_bootstrap_once, seeds), dtype=float)
    else:
        _init_bootstrap_worker(y_true, arrays, teams)
        rank_samples = np.asarray([_bootstrap_once(seed) for seed in seeds], dtype=float)

    groups = {}
    for group, filt in GROUPS.items():
        keep = [j for j, team in enumerate(teams) if filt(team)]
        group_teams = [teams[j] for j in keep]
        group_samples = rank_samples[:, keep]
        groups[group] = {
            "n": len(group_teams),
            "rank_intervals": rank_summary(group_samples, group_teams),
            "decisive_pairwise_probabilities": pairwise_win_probability(group_samples, group_teams),
        }

    return {
        "n_rows": int(n),
        "n_teams": len(teams),
        "n_bootstrap": n_boot,
        "observed_ranks": observed_ranks,
        "groups": groups,
    }


def markdown_summary(result: dict) -> str:
    lines = [
        "# Public-Validation Rank Uncertainty",
        "",
        "Row-bootstrap sensitivity over the public-validation estimand; not an official-test ranking.",
        "",
    ]
    for subset, obj in result["subsets"].items():
        lines += [
            f"## {subset}",
            "",
            "| Team | observed rank | median boot rank | 95% rank interval | P(top 5) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        intervals = obj["groups"]["trained_19"]["rank_intervals"]
        for team, obs_rank in sorted(obj["observed_ranks"].items(), key=lambda kv: kv[1]):
            if team not in intervals:
                continue
            r = intervals[team]
            lines.append(
                f"| {team} | {obs_rank} | {r['rank_median']:.1f} | [{r['rank_ci95_lo']:.0f}, {r['rank_ci95_hi']:.0f}] | {r['top5_probability']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("CAPSULE_ARTIFACT_ROOT", "."))
    parser.add_argument("--cv2024-results-dir", default=os.environ.get("CV2024_RESULTS_DIR", "external/cv2024_repo/Results"))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 1) // 2)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-json", default="results/cv2024_rank_uncertainty.json")
    parser.add_argument("--write-md", default="results/cv2024_rank_uncertainty.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    results_dir = Path(args.cv2024_results_dir)
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    sanity_check = load_official_sanity(results_dir)
    day1 = json.loads((root / "results/cv2024_rescored_orig.json").read_text())
    valid_teams = [t for t in day1["teams"] if t["valid_for_rescore"]]
    rng = np.random.default_rng(args.seed)

    output = {
        "description": (
            "Row-bootstrap rank uncertainty for released public-validation "
            "submission re-scoring. Combined metric arithmetic mirrors the "
            "CV2024 validation metric definition; this is not an official-test "
            "leaderboard."
        ),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "subsets": {},
    }
    for subset in ["orig_public_val", "le6", "le6_plus_internal"]:
        output["subsets"][subset] = analyze_subset(
            root, results_dir, subset, valid_teams, sanity_check, args.n_bootstrap, rng, args.jobs
        )

    for path in [args.write_json, args.write_md]:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / args.write_json).write_text(json.dumps(output, indent=2) + "\n")
    (root / args.write_md).write_text(markdown_summary(output) + "\n")
    print(f"Saved {root / args.write_json}")


if __name__ == "__main__":
    main()
