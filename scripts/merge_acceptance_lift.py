#!/usr/bin/env python3
"""Merge acceptance-lift single-seed counterfactual runs.

The worker shards write one JSON per seed. This script merges those shards
and produces paper-facing summaries for the two experiment families:

1. strict `le6_plus_internal` evaluation for baseline vs le6 training.
2. n=10 completion for random and Comp-A matched arms.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Iterable


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def find_part(root: Path, stem: str) -> Path:
    candidates = sorted((root / "results" / "acceptance_lift").rglob(f"{stem}.json"))
    if not candidates:
        raise FileNotFoundError(f"missing shard {stem}.json under results/acceptance_lift")
    if len(candidates) > 1:
        raise RuntimeError(f"multiple shards for {stem}: {candidates}")
    return candidates[0]


def rel_display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def merge_runs(parts: Iterable[Path], output: Path, root: Path) -> dict:
    parts = list(parts)
    loaded = [load_json(p) for p in parts]
    if not loaded:
        raise ValueError(f"no parts for {output}")
    base = loaded[0]
    runs = []
    seen = set()
    for data in loaded:
        for run in data.get("runs", []):
            seed = int(run["seed"])
            if seed in seen:
                raise RuntimeError(f"duplicate seed {seed} for {output}")
            seen.add(seed)
            runs.append(run)
    runs.sort(key=lambda r: int(r["seed"]))
    base["runs"] = runs
    args = base.setdefault("args", {})
    args["seeds"] = [int(r["seed"]) for r in runs]
    args["output"] = rel_display(output, root)
    args["acceptance_lift_merged"] = True
    meta = base.setdefault("meta", {})
    meta["acceptance_lift_merge_parts"] = [rel_display(p, root) for p in parts]
    meta["acceptance_lift_merged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["acceptance_lift_output"] = rel_display(output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(base, indent=2, default=str) + "\n")
    return base


def require_compatible(data: dict, label: str, root: Path) -> None:
    keys = (
        "train_csv_md5",
        "dedup_val_csv_md5",
        "script_sha256",
        "batch",
    )
    meta_values = {key: set() for key in keys}
    train_paths = set()
    eval_paths = set()
    for run_source in data.get("meta", {}).get("acceptance_lift_merge_parts", []):
        part = Path(run_source)
        if not part.is_absolute():
            part = root / part
        part_data = load_json(part)
        meta = part_data.get("meta", {})
        args = part_data.get("args", {})
        train_paths.add(args.get("train_csv"))
        eval_paths.add(args.get("dedup_val_csv"))
        for key in keys:
            meta_values[key].add(meta.get(key))
    bad = {key: vals for key, vals in meta_values.items() if len(vals) != 1}
    if bad:
        raise RuntimeError(f"incompatible {label} merge metadata: {bad}")
    data.setdefault("meta", {})["acceptance_lift_train_csv_paths"] = sorted(train_paths)
    data.setdefault("meta", {})["acceptance_lift_eval_csv_paths"] = sorted(eval_paths)


def vals(data: dict, metric: str = "bal_acc", split: str = "dedup_val") -> list[float]:
    return [float(run["last"][split][metric]) for run in data.get("runs", [])]


def mean_sd(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def row(label: str, data: dict, split: str) -> dict:
    xs = vals(data, split=split)
    m, s = mean_sd(xs)
    return {
        "label": label,
        "n": len(xs),
        "seeds": [int(r["seed"]) for r in data.get("runs", [])],
        f"{split}_bal_acc_mean": m,
        f"{split}_bal_acc_sd": s,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    out_dir = root / "results" / "acceptance_lift"
    out_dir.mkdir(parents=True, exist_ok=True)

    strict_outputs = {}
    strict_parts = {}
    for arm in ("strict_baseline", "strict_le6"):
        parts = []
        missing = []
        for seed in range(10):
            try:
                parts.append(find_part(root, f"{arm}_s{seed}"))
            except FileNotFoundError:
                missing.append(seed)
        if missing and not args.allow_incomplete:
            raise FileNotFoundError(f"{arm} missing seeds {missing}")
        strict_parts[arm] = parts
        strict_outputs[arm] = merge_runs(
            parts,
            out_dir / f"phase5_v5_{arm}_le6_plus_internal_n{len(parts)}.json",
            root,
        )

    matched_specs = {
        "random": {"output": out_dir / "phase5_v4_random_s0_n10.json"},
        "compA": {"output": out_dir / "phase5_v4_compA_s0_n10.json"},
    }
    matched_outputs = {}
    for arm, spec in matched_specs.items():
        parts = []
        missing = []
        for seed in range(10):
            try:
                parts.append(find_part(root, f"{arm}_s{seed}"))
            except FileNotFoundError:
                missing.append(seed)
        if missing and not args.allow_incomplete:
            raise FileNotFoundError(f"{arm} missing seeds {missing}")
        matched_outputs[arm] = merge_runs(parts, spec["output"], root)

    for label, data in {
        "strict_baseline": strict_outputs["strict_baseline"],
        "strict_le6": strict_outputs["strict_le6"],
        "random": matched_outputs["random"],
        "compA": matched_outputs["compA"],
    }.items():
        require_compatible(data, label, root)

    summary = {
        "strict_le6_plus_internal": {
            "estimand": "balanced accuracy on le6_plus_internal public validation; runner key is dedup_val",
            "baseline": row("baseline", strict_outputs["strict_baseline"], "dedup_val"),
            "le6": row("le6", strict_outputs["strict_le6"], "dedup_val"),
        },
        "matched_arm_completion": {
            "estimand": "balanced accuracy on original CV2024 public validation; runner key is orig_val",
            "random": row("random", matched_outputs["random"], "orig_val"),
            "compA": row("compA", matched_outputs["compA"], "orig_val"),
        },
    }
    b = summary["strict_le6_plus_internal"]["baseline"]["dedup_val_bal_acc_mean"]
    l = summary["strict_le6_plus_internal"]["le6"]["dedup_val_bal_acc_mean"]
    summary["strict_le6_plus_internal"]["delta_le6_minus_baseline"] = l - b

    summary_path = out_dir / "acceptance_lift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    md = out_dir / "acceptance_lift_summary.md"
    md.write_text(
        "\n".join(
            [
                "# Acceptance-Lift Experiment Summary",
                "",
                "## Strict le6_plus_internal Validation",
                "",
                "Estimand: balanced accuracy on le6_plus_internal public validation.",
                "",
                f"- baseline n={summary['strict_le6_plus_internal']['baseline']['n']}: "
                f"{b:.4f}",
                f"- le6 n={summary['strict_le6_plus_internal']['le6']['n']}: {l:.4f}",
                f"- delta le6-baseline: {l - b:+.4f}",
                "",
                "## Matched-Arm Completion",
                "",
                "Estimand: balanced accuracy on original CV2024 public validation.",
                "",
                f"- random n={summary['matched_arm_completion']['random']['n']}: "
                f"{summary['matched_arm_completion']['random']['orig_val_bal_acc_mean']:.4f}",
                f"- Comp-A n={summary['matched_arm_completion']['compA']['n']}: "
                f"{summary['matched_arm_completion']['compA']['orig_val_bal_acc_mean']:.4f}",
                "",
            ]
        )
        + "\n"
    )
    print(f"wrote {summary_path}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
