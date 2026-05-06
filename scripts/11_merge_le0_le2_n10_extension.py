#!/usr/bin/env python3
"""Merge le0/le2 per-seed extension shards into n=10 summaries.

This script is intentionally strict: it refuses to write paper-facing
summaries unless all seeds 0-9 are present exactly once for both threshold
arms and the baseline reference contains the same ten seeds.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path


ARMS = {
    "le0": {
        "label": "le0 exact removed",
        "train_rows": 21241,
        "kvasir_kept": 10645,
    },
    "le2": {
        "label": "le2 strict near-duplicate",
        "train_rows": 10825,
        "kvasir_kept": 229,
    },
}
EXPECTED_SEEDS = set(range(10))


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def mean_sd(xs: list[float]) -> tuple[float, float]:
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def runs_by_seed(data: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for run in data.get("runs", []):
        seed = int(run["seed"])
        if seed in out:
            raise RuntimeError(f"duplicate seed {seed} inside one JSON")
        out[seed] = run
    return out


def metric_values(data: dict, split: str = "orig_val", metric: str = "bal_acc") -> dict[int, float]:
    values: dict[int, float] = {}
    for run in data.get("runs", []):
        values[int(run["seed"])] = float(run["last"][split][metric])
    return values


def reference_baseline(root: Path) -> dict:
    candidates = [
        root / "results" / "baseline" / "phase5_v5_baseline_n10.json",
        root / "results" / "counterfactual_n10" / "phase5_v4_baseline_n10.json",
    ]
    for path in candidates:
        if path.exists():
            data = load_json(path)
            data.setdefault("meta", {})["reference_path"] = str(path.relative_to(root))
            return data
    raise FileNotFoundError(f"missing n=10 baseline reference; tried {[str(p) for p in candidates]}")


def merge_arm(root: Path, arm: str) -> dict:
    parts = [
        root / "results" / "le0_le2_extension" / f"phase5_v4_{arm}_s{seed}_n1.json"
        for seed in sorted(EXPECTED_SEEDS)
    ]
    missing_paths = [p for p in parts if not p.exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"missing {arm} shard(s): {[str(p.relative_to(root)) for p in missing_paths]}"
        )

    loaded = [load_json(p) for p in parts]
    merged: dict[int, dict] = {}
    for path, data in zip(parts, loaded):
        shard_runs = runs_by_seed(data)
        if len(shard_runs) != 1:
            raise RuntimeError(f"{path.relative_to(root)} must contain exactly one run")
        for seed, run in shard_runs.items():
            expected_seed = int(path.name.split("_s", 1)[1].split("_", 1)[0])
            if seed != expected_seed:
                raise RuntimeError(
                    f"{path.relative_to(root)} seed mismatch: JSON seed={seed}, name seed={expected_seed}"
                )
            if seed in merged:
                raise RuntimeError(f"duplicate seed {seed} while merging {arm}")
            merged[seed] = run

    missing = sorted(EXPECTED_SEEDS - set(merged))
    extra = sorted(set(merged) - EXPECTED_SEEDS)
    if missing or extra:
        raise RuntimeError(f"{arm} seed mismatch: missing={missing}, extra={extra}")

    base = copy.deepcopy(loaded[0])
    base["runs"] = [merged[s] for s in sorted(EXPECTED_SEEDS)]
    base.setdefault("args", {})["seeds"] = sorted(EXPECTED_SEEDS)
    base["args"]["output"] = f"le0_le2_extension/phase5_v4_{arm}_n10.json"
    meta = base.setdefault("meta", {})
    meta["le0_le2_n10_merge"] = True
    meta["le0_le2_n10_merge_parts"] = [str(p.relative_to(root)) for p in parts]
    meta["le0_le2_n10_merged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    out_path = root / "results" / "le0_le2_extension" / f"phase5_v4_{arm}_n10.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(base, indent=2, default=str) + "\n")
    return base


def summarize_arm(arm: str, data: dict, baseline_values: dict[int, float]) -> dict:
    values = metric_values(data)
    if set(values) != EXPECTED_SEEDS:
        raise RuntimeError(f"{arm} values do not cover seeds 0-9 exactly")
    arm_values = [values[s] for s in sorted(EXPECTED_SEEDS)]
    deltas = [values[s] - baseline_values[s] for s in sorted(EXPECTED_SEEDS)]
    mean_value, sd_value = mean_sd(arm_values)
    mean_delta, sd_delta = mean_sd(deltas)
    return {
        "arm": arm,
        "label": ARMS[arm]["label"],
        "train_rows": ARMS[arm]["train_rows"],
        "kvasir_kept": ARMS[arm]["kvasir_kept"],
        "n": 10,
        "seeds": sorted(EXPECTED_SEEDS),
        "orig_val_bal_acc_mean": mean_value,
        "orig_val_bal_acc_sd": sd_value,
        "delta_vs_baseline_mean": mean_delta,
        "delta_vs_baseline_sd": sd_delta,
        "delta_vs_baseline_se": sd_delta / math.sqrt(len(deltas)),
        "source_json": f"results/le0_le2_extension/phase5_v4_{arm}_n10.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()

    baseline = reference_baseline(root)
    baseline_values = metric_values(baseline)
    if set(baseline_values) != EXPECTED_SEEDS:
        raise RuntimeError("baseline reference must contain seeds 0-9 exactly")

    rows = []
    for arm in ARMS:
        merged = merge_arm(root, arm)
        rows.append(summarize_arm(arm, merged, baseline_values))

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "complete",
        "baseline_reference": baseline.get("meta", {}).get("reference_path"),
        "rows": rows,
    }
    out_json = root / "results" / "le0_le2_extension" / "le0_le2_n10_extension_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# le0/le2 n=10 Extension Summary",
        "",
        "| Arm | n | arm mean | sd | delta vs baseline | Source |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {arm} | {n} | {mean:.4f} | {sd:.4f} | {delta:+.4f} | `{source}` |".format(
                arm=row["arm"],
                n=row["n"],
                mean=row["orig_val_bal_acc_mean"],
                sd=row["orig_val_bal_acc_sd"],
                delta=row["delta_vs_baseline_mean"],
                source=row["source_json"],
            )
        )
    out_md = root / "results" / "le0_le2_extension" / "le0_le2_n10_extension_summary.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
