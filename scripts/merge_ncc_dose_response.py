#!/usr/bin/env python3
"""Merge NCC dose-response shards and write n=10 summaries.

The paper reports NCC-99/95/90/85/80 on the same ten paired seeds. Some
thresholds were produced as two shards: seeds 0-3 (`*_n4.json`) and seeds
4-9 (`*_seeds49.json`). This script merges those shards into canonical
`*_n10.json` files and writes a compact summary used by the smoke test.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path


THRESHOLDS = ("99", "95", "90", "85", "80")
EXPECTED_SEEDS = set(range(10))


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def runs_by_seed(data: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for run in data.get("runs", []):
        seed = int(run["seed"])
        if seed in out:
            raise RuntimeError(f"duplicate seed {seed} inside {data.get('args', {}).get('output')}")
        out[seed] = run
    return out


def last_metric(run: dict, split: str = "orig_val", metric: str = "bal_acc") -> float:
    if "last" in run:
        return float(run["last"][split][metric])
    return float(run["history"][-1][split][metric])


def mean_sd(xs: list[float]) -> tuple[float, float]:
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def find_part(root: Path, threshold: str, suffix: str) -> Path:
    candidates = [
        root / "results" / "mechanism_probes" / f"phase8_ncc{threshold}_v5_{suffix}.json",
        root / "results" / "r10_experiments" / f"phase8_ncc{threshold}_v5_{suffix}.json",
        root.parent / "results" / "mechanism_probes" / f"phase8_ncc{threshold}_v5_{suffix}.json",
        root.parent / "results" / "r10_experiments" / f"phase8_ncc{threshold}_v5_{suffix}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing NCC-{threshold} {suffix}; tried {[str(c) for c in candidates]}")


def merge_threshold(root: Path, threshold: str) -> dict:
    parts = [find_part(root, threshold, "n4"), find_part(root, threshold, "seeds49")]
    loaded = [load_json(p) for p in parts]
    merged: dict[int, dict] = {}
    for path, data in zip(parts, loaded):
        for seed, run in runs_by_seed(data).items():
            if seed in merged:
                raise RuntimeError(f"duplicate seed {seed} while merging NCC-{threshold}")
            merged[seed] = run
    missing = sorted(EXPECTED_SEEDS - set(merged))
    extra = sorted(set(merged) - EXPECTED_SEEDS)
    if missing or extra:
        raise RuntimeError(f"NCC-{threshold} seed mismatch: missing={missing}, extra={extra}")

    out_path = root / "results" / "mechanism_probes" / f"phase8_ncc{threshold}_v5_n10.json"
    base = copy.deepcopy(loaded[0])
    base["runs"] = [merged[s] for s in sorted(EXPECTED_SEEDS)]
    base.setdefault("args", {})["seeds"] = sorted(EXPECTED_SEEDS)
    base["args"]["output"] = str(out_path.relative_to(root / "results"))
    meta = base.setdefault("meta", {})
    meta["ncc_merge"] = True
    meta["ncc_merge_parts"] = [
        str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
        for p in parts
    ]
    meta["ncc_merged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(base, indent=2, default=str) + "\n")
    return base


def values_by_seed(data: dict, split: str = "orig_val") -> dict[int, float]:
    return {int(r["seed"]): last_metric(r, split=split) for r in data.get("runs", [])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()

    baseline = load_json(root / "results" / "baseline" / "phase5_v5_baseline_n10.json")
    le6 = load_json(root / "results" / "baseline" / "phase5_v5_le6_n10.json")
    baseline_values = values_by_seed(baseline)
    le6_values = values_by_seed(le6)
    if set(baseline_values) != EXPECTED_SEEDS or set(le6_values) != EXPECTED_SEEDS:
        raise RuntimeError("baseline/le6 references must both contain seeds 0-9")

    rows = []
    merged_by_threshold = {}
    for threshold in THRESHOLDS:
        merged = merge_threshold(root, threshold)
        vals = values_by_seed(merged)
        arm_values = [vals[s] for s in sorted(EXPECTED_SEEDS)]
        deltas = [vals[s] - baseline_values[s] for s in sorted(EXPECTED_SEEDS)]
        mean_value, sd_value = mean_sd(arm_values)
        mean_delta, sd_delta = mean_sd(deltas)
        rows.append(
            {
                "threshold": f"NCC >= 0.{threshold}",
                "n": len(arm_values),
                "seeds": sorted(vals),
                "orig_val_bal_acc_mean": mean_value,
                "orig_val_bal_acc_sd": sd_value,
                "delta_vs_baseline_mean": mean_delta,
                "delta_vs_baseline_sd": sd_delta,
                "delta_vs_baseline_se": sd_delta / math.sqrt(len(deltas)),
                "source_json": f"results/mechanism_probes/phase8_ncc{threshold}_v5_n10.json",
            }
        )
        merged_by_threshold[threshold] = vals

    le6_deltas = [le6_values[s] - baseline_values[s] for s in sorted(EXPECTED_SEEDS)]
    le6_mean, le6_sd = mean_sd([le6_values[s] for s in sorted(EXPECTED_SEEDS)])
    le6_delta_mean, le6_delta_sd = mean_sd(le6_deltas)
    rows.append(
        {
            "threshold": "le6",
            "n": 10,
            "seeds": sorted(EXPECTED_SEEDS),
            "orig_val_bal_acc_mean": le6_mean,
            "orig_val_bal_acc_sd": le6_sd,
            "delta_vs_baseline_mean": le6_delta_mean,
            "delta_vs_baseline_sd": le6_delta_sd,
            "delta_vs_baseline_se": le6_delta_sd / math.sqrt(10),
            "source_json": "results/baseline/phase5_v5_le6_n10.json",
        }
    )

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "baseline": {
            "n": 10,
            "orig_val_bal_acc_mean": statistics.mean(baseline_values.values()),
            "orig_val_bal_acc_sd": statistics.stdev(baseline_values.values()),
            "source_json": "results/baseline/phase5_v5_baseline_n10.json",
        },
        "rows": rows,
    }
    out_json = root / "results" / "mechanism_probes" / "ncc_dose_response_n10_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# NCC Dose-Response n=10 Summary",
        "",
        "| Threshold | n | arm mean | sd | delta vs baseline | Source |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {threshold} | {n} | {mean:.4f} | {sd:.4f} | {delta:+.4f} | `{source}` |".format(
                threshold=row["threshold"],
                n=row["n"],
                mean=row["orig_val_bal_acc_mean"],
                sd=row["orig_val_bal_acc_sd"],
                delta=row["delta_vs_baseline_mean"],
                source=row["source_json"],
            )
        )
    out_md = root / "results" / "mechanism_probes" / "ncc_dose_response_n10_summary.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
