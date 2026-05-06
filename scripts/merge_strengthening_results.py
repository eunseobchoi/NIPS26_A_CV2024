#!/usr/bin/env python3
"""Merge strengthening-experiment shards into paper-facing aggregates."""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path


SPLITS = ("orig_val", "dedup_val", "kvasir_s1")


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def mean_sd(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def runs_by_seed(data: dict) -> dict[int, dict]:
    out = {}
    for run in data.get("runs", []):
        seed = int(run["seed"])
        if seed in out:
            raise RuntimeError(f"duplicate seed {seed} inside one JSON")
        out[seed] = run
    return out


def merge_parts(parts: list[Path], output: Path, expected: set[int]) -> dict:
    loaded = [load_json(p) for p in parts]
    if not loaded:
        raise ValueError(f"no input parts for {output}")
    merged_runs = {}
    for path, data in zip(parts, loaded):
        for seed, run in runs_by_seed(data).items():
            if seed in merged_runs:
                raise RuntimeError(f"duplicate seed {seed} while merging {output}")
            merged_runs[seed] = run
    missing = sorted(expected - set(merged_runs))
    extra = sorted(set(merged_runs) - expected)
    if missing or extra:
        raise RuntimeError(f"{output} seed mismatch: missing={missing}, extra={extra}")

    base = copy.deepcopy(loaded[0])
    base["runs"] = [merged_runs[s] for s in sorted(expected)]
    base.setdefault("args", {})["seeds"] = sorted(expected)
    base["args"]["output"] = str(output.relative_to(output.parents[1]))
    meta = base.setdefault("meta", {})
    meta["strengthening_merged"] = True
    meta["strengthening_merge_parts"] = [
        str(p.relative_to(output.parents[1])) for p in parts
    ]
    meta["strengthening_merged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(base, indent=2, default=str) + "\n")
    return base


def metric_values(data: dict, split: str = "orig_val", metric: str = "bal_acc") -> dict[int, float]:
    values = {}
    for run in data.get("runs", []):
        values[int(run["seed"])] = float(run["last"][split][metric])
    return values


def row(label: str, data: dict) -> dict:
    result = {
        "label": label,
        "n": len(data.get("runs", [])),
        "seeds": [int(r["seed"]) for r in data.get("runs", [])],
    }
    for split in SPLITS:
        xs = list(metric_values(data, split=split).values())
        m, s = mean_sd(xs)
        result[f"{split}_bal_acc_mean"] = m
        result[f"{split}_bal_acc_sd"] = s
    return result


def paired_delta(label: str, left: dict, right: dict, split: str = "orig_val") -> dict:
    a = metric_values(left, split=split)
    b = metric_values(right, split=split)
    seeds = sorted(set(a) & set(b))
    deltas = [a[s] - b[s] for s in seeds]
    m, s = mean_sd(deltas)
    se = s / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    return {
        "label": label,
        "split": split,
        "n": len(deltas),
        "seeds": seeds,
        "mean_delta": m,
        "sd_delta": s,
        "se_delta": se,
    }


def required_load(root: Path, label: str, rels: list[str]) -> dict:
    tried = []
    for rel in rels:
        path = root / rel
        tried.append(str(path))
        if path.exists():
            data = load_json(path)
            data.setdefault("meta", {})["strengthening_reference_path"] = rel
            return data
    raise FileNotFoundError(f"missing required {label} reference; tried {tried}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    args = ap.parse_args()
    root = args.root.resolve()

    compc_parts = [root / "results" / "strengthening" / f"compC_s{s}.json" for s in range(10)]
    pathb_parts = [root / "results" / "mechanism_probes" / "phase5_exp1_le6_kvfree_s1_n4.json"]
    pathb_parts += [root / "results" / "mechanism_probes" / f"pathb_exp1_s{s}.json" for s in range(4, 10)]
    for path in compc_parts + pathb_parts:
        if not path.exists():
            raise FileNotFoundError(path)

    compc = merge_parts(
        compc_parts,
        root / "results" / "strengthening" / "phase5_v5_compC_aiims_ulcer_s0_n10.json",
        set(range(10)),
    )
    pathb = merge_parts(
        pathb_parts,
        root / "results" / "mechanism_probes" / "phase5_exp1_le6_kvfree_s1_n10.json",
        set(range(10)),
    )

    summary = {
        "compC_aiims_ulcer_oversampled": row("Comp-C AIIMS-only doubled Ulcer", compc),
        "pathb_exp1_le6_kvfree_s1": row("Path-B Exp.1 le6_kvfree_s1", pathb),
        "contrasts": [],
    }
    references = {
        "le6": required_load(root, "le6", [
            "results/baseline/phase5_v5_le6_n10.json",
            "results/counterfactual_n10/phase5_v4_le6_n10.json",
        ]),
        "compA": required_load(root, "Comp-A", [
            "results/acceptance_lift/phase5_v4_compA_s0_n10.json",
        ]),
        "compB": required_load(root, "Comp-B", [
            "results/counterfactual_n10/phase5_v4_compB_s0_n10.json",
        ]),
    }
    summary["reference_paths"] = {
        key: value.get("meta", {}).get("strengthening_reference_path")
        for key, value in references.items()
    }
    summary["contrasts"].append(paired_delta("Path-B Exp.1 - le6", pathb, references["le6"]))
    summary["contrasts"].append(paired_delta("Comp-C - Comp-A", compc, references["compA"]))
    summary["contrasts"].append(paired_delta("Comp-C - Comp-B", compc, references["compB"]))

    out_json = root / "results" / "strengthening" / "strengthening_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Strengthening Experiment Summary",
        "",
        "## Comp-C AIIMS-only doubled Ulcer",
        "",
        f"- n={summary['compC_aiims_ulcer_oversampled']['n']}",
        f"- orig-val balanced accuracy: {summary['compC_aiims_ulcer_oversampled']['orig_val_bal_acc_mean']:.4f} "
        f"+/- {summary['compC_aiims_ulcer_oversampled']['orig_val_bal_acc_sd']:.4f}",
        "",
        "## Path-B Exp.1 le6_kvfree_s1",
        "",
        f"- n={summary['pathb_exp1_le6_kvfree_s1']['n']}",
        f"- orig-val balanced accuracy: {summary['pathb_exp1_le6_kvfree_s1']['orig_val_bal_acc_mean']:.4f} "
        f"+/- {summary['pathb_exp1_le6_kvfree_s1']['orig_val_bal_acc_sd']:.4f}",
        "",
        "## Paired Contrasts",
        "",
    ]
    for contrast in summary["contrasts"]:
        lines.append(
            f"- {contrast['label']} ({contrast['split']}, n={contrast['n']}): "
            f"mean delta = {contrast['mean_delta']:+.4f} "
            f"(SE = {contrast['se_delta']:.4f}; SD across paired seeds = {contrast['sd_delta']:.4f})"
        )
    out_md = root / "results" / "strengthening" / "strengthening_summary.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
