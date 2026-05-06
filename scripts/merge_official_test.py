#!/usr/bin/env python3
"""Merge official-test direct-evaluation shards."""

from __future__ import annotations

import json
import math
import re
import argparse
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/official_test"
OUT_JSON = ROOT / "results/official_test/official_test_direct_eval_summary.json"
OUT_MD = ROOT / "results/official_test/official_test_direct_eval_summary.md"

METRICS = ["bal_acc", "mean_auc", "combined", "acc", "f1_macro"]
ARM_ORDER = ["baseline", "random10596_s0", "le6"]


def sd(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def se(xs: list[float]) -> float:
    return sd(xs) / math.sqrt(len(xs)) if xs else float("nan")


def summarize_values(xs: list[float]) -> dict:
    return {
        "n": len(xs),
        "mean": mean(xs) if xs else None,
        "sd": sd(xs) if xs else None,
        "se": se(xs) if xs else None,
        "median": median(xs) if xs else None,
        "min": min(xs) if xs else None,
        "max": max(xs) if xs else None,
    }


def parse_file(path: Path):
    m = re.match(r"(.+)_official_test_s(\d+)\.json$", path.name)
    if not m:
        return None
    arm = m.group(1)
    seed = int(m.group(2))
    obj = json.loads(path.read_text())
    if len(obj.get("runs", [])) != 1:
        raise ValueError(f"{path} expected exactly one run")
    run = obj["runs"][0]
    last = run["last"]["official_test"]
    try:
        rel_path = str(path.relative_to(ROOT))
    except ValueError:
        rel_path = path.name
    return {
        "arm": arm,
        "seed": seed,
        "path": rel_path,
        "metrics": {k: float(last[k]) for k in METRICS if last.get(k) is not None},
        "per_class": last.get("per_class", {}),
        "n_test": int(last["n_test"]),
        "meta": obj.get("meta", {}),
    }


def paired_contrast(records: dict[str, dict[int, dict]], a: str, b: str, metric: str) -> dict:
    common = sorted(set(records.get(a, {})) & set(records.get(b, {})))
    diffs = [
        records[b][s]["metrics"][metric] - records[a][s]["metrics"][metric]
        for s in common
        if metric in records[b][s]["metrics"] and metric in records[a][s]["metrics"]
    ]
    out = summarize_values(diffs)
    out.update({"a": a, "b": b, "metric": metric, "common_seeds": common})
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=str(RESULT_DIR))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    args = parser.parse_args(argv)
    result_dir = Path(args.result_dir)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)

    result_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[int, dict]] = {}
    skipped = []
    for path in sorted(result_dir.glob("*_official_test_s*.json")):
        rec = parse_file(path)
        if rec is None:
            skipped.append(str(path))
            continue
        records.setdefault(rec["arm"], {})[rec["seed"]] = rec

    arms = {}
    for arm, by_seed in sorted(records.items()):
        rows = [by_seed[s] for s in sorted(by_seed)]
        arms[arm] = {
            "n": len(rows),
            "seeds": sorted(by_seed),
            "n_test_values": sorted(set(r["n_test"] for r in rows)),
            "metrics": {
                metric: summarize_values([r["metrics"][metric] for r in rows if metric in r["metrics"]])
                for metric in METRICS
            },
            "rows": rows,
        }

    contrasts = {}
    for a, b in [
        ("baseline", "le6"),
        ("random10596_s0", "le6"),
        ("baseline", "random10596_s0"),
    ]:
        if a in records and b in records:
            contrasts[f"{b}_minus_{a}"] = {
                metric: paired_contrast(records, a, b, metric) for metric in METRICS
            }

    out = {
        "description": "DINOv2-L/14 LoRA direct evaluation on the released CV2024 official AIIMS test images.",
        "arms": arms,
        "contrasts": contrasts,
        "skipped": skipped,
    }
    out_json.write_text(json.dumps(out, indent=2) + "\n")

    lines = [
        "# Official AIIMS-test direct evaluation",
        "",
        "| Arm | n | bal acc | mean AUC | combined | accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        a = arms[arm]
        lines.append(
            "| {arm} | {n} | {ba:.4f}±{ba_se:.4f} | {auc:.4f}±{auc_se:.4f} | {comb:.4f}±{comb_se:.4f} | {acc:.4f}±{acc_se:.4f} |".format(
                arm=arm,
                n=a["n"],
                ba=a["metrics"]["bal_acc"]["mean"],
                ba_se=a["metrics"]["bal_acc"]["se"],
                auc=a["metrics"]["mean_auc"]["mean"],
                auc_se=a["metrics"]["mean_auc"]["se"],
                comb=a["metrics"]["combined"]["mean"],
                comb_se=a["metrics"]["combined"]["se"],
                acc=a["metrics"]["acc"]["mean"],
                acc_se=a["metrics"]["acc"]["se"],
            )
        )
    lines += ["", "## Paired Contrasts", ""]
    lines.append("| Contrast | metric | n | mean diff | SE |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for cname, vals in contrasts.items():
        for metric in ["bal_acc", "mean_auc", "combined", "acc"]:
            v = vals[metric]
            lines.append(
                f"| {cname} | {metric} | {v['n']} | {v['mean']:+.4f} | {v['se']:.4f} |"
            )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    for arm in ARM_ORDER:
        if arm in arms:
            print(arm, arms[arm]["n"], arms[arm]["metrics"]["combined"])


if __name__ == "__main__":
    main()
