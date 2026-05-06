"""Consolidate phase5 counterfactual results into one summary table.

Reads:
  phase5_counterfactual_le0_seeds01.json
  phase5_counterfactual_le2_seeds01.json
  phase5_counterfactual_le6_seeds01.json
  phase5_counterfactual_le6_seeds23.json
  phase5_random10596_seeds01.json
  phase5_random10596_s0_seeds01.json  (optional 2nd)

Outputs LaTeX table row lines to stdout.
"""
import os
import json
import glob
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CAPSULE_ROOT", ".") + "/results")

CONDITIONS = {
    "Original (contaminated)": [],  # baseline from text, n=37607
    "le0": ["phase5_counterfactual_le0_seeds01.json"],
    "le2": ["phase5_counterfactual_le2_seeds01.json"],
    "le6": ["phase5_counterfactual_le6_seeds01.json", "phase5_counterfactual_le6_seeds23.json"],
    "random10596_s42": ["phase5_random10596_seeds01.json"],
    "random10596_s0": ["phase5_random10596_s0_seeds01.json"],
}

SIZES = {
    "Original (contaminated)": 37607,
    "le0": 21241,
    "le2": 10825,
    "le6": 10596,
    "random10596_s42": 10596,
    "random10596_s0": 10596,
}


def aggregate(files):
    records = []
    for fn in files:
        p = OUT / fn
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        for r in d.get("runs", []):
            last = r["last"]
            records.append({
                "seed": r["seed"],
                "orig_val": last["orig_val"]["bal_acc"],
                "dedup_val": last["dedup_val"]["bal_acc"],
                "kvasir_s1": last["kvasir_s1"]["bal_acc"],
            })
    return records


baseline_orig = 0.816
lines = []
print(f"{'Condition':<30} {'n':>6} {'OrigVal':>12} {'DedupVal':>12} {'KvasirS1':>12} {'Δ_orig':>10}")
for cond, files in CONDITIONS.items():
    rec = aggregate(files) if files else [{"orig_val": baseline_orig, "dedup_val": None, "kvasir_s1": 0.55, "seed": -1}]
    if not rec:
        print(f"{cond:<30} {SIZES.get(cond, 0):>6} NO_DATA")
        continue
    o = np.array([r["orig_val"] for r in rec])
    dv = np.array([r["dedup_val"] for r in rec if r["dedup_val"] is not None])
    k = np.array([r["kvasir_s1"] for r in rec if r["kvasir_s1"] is not None])
    n_seeds = len(o)
    fmt = lambda m, s: f"{m:.4f}±{s:.4f}" if n_seeds > 1 else f"{m:.4f}"
    orig_str = fmt(o.mean(), o.std())
    dv_str = fmt(dv.mean(), dv.std()) if len(dv) else "--"
    k_str = fmt(k.mean(), k.std()) if len(k) else "--"
    delta_str = f"{o.mean() - baseline_orig:+.4f}" if cond != "Original (contaminated)" else "  -    "
    print(f"{cond:<30} {SIZES.get(cond, 0):>6} {orig_str:>12} {dv_str:>12} {k_str:>12} {delta_str:>10}  (n_seeds={n_seeds})")
