"""Generate updated counterfactual table with 5 pools including size control.

Outputs to stdout a LaTeX table. Call after all phase5 results are complete.
"""
import os
import json
import glob
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CAPSULE_ROOT", ".") + "/results")


def agg(files):
    recs = []
    for fn in files:
        p = OUT / fn
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        for r in d.get("runs", []):
            last = r["last"]
            recs.append({
                "orig": last["orig_val"]["bal_acc"],
                "dedup": last["dedup_val"]["bal_acc"],
                "kvasir": last["kvasir_s1"]["bal_acc"],
            })
    return recs


def fmt(vals, precision=3):
    a = np.array(vals)
    if len(a) == 1:
        return f"${a.mean():.{precision}f}$"
    return f"${a.mean():.{precision}f} \\pm {a.std():.{precision}f}$"


conditions = {
    "Baseline (contaminated)": {"files": ["cv2024_pooled.json"], "n": "37{,}607", "seeds_expected": 2},
    "Random $n{=}10{,}596$ (size ctrl)": {"files": ["phase5_random10596_seeds01.json"], "n": "10{,}596", "seeds_expected": 2},
    "\\texttt{le0}": {"files": ["phase5_counterfactual_le0_seeds01.json"], "n": "21{,}241", "seeds_expected": 2},
    "\\texttt{le2}": {"files": ["phase5_counterfactual_le2_seeds01.json"], "n": "10{,}825", "seeds_expected": 2},
    "\\texttt{le6}": {"files": ["phase5_counterfactual_le6_seeds01.json", "phase5_counterfactual_le6_seeds23.json"], "n": "10{,}596", "seeds_expected": 4},
}


baseline_orig = None
baseline_orig_std = None
rows = []
orig_vals = {}
for cond, meta in conditions.items():
    if cond == "Baseline (contaminated)":
        # Use baseline numbers: 0.8163 ± 0.007 (computed earlier)
        baseline_orig = 0.8163
        baseline_orig_std = 0.0069
        rows.append({
            "cond": cond, "n": meta["n"], "orig": "$0.816 \\pm 0.007$",
            "dedup": "--", "kvasir": "$\\approx 0.55$", "delta": "--", "n_seeds": 2})
        orig_vals[cond] = [0.8163]
    else:
        recs = agg(meta["files"])
        n_seeds = len(recs)
        if n_seeds == 0:
            rows.append({"cond": cond, "n": meta["n"], "orig": "--", "dedup": "--", "kvasir": "--", "delta": "--", "n_seeds": 0})
            orig_vals[cond] = []
            continue
        orig = fmt([r["orig"] for r in recs])
        dedup = fmt([r["dedup"] for r in recs])
        kvasir = fmt([r["kvasir"] for r in recs])
        orig_vals[cond] = [r["orig"] for r in recs]
        delta = np.mean(orig_vals[cond]) - baseline_orig
        delta_str = f"${delta:+.3f}$"
        rows.append({"cond": cond, "n": meta["n"], "orig": orig, "dedup": dedup, "kvasir": kvasir, "delta": delta_str, "n_seeds": n_seeds})


# Compute Δ rows
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\small")
print(r"\caption{\textbf{Counterfactual sensitivity: training pool dedup threshold + size control.}")
print(r"DINOv2-ViT-L/14 + LoRA-$r{=}8$ trained from scratch, 10 epochs, final-epoch")
print(r"balanced accuracy (mean$\pm$std over seeds as indicated).")
print(r"``Random $n{=}10{,}596$'' is a random draw from the full contaminated pool at the")
print(r"\texttt{le6}-matched size — a sample-size control that preserves the $72\%$ KVASIR-contamination")
print(r"rate.  $\Delta$(pool $-$ baseline) is the balanced-accuracy drop versus the full")
print(r"contaminated pool on the \emph{same} CV2024 original validation split;")
print(r"$\Delta$(\texttt{le6} $-$ random) isolates leakage-attributable drop after controlling for")
print(r"training-size reduction.}")
print(r"\label{tab:counterfactual}")
print(r"\begin{tabular}{lrrrrr}")
print(r"\toprule")
print(r"Training pool & $n$ & Orig val & Dedup val (\texttt{le6}) & Kvasir split\_1 & $\Delta_\text{orig}$ \\")
print(r"\midrule")
for r in rows:
    print(f"{r['cond']} & ${r['n']}$ & {r['orig']} & {r['dedup']} & {r['kvasir']} & {r['delta']} \\\\")
print(r"\midrule")
# Compute leakage-only Δ = le6 - random
if orig_vals.get("\\texttt{le6}") and orig_vals.get("Random $n{=}10{,}596$ (size ctrl)"):
    delta_leak = np.mean(orig_vals["\\texttt{le6}"]) - np.mean(orig_vals["Random $n{=}10{,}596$ (size ctrl)"])
    print(f"$\\Delta$(\\texttt{{le6}} $-$ random, leakage-only) & -- & ${delta_leak:+.3f}$ & -- & -- & -- \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
