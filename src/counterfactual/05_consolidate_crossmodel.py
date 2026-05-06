"""Consolidate cross-model counterfactual results.

Prefer n=10 artifacts for the paper table:
  phase5_crossmodel_{dinov2B,dinov2S,resnet50,convnextT}_{baseline,random,le6}_n10.json

Falls back to older n=4/seeds01 files only for legacy package checks.
Outputs a LaTeX table row summary to stdout.
"""
import os
import json
import statistics
from pathlib import Path

OUT = Path(os.environ.get("CAPSULE_ROOT", ".") + "/results")


def _run_metric(run, key="orig_val"):
    if "last" in run:
        return run["last"][key]["bal_acc"]
    return run["history"][-1][key]["bal_acc"]


def _agg(files, key="orig_val"):
    vals = []
    for f in files:
        p = OUT / f
        if not p.exists():
            return None
        with open(p) as f2:
            d = json.load(f2)
        for r in d.get("runs", []):
            vals.append(_run_metric(r, key=key))
    return vals


BACKBONES = ["dinov2_vitl14", "dinov2_vitb14", "dinov2_vits14", "resnet50", "convnext_tiny"]
LABELS = {
    "dinov2_vitl14": "DINOv2-L/14 (LoRA r=8)",
    "dinov2_vitb14": "DINOv2-B/14 (LoRA r=8)",
    "dinov2_vits14": "DINOv2-S/14 (LoRA r=8)",
    "resnet50":      "ResNet-50 (linear probe)",
    "convnext_tiny":  "ConvNeXt-Tiny (linear probe)",
}
POOLS = [
    ("baseline", "Baseline (contaminated, $n{=}37{,}607$)"),
    ("random",   "Random $n{=}10{,}596$ (size ctrl)"),
    ("le6",      "\\texttt{le6} ($n{=}10{,}596$, no KVASIR)"),
]


def fmt(arr):
    if arr is None or len(arr) == 0: return "--"
    if len(arr) == 1: return f"${arr[0]:.3f}$"
    return f"${statistics.mean(arr):.3f}\\pm{statistics.stdev(arr):.3f}$"


def get_result(backbone, pool):
    """Return numpy array of orig_val bal_acc across seeds."""
    if backbone == "dinov2_vitl14":
        if pool == "baseline":
            return _agg(["baseline/phase5_v5_baseline_n10.json"])
        elif pool == "random":
            return _agg(["acceptance_lift/phase5_v4_random_s0_n10.json"])
        elif pool == "le6":
            return _agg(["baseline/phase5_v5_le6_n10.json"])
    else:
        short = {
            "dinov2_vitb14": "dinov2B",
            "dinov2_vits14": "dinov2S",
            "resnet50": "resnet50",
            "convnext_tiny": "convnextT",
        }[backbone]
        n10 = OUT / "crossmodel" / f"phase5_crossmodel_{short}_{pool}_n10.json"
        if n10.exists():
            return _agg([f"crossmodel/phase5_crossmodel_{short}_{pool}_n10.json"])
        n4 = OUT / "crossmodel" / f"phase5_crossmodel_{short}_{pool}_n4.json"
        if n4.exists():
            return _agg([f"crossmodel/phase5_crossmodel_{short}_{pool}_n4.json"])
        return _agg([f"crossmodel/phase5_crossmodel_{short}_{pool}_seeds01.json"])


def main():
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\caption{\textbf{Cross-model counterfactual generalization.}  Final-epoch")
    print(r"balanced accuracy on the original CV2024 public validation split (mean$\pm$std")
    print(r"over seeds) for five model families trained on three pools.  $\Delta$(pool $-$ baseline)")
    print(r"isolates the drop due to removing data; the last column subtracts the size-only")
    print(r"drop (``random'') from the \texttt{le6} drop to yield the \emph{not size-attributable}")
    print(r"residual.  The monotone pattern reproduces across all five backbones.}")
    print(r"\label{tab:crossmodel}")
    print(r"\begin{tabular}{l|rrr|rr|r}")
    print(r"\toprule")
    print(r"Backbone & Baseline & Random & \texttt{le6} & $\Delta_\text{random}$ & $\Delta_\text{\texttt{le6}}$ & residual \\")
    print(r"\midrule")
    for bb in BACKBONES:
        base = get_result(bb, "baseline")
        rnd  = get_result(bb, "random")
        l6   = get_result(bb, "le6")
        if base is None or rnd is None or l6 is None:
            print(f"{LABELS[bb]} & {fmt(base)} & {fmt(rnd)} & {fmt(l6)} & ? & ? & ? \\\\")
            continue
        d_rnd = statistics.mean(rnd) - statistics.mean(base)
        d_l6  = statistics.mean(l6) - statistics.mean(base)
        residual = d_l6 - d_rnd
        print(f"{LABELS[bb]} & {fmt(base)} & {fmt(rnd)} & {fmt(l6)} & ${d_rnd:+.3f}$ & ${d_l6:+.3f}$ & $\\mathbf{{{residual:+.3f}}}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
