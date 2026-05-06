"""Filter-pass rate vs severity (mechanistic diagnosis figure).

Plots the fraction of test samples passing the entropy filter
(tau = alpha * log K, alpha typically 0.3-0.5) as a function of
corruption severity, for 4 corruption types. Demonstrates the
'confidence-filter paradox': as severity increases, entropy rises
and the filter silences more of the adaptation signal.
"""
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import json

ROOT = Path(os.environ.get("CAPSULE_ROOT", "."))
OUT = ROOT / "results"
FIG_DIR = Path(os.environ.get("CAPSULE_FIGURES_DIR", ROOT / "figures"))


def main():
    src_log = Path(os.environ.get("TTA_FILTER_LOG", ROOT / "results/stage6_corruption.txt"))
    if not src_log.exists():
        shipped = ROOT / "artifacts/summaries/filter_pass_summary.json"
        if shipped.exists():
            OUT.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(shipped, OUT / "filter_pass_summary.json")
            print(f"{src_log} not found; copied shipped summary to {OUT / 'filter_pass_summary.json'}")
            return
        raise SystemExit(
            f"{src_log} not found. Set TTA_FILTER_LOG or run the TTA log-producing job first."
        )
    data = defaultdict(list)
    with open(src_log) as f:
        for line in f:
            m = re.search(r"\[(\S+) sev=(\d+) fold=(\d+) (\S+)\] bal=([\d.]+) ent=([\d.]+) pass=([\d.]+)", line)
            if not m: continue
            corr, sev, fold, method, bal, ent, p = m.groups()
            data[(corr, int(sev))].append({
                "fold": int(fold), "method": method,
                "bal": float(bal), "ent": float(ent), "pass": float(p)
            })

    corruptions = sorted({k[0] for k in data})
    sevs = sorted({k[1] for k in data})

    # Save summary JSON
    summary = {}
    for corr in corruptions:
        summary[corr] = {}
        for sev in sevs:
            rows = data.get((corr, sev), [])
            if not rows: continue
            summary[corr][sev] = {
                "n_cells": len(rows),
                "entropy_mean": float(np.mean([r["ent"] for r in rows])),
                "pass_mean": float(np.mean([r["pass"] for r in rows])),
                "pass_std": float(np.std([r["pass"] for r in rows])),
                "bal_mean": float(np.mean([r["bal"] for r in rows])),
            }
    with open(OUT / "filter_pass_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        colors = {"colorjitter_blur": "#4C72B0", "jpeg": "#55A868",
                  "motion_blur": "#C44E52", "noise": "#CCB974",
                  "posterize": "#8172B2"}
        for corr in corruptions:
            x = [s for s in sevs if (corr, s) in data]
            ent_y = [summary[corr][s]["entropy_mean"] for s in x]
            pass_y = [summary[corr][s]["pass_mean"] for s in x]
            ax1.plot(x, ent_y, marker="o", color=colors.get(corr, "gray"),
                     label=corr, linewidth=2)
            ax2.plot(x, pass_y, marker="o", color=colors.get(corr, "gray"),
                     label=corr, linewidth=2)
        ax1.axhline(np.log(11), color="black", linestyle=":", alpha=0.5,
                    label=r"$\log K \approx 2.40$")
        ax1.set_xlabel("Corruption severity")
        ax1.set_ylabel("Mean pre-adaptation entropy (nats)")
        ax1.set_title("(a) Entropy rises with severity")
        ax1.legend(fontsize=8, loc="lower right")
        ax1.grid(True, alpha=0.3)
        ax2.set_xlabel("Corruption severity")
        ax2.set_ylabel("Fraction of samples passing entropy filter\n($\\tau = 0.4\\log K$)")
        ax2.set_title("(b) Filter-pass rate collapses")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.02, 0.25)
        plt.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "fig_filter_pass.png", dpi=150, bbox_inches="tight")
        print(f"Saved {FIG_DIR / 'fig_filter_pass.png'}")
    except Exception as e:
        print(f"Matplotlib error: {e}")


if __name__ == "__main__":
    main()
