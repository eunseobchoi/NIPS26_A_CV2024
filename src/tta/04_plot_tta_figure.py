"""Plot TTA balanced accuracy vs corruption severity per method, with
no-adapt baseline and 1/K chance line."""
import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

RESULTS = Path(os.environ.get("CAPSULE_ROOT", ".") + "/results")
PAPER = Path(os.environ.get("CAPSULE_ROOT", ".") + "/paper/tmlr")


def main():
    with open(RESULTS / "kvasir_tta_bench.json") as f:
        d = json.load(f)
    runs = d["runs"]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["method"], r["severity"])].append(r["bal_acc"])

    methods = sorted({m for m, _ in groups.keys()})
    sevs = sorted({s for _, s in groups.keys()})

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {"no_adapt": "#3f3f3f", "ln_adapt": "#4C72B0",
                  "head_ttt": "#55A868", "sar_official": "#DD8452"}
        markers = {"no_adapt": "o", "ln_adapt": "s",
                   "head_ttt": "^", "sar_official": "D"}
        for m in methods:
            means = []
            stds = []
            for s in sevs:
                vals = groups.get((m, s), [])
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            ax.errorbar(sevs, means, yerr=stds,
                        marker=markers.get(m, "o"),
                        color=colors.get(m, "gray"),
                        linewidth=2, capsize=3,
                        label=m.replace("_", "-"))

        K = 11
        ax.axhline(1/K, color="gray", linestyle=":", alpha=0.6,
                   label=f"1/K chance (1/{K})")
        ax.set_xlabel("Corruption severity")
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(0.05, 0.35)
        ax.set_xticks(sevs)
        ax.set_title("TTA on Kvasir official frame-list split (fold-0 LoRA ckpt)")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(PAPER / "fig_tta.png", dpi=150, bbox_inches="tight")
        print(f"Saved {PAPER / 'fig_tta.png'}")
    except Exception as e:
        print(f"Matplotlib: {e}")


if __name__ == "__main__":
    main()
