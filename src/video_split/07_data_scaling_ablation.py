"""Plot and summarize the data scaling ablation."""
import os
import json
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("CAPSULE_ROOT", "."))
RESULTS = ROOT / "results"
PAPER = Path(os.environ.get("FIGURE_DIR", ROOT / "figures"))
PAPER.mkdir(parents=True, exist_ok=True)


def main():
    in_path = RESULTS / "side_data_scaling.json"
    if not in_path.exists():
        in_path = RESULTS / "split_robustness/side_data_scaling.json"
    with open(in_path) as f:
        d = json.load(f)
    runs = d["runs"]

    # Aggregate by scale
    from collections import defaultdict
    by_scale = defaultdict(list)
    for r in runs:
        by_scale[r["scale"]].append(r["best_metrics"])

    scales = sorted(by_scale.keys())
    means, stds, n_trains, nulls = [], [], [], []
    for s in scales:
        ms = by_scale[s]
        means.append(np.mean([m["bal_acc"] for m in ms]))
        stds.append(np.std([m["bal_acc"] for m in ms]))
        nulls.append(np.mean([m["null_acc"] for m in ms]))
        n_trains.append(int(np.mean([r["n_train"] for r in runs if r["scale"] == s])))

    # LaTeX table
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{\textbf{Data scaling ablation}, LoRA $r{=}8$ on official two-fold split "
        r"(fold 0 train $\to$ fold 1 test). Mean $\pm$ std over 2 seeds. ``Null'' is always-predict-Normal accuracy on the test fold.}",
        r"\label{tab:scaling}",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"Scale & $n_{\mathrm{train}}$ & Balanced accuracy & Null (test fold) \\",
        r"\midrule",
    ]
    for i, s in enumerate(scales):
        lines.append(f"{s:.2f} & {n_trains[i]:,} & ${means[i]:.3f} \\pm {stds[i]:.3f}$ & {nulls[i]:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    (PAPER / "table_scaling.tex").write_text("\n".join(lines))
    print(f"Saved table_scaling.tex")

    # Figure
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.errorbar(n_trains, means, yerr=stds, marker="o", linewidth=2,
                    color="#C44E52", label="LoRA r=8 (balanced acc)")
        ax.axhline(1 / 11, color="gray", linestyle=":", alpha=0.8,
                   label="balanced chance = 1/11")
        ax.set_xscale("log")
        ax.set_xlabel("Training frames (log scale)")
        ax.set_ylabel("Balanced accuracy (test, fold 1)")
        ax.set_ylim(0.05, 0.34)
        ax.set_title("More frames from the same videos give limited gains")
        ax.legend()
        plt.tight_layout()
        fig.savefig(PAPER / "fig_scaling.png", dpi=150, bbox_inches="tight")
        print(f"Saved fig_scaling.png")
    except Exception as e:
        print(f"Matplotlib failed: {e}")

    # Print summary
    print("\n=== Data scaling summary ===")
    for i, s in enumerate(scales):
        print(f"  scale={s:.2f} n={n_trains[i]:5d} bal={means[i]:.4f} ± {stds[i]:.4f} null={nulls[i]:.4f}")
    print(f"\n=> Bal_acc saturates at ~{max(means):.3f}, with 10x data increase "
          f"({n_trains[0]:,} → {n_trains[-1]:,}) yielding only {(max(means)-min(means)):.3f} improvement.")
    print(f"=> Null baseline ({np.mean(nulls):.3f}) remains unreached at any scale.")


if __name__ == "__main__":
    main()
