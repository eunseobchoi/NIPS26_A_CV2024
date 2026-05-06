"""Regenerate shipped paper figures from bundled JSON/CSV artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _pool_bal(path: Path) -> np.ndarray:
    data = json.load(path.open())
    return np.array([run["last"]["orig_val"]["bal_acc"] for run in data["runs"]])


def _artifact_root(counterfactual_dir: Path) -> Path:
    resolved = counterfactual_dir.resolve()
    if resolved.name == "counterfactual" and resolved.parent.name == "results":
        return resolved.parent.parent
    return Path.cwd().resolve()


def fig_counterfactual(root: Path, out_dir: Path) -> None:
    baseline_dir = root / "results" / "baseline"
    v4_dir = root / "results" / "counterfactual_n10"
    lift_dir = root / "results" / "acceptance_lift"
    strengthening_dir = root / "results" / "strengthening"

    base = _pool_bal(baseline_dir / "phase5_v5_baseline_n10.json")
    rand = _pool_bal(lift_dir / "phase5_v4_random_s0_n10.json")
    cmat = np.concatenate(
        [
            _pool_bal(v4_dir / "phase5_v4_compmatched_s0_n4.json"),
            _pool_bal(v4_dir / "phase5_v4_compmatched_s0_seeds49.json"),
        ]
    )
    comp_a = _pool_bal(lift_dir / "phase5_v4_compA_s0_n10.json")
    comp_b = _pool_bal(v4_dir / "phase5_v4_compB_s0_n10.json")
    comp_c = _pool_bal(strengthening_dir / "phase5_v5_compC_aiims_ulcer_s0_n10.json")
    le6 = _pool_bal(baseline_dir / "phase5_v5_le6_n10.json")

    def mean_ci(vals: np.ndarray) -> tuple[float, float]:
        return float(vals.mean()), float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))

    path_groups = [
        ("Baseline", base, "#4d4d4d"),
        ("Size only\nrandom", rand, "#b58b00"),
        ("+ class\nmatched", cmat, "#4d72b8"),
        ("- KV Ulcer\nComp-A", comp_a, "#8c2d4f"),
        ("All KV\nremoved", le6, "#c83e4d"),
    ]
    path_means = np.array([mean_ci(vals)[0] for _, vals, _ in path_groups])
    path_ci = np.array([mean_ci(vals)[1] for _, vals, _ in path_groups])
    path_colors = [color for _, _, color in path_groups]
    step_labels = [
        ("size", path_means[1] - path_means[0], "#8a6b00"),
        ("class", path_means[2] - path_means[1], "#2a4c9a"),
        ("KV Ulcer", path_means[3] - path_means[2], "#7a2445"),
        ("other KV", path_means[4] - path_means[3], "#9c2f3c"),
    ]

    fig = plt.figure(figsize=(7.6, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.0], wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    xpos = np.arange(len(path_groups))
    ax.plot(xpos, path_means, color="#4a4a4a", lw=1.2, zorder=1)
    for i, (mean, ci, color) in enumerate(zip(path_means, path_ci, path_colors)):
        ax.errorbar(
            i,
            mean,
            yerr=ci,
            fmt="o",
            ms=6.5,
            color=color,
            ecolor="#333333",
            elinewidth=0.8,
            capsize=2.5,
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=3,
        )
        ax.text(i, mean + 0.014, f"{mean:.3f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    for i, (name, delta, color) in enumerate(step_labels):
        midx = i + 0.5
        y = max(path_means[i], path_means[i + 1]) + 0.035
        ax.annotate(
            "",
            xy=(i + 1, path_means[i + 1] + 0.006),
            xytext=(i, path_means[i] - 0.006),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0, "shrinkA": 4, "shrinkB": 4},
        )
        ax.text(
            midx,
            y,
            f"{name}\n{delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=color,
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": color, "lw": 0.45, "alpha": 0.95},
        )

    ax.set_xticks(xpos)
    ax.set_xticklabels([label for label, _, _ in path_groups], fontsize=7.8)
    ax.set_ylabel("Original-val balanced accuracy")
    ax.set_ylim(0.585, 0.875)
    ax.set_xlim(-0.35, len(path_groups) - 0.65)
    ax.grid(axis="y", alpha=0.22, lw=0.6)
    ax.set_title("(a) Fixed-list accounting path", loc="left", fontsize=10.2, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    ctrl_groups = [
        ("Comp-A\nAIIMS\nUlcer", comp_a, "#8c2d4f"),
        ("Comp-C\nAIIMS\nUlcer x2", comp_c, "#bd6b8a"),
        ("Comp-B\nKV Ulcer\nx2", comp_b, "#27734d"),
    ]
    ctrl_means = np.array([mean_ci(vals)[0] for _, vals, _ in ctrl_groups])
    ctrl_ci = np.array([mean_ci(vals)[1] for _, vals, _ in ctrl_groups])
    ctrl_colors = [color for _, _, color in ctrl_groups]
    cx = np.arange(len(ctrl_groups))
    ax2.bar(cx, ctrl_means, yerr=ctrl_ci, capsize=2.5, color=ctrl_colors, edgecolor="black", linewidth=0.5, width=0.62)
    for i, mean in enumerate(ctrl_means):
        ax2.text(i, mean + 0.006, f"{mean:.3f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax2.set_xticks(cx)
    ax2.set_xticklabels([label for label, _, _ in ctrl_groups], fontsize=7.2)
    ax2.set_ylim(0.625, 0.775)
    ax2.set_ylabel("Balanced accuracy", labelpad=2)
    ax2.grid(axis="y", alpha=0.22, lw=0.6)
    ax2.set_title("(b) Ulcer-source controls", loc="left", fontsize=10.2, fontweight="bold")

    def bracket(axis: plt.Axes, x0: float, x1: float, y: float, text: str, color: str) -> None:
        h = 0.004
        axis.plot([x0, x0, x1, x1], [y, y + h, y + h, y], color=color, lw=0.8, clip_on=False)
        axis.text((x0 + x1) / 2, y + h + 0.002, text, ha="center", va="bottom", fontsize=7.0, color=color)

    bracket(ax2, 0, 1, 0.694, "Comp-C - A = -0.004", "#7a2445")
    bracket(ax2, 1, 2, 0.757, "B - C = +0.084", "#27734d")

    plt.tight_layout(pad=0.25)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_counterfactual.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_dir / "fig_counterfactual.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"saved {out_dir / 'fig_counterfactual.pdf'}")
    print(f"  baseline mean = {base.mean():.4f}")
    print(f"  le6 mean      = {le6.mean():.4f}")
    print(f"  Delta le6     = {le6.mean() - base.mean():+.4f}")


def fig_hamming(audit_summary: Path | None, out_dir: Path) -> None:
    if audit_summary is None:
        return
    annotations = audit_summary.resolve().parents[1] / "annotations"
    sources = ["KVASIR", "SEE-AI", "KID", "AIIMS"]
    colors = {"KVASIR": "#c83e4d", "SEE-AI": "#4d72b8", "KID": "#4d8f5a", "AIIMS": "#c7a03a"}
    data: dict[str, np.ndarray] = {}
    flagged: dict[str, int] = {}
    for src in sources:
        path = annotations / f"cv2024_{src}_phash_annotated.csv"
        df = pd.read_csv(path)
        max_h = df[["min_phash_dist_to_kvasir", "min_dhash_dist_to_kvasir"]].max(axis=1).to_numpy()
        data[src] = max_h
        flagged[src] = int((max_h <= 6).sum())

    fig, ax = plt.subplots(figsize=(7.4, 3.05))
    bins = np.arange(-0.5, 34.5, 1)
    ax.axvspan(-0.5, 6.5, color="#c83e4d", alpha=0.10, label="decision region: max <= 6")
    for src in sources:
        ax.hist(
            data[src],
            bins=bins,
            density=True,
            alpha=0.55 if src == "KVASIR" else 0.36,
            color=colors[src],
            edgecolor=colors[src],
            linewidth=0.45,
            label=f"CV2024-{src} (n={len(data[src]):,})",
        )
    ax.axvline(6.5, color="#222222", linestyle=":", lw=1.1)
    ax.axvline(32, color="#555555", linestyle="--", lw=0.9)
    ymax = ax.get_ylim()[1]
    controls = sum(flagged[src] for src in sources if src != "KVASIR")
    n_controls = sum(len(data[src]) for src in sources if src != "KVASIR")
    ax.text(0.55, ymax * 0.88, "KVASIR\n38,592 / 38,592\ninside rule", ha="left", va="top", color="#9c2f3c", fontsize=9.0, fontweight="bold")
    ax.text(9.0, ymax * 0.76, f"non-KVASIR controls\n{controls} / {n_controls:,}\ninside rule", ha="left", va="top", color="#333333", fontsize=8.4)
    ax.text(26.8, ymax * 0.18, "random 64-bit\nmean = 32", ha="left", va="bottom", fontsize=8.0, color="#444444")
    ax.set_xlabel("max(pHash Hamming, dHash Hamming) to nearest Kvasir-Capsule frame")
    ax.set_ylabel("Density")
    ax.set_xlim(-0.5, 34)
    ax.set_title("Operational duplicate rule: pHash <= 6 and dHash <= 6", fontsize=11.0)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.legend(loc="upper right", fontsize=7.8, frameon=True, framealpha=0.95)
    fig.tight_layout(pad=0.35)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_hamming_hist.png", dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"saved {out_dir / 'fig_hamming_hist.png'}")


def fig_ncc(ncc_summary: Path, out_dir: Path) -> None:
    ncc_csv = ncc_summary.with_name("cv2024_KVASIR_ncc_full.csv")
    if not ncc_csv.exists():
        print(f"skip fig_ncc: {ncc_csv} not found")
        return

    df = pd.read_csv(ncc_csv)
    df["max_h"] = df[["phash_dist", "dhash_dist"]].max(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.75), gridspec_kw={"width_ratios": [1.12, 1]})
    ax = axes[0]
    ax.hist(
        df["ncc"],
        bins=np.linspace(0.5, 1.0, 101),
        color="#4a90d9",
        edgecolor="#1f4a7f",
        linewidth=0.3,
    )
    ax.axvline(0.99, color="#d94a4a", lw=1.0, ls="--")
    ax.axvline(0.95, color="#d9a54a", lw=1.0, ls=":")
    ax.set_xlabel("Normalized cross-correlation")
    ax.set_ylabel("Count (log-scale)")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 1.01)
    n99 = int((df["ncc"] >= 0.99).sum())
    n95 = int((df["ncc"] >= 0.95).sum())
    ax.text(0.988, 10**3.75, f"NCC >= 0.99\n{n99:,} ({100*n99/len(df):.1f}%)", ha="right", va="top", fontsize=8.0, color="#b23333")
    ax.text(0.948, 10**2.25, f">= 0.95\n{n95:,} ({100*n95/len(df):.1f}%)", ha="right", va="top", fontsize=8.0, color="#9b741f")
    ax.set_title(f"(a) Pixel NCC over N={len(df):,} flagged pairs", loc="left", fontweight="bold")

    ax = axes[1]
    bands = list(range(7))
    rates = []
    counts = []
    for band in bands:
        sub = df[df["max_h"] == band]
        counts.append(len(sub))
        rates.append(100 * (sub["ncc"] >= 0.99).mean() if len(sub) else 0)
    xpos = np.arange(len(bands))
    bar_colors = ["#4a90d9" if count >= 30 else "#b9b9b9" for count in counts]
    ax.bar(xpos, rates, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.axvspan(2.5, 6.5, color="#d9a54a", alpha=0.18, label="contested band")
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(b) for b in bands])
    ax.set_xlabel("max(pHash Hamming, dHash Hamming)")
    ax.set_ylabel("NCC >= 0.99 (%)")
    ax.set_ylim(70, 106)
    for i, (count, rate) in enumerate(zip(counts, rates)):
        label_y = min(max(rate, 70) + 0.6, 101.5)
        label = f"n={count:,}" if count >= 1000 else f"n={count}"
        if 0 < count < 30:
            label += "\nsparse"
        ax.text(i, label_y, label, ha="center", va="bottom", fontsize=6.1)
    ax.legend(loc="lower left", fontsize=7)
    ax.text(6, 71.2, "max=6 empty", ha="center", va="bottom", fontsize=6.3, color="#555555")
    ax.set_title("(b) NCC >= 0.99 rate by joint-rule band", loc="left", fontweight="bold")

    plt.tight_layout(pad=0.3, w_pad=0.5)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_ncc.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_dir / "fig_ncc.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved {out_dir / 'fig_ncc.pdf'}")


def fig_filter_pass(filter_pass: Path | None, out_dir: Path) -> None:
    if filter_pass is None:
        return
    summary = json.load(filter_pass.open())
    colors = {"colorjitter_blur": "#4d72b8", "jpeg": "#4d8f5a", "motion_blur": "#c83e4d", "noise": "#c7a03a"}
    labels = {"colorjitter_blur": "color+jitter blur", "jpeg": "JPEG", "motion_blur": "motion blur", "noise": "noise"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.85), sharex=True)
    for corr in sorted(summary):
        sevs = sorted(int(s) for s in summary[corr])
        ent = [summary[corr][str(s)]["entropy_mean"] for s in sevs]
        passed = [summary[corr][str(s)]["pass_mean"] for s in sevs]
        axes[0].plot(sevs, ent, marker="o", lw=1.8, ms=4.2, color=colors.get(corr, "#666666"), label=labels.get(corr, corr))
        axes[1].plot(sevs, passed, marker="o", lw=1.8, ms=4.2, color=colors.get(corr, "#666666"), label=labels.get(corr, corr))
    axes[0].axhline(np.log(11), color="#333333", linestyle=":", lw=0.9, label="log K")
    axes[0].set_ylabel("Mean entropy")
    axes[0].set_title("(a) Entropy rises", loc="left", fontweight="bold")
    axes[1].set_ylabel("Entropy-filter pass rate")
    axes[1].set_ylim(-0.01, 0.23)
    axes[1].set_title("(b) Filter pass collapses", loc="left", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Corruption severity")
        ax.set_xticks([0, 3, 5])
        ax.grid(alpha=0.25, lw=0.6)
    axes[1].annotate(
        "0% for\nmotion blur\nat severity 5",
        xy=(5, 0.0),
        xytext=(3.35, 0.115),
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#9c2f3c"},
        bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.84},
        fontsize=8.0,
        color="#9c2f3c",
        ha="left",
    )
    axes[0].legend(loc="lower right", fontsize=7.4, frameon=True)
    axes[1].legend(loc="upper right", fontsize=7.4, frameon=True)
    fig.tight_layout(pad=0.3, w_pad=0.75)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_filter_pass.png", dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"saved {out_dir / 'fig_filter_pass.png'}")


def fig_scaling(scaling: Path | None, out_dir: Path) -> None:
    if scaling is None:
        return
    data = json.load(scaling.open())
    by_scale: dict[float, list[dict]] = {}
    for run in data["runs"]:
        by_scale.setdefault(float(run["scale"]), []).append(run)
    scales = sorted(by_scale)
    x = [np.mean([r["n_train"] for r in by_scale[s]]) for s in scales]
    y = [np.mean([r["best_metrics"]["bal_acc"] for r in by_scale[s]]) for s in scales]
    err = [np.std([r["best_metrics"]["bal_acc"] for r in by_scale[s]], ddof=0) for s in scales]
    fig, ax = plt.subplots(figsize=(5.7, 3.25))
    ax.errorbar(x, y, yerr=err, marker="o", lw=2.0, ms=5.0, capsize=3.0, color="#c83e4d", label="LoRA r=8")
    ax.axhline(1 / 11, color="#777777", linestyle=":", lw=1.1, label="balanced chance = 1/11")
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.010, f"{yi:.3f}", ha="center", va="bottom", fontsize=8.0)
    ax.set_xscale("log")
    ax.set_xlabel("Training frames from fixed video pool (log scale)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.05, 0.34)
    ax.set_title("Same-video frame scaling saturates below 0.30", fontsize=11.0)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", fontsize=8.0, frameon=True)
    fig.tight_layout(pad=0.35)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_scaling.png", dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"saved {out_dir / 'fig_scaling.png'}")


def fig_videocount(videocount: Path | None, out_dir: Path) -> None:
    if videocount is None:
        return
    data = json.load(videocount.open())
    by_v: dict[int, list[dict]] = {}
    for run in data["runs"]:
        by_v.setdefault(int(run["n_videos"]), []).append(run)
    vs = sorted(by_v)
    y = [np.mean([r["last_metrics"]["bal_acc"] for r in by_v[v]]) for v in vs]
    err = [np.std([r["last_metrics"]["bal_acc"] for r in by_v[v]], ddof=0) for v in vs]
    unique = [np.mean([r["n_train_unique"] for r in by_v[v]]) for v in vs]
    fig, ax = plt.subplots(figsize=(5.9, 3.35))
    ax.errorbar(vs, y, yerr=err, marker="o", lw=2.0, ms=5.0, capsize=3.0, color="#2e7b8f", label="Final-epoch bal. acc.")
    ax.axhline(1 / 11, color="#777777", linestyle=":", lw=1.1, label="chance 1/K")
    for v, yi in zip(vs, y):
        ax.text(v, yi + 0.010, f"{yi:.3f}", ha="center", va="bottom", fontsize=8.0)
    ax.text(0.03, 0.95, "oversampled frame budget fixed at ~23K", transform=ax.transAxes, ha="left", va="top", fontsize=8.3, bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#2e7b8f", "lw": 0.6})
    ax.text(
        0.97,
        0.25,
        f"unique frames: {int(unique[0]):,} -> {int(unique[-1]):,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.82},
    )
    ax.set_xlabel("Number of training videos")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xticks(vs)
    ax.set_ylim(0.05, 0.30)
    ax.set_title("Training-set diversity at fixed budget", fontsize=11.0)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower right", fontsize=8.0, frameon=True)
    fig.tight_layout(pad=0.35)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_videocount.png", dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"saved {out_dir / 'fig_videocount.png'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-summary", type=Path)
    parser.add_argument("--ncc-summary", type=Path, required=True)
    parser.add_argument("--per-patient", type=Path)
    parser.add_argument("--counterfactual-dir", type=Path, default=Path("results/counterfactual"))
    parser.add_argument("--multibackbone-dir", type=Path)
    parser.add_argument("--tta-json", type=Path)
    parser.add_argument("--filter-pass", type=Path)
    parser.add_argument("--scaling", type=Path)
    parser.add_argument("--videocount", type=Path)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = _artifact_root(args.counterfactual_dir)
    fig_hamming(args.audit_summary, args.out)
    fig_counterfactual(root, args.out)
    fig_ncc(args.ncc_summary, args.out)
    fig_filter_pass(args.filter_pass, args.out)
    fig_scaling(args.scaling, args.out)
    fig_videocount(args.videocount, args.out)


if __name__ == "__main__":
    main()
