#!/usr/bin/env python3
"""
Cluster-bootstrap CV2024 within-split re-exposure rate at the
Kvasir-Capsule source-video level (41 unique videos in the
KVASIR validation slice).

Honest CI replacement for the frame-level Wilson [11.35%, 12.53%].
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def video_id(fn: str) -> str:
    m = re.match(r"^([0-9a-f]+)_", fn)
    return m.group(1) if m else ""


def main():
    csv_path = "/home/user/main/capsule_tta/results/cv2024_KVASIR_phash_annotated.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    train = [r for r in rows if r["cv_split"] == "training"]
    val = [r for r in rows if r["cv_split"] == "validation"]
    train_phash = set(r["phash"] for r in train)

    # Per-video frame-level: (n_total, n_flagged) per video
    per_video = defaultdict(lambda: {"n": 0, "k": 0})
    for r in val:
        v = video_id(r["filename"])
        per_video[v]["n"] += 1
        if r["phash"] in train_phash:
            per_video[v]["k"] += 1

    videos = sorted(per_video.keys())
    n_videos = len(videos)
    n_total = sum(s["n"] for s in per_video.values())
    k_total = sum(s["k"] for s in per_video.values())
    print(f"videos={n_videos}, total val frames={n_total}, flagged={k_total}, "
          f"point rate={100*k_total/n_total:.3f}%")

    # Print per-video breakdown
    print("\nPer-video flagged counts (sorted by k):")
    print(f"  {'video_id':>20s} {'n':>6s} {'k':>6s} {'rate':>7s}")
    for v in sorted(videos, key=lambda v: -per_video[v]["k"]):
        s = per_video[v]
        print(f"  {v:>20s} {s['n']:>6d} {s['k']:>6d} {100*s['k']/s['n']:>6.2f}%")

    # Cluster bootstrap: resample videos with replacement
    rng = np.random.default_rng(42)
    n_boot = 20000
    rates = []
    for b in range(n_boot):
        sample_videos = rng.choice(videos, size=n_videos, replace=True)
        n_b = sum(per_video[v]["n"] for v in sample_videos)
        k_b = sum(per_video[v]["k"] for v in sample_videos)
        rates.append(k_b / n_b if n_b > 0 else 0.0)
    rates = np.array(rates)
    pct_lo = float(np.quantile(rates, 0.025))
    pct_hi = float(np.quantile(rates, 0.975))
    pct_mean = float(rates.mean())
    pct_median = float(np.median(rates))
    print(f"\nCluster bootstrap (n_boot={n_boot}, video-level resample):")
    print(f"  point rate: {100*k_total/n_total:.3f}%")
    print(f"  bootstrap mean: {100*pct_mean:.3f}%")
    print(f"  bootstrap median: {100*pct_median:.3f}%")
    print(f"  95% percentile CI: [{100*pct_lo:.3f}%, {100*pct_hi:.3f}%]")

    # Compare against frame-level Wilson [11.347, 12.527]
    # And against ISIC NCC Wilson upper 0.029%
    isic_upper = 0.000288  # 0.029%
    print(f"\nGap vs ISIC NCC Wilson upper ({100*isic_upper:.3f}%):")
    print(f"  point estimate gap: {(k_total/n_total)/isic_upper:.0f}x")
    print(f"  cluster lower / ISIC upper: {pct_lo/isic_upper:.0f}x")

    # Also Wilson for reference
    from scipy import stats
    z = stats.norm.isf(0.025)
    p = k_total / n_total
    den = 1 + z*z/n_total
    center = (p + z*z/(2*n_total)) / den
    half = (z/den) * np.sqrt(p*(1-p)/n_total + z*z/(4*n_total*n_total))
    print(f"\nFrame-level Wilson 95% CI (anti-conservative): "
          f"[{100*(center-half):.3f}%, {100*(center+half):.3f}%]")

    out = {
        "n_videos": n_videos,
        "n_val_frames": n_total,
        "n_flagged_val_frames": k_total,
        "point_rate_pct": round(100 * k_total / n_total, 4),
        "cluster_bootstrap": {
            "n_boot": n_boot,
            "mean_pct": round(100 * pct_mean, 4),
            "median_pct": round(100 * pct_median, 4),
            "ci_2.5_pct": round(100 * pct_lo, 4),
            "ci_97.5_pct": round(100 * pct_hi, 4),
        },
        "frame_level_wilson_ci_pct": [
            round(100 * (center - half), 4),
            round(100 * (center + half), 4),
        ],
        "per_video": {v: {"n": s["n"], "k": s["k"]}
                       for v, s in per_video.items()},
        "comparison_isic_ncc_wilson_upper_pct": 0.0288,
        "ratio_point_vs_isic_upper": round((k_total/n_total)/isic_upper, 1),
        "ratio_cluster_lower_vs_isic_upper": round(pct_lo/isic_upper, 1),
    }
    out_path = Path("results/cv2024_cluster_bootstrap.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
