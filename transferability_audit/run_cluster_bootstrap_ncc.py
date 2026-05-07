#!/usr/bin/env python3
"""
NCC-confirmed cluster bootstrap: re-resamples 41 KVASIR source videos,
but uses the 540 NCC-confirmed (NCC>=0.99) subset of the 1381 pHash-exact
val rows. Provides the matched-NCC-endpoint cluster CI that is referenced
in the appendix as "cluster (proportional)".

Pipeline:
  1. Load `cv2024_KVASIR_internal_train_val_phash_exact_pairs.csv` (1,381 rows)
  2. For each (val, train) pair, compute NCC on 256x256 grayscale images
  3. Bucket by source-video prefix, count k_NCC >= 0.99 per video
  4. Cluster bootstrap (resample videos with replacement) over the
     val-only denominators (per-video n totals from the cluster_bootstrap.json)
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PAIRS_CSV = ("/home/user/main/capsule_tta/submission_FINAL/artifacts/"
             "annotations/cv2024_KVASIR_internal_train_val_phash_exact_pairs.csv")
DATA_ROOT = "/home/user/main/capsule_tta/data/cv2024"  # CSV path includes /Dataset/ already
CB_JSON = ("/home/user/main/capsule_tta/transferability_audit/results/"
           "cv2024_cluster_bootstrap.json")
OUT = ("/home/user/main/capsule_tta/transferability_audit/results/"
       "cv2024_cluster_bootstrap_ncc.json")


def video_id(fn: str) -> str:
    m = re.match(r"^([0-9a-f]+)_", fn)
    return m.group(1) if m else ""


def load_g(path):
    try:
        return np.asarray(
            Image.open(path).convert("L").resize((256, 256), Image.BILINEAR),
            dtype=np.float32)
    except Exception:
        return None


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-9 else 0.0


def main():
    # Load the 1,381 pairs
    pairs = []
    with open(PAIRS_CSV) as f:
        for r in csv.DictReader(f):
            pairs.append({
                "val_filename": r["val_filename"],
                "val_path": r["val_path"].replace("<CV2024_ROOT>", DATA_ROOT),
                "train_path": r["train_path"].replace("<CV2024_ROOT>", DATA_ROOT),
            })
    print(f"Loaded {len(pairs)} pHash-exact val-train pairs", file=sys.stderr)

    # Compute NCC per pair
    cache = {}

    def get(p):
        if p not in cache:
            cache[p] = load_g(p)
        return cache[p]

    ncc_results = []
    for idx, p in enumerate(pairs, 1):
        a = get(p["val_path"])
        b = get(p["train_path"])
        if a is None or b is None:
            ncc_results.append((p["val_filename"], None))
            continue
        ncc_results.append((p["val_filename"], ncc(a, b)))
        if idx % 200 == 0:
            print(f"  NCC {idx}/{len(pairs)}", file=sys.stderr, flush=True)

    valid = [(fn, n) for fn, n in ncc_results if n is not None]
    skipped = len(ncc_results) - len(valid)
    ge99 = sum(1 for fn, n in valid if n >= 0.99)
    print(f"NCC done: {len(valid)} pairs, {skipped} skipped, "
          f"NCC>=0.99 = {ge99}", file=sys.stderr)

    # Per-video k_NCC counts (val rows)
    per_video_k_ncc = defaultdict(int)
    for fn, n in valid:
        if n is not None and n >= 0.99:
            per_video_k_ncc[video_id(fn)] += 1

    # Load the per-video n totals from existing cluster bootstrap
    with open(CB_JSON) as f:
        cb = json.load(f)
    per_video_n = {v: d["n"] for v, d in cb["per_video"].items()}

    # Build per-video (n, k_NCC) — videos not in NCC list have k_NCC=0
    videos = sorted(per_video_n.keys())
    per_video = {}
    for v in videos:
        per_video[v] = {"n": per_video_n[v], "k_ncc": per_video_k_ncc.get(v, 0)}

    n_total = sum(d["n"] for d in per_video.values())
    k_total = sum(d["k_ncc"] for d in per_video.values())
    print(f"\nPer-video n_total={n_total}, k_NCC_total={k_total}, "
          f"point rate={100*k_total/n_total:.3f}%", file=sys.stderr)

    # Cluster bootstrap
    rng = np.random.default_rng(42)
    n_boot = 20000
    rates = []
    for b in range(n_boot):
        sample = rng.choice(videos, size=len(videos), replace=True)
        n_b = sum(per_video[v]["n"] for v in sample)
        k_b = sum(per_video[v]["k_ncc"] for v in sample)
        rates.append(k_b / n_b if n_b > 0 else 0.0)
    rates = np.array(rates)
    pct_lo = float(np.quantile(rates, 0.025))
    pct_hi = float(np.quantile(rates, 0.975))
    pct_mean = float(rates.mean())
    pct_median = float(np.median(rates))
    print(f"Cluster bootstrap NCC-confirmed (n_boot={n_boot}):", file=sys.stderr)
    print(f"  point: {100*k_total/n_total:.3f}%", file=sys.stderr)
    print(f"  mean: {100*pct_mean:.3f}%", file=sys.stderr)
    print(f"  percentile 95% CI: [{100*pct_lo:.3f}%, {100*pct_hi:.3f}%]",
          file=sys.stderr)

    # Compare against ISIC NCC Wilson upper 0.0288%
    isic_upper = 0.000288
    print(f"  cluster lower / ISIC NCC upper: {pct_lo/isic_upper:.0f}x",
          file=sys.stderr)
    print(f"  point / ISIC NCC point (0.008%): "
          f"{(k_total/n_total)/0.0000789:.0f}x", file=sys.stderr)

    out = {
        "n_videos": len(videos),
        "n_val_frames": n_total,
        "k_NCC_confirmed": k_total,
        "point_rate_pct": round(100 * k_total / n_total, 4),
        "skipped_nccs": skipped,
        "cluster_bootstrap_ncc": {
            "n_boot": n_boot,
            "mean_pct": round(100 * pct_mean, 4),
            "median_pct": round(100 * pct_median, 4),
            "ci_2.5_pct": round(100 * pct_lo, 4),
            "ci_97.5_pct": round(100 * pct_hi, 4),
        },
        "per_video_ncc_breakdown": {
            v: per_video[v] for v in videos
        },
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
