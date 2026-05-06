"""Per-video leakage profile (Kvasir-Capsule labeled videos).

Kvasir-Capsule labels each frame with a filename prefixed by a video
hash (e.g., "3c8d5f0b90d7475d_5043.jpg"), giving a video-level grouping.
The Nature article (Smedsrud 2021) does not publish patient
identifiers, so we treat these prefixes as video-level groups only.
For each CV2024-KVASIR near-duplicate pair, attribute to a Kvasir
video. This produces a per-video-group leakage distribution:
Is leakage uniform across the 43 labeled Kvasir videos, or
concentrated in a few?

This is a methodologically novel contribution: prior audits just
report pooled counts. Per-video attribution lets downstream users
assess clinical representativeness of the Kvasir-origin-removed public corpus.
"""
import os
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
OUT = ROOT / "results"
FIG_DIR = Path(os.environ.get("CAPSULE_FIGURES_DIR", ROOT / "figures"))


def main():
    # Load per-file pHash annotations for KVASIR source
    df = pd.read_csv(OUT / "cv2024_KVASIR_phash_annotated.csv")
    print(f"CV2024-KVASIR: {len(df)} files")

    # Extract video ID from nearest_kvasir_file (prefix before "_")
    df["kvasir_video"] = df["nearest_kvasir_file"].str.split("_").str[0]

    # Only count files flagged as near-duplicate (pHash AND dHash ≤ 6)
    flagged = df[(df["min_phash_dist_to_kvasir"] <= 6) &
                 (df["min_dhash_dist_to_kvasir"] <= 6)]
    print(f"Flagged near-dup: {len(flagged)}")

    # Per-Kvasir-video leakage: how many CV2024 files trace back to each Kvasir video
    per_video_count = flagged["kvasir_video"].value_counts()
    print(f"\nDistinct Kvasir videos appearing in CV2024-KVASIR: {len(per_video_count)}")
    print(f"(Kvasir-Capsule has 43 total videos; 14-class labeled frames draw from all 43)")

    # Kvasir total video count
    # Walk all kvasir folders to count distinct videos
    KVASIR_ROOT = Path(os.environ.get("KVASIR_ROOT", ROOT / "data/kvasir_capsule/labelled_images"))
    if (KVASIR_ROOT / "labelled_images").is_dir():
        KVASIR_ROOT = KVASIR_ROOT / "labelled_images"
    all_kvasir_videos = set()
    video_frame_count = Counter()
    for folder in KVASIR_ROOT.iterdir():
        if folder.is_dir():
            for fn in folder.glob("*.jpg"):
                vid = fn.stem.split("_")[0]
                all_kvasir_videos.add(vid)
                video_frame_count[vid] += 1
    print(f"Total distinct Kvasir-Capsule videos (14-class): {len(all_kvasir_videos)}")

    # Distribution
    coverage = {}
    for vid in sorted(all_kvasir_videos):
        nfr = video_frame_count[vid]
        nflagged = per_video_count.get(vid, 0)
        coverage[vid] = {"total_frames": nfr, "in_cv2024": int(nflagged),
                          "frac": nflagged / max(nfr, 1)}

    # Summary stats
    fracs = [c["frac"] for c in coverage.values()]
    print(f"\n=== Video/exam-prefix reuse profile ===")
    print(f"Videos with >90% frames leaked to CV2024: "
          f"{sum(1 for f in fracs if f>0.9)}/{len(fracs)}")
    print(f"Videos with >50% frames leaked:           "
          f"{sum(1 for f in fracs if f>0.5)}/{len(fracs)}")
    print(f"Videos with 0% frames leaked:             "
          f"{sum(1 for f in fracs if f==0)}/{len(fracs)}")
    print(f"Mean fraction leaked per video:           {np.mean(fracs):.3f}")
    print(f"Std:                                       {np.std(fracs):.3f}")

    # Top 10 most leaked videos
    print(f"\nTop 10 most-leaked Kvasir videos (by fraction):")
    top = sorted(coverage.items(), key=lambda x: -x[1]["frac"])[:10]
    for vid, c in top:
        print(f"  {vid[:16]}  {c['in_cv2024']:>5}/{c['total_frames']:<5} = {c['frac']:.3f}")

    print(f"\nBottom 10 least-leaked Kvasir videos (>0 frames):")
    bottom = [x for x in sorted(coverage.items(), key=lambda x: x[1]["frac"]) if x[1]["total_frames"] > 0][:10]
    for vid, c in bottom:
        print(f"  {vid[:16]}  {c['in_cv2024']:>5}/{c['total_frames']:<5} = {c['frac']:.3f}")

    # Save profile
    with open(OUT / "per_patient_leakage.json", "w") as f:
        json.dump({
            "n_kvasir_videos": len(all_kvasir_videos),
            "n_cv2024_kvasir_flagged": len(flagged),
            "n_distinct_kvasir_videos_in_cv2024": len(per_video_count),
            "mean_fraction_leaked": float(np.mean(fracs)),
            "std_fraction_leaked": float(np.std(fracs)),
            "videos_100pct_leaked": int(sum(1 for f in fracs if f >= 0.99)),
            "videos_90pct_leaked": int(sum(1 for f in fracs if f > 0.9)),
            "videos_50pct_leaked": int(sum(1 for f in fracs if f > 0.5)),
            "videos_0pct_leaked": int(sum(1 for f in fracs if f == 0)),
            "per_video": coverage,
        }, f, indent=2)
    print(f"\nSaved {OUT / 'per_patient_leakage.json'}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        # Histogram of fraction leaked
        ax1.hist(fracs, bins=20, color="#C44E52", edgecolor="black")
        ax1.set_xlabel("Fraction of video frames in CV2024 (flagged)")
        ax1.set_ylabel("Number of Kvasir-Capsule video prefixes")
        ax1.set_title(f"Video/exam-prefix reuse distribution (n={len(fracs)})")
        ax1.grid(True, alpha=0.3)
        # Scatter: n_total vs n_in_cv
        x = [c["total_frames"] for c in coverage.values()]
        y = [c["in_cv2024"] for c in coverage.values()]
        ax2.scatter(x, y, alpha=0.6, color="#4C72B0")
        ax2.plot([0, max(x)], [0, max(x)], "k--", alpha=0.5, label="y=x (100% leaked)")
        ax2.set_xlabel("Video-prefix total frames in Kvasir-Capsule")
        ax2.set_ylabel("Video-prefix frames near-duplicated in CV2024")
        ax2.set_title("Video/exam-prefix reuse")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")
        ax2.set_ylim(-40, max(y) * 1.10)
        plt.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "fig_per_patient_leakage.png", dpi=150, bbox_inches="tight")
        print(f"Saved {FIG_DIR / 'fig_per_patient_leakage.png'}")
    except Exception as e:
        print(f"Matplotlib: {e}")


if __name__ == "__main__":
    main()
