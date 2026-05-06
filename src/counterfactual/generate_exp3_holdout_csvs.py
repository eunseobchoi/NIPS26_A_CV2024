"""Exp 3 setup -- identify top-k leaked Kvasir videos and build hold-out CSVs.

Purpose (Path B Exp 3):
  Build the training/test CSVs for the video-hold-out membership-inference
  probe. Uses the CV2024 full-pool baseline CSV (37,607 rows including
  27K KVASIR rows) -- NOT base -- because base has already dedup-removed
  KVASIR rows, leaving nothing to hold out.
  Steps:
    1. From `cv2024_KVASIR_phash_annotated.csv`, identify Kvasir videos
       with highest frame-count that appear in CV2024 training rows.
       Video id = first `_`-separated prefix of Kvasir filename.
    2. Select top-K videos (default K=3) preferring class diversity over
       {Angioectasia, Erosion, Normal} when possible.
    3. Read `cv2024_training_baseline_fullpool.csv`. Drop every KVASIR row
       whose `image_path` basename maps to any chosen video_id.
       Output: `cv2024_training_baseline_minus_top3videos.csv`.
    4. Build hold-out EVAL csv from removed rows. Output:
       `cv2024_holdout_top3videos_test.csv`.
    5. Deterministic (rank-based selection with priority-class preference).

Downstream consumer:
  `phase5_exp3_mi_probe.py`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
CV2024_ROOT = Path(os.environ.get("CV2024_ROOT", ROOT / "data/cv2024/Dataset"))
if (CV2024_ROOT / "Dataset").is_dir():
    CV2024_ROOT = CV2024_ROOT / "Dataset"
BASE_CSV = Path(os.environ.get(
    "BASE_CSV",
    ROOT / "results/cv2024_training_baseline_fullpool.csv",
))
PHASH_CSV = Path(os.environ.get(
    "PHASH_CSV",
    ROOT / "artifacts/annotations/cv2024_KVASIR_phash_annotated.csv",
))
OUT_TRAIN = ROOT / "results/cv2024_training_baseline_minus_top3videos.csv"
OUT_HOLD = ROOT / "results/cv2024_holdout_top3videos_test.csv"
OUT_MANIFEST = ROOT / "results/cv2024_holdout_top3videos_manifest.json"

CV2024_CLASSES = (
    "Angioectasia", "Bleeding", "Erosion", "Erythema", "Foreign Body",
    "Lymphangiectasia", "Normal", "Polyp", "Ulcer", "Worms",
)

PRIORITY_CLASSES = ("Angioectasia", "Erosion", "Normal")


def video_id(filename: str) -> str:
    return str(filename).split("_")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="top-K videos to hold out")
    ap.add_argument("--min_cv_rows", type=int, default=200,
                    help="minimum base rows attributed to a video for eligibility")
    args = ap.parse_args()

    if BASE_CSV.exists():
        base = pd.read_csv(BASE_CSV)
    else:
        base_xlsx = CV2024_ROOT / "training" / "training_data.xlsx"
        if not base_xlsx.exists():
            raise FileNotFoundError(
                f"Need {BASE_CSV} or raw CV2024 training_data.xlsx at {base_xlsx}"
            )
        base = pd.read_excel(base_xlsx)
        BASE_CSV.parent.mkdir(parents=True, exist_ok=True)
        base.to_csv(BASE_CSV, index=False)
        print(f"Generated baseline full-pool CSV: {BASE_CSV}")
    phash = pd.read_csv(PHASH_CSV,
                        usecols=["filename", "cv_dataset", "nearest_kvasir_file"])
    kvasir_rows = phash[phash["cv_dataset"] == "KVASIR"].copy()
    kvasir_rows["kvasir_video"] = kvasir_rows["nearest_kvasir_file"].dropna().astype(str).apply(video_id)

    video_counts = Counter(kvasir_rows["kvasir_video"].dropna().tolist())
    print(f"Total KVASIR CV2024 rows: {len(kvasir_rows)}")
    print(f"Unique Kvasir video_ids mapped: {len(video_counts)}")
    print("Top-10 videos by row-count:")
    for v, c in video_counts.most_common(10):
        print(f"  {v}: {c} CV2024 rows")

    base_basenames = base["image_path"].apply(
        lambda p: Path(str(p).replace("\\", "/")).name)
    base_kvasir_mask = base["Dataset"] == "KVASIR"

    filename_to_video = dict(zip(
        kvasir_rows["filename"].astype(str),
        kvasir_rows["kvasir_video"]))

    base_video_counter = Counter()
    base_video_to_class = defaultdict(Counter)
    for idx, row in base[base_kvasir_mask].iterrows():
        bn = Path(str(row["image_path"]).replace("\\", "/")).name
        v = filename_to_video.get(bn)
        if v is None:
            continue
        base_video_counter[v] += 1
        for c in CV2024_CLASSES:
            if row.get(c, 0) == 1:
                base_video_to_class[v][c] += 1
                break

    print(f"\nbase KVASIR rows mapped to videos: "
          f"{sum(base_video_counter.values())} "
          f"across {len(base_video_counter)} videos")
    print("Top-10 base videos by base row-count (dominant class in parens):")
    for v, c in base_video_counter.most_common(10):
        dom = base_video_to_class[v].most_common(1)[0] if base_video_to_class[v] else ("?", 0)
        print(f"  {v}: {c} base rows  dom-class={dom[0]} ({dom[1]})")

    eligible = [(v, c) for v, c in base_video_counter.most_common()
                if c >= args.min_cv_rows]
    print(f"\nEligible videos (>={args.min_cv_rows} base rows): {len(eligible)}")

    # For each priority class, pick the video with highest rows where that
    # class is DOMINANT (strict) or contributes most overall. This yields
    # class-diversity across {Angioectasia, Erosion, Normal}.
    by_class_rank = defaultdict(list)  # class -> [(video, n_class_rows, total_rows)]
    for v, _ in eligible:
        for c, n_c in base_video_to_class[v].items():
            total = base_video_counter[v]
            by_class_rank[c].append((v, n_c, total))
    for c in by_class_rank:
        by_class_rank[c].sort(key=lambda t: -t[1])  # desc by class-row count

    chosen = []
    chosen_classes = set()
    for pc in PRIORITY_CLASSES:
        if len(chosen) >= args.k:
            break
        for v, n_c, total in by_class_rank.get(pc, []):
            if v not in chosen and n_c >= args.min_cv_rows // 2:
                chosen.append(v)
                chosen_classes.add(pc)
                break
    # Back-fill from overall most-common if still short
    for v, c in eligible:
        if len(chosen) >= args.k:
            break
        if v not in chosen:
            chosen.append(v)
            dom = base_video_to_class[v].most_common(1)[0][0] if base_video_to_class[v] else None
            if dom:
                chosen_classes.add(dom)
    print(f"\nChosen top-{args.k} videos: {chosen}")
    print(f"Spanning classes: {sorted(chosen_classes)}")
    for v in chosen:
        dist = dict(base_video_to_class[v].most_common(4))
        print(f"  {v}: total={base_video_counter[v]} classes={dist}")

    chosen_set = set(chosen)
    mask_remove = pd.Series(False, index=base.index)
    for i, bn in enumerate(base_basenames):
        if not base_kvasir_mask.iloc[i]:
            continue
        v = filename_to_video.get(bn)
        if v in chosen_set:
            mask_remove.iloc[i] = True
    n_removed = int(mask_remove.sum())
    train_reduced = base[~mask_remove].reset_index(drop=True)
    holdout = base[mask_remove].reset_index(drop=True)
    print(f"\nRemoved from base: {n_removed} rows "
          f"({n_removed / len(base) * 100:.2f}%)")
    print(f"train_reduced: {len(train_reduced)} rows")
    print(f"holdout (membership test set): {len(holdout)} rows")

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    train_reduced.to_csv(OUT_TRAIN, index=False)
    holdout.to_csv(OUT_HOLD, index=False)

    manifest = {
        "k": args.k,
        "chosen_video_ids": chosen,
        "classes_covered": sorted(chosen_classes),
        "n_base_original": len(base),
        "n_removed": n_removed,
        "n_train_reduced": len(train_reduced),
        "n_holdout": len(holdout),
        "train_reduced_md5": hashlib.md5(OUT_TRAIN.read_bytes()).hexdigest(),
        "holdout_md5": hashlib.md5(OUT_HOLD.read_bytes()).hexdigest(),
        "base_md5": hashlib.md5(BASE_CSV.read_bytes()).hexdigest(),
        "phash_md5": hashlib.md5(PHASH_CSV.read_bytes()).hexdigest(),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved {OUT_TRAIN}")
    print(f"Saved {OUT_HOLD}")
    print(f"Saved {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
