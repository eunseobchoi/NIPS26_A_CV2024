"""Generate Kvasir-origin-removed CV2024 training/validation splits.

Uses pHash audit output to identify CV2024 files that are
near-duplicates of Kvasir-Capsule frames, and produces cleaned CSV
splits suitable for release alongside the paper.

Thresholds (documented in paper):
- pHash ≤ 6 AND dHash ≤ 6 = operational near-duplicate threshold
- pHash ≤ 2 AND dHash ≤ 2 = nearly identical

Output:
- cv2024_training_dedup_le6.csv  (conservative: removes pHash≤6 ∧ dHash≤6)
- cv2024_training_dedup_le2.csv  (aggressive: removes pHash≤2 ∧ dHash≤2)
- cv2024_validation_dedup_le6.csv
- cv2024_validation_dedup_le2.csv
- README_dedup.md (documentation)
"""
import os
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("CAPSULE_ROOT", "."))
OUT = ROOT / "results"


def load_cv2024_xlsx():
    """Load CV2024 training and validation XLSX, add source path info."""
    frames = []
    for split in ("training", "validation"):
        xlsx = ROOT / f"data/cv2024/Dataset/{split}/{split}_data.xlsx"
        df = pd.read_excel(xlsx)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_phash_annotated():
    """Load the annotated CSV from phash_audit.py"""
    dfs = []
    for src in ("KVASIR", "SEE-AI", "KID", "AIIMS"):
        path = OUT / f"cv2024_{src}_phash_annotated.csv"
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
    if not dfs:
        raise SystemExit("Run phash_audit.py first to generate annotated CSVs.")
    return pd.concat(dfs, ignore_index=True)


def main():
    annotated = load_phash_annotated()
    full = load_cv2024_xlsx()
    print(f"CV2024 total: {len(full)}  (training {(full['split']=='training').sum()}, "
          f"validation {(full['split']=='validation').sum()})")
    print(f"Annotated (KVASIR source only): {len(annotated)}")

    # Determine which files are flagged
    for threshold_name, t_ph, t_dh in [("le6", 6, 6), ("le2", 2, 2), ("le0", 0, 0)]:
        flagged = annotated[
            (annotated["min_phash_dist_to_kvasir"] <= t_ph) &
            (annotated["min_dhash_dist_to_kvasir"] <= t_dh)
        ]
        flagged_names = set(flagged["filename"])
        print(f"\n=== Threshold pHash≤{t_ph} ∧ dHash≤{t_dh} ===")
        print(f"  Flagged KVASIR-source files: {len(flagged_names)} / {len(annotated)} "
              f"({100*len(flagged_names)/len(annotated):.1f}%)")
        by_split = Counter(flagged["cv_split"])
        print(f"  By split: {dict(by_split)}")
        # Build dedup splits
        cleaned = full.copy()
        # Match by filename from image_path
        cleaned["filename"] = cleaned["image_path"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
        before = len(cleaned)
        # Only drop rows where source is KVASIR AND filename is flagged
        # (files from SEE-AI/KID/AIIMS can't be duplicates of Kvasir-Capsule
        # by construction)
        drop_mask = (cleaned["Dataset"] == "KVASIR") & \
                    (cleaned["filename"].isin(flagged_names))
        cleaned_final = cleaned[~drop_mask]
        after = len(cleaned_final)
        print(f"  Dropped: {before - after} / {before}  remaining: {after}")
        # Save per-split
        for split in ("training", "validation"):
            sub = cleaned_final[cleaned_final["split"] == split]
            out = OUT / f"cv2024_{split}_dedup_{threshold_name}.csv"
            cols = [c for c in sub.columns
                    if c not in ("filename", "split")]
            sub[cols].to_csv(out, index=False)
            print(f"    {out.name}: {len(sub)} rows")

    # Write README
    readme = f"""# Deduplicated CV2024 splits (auxiliary release)

This directory contains cleaned versions of the Capsule Vision 2024 Challenge
training and validation splits, with files whose pixel content is a
near-duplicate of Kvasir-Capsule frames removed.

## Methodology

Multi-hash perceptual audit:
- pHash + dHash (imagehash library, 64-bit, hash_size=8)
- Near-duplicate: minimum Hamming distance to any Kvasir-Capsule frame
  satisfies pHash ≤ 6 AND dHash ≤ 6 (operational threshold validated by
  released NCC/PDQ/learned-feature and threshold-sensitivity checks)

References:
- Zauner 2010 (pHash dissertation)
- Barz & Denzler 2020 ciFAIR (methodology template)
- Wahlang et al. 2024 (EndoExtend24 — documents CV2024 = Kvasir-Capsule cropped variant)

## Files

| File | Threshold | Use case |
|------|-----------|----------|
| `cv2024_training_dedup_le0.csv` | pHash=0 ∧ dHash=0 (exact) | Most conservative, byte-near-identical only |
| `cv2024_training_dedup_le2.csv` | pHash≤2 ∧ dHash≤2 | Strict near-duplicate |
| `cv2024_training_dedup_le6.csv` | pHash≤6 ∧ dHash≤6 | Operational near-duplicate threshold |
| (same for validation) | | |

Use `cv2024_*_dedup_le6.csv` for standard near-duplicate-free training/evaluation.

## Audit artifact

`phash_audit.json` contains the Hamming-distance histograms per CV2024 source,
and `cv2024_*_phash_annotated.csv` contains per-file minimum distances to
Kvasir.
"""
    (OUT / "README_dedup.md").write_text(readme)
    print(f"\nWrote {OUT / 'README_dedup.md'}")


if __name__ == "__main__":
    main()
