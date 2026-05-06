#!/usr/bin/env python3
"""Generate le6_kvfree_s1 training CSV for the same-source re-exposure probe.

Purpose (Path B Exp 1):
  Produce a training pool with the SAME source domain (Kvasir-Capsule) and
  SAME class-prior as le6 but DIFFERENT pixels: Kvasir-Capsule official
  split_1 frames that never appeared in CV2024. Comparing (A) le6 vs
  (B) le6 + split_1-unseen probes same-source re-exposure separately from
  literal CV2024 frame reuse.

Inputs:
  - artifacts/csvs/cv2024_training_dedup_le6.csv
  - data/official_splits/split_1.csv (Smedsrud 2021 official two-fold split)
  - artifacts/annotations/cv2024_KVASIR_phash_annotated.csv (identifies which
    Kvasir-Capsule frames appear in CV2024 via nearest_kvasir_file column)

Output:
  - artifacts/csvs/cv2024_training_le6_kvfree_s1.csv (10,880 rows; MD5 recorded in log)

Deterministic (random_state=0 on per-class sampling).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
LE6_CSV = Path(os.environ.get(
    "LE6_CSV",
    ROOT / "artifacts/csvs/cv2024_training_dedup_le6.csv",
))
SPLIT1_CSV = ROOT / "data/official_splits/split_1.csv"
PHASH_CSV = Path(os.environ.get(
    "PHASH_CSV",
    ROOT / "artifacts/annotations/cv2024_KVASIR_phash_annotated.csv",
))
OUT_CSV = Path(os.environ.get(
    "OUT_CSV",
    ROOT / "artifacts/csvs/cv2024_training_le6_kvfree_s1.csv",
))

CLASS_COLS = (
    "Angioectasia", "Bleeding", "Erosion", "Erythema", "Foreign Body",
    "Lymphangiectasia", "Normal", "Polyp", "Ulcer", "Worms",
)

LABEL_MAP = {
    "Angiectasia": "Angioectasia",
    "Blood": "Bleeding",
    "Erosion": "Erosion",
    "Erythematous": "Erythema",
    "Foreign Bodies": "Foreign Body",
    "Lymphangiectasia": "Lymphangiectasia",
    "Normal": "Normal",
    "Ulcer": "Ulcer",
    "Reduced Mucosal View": None,
    "Ileo-cecal valve": None,
    "Pylorus": None,
}

LABEL_TO_FOLDER = {
    "Angiectasia": "angiectasia",
    "Blood": "blood_fresh",
    "Erosion": "erosion",
    "Erythematous": "erythema",
    "Foreign Bodies": "foreign_body",
    "Lymphangiectasia": "lymphangiectasia",
    "Normal": "normal_clean_mucosa",
    "Ulcer": "ulcer",
}

TARGET_PER_CLASS = 200
SEED = 0


def main() -> None:
    le6 = pd.read_csv(LE6_CSV)
    s1 = pd.read_csv(SPLIT1_CSV)
    phash = pd.read_csv(PHASH_CSV, usecols=["filename", "nearest_kvasir_file"])

    cv2024_kvasir_hits = set(phash["nearest_kvasir_file"].dropna().astype(str))
    s1_unseen = s1[~s1["filename"].isin(cv2024_kvasir_hits)].copy()
    s1_unseen["cv_label"] = s1_unseen["label"].map(LABEL_MAP)
    s1_mapped = s1_unseen[s1_unseen["cv_label"].notna()].copy()

    selected = []
    for _, group in s1_mapped.groupby("cv_label"):
        k = min(TARGET_PER_CLASS, len(group))
        selected.append(group.sample(n=k, random_state=SEED))
    addons = pd.concat(selected, ignore_index=True)

    addon_rows = []
    for _, r in addons.iterrows():
        folder = LABEL_TO_FOLDER[r["label"]]
        row = {
            "image_path": f"kvasir_capsule_split_1\\{folder}\\{r['filename']}",
            "Dataset": "KVASIR_SPLIT1",
        }
        for c in CLASS_COLS:
            row[c] = 1 if c == r["cv_label"] else 0
        addon_rows.append(row)
    addon_df = pd.DataFrame(addon_rows, columns=le6.columns.tolist())

    combined = pd.concat([le6, addon_df], ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)

    md5 = hashlib.md5(OUT_CSV.read_bytes()).hexdigest()
    print(f"Saved {OUT_CSV} | rows={len(combined)} | md5={md5}")


if __name__ == "__main__":
    main()
