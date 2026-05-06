"""Label mapping audit: do CV2024-KVASIR files have the same class label
as their Kvasir-Capsule near-duplicate?

Kvasir-Capsule labels (14-class): angiectasia, blood_fresh, blood_hematin,
erosion, erythema, foreign_body, ileocecal_valve, lymphangiectasia,
normal_clean_mucosa, polyp, pylorus, reduced_mucosal_view, ulcer,
ampulla_of_vater.

CV2024 labels (10-class): Angioectasia, Bleeding, Erosion, Erythema,
Foreign Body, Lymphangiectasia, Normal, Polyp, Ulcer, Worms.

For each pHash-flagged pair, retrieve:
  - Kvasir frame's class (from folder name)
  - CV2024 frame's class (from XLSX one-hot)
  - Whether they match under the canonical mapping

Outputs:
  - Per-class agreement matrix
  - Fraction of disagreements
  - Specific disagreement examples (useful for paper figure)
"""
import os
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
OUT = ROOT / "results"
CV2024_ROOT = Path(os.environ.get("CV2024_ROOT", ROOT / "data/cv2024/Dataset"))
if (CV2024_ROOT / "Dataset").is_dir():
    CV2024_ROOT = CV2024_ROOT / "Dataset"
KVASIR_ROOT = Path(os.environ.get("KVASIR_ROOT", ROOT / "data/kvasir_capsule/labelled_images"))
if (KVASIR_ROOT / "labelled_images").is_dir():
    KVASIR_ROOT = KVASIR_ROOT / "labelled_images"

# Canonical mapping: Kvasir-Capsule 14-class folder -> CV2024 10-class label (or None if not in CV2024)
KVASIR_TO_CV = {
    "angiectasia": "Angioectasia",
    "blood_fresh": "Bleeding",
    "blood_hematin": "Bleeding",
    "erosion": "Erosion",
    "erythema": "Erythema",
    "foreign_body": "Foreign Body",
    "ileocecal_valve": None,
    "lymphangiectasia": "Lymphangiectasia",
    "normal_clean_mucosa": "Normal",
    "polyp": "Polyp",
    "pylorus": None,
    "reduced_mucosal_view": None,
    "ulcer": "Ulcer",
    "ampulla_of_vater": None,
}

CV_CLASSES = ["Angioectasia", "Bleeding", "Erosion", "Erythema",
              "Foreign Body", "Lymphangiectasia", "Normal", "Polyp",
              "Ulcer", "Worms"]


def index_kvasir_by_filename():
    """Return {filename: kvasir_class_folder}"""
    idx = {}
    for folder in KVASIR_ROOT.iterdir():
        if folder.is_dir():
            for fn in folder.glob("*.jpg"):
                idx[fn.name] = folder.name
    return idx


def load_cv2024_labels():
    """Return df with columns: image_path, Dataset, filename, cv_label (single-class str), cv_split."""
    frames = []
    for split in ("training", "validation"):
        xlsx = CV2024_ROOT / split / f"{split}_data.xlsx"
        df = pd.read_excel(xlsx)
        df = df[df["Dataset"] == "KVASIR"]
        for _, row in df.iterrows():
            cv_lbl = None
            for c in CV_CLASSES:
                if c in row and row[c] == 1:
                    cv_lbl = c
                    break
            rel = row["image_path"].replace("\\", "/")
            frames.append({
                "image_path": rel,
                "filename": Path(rel).name,
                "cv_split": split,
                "cv_label": cv_lbl,
            })
    return pd.DataFrame(frames)


def main():
    print("Indexing Kvasir-Capsule...")
    kv_idx = index_kvasir_by_filename()
    print(f"  {len(kv_idx)} Kvasir labeled frames")

    print("Loading CV2024-KVASIR labels...")
    cv_df = load_cv2024_labels()
    print(f"  {len(cv_df)} CV2024-KVASIR frames")

    # For each CV2024-KVASIR file, find its Kvasir folder/label
    cv_df["kvasir_folder"] = cv_df["filename"].map(kv_idx)
    matched = cv_df.dropna(subset=["kvasir_folder"])
    print(f"  {len(matched)} filename-matched pairs (=base overlap)")

    # For each matched pair, compute expected CV label vs actual CV label
    def canon(row):
        kv_folder = row["kvasir_folder"]
        if kv_folder not in KVASIR_TO_CV:
            return "UNKNOWN"
        mapped = KVASIR_TO_CV[kv_folder]
        return mapped if mapped else f"(kvasir-only: {kv_folder})"

    matched["kvasir_mapped_label"] = matched.apply(canon, axis=1)

    # Drop pairs where kvasir class isn't in CV schema (they wouldn't appear in CV2024 anyway, but let's see)
    valid = matched[matched["kvasir_mapped_label"].isin(CV_CLASSES)]
    print(f"  {len(valid)} with mappable Kvasir labels")

    # Confusion
    agree = (valid["kvasir_mapped_label"] == valid["cv_label"]).sum()
    disagree = len(valid) - agree
    print(f"\nAgree:    {agree}/{len(valid)} ({100*agree/len(valid):.1f}%)")
    print(f"Disagree: {disagree}/{len(valid)} ({100*disagree/len(valid):.1f}%)")

    # Disagreement breakdown
    if disagree > 0:
        disag = valid[valid["kvasir_mapped_label"] != valid["cv_label"]]
        print(f"\nDisagreement breakdown (top 15 Kvasir→CV confusion patterns):")
        conf = disag.groupby(["kvasir_mapped_label", "cv_label"]).size().reset_index(name="count")
        conf = conf.sort_values("count", ascending=False)
        for _, row in conf.head(15).iterrows():
            print(f"  Kvasir-mapped={row['kvasir_mapped_label']:<18} → CV2024={row['cv_label']:<18} count={row['count']}")

    # Also: Kvasir classes that don't map (pylorus, ileocecal_valve etc.) — how many appear in CV2024?
    non_mappable = matched[~matched["kvasir_mapped_label"].isin(CV_CLASSES)]
    print(f"\nNon-mappable Kvasir labels in CV2024 (Kvasir-only classes):")
    if len(non_mappable) > 0:
        print(f"  Total: {len(non_mappable)} frames")
        print(non_mappable["kvasir_mapped_label"].value_counts().head(10).to_string())

    # Save
    summary = {
        "n_matched": len(matched),
        "n_valid": len(valid),
        "agree": int(agree),
        "disagree": int(disagree),
        "agree_frac": float(agree / max(len(valid), 1)),
        "confusion_top15": [
            {"kvasir_mapped": r["kvasir_mapped_label"], "cv_label": r["cv_label"], "count": int(r["count"])}
            for _, r in conf.head(15).iterrows()
        ] if disagree > 0 else [],
        "non_mappable_count": len(non_mappable),
        "non_mappable_breakdown": non_mappable["kvasir_mapped_label"].value_counts().to_dict() if len(non_mappable) > 0 else {},
    }
    with open(OUT / "label_mapping_audit.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    # Save detailed CSV
    valid[["filename", "cv_split", "cv_label", "kvasir_mapped_label", "kvasir_folder"]].to_csv(
        OUT / "label_mapping_details.csv", index=False)
    print(f"\nSaved summary to {OUT / 'label_mapping_audit.json'}")
    print(f"Saved details to {OUT / 'label_mapping_details.csv'}")


if __name__ == "__main__":
    main()
