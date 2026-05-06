"""CV2024 internal train/val split-overlap audit.

Tests whether CV2024 training and validation KVASIR-source files
contain near-duplicate pairs. This matters because even ignoring
Kvasir-Capsule, a model trained on CV2024 training is then evaluated
on CV2024 validation; if the two splits share frames at the pixel
level, the validation score is inflated.

Result (saved to results/cv2024_internal_leak.json):
  - 11.9% of CV2024-KVASIR validation is pHash-exact-duplicate of training
  - 27.5% at pHash <= 2
  - 54.8% at pHash <= 6 (near-duplicate)
  - Median train→val Hamming: 6.0

This is a separate finding from the Kvasir→CV2024 leakage: it
shows CV2024's internal partitioning does not prevent re-sampling
the same video-prefix frames across the train/val boundary.
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
OUT = ROOT / "results"
CV2024_ROOT = Path(os.environ.get("CV2024_ROOT", ROOT / "data/cv2024/Dataset"))
if (CV2024_ROOT / "Dataset").is_dir():
    CV2024_ROOT = CV2024_ROOT / "Dataset"

_POP16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)


def popcount_u64_2d(arr):
    f = arr.view(np.uint16).reshape(*arr.shape, 4)
    return _POP16[f].astype(np.int16).sum(axis=-1)


def main():
    with open(OUT / "hashes_cv2024.json") as f:
        cv = json.load(f)

    train = [x for x in cv if x.get("cv_split") == "training" and x.get("cv_dataset") == "KVASIR"]
    val   = [x for x in cv if x.get("cv_split") == "validation" and x.get("cv_dataset") == "KVASIR"]
    print(f"CV2024-KVASIR train: {len(train)}, val: {len(val)}")
    fn_overlap = len(set(x["filename"] for x in train) & set(x["filename"] for x in val))
    print(f"Filename overlap: {fn_overlap}")

    tr_int = np.array([int(x["phash"], 16) for x in train], dtype=np.uint64)
    va_int = np.array([int(x["phash"], 16) for x in val], dtype=np.uint64)

    min_d = np.full(len(va_int), 64, dtype=np.int16)
    min_idx = np.zeros(len(va_int), dtype=np.int64)
    chunk = 512
    for i in range(0, len(va_int), chunk):
        v = va_int[i:i+chunk]
        xor = v[:, None] ^ tr_int[None, :]
        pc = popcount_u64_2d(xor)
        j = pc.argmin(axis=1)
        min_idx[i:i+chunk] = j
        min_d[i:i+chunk] = pc[np.arange(len(v)), j]

    thresholds = {}
    for t in (0, 1, 2, 3, 6, 10, 20, 32):
        n = int((min_d <= t).sum())
        thresholds[f"le{t}"] = {"n": n, "frac": float(n / len(va_int))}

    # Per-class breakdown
    import pandas as pd
    tr_fns = set(x["filename"] for x in train)
    va_fns = set(x["filename"] for x in val)

    # CV2024 val label info
    cv_val_anno = pd.read_excel(CV2024_ROOT / "validation/validation_data.xlsx")
    cv_val_anno = cv_val_anno[cv_val_anno["Dataset"] == "KVASIR"]
    cv_val_anno["filename"] = cv_val_anno["image_path"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
    class_cols = [c for c in cv_val_anno.columns if c in
                  {"Angioectasia","Bleeding","Erosion","Erythema","Foreign Body",
                   "Lymphangiectasia","Normal","Polyp","Ulcer","Worms"}]
    def get_label(row):
        for c in class_cols:
            if row.get(c) == 1: return c
        return None
    cv_val_anno["label"] = cv_val_anno.apply(get_label, axis=1)

    # Attach min_d to each val file
    val_df = pd.DataFrame({
        "filename": [x["filename"] for x in val],
        "min_hamming_to_train": min_d.tolist(),
    })
    val_df = val_df.merge(cv_val_anno[["filename", "label"]], on="filename", how="left")

    # Per-class flagging rate at ≤6
    rows = []
    for lbl, g in val_df.groupby("label"):
        rows.append({
            "label": lbl, "n": len(g),
            "flagged_le0": int((g["min_hamming_to_train"] == 0).sum()),
            "flagged_le6": int((g["min_hamming_to_train"] <= 6).sum()),
            "median_hamming": float(g["min_hamming_to_train"].median()),
        })
    per_class = pd.DataFrame(rows)
    print("\nPer-class CV2024 val KVASIR → train near-duplicate rate:")
    print(per_class.to_string())

    summary = {
        "n_train": len(train), "n_val": len(val),
        "filename_overlap": fn_overlap,
        "phash_thresholds": thresholds,
        "median_hamming_val_to_train": float(np.median(min_d)),
        "mean_hamming_val_to_train": float(min_d.mean()),
        "per_class": per_class.to_dict(orient="records"),
    }
    with open(OUT / "cv2024_internal_leak.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved {OUT / 'cv2024_internal_leak.json'}")


if __name__ == "__main__":
    main()
