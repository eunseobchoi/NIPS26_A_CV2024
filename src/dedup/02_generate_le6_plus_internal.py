"""Generate le6_plus_internal.csv: le6 validation PLUS additional internal leakage removal.

le6 already removes KVASIR-origin near-duplicates. This script additionally removes:
1. SEE-AI within-source filename collisions (171 files)
2. SEE-AI within-source pHash-exact internal pairs
3. KID/AIIMS within-source pHash-exact internal pairs

Produces a stricter validation set that removes BOTH Kvasir leakage AND intra-source leakage.
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CAPSULE_ROOT", "."))
OUT = ROOT / "results"


def main():
    # Start from le6-dedup validation (KVASIR already removed)
    le6_val = pd.read_csv(OUT / "cv2024_validation_dedup_le6.csv")
    print(f"Starting le6 val: {len(le6_val)} (KVASIR removed)")

    # Load the internal per-source pHash data
    with open(OUT / "cv2024_internal_cross_source.json") as f:
        cs = json.load(f)

    # Load the raw CV2024 hash cache to find which files are flagged
    with open(OUT / "hashes_cv2024.json") as f:
        cv = json.load(f)

    # Build (filename, cv_dataset, cv_split) → phash/dhash map
    cv_index = {x["filename"]: x for x in cv if x.get("phash")}

    # Per-source: training pHashes and validation pHashes
    _POP16 = np.array([bin(i).count("1") for i in range(65536)], dtype=np.uint8)
    def popcount(arr):
        f = arr.view(np.uint16).reshape(*arr.shape, 4)
        return _POP16[f].astype(np.int16).sum(axis=-1)

    drop_files = set()

    for src in ("SEE-AI", "KID", "AIIMS"):
        tr = [x for x in cv if x.get("cv_split") == "training" and x.get("cv_dataset") == src and x.get("phash")]
        va = [x for x in cv if x.get("cv_split") == "validation" and x.get("cv_dataset") == src and x.get("phash")]
        if not tr or not va:
            continue

        # (a) identical filename overlap
        tr_names = set(x["filename"] for x in tr)
        va_names = set(x["filename"] for x in va)
        overlap = tr_names & va_names
        print(f"{src}: filename overlap {len(overlap)} files")
        drop_files.update(overlap)

        # (b) pHash-exact internal pairs
        tr_ph = np.array([int(x["phash"], 16) for x in tr], dtype=np.uint64)
        va_ph = np.array([int(x["phash"], 16) for x in va], dtype=np.uint64)
        chunk = 256
        for i in range(0, len(va_ph), chunk):
            vh = va_ph[i : i + chunk]
            xor = vh[:, None] ^ tr_ph[None, :]
            pc = popcount(xor)
            min_d = pc.min(axis=1)
            for k in range(len(vh)):
                if min_d[k] == 0:
                    drop_files.add(va[i + k]["filename"])
        print(f"{src}: total drop set now {len(drop_files)} files (across all sources so far)")

    # Apply drop
    # CV2024 val XLSX has filename column — load original val to compute after le6
    cv2024_root = Path(os.environ.get("CV2024_ROOT", ROOT / "data/cv2024/Dataset"))
    if (cv2024_root / "Dataset").is_dir():
        cv2024_root = cv2024_root / "Dataset"
    val_xlsx = cv2024_root / "validation/validation_data.xlsx"
    if not val_xlsx.exists():
        # Run on edge-device where data is not available — load from CSV directly
        print(f"WARNING: {val_xlsx} not found; dropping from le6_val by filename match")
        # le6_val may or may not have the filename column depending on pipeline
        fn_col = "image_path" if "image_path" in le6_val.columns else "filename"
        if fn_col in le6_val.columns:
            # Extract basename — handle both forward and backslash separators
            def _bn(x):
                if not x:
                    return None
                s = str(x).replace("\\", "/")
                return s.rsplit("/", 1)[-1]
            le6_val["_basename"] = le6_val[fn_col].apply(_bn)
            before = len(le6_val)
            le6_plus = le6_val[~le6_val["_basename"].isin(drop_files)].drop(columns=["_basename"])
            print(f"le6_plus_internal: {before} → {len(le6_plus)} ({before - len(le6_plus)} more removed)")
        else:
            print(f"ERROR: no filename-like column in le6_val, cols={le6_val.columns.tolist()}")
            return
    else:
        df_val = pd.read_excel(val_xlsx)
        fn_col = next((c for c in df_val.columns if "image" in c.lower() or "path" in c.lower() or "file" in c.lower()), None)
        print(f"using filename column: {fn_col}")
        def _bn(x):
            if not x:
                return None
            s = str(x).replace("\\", "/")
            return s.rsplit("/", 1)[-1]
        df_val["_basename"] = df_val[fn_col].apply(_bn)
        le6_val["_basename"] = le6_val.get("image_path", le6_val.iloc[:, 0]).apply(_bn)
        le6_plus = le6_val[~le6_val["_basename"].isin(drop_files)].drop(columns=["_basename"], errors="ignore")
        print(f"le6_plus_internal: {len(le6_val)} → {len(le6_plus)}")

    le6_plus.to_csv(OUT / "cv2024_validation_le6_plus_internal.csv", index=False)
    print(f"Saved: {OUT / 'cv2024_validation_le6_plus_internal.csv'}")

    # Summary
    summary = {
        "le6_val_rows": len(le6_val),
        "le6_plus_internal_rows": len(le6_plus),
        "extra_removed": len(le6_val) - len(le6_plus),
        "drop_files_identified": len(drop_files),
    }
    with open(OUT / "le6_plus_internal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
