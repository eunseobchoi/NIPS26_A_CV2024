"""Full NCC verification on ALL 38,592 CV2024-KVASIR flagged pairs.

Addresses r2 §4.4 imbalance: external audit 100% is hash-verified
but only 60.7% trivially pixel-identical; remaining 39.3% hash-flagged
pairs need explicit NCC check.

For each CV2024-KVASIR file, compute NCC to its nearest-pHash
Kvasir frame (attribution already in cv2024_KVASIR_phash_annotated.csv).
Report pixel-verified rate by Hamming band.
"""
import os
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
OUT = ROOT / "results"
KVASIR = Path(os.environ.get("KVASIR_ROOT", ROOT / "data/kvasir_capsule/labelled_images"))
if (KVASIR / "labelled_images").is_dir():
    KVASIR = KVASIR / "labelled_images"


def ncc(p1, p2):
    a = np.array(Image.open(p1).convert("L").resize((96, 96))).astype(np.float32)
    b = np.array(Image.open(p2).convert("L").resize((96, 96))).astype(np.float32)
    am = a - a.mean(); bm = b - b.mean()
    return float((am*bm).sum() / ((np.sqrt((am**2).sum())*np.sqrt((bm**2).sum())) + 1e-12))


def main():
    df = pd.read_csv(OUT / "cv2024_KVASIR_phash_annotated.csv")
    print(f"CV2024-KVASIR annotated: {len(df)} files")

    # Find nearest Kvasir file path
    kv_idx = {}
    for folder in KVASIR.iterdir():
        if folder.is_dir():
            for fn in folder.glob("*.jpg"):
                kv_idx.setdefault(fn.name, str(fn))
    print(f"Kvasir labeled frames: {len(kv_idx)}")

    # Only process flagged pairs: pHash ≤ 6 AND dHash ≤ 6 (all 38592 KVASIR)
    flagged = df[(df["min_phash_dist_to_kvasir"] <= 6) &
                 (df["min_dhash_dist_to_kvasir"] <= 6)]
    print(f"Flagged (joint ≤6): {len(flagged)}")
    # Also need nearest Kvasir filename
    if "nearest_kvasir_file" not in flagged.columns:
        raise SystemExit("Missing nearest_kvasir_file column in annotation CSV")

    t0 = time.perf_counter()
    results = []
    for i, (_, row) in enumerate(flagged.iterrows()):
        cv_path = row["path"]
        kv_name = row["nearest_kvasir_file"]
        kv_path = kv_idx.get(kv_name)
        if kv_path is None:
            continue
        try:
            sim = ncc(cv_path, kv_path)
        except Exception:
            sim = None
        results.append({
            "filename": row["filename"],
            "cv_split": row["cv_split"],
            "kvasir_file": kv_name,
            "phash_dist": int(row["min_phash_dist_to_kvasir"]),
            "dhash_dist": int(row["min_dhash_dist_to_kvasir"]),
            "ncc": sim,
        })
        if (i+1) % 2000 == 0:
            dt = time.perf_counter() - t0
            eta = (len(flagged) - i - 1) / (i+1) * dt
            print(f"  {i+1}/{len(flagged)}  elapsed {dt:.0f}s  ETA {eta/60:.1f}min", flush=True)

    out = pd.DataFrame(results).dropna(subset=["ncc"])
    out.to_csv(OUT / "cv2024_KVASIR_ncc_full.csv", index=False)

    print(f"\n=== NCC summary (n={len(out)}) ===")
    print(f"  mean: {out['ncc'].mean():.4f}")
    print(f"  median: {out['ncc'].median():.4f}")
    print(f"  NCC >= 0.99: {(out['ncc'] >= 0.99).sum()} ({100*(out['ncc'] >= 0.99).mean():.2f}%)")
    print(f"  NCC >= 0.95: {(out['ncc'] >= 0.95).sum()} ({100*(out['ncc'] >= 0.95).mean():.2f}%)")
    print(f"  NCC >= 0.80: {(out['ncc'] >= 0.80).sum()} ({100*(out['ncc'] >= 0.80).mean():.2f}%)")

    print(f"\n=== Per Hamming band (pHash+dHash joint) ===")
    out["max_hamming"] = out[["phash_dist","dhash_dist"]].max(axis=1)
    for band in [0, 2, 4, 6]:
        sub = out[out["max_hamming"] == band]
        if len(sub) > 0:
            pct_99 = 100*(sub["ncc"] >= 0.99).mean()
            pct_95 = 100*(sub["ncc"] >= 0.95).mean()
            print(f"  max-hamming == {band}: n={len(sub)}  NCC≥0.99 {pct_99:.1f}%  NCC≥0.95 {pct_95:.1f}%")

    summary = {
        "n_flagged": len(flagged), "n_ncc_done": len(out),
        "ncc_mean": float(out["ncc"].mean()),
        "ncc_ge_99": int((out["ncc"] >= 0.99).sum()),
        "ncc_ge_95": int((out["ncc"] >= 0.95).sum()),
        "ncc_ge_99_frac": float((out["ncc"] >= 0.99).mean()),
        "ncc_ge_95_frac": float((out["ncc"] >= 0.95).mean()),
        "per_band": {},
    }
    for band in [0, 1, 2, 3, 4, 5, 6]:
        sub = out[out["max_hamming"] == band]
        if len(sub) > 0:
            summary["per_band"][band] = {
                "n": len(sub),
                "ncc_ge_99": int((sub["ncc"] >= 0.99).sum()),
                "ncc_ge_99_frac": float((sub["ncc"] >= 0.99).mean()),
                "ncc_ge_95": int((sub["ncc"] >= 0.95).sum()),
                "ncc_ge_95_frac": float((sub["ncc"] >= 0.95).mean()),
            }
    with open(OUT / "cv2024_KVASIR_ncc_full_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT / 'cv2024_KVASIR_ncc_full_summary.json'}")


if __name__ == "__main__":
    main()
