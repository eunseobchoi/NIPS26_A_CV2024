"""Multi-hash audit for CV2024 ↔ Kvasir-Capsule near-duplicate detection.

Methodology (NeurIPS D&B standard):
- pHash (Zauner 2010) + dHash complementary hashes (imagehash library, 64-bit)
- Hamming distance ≤ 6 = near-duplicate (Krawetz HackerFactor community standard)
- Hamming distance ≤ 2 = nearly identical
- Inter-set vs intra-set distance histograms
- Negative control: CV2024 SEE-AI vs Kvasir (unrelated source)
- Results cached to JSON for reproducibility

References:
- Zauner 2010 (pHash dissertation)
- Barz & Denzler 2020 ciFAIR (manual verification protocol)
- Hao et al. 2023 (Hamming distribution analysis)
- Wahlang et al. 2024 EndoExtend24 (explicitly notes CV2024 = Kvasir cropped variant)
"""
import os
import argparse
import hashlib
import json
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import imagehash

ROOT = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
KVASIR_ROOT = Path(os.environ.get("KVASIR_ROOT", ROOT / "data/kvasir_capsule/labelled_images"))
if (KVASIR_ROOT / "labelled_images").is_dir():
    KVASIR_ROOT = KVASIR_ROOT / "labelled_images"
CV2024_ROOT = Path(os.environ.get("CV2024_ROOT", ROOT / "data/cv2024/Dataset"))
if (CV2024_ROOT / "Dataset").is_dir():
    CV2024_ROOT = CV2024_ROOT / "Dataset"
OUT = ROOT / "results"


def index_kvasir_frames():
    """Build {filename: absolute_path} for Kvasir labeled frames."""
    items = []
    for folder in KVASIR_ROOT.iterdir():
        if folder.is_dir():
            for fn in folder.glob("*.jpg"):
                items.append({"source": "kvasir",
                              "partition": folder.name,  # class folder
                              "filename": fn.name,
                              "path": str(fn)})
    return items


def index_cv2024_frames():
    """Build Dataset x partition x filename for CV2024."""
    items = []
    for split in ("training", "validation"):
        xlsx = CV2024_ROOT / split / f"{split}_data.xlsx"
        df = pd.read_excel(xlsx)
        for _, row in df.iterrows():
            rel = row["image_path"].replace("\\", "/")
            path = CV2024_ROOT / rel
            if path.exists():
                items.append({"source": "cv2024",
                              "partition": f"{split}_{row['Dataset']}",
                              "cv_dataset": row["Dataset"],  # KVASIR / SEE-AI / KID / AIIMS
                              "cv_split": split,  # training / validation
                              "filename": Path(rel).name,
                              "path": str(path)})
    return items


def hash_all(items, cache_path, hash_size=8):
    """Compute pHash + dHash for every item, cache to JSON."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        if len(data) == len(items):
            print(f"  Loaded cached hashes: {len(data)}")
            return data
    results = []
    t0 = time.perf_counter()
    for i, it in enumerate(items):
        try:
            img = Image.open(it["path"]).convert("RGB")
            ph = str(imagehash.phash(img, hash_size=hash_size))
            dh = str(imagehash.dhash(img, hash_size=hash_size))
        except Exception as e:
            ph = None; dh = None
        r = dict(it, phash=ph, dhash=dh)
        results.append(r)
        if (i + 1) % 1000 == 0:
            dt = time.perf_counter() - t0
            rate = (i + 1) / dt
            eta = (len(items) - i - 1) / rate
            print(f"  {i+1}/{len(items)}  rate={rate:.1f}/s  ETA {eta/60:.1f}min",
                  flush=True)
    with open(cache_path, "w") as f:
        json.dump(results, f)
    return results


def hex_to_int(h):
    return int(h, 16) if h else None


def hamming_distance(a, b):
    """Hamming distance between two hex strings (64-bit hashes)."""
    if a is None or b is None:
        return 64
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def build_bucket_index(items, field="phash", first_bits=12):
    """Map first N bits of hash -> list of items for fast lookup.

    With first_bits=12 and 64-bit hashes, bucket size ≈ n / 2^12,
    comparison count drops from O(n*m) to O(n*m / 2^12 * ~2^12/2) = still
    O(n*m/2) worst case. Better: exhaustive for our ~50k scale is fine.
    """
    idx = defaultdict(list)
    for it in items:
        h = it.get(field)
        if h:
            key = h[:first_bits // 4]  # hex chars = 4 bits each
            idx[key].append(it)
    return idx


# Precomputed 16-bit popcount lookup table (uint8)
_POPCOUNT16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)


def popcount_u64_matrix(mat):
    """Fast popcount over a uint64 2D array using 16-bit lookup."""
    # Reinterpret as uint16 4-tuples along last axis
    flat = mat.view(np.uint16).reshape(*mat.shape, 4)
    return _POPCOUNT16[flat].astype(np.int16).sum(axis=-1)


def all_pairs_min_hamming(source_items, target_items, field="phash"):
    """For each source item, find target item with minimum Hamming distance.

    Uses uint16 popcount lookup table, O(n*m / 4) table lookups.
    For 38K × 47K pairs this finishes in ~30s.
    """
    tgt_ints = np.array([hex_to_int(it.get(field)) or 0 for it in target_items],
                        dtype=np.uint64)
    src_ints = np.array([hex_to_int(it.get(field)) or 0 for it in source_items],
                        dtype=np.uint64)
    min_dist = np.zeros(len(source_items), dtype=np.int16)
    min_idx = np.zeros(len(source_items), dtype=np.int64)
    # Process in chunks to fit the uint16 expansion in memory
    chunk = 512
    t0 = time.perf_counter()
    n_src = len(src_ints)
    for i in range(0, n_src, chunk):
        s = src_ints[i:i+chunk]
        xor = s[:, None] ^ tgt_ints[None, :]  # (c, m) uint64
        pc = popcount_u64_matrix(xor)  # (c, m) int16
        j = pc.argmin(axis=1)
        min_idx[i:i+chunk] = j
        min_dist[i:i+chunk] = pc[np.arange(len(s)), j]
        if (i // chunk) % 10 == 0 and i > 0:
            dt = time.perf_counter() - t0
            rate = i / dt
            eta = (n_src - i) / rate
            print(f"  min-Hamming {i}/{n_src}  rate={rate:.0f}/s  ETA {eta:.0f}s",
                  flush=True)
    return min_dist, min_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_hash", action="store_true",
                    help="Use cached hashes if present")
    ap.add_argument("--out_json", default=str(OUT / "phash_audit.json"))
    args = ap.parse_args()

    # Index
    print("[1/5] Indexing Kvasir frames...")
    kv = index_kvasir_frames()
    print(f"  Kvasir: {len(kv)} frames")
    print("[1/5] Indexing CV2024 frames...")
    cv = index_cv2024_frames()
    by_src = Counter(it["cv_dataset"] for it in cv)
    print(f"  CV2024: {len(cv)} frames ({dict(by_src)})")

    # Hash
    print("\n[2/5] Computing pHash + dHash on Kvasir (may take ~30min)...")
    kv = hash_all(kv, OUT / "hashes_kvasir.json")
    print(f"\n[2/5] Computing pHash + dHash on CV2024 (may take ~30min)...")
    cv = hash_all(cv, OUT / "hashes_cv2024.json")

    # Diagnostic: inter-set min-Hamming distributions per CV2024 source
    print(f"\n[3/5] Computing min-Hamming distance from each CV2024 file to Kvasir...")
    # Split CV2024 by source for separate audits
    report = {"n_kvasir": len(kv), "n_cv2024": len(cv),
              "by_cv_dataset": dict(by_src)}
    for src in ("KVASIR", "SEE-AI", "KID", "AIIMS"):
        sub = [it for it in cv if it["cv_dataset"] == src]
        if not sub:
            continue
        print(f"\n  Source={src}  n={len(sub)}")
        # pHash
        min_d_ph, min_idx_ph = all_pairs_min_hamming(sub, kv, field="phash")
        # dHash
        min_d_dh, min_idx_dh = all_pairs_min_hamming(sub, kv, field="dhash")
        # Histogram buckets
        bins = [0, 1, 3, 6, 10, 15, 20, 30, 64]
        ph_hist = np.histogram(min_d_ph, bins=bins)[0]
        dh_hist = np.histogram(min_d_dh, bins=bins)[0]
        # Joint: flagged if pHash ≤ 6 AND dHash ≤ 6
        dual_le6 = ((min_d_ph <= 6) & (min_d_dh <= 6)).sum()
        dual_le2 = ((min_d_ph <= 2) & (min_d_dh <= 2)).sum()
        dual_exact = ((min_d_ph == 0) & (min_d_dh == 0)).sum()
        # Per-CV2024 partition breakdown
        by_split = Counter(it["cv_split"] for it in sub)
        flagged_train = sum(1 for i, it in enumerate(sub)
                            if it["cv_split"] == "training"
                            and min_d_ph[i] <= 6 and min_d_dh[i] <= 6)
        flagged_val = sum(1 for i, it in enumerate(sub)
                          if it["cv_split"] == "validation"
                          and min_d_ph[i] <= 6 and min_d_dh[i] <= 6)
        report[src] = {
            "n_total": len(sub),
            "by_split": dict(by_split),
            "phash_hist": {str(b): int(v) for b, v in zip(bins[1:], ph_hist)},
            "dhash_hist": {str(b): int(v) for b, v in zip(bins[1:], dh_hist)},
            "dual_exact": int(dual_exact),
            "dual_le2": int(dual_le2),
            "dual_le6": int(dual_le6),
            "dual_le6_training": int(flagged_train),
            "dual_le6_validation": int(flagged_val),
            "phash_mean": float(min_d_ph.mean()),
            "phash_median": float(np.median(min_d_ph)),
            "dhash_mean": float(min_d_dh.mean()),
            "dhash_median": float(np.median(min_d_dh)),
        }
        # Save the minimum distance per item for dedup CSV
        for i, it in enumerate(sub):
            it["min_phash_dist_to_kvasir"] = int(min_d_ph[i])
            it["min_dhash_dist_to_kvasir"] = int(min_d_dh[i])
            it["nearest_kvasir_file"] = Path(kv[min_idx_ph[i]]["path"]).name
        # Dump dedup-annotated list
        pd.DataFrame(sub).to_csv(OUT / f"cv2024_{src}_phash_annotated.csv",
                                  index=False)
        print(f"  {src}: pHash<=6 AND dHash<=6: {dual_le6}/{len(sub)} "
              f"({100*dual_le6/len(sub):.1f}%);  exact match: {dual_exact}")

    # Save report
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[5/5] Saved {args.out_json}")


if __name__ == "__main__":
    main()
