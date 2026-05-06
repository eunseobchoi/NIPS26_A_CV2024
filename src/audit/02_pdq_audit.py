"""PDQ + pHash combined audit.

PDQ is a 256-bit perceptual hash used for near-duplicate matching. We run
it on the full corpus alongside our existing pHash + dHash to:
(a) corroborate that 100% CV2024-KVASIR flagging is not a pHash artifact;
(b) check whether PDQ flags additional near-duplicates in the
    negative-control sources (SEE-AI, KID, AIIMS);
(c) tighten the "lower bound" framing — if PDQ also gets ~0% on
    negative controls, the audit is more defensible.

PDQ threshold used here: <=90 bits Hamming on the 256-bit hash.
"""
import os
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pdqhash
from PIL import Image

ROOT = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
KVASIR_ROOT = Path(os.environ.get("KVASIR_ROOT", ROOT / "data/kvasir_capsule/labelled_images"))
if (KVASIR_ROOT / "labelled_images").is_dir():
    KVASIR_ROOT = KVASIR_ROOT / "labelled_images"
CV2024_ROOT = Path(os.environ.get("CV2024_ROOT", ROOT / "data/cv2024/Dataset"))
if (CV2024_ROOT / "Dataset").is_dir():
    CV2024_ROOT = CV2024_ROOT / "Dataset"
OUT = ROOT / "results"

PDQ_THRESHOLD_BITS = 90


def pdq_hash_256(img_np):
    """Returns (256-element uint8 array, quality)."""
    h, q = pdqhash.compute(img_np)
    return np.array(h, dtype=np.uint8), int(q)


def hamming_pdq(h1, h2):
    return int(np.count_nonzero(h1 != h2))


_POP16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)
_POP8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def hash_list_to_packed(hashes_list):
    """Pack list of 256-element uint8 hashes into (n, 4) uint64 matrix for fast XOR+popcount."""
    arr = np.array(hashes_list, dtype=np.uint8)  # (n, 256)
    # Pack bits into uint64: 4 × 64-bit per hash
    packed = np.packbits(arr, axis=1, bitorder="big").view(np.uint64).reshape(arr.shape[0], 4)
    return packed


def popcount_u64_2d(mat):
    """Popcount over uint64 2D array."""
    flat = mat.view(np.uint16).reshape(*mat.shape, 4)
    return _POP16[flat].astype(np.int16).sum(axis=-1)


def index_kvasir():
    items = []
    for folder in KVASIR_ROOT.iterdir():
        if folder.is_dir():
            for fn in folder.glob("*.jpg"):
                items.append({"source": "kvasir", "filename": fn.name,
                              "path": str(fn), "class": folder.name})
    return items


def index_cv2024():
    items = []
    for split in ("training", "validation"):
        xlsx = CV2024_ROOT / split / f"{split}_data.xlsx"
        df = pd.read_excel(xlsx)
        for _, row in df.iterrows():
            rel = row["image_path"].replace("\\", "/")
            path = CV2024_ROOT / rel
            if path.exists():
                items.append({"source": "cv2024",
                              "cv_dataset": row["Dataset"],
                              "cv_split": split,
                              "filename": Path(rel).name,
                              "path": str(path)})
    return items


def compute_all_pdq(items, cache_path):
    """Compute PDQ hashes with caching."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        if len(data) == len(items):
            print(f"  Loaded cached PDQ hashes: {len(data)}")
            return data
    results = []
    t0 = time.perf_counter()
    for i, it in enumerate(items):
        try:
            img = np.array(Image.open(it["path"]).convert("RGB"))
            h, q = pdq_hash_256(img)
            it2 = dict(it, pdq_hex=h.tobytes().hex(), pdq_quality=q)
        except Exception as e:
            it2 = dict(it, pdq_hex=None, pdq_quality=0)
        results.append(it2)
        if (i + 1) % 500 == 0:
            dt = time.perf_counter() - t0
            rate = (i + 1) / dt
            eta = (len(items) - i - 1) / rate
            print(f"  {i+1}/{len(items)}  rate={rate:.1f}/s  ETA {eta/60:.1f}min", flush=True)
    with open(cache_path, "w") as f:
        json.dump(results, f)
    return results


def all_pairs_min_hamming_pdq(source, target):
    """Returns (min_dist array, min_idx array). Uses bit-packed uint8 directly."""
    # Each hash is 32 uint8 bytes (256 bits)
    src_raw = np.array([list(bytes.fromhex(h["pdq_hex"])) for h in source], dtype=np.uint8)
    tgt_raw = np.array([list(bytes.fromhex(h["pdq_hex"])) for h in target], dtype=np.uint8)
    # src_raw shape: (n_src, 32), tgt_raw shape: (n_tgt, 32)

    n_src = src_raw.shape[0]
    min_d = np.full(n_src, 256, dtype=np.int16)
    min_idx = np.zeros(n_src, dtype=np.int64)
    chunk = 64
    t0 = time.perf_counter()
    for i in range(0, n_src, chunk):
        s = src_raw[i:i+chunk]  # (c, 32)
        # XOR: (c, 1, 32) ^ (1, n_tgt, 32) -> (c, n_tgt, 32)
        xor = s[:, None, :] ^ tgt_raw[None, :, :]  # uint8
        # popcount via 8-bit lookup (fast)
        pc = _POP8[xor].sum(axis=-1, dtype=np.int16)  # (c, n_tgt)
        j = pc.argmin(axis=1)
        min_idx[i:i+chunk] = j
        min_d[i:i+chunk] = pc[np.arange(len(s)), j]
        if (i // chunk) % 20 == 0 and i > 0:
            dt = time.perf_counter() - t0
            rate = i / dt
            eta = (n_src - i) / rate
            print(f"  PDQ min-Hamming {i}/{n_src}  rate={rate:.0f}/s  ETA {eta:.0f}s", flush=True)
    return min_d, min_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default=str(OUT / "pdq_audit.json"))
    args = ap.parse_args()

    print("[1/3] Index...")
    kv = index_kvasir()
    cv = index_cv2024()
    print(f"  Kvasir: {len(kv)}  CV2024: {len(cv)}")

    print("\n[2/3] Compute PDQ hashes...")
    print("  Kvasir:")
    kv = compute_all_pdq(kv, OUT / "pdq_hashes_kvasir.json")
    print("  CV2024:")
    cv = compute_all_pdq(cv, OUT / "pdq_hashes_cv2024.json")

    print("\n[3/3] Min-Hamming per CV2024 source...")
    report = {"n_kvasir": len(kv), "n_cv2024": len(cv),
              "threshold_near_dup": PDQ_THRESHOLD_BITS,
              "hash_bits": 256}
    for src in ("KVASIR", "SEE-AI", "KID", "AIIMS"):
        sub = [it for it in cv if it["cv_dataset"] == src]
        if not sub:
            continue
        print(f"\n  {src} n={len(sub)}")
        min_d, min_idx = all_pairs_min_hamming_pdq(sub, kv)
        # Count at PDQ threshold
        flagged_pdq_90 = int((min_d <= PDQ_THRESHOLD_BITS).sum())
        flagged_pdq_50 = int((min_d <= 50).sum())
        flagged_pdq_30 = int((min_d <= 30).sum())
        flagged_exact = int((min_d == 0).sum())
        # Histogram
        bins = [0, 10, 30, 50, 70, 90, 120, 150, 180, 256]
        hist = np.histogram(min_d, bins=bins)[0]
        report[src] = {
            "n": len(sub),
            "pdq_exact": flagged_exact,
            "pdq_le30": flagged_pdq_30,
            "pdq_le50": flagged_pdq_50,
            "pdq_le90": flagged_pdq_90,
            "pdq_mean": float(min_d.mean()),
            "pdq_median": float(np.median(min_d)),
            "pdq_hist_bins": bins,
            "pdq_hist": [int(v) for v in hist],
        }
        print(f"    exact: {flagged_exact}/{len(sub)} ({100*flagged_exact/len(sub):.1f}%)")
        print(f"    le30: {flagged_pdq_30}/{len(sub)} ({100*flagged_pdq_30/len(sub):.1f}%)")
        print(f"    le90 (Meta near-dup): {flagged_pdq_90}/{len(sub)} "
              f"({100*flagged_pdq_90/len(sub):.1f}%)")
        print(f"    mean Hamming: {min_d.mean():.1f}, median: {np.median(min_d):.0f}")
        # Save per-file annotated
        for i, it in enumerate(sub):
            it["pdq_min_dist_to_kvasir"] = int(min_d[i])
            it["pdq_nearest_kvasir_file"] = Path(kv[min_idx[i]]["path"]).name
        pd.DataFrame(sub).to_csv(OUT / f"cv2024_{src}_pdq_annotated.csv", index=False)

    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {args.out_json}")


if __name__ == "__main__":
    main()
