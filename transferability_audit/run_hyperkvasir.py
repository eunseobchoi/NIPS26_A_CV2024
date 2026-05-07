#!/usr/bin/env python3
"""
HyperKvasir audit:
  1. Compute pHash + dHash on all labeled images
  2. Cross-bench against CV2024-KVASIR pHashes (Kvasir-Capsule slice)
     to test framework's positive sensitivity to overlap between two
     Kvasir-family endoscopy datasets from the same lab (Simula).

Outputs:
  - results/hyperkvasir_phash_annotated.csv (per-image hashes)
  - results/hyperkvasir_audit.json (intra-set + cross-bench summary)
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image


def hex_to_u64(h):
    return np.uint64(int(h, 16))


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + (
        (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def cross_bench_le6(rows_a, rows_b, threshold=6, chunk_size=512):
    """Find joint pHash+dHash <= threshold pairs between A and B."""
    p_a = np.array([hex_to_u64(r["phash"]) for r in rows_a], dtype=np.uint64)
    d_a = np.array([hex_to_u64(r["dhash"]) for r in rows_a], dtype=np.uint64)
    p_b = np.array([hex_to_u64(r["phash"]) for r in rows_b], dtype=np.uint64)
    d_b = np.array([hex_to_u64(r["dhash"]) for r in rows_b], dtype=np.uint64)
    ids_a = [r["image_id"] for r in rows_a]
    ids_b = [r["image_id"] for r in rows_b]
    pairs = []
    for s in range(0, len(rows_a), chunk_size):
        e = min(s + chunk_size, len(rows_a))
        ph_xor = p_a[s:e, None] ^ p_b[None, :]
        dh_xor = d_a[s:e, None] ^ d_b[None, :]
        joint = popcount64(ph_xor) + popcount64(dh_xor)
        ai, bi = np.where(joint <= threshold)
        for i_local, j in zip(ai, bi):
            i = s + int(i_local)
            pairs.append((ids_a[i], ids_b[int(j)], int(joint[i_local, j])))
        if (e // chunk_size) % 5 == 0:
            print(f"    {e}/{len(rows_a)}", file=sys.stderr, flush=True)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True,
                    help="Top dir with HyperKvasir labeled images "
                         "(any nested *.jpg)")
    ap.add_argument("--cv2024_kvasir_csv",
                    default="/home/user/main/capsule_tta/results/"
                            "cv2024_KVASIR_phash_annotated.csv")
    ap.add_argument("--out", default="results/hyperkvasir_audit.json")
    ap.add_argument("--annotated_out",
                    default="results/hyperkvasir_phash_annotated.csv")
    args = ap.parse_args()

    images_dir = Path(args.images)
    image_paths = sorted(images_dir.rglob("*.jpg"))
    print(f"Found {len(image_paths)} jpg files", file=sys.stderr)

    # Compute pHash + dHash
    rows = []
    skipped = 0
    for idx, path in enumerate(image_paths, 1):
        try:
            img = Image.open(path)
            ph = imagehash.phash(img, hash_size=8)
            dh = imagehash.dhash(img, hash_size=8)
            # Use relative path as image_id (preserve subdir labels)
            rel = path.relative_to(images_dir)
            rows.append({
                "image_id": str(rel),
                "phash": str(ph),
                "dhash": str(dh),
            })
        except Exception:
            skipped += 1
        if idx % 2000 == 0:
            print(f"  hashed {idx}/{len(image_paths)}", file=sys.stderr,
                  flush=True)
    print(f"  done. hashed {len(rows)} / {len(image_paths)}, skipped {skipped}",
          file=sys.stderr)

    # Save annotated CSV
    Path(args.annotated_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.annotated_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "phash", "dhash"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {args.annotated_out}", file=sys.stderr)

    # Intra-HyperKvasir pHash-exact and joint <= 6
    print("Intra-HyperKvasir analysis...", file=sys.stderr)
    phash_to_ids = defaultdict(list)
    for r in rows:
        phash_to_ids[r["phash"]].append(r["image_id"])
    intra_exact_groups = {h: ids for h, ids in phash_to_ids.items() if len(ids) > 1}
    intra_exact_extra = sum(len(ids) for ids in intra_exact_groups.values()) - len(intra_exact_groups)

    # Intra joint <= 6 (split-pair to avoid n^2 memory)
    half = len(rows) // 2
    intra_pairs = cross_bench_le6(rows[:half], rows[half:])
    intra_joint_le6 = len(intra_pairs)
    print(f"  intra pHash-exact extra rows: {intra_exact_extra} "
          f"({100*intra_exact_extra/len(rows):.3f}%)", file=sys.stderr)
    print(f"  intra joint<=6 (split-pair): {intra_joint_le6}",
          file=sys.stderr)

    # Cross-bench: HyperKvasir x CV2024-KVASIR
    print(f"Loading CV2024-KVASIR pHashes from {args.cv2024_kvasir_csv}...",
          file=sys.stderr)
    cv_rows = []
    with open(args.cv2024_kvasir_csv) as f:
        for r in csv.DictReader(f):
            cv_rows.append({
                "image_id": r["filename"],
                "phash": r["phash"],
                "dhash": r["dhash"],
            })
    print(f"  loaded {len(cv_rows)} CV2024-KVASIR rows", file=sys.stderr)

    print("Cross-bench HyperKvasir x CV2024-KVASIR (joint <= 6)...",
          file=sys.stderr)
    cross_pairs = cross_bench_le6(rows, cv_rows)
    print(f"  cross-bench joint<=6: {len(cross_pairs)} pairs",
          file=sys.stderr)

    # pHash-exact specifically
    cv_phash_set = set(r["phash"] for r in cv_rows)
    hk_exact = sum(1 for r in rows if r["phash"] in cv_phash_set)
    print(f"  cross-bench pHash-exact (any joint=0 with CV2024 KVASIR): "
          f"{hk_exact} HyperKvasir rows", file=sys.stderr)

    summary = {
        "benchmark": "HyperKvasir labeled images",
        "n_images_indexed": len(rows),
        "n_skipped": skipped,
        "intra_set": {
            "phash_exact_groups": len(intra_exact_groups),
            "phash_exact_extra_rows": intra_exact_extra,
            "phash_exact_extra_pct": round(
                100 * intra_exact_extra / max(len(rows), 1), 3
            ),
            "joint_le6_split_pair_count": intra_joint_le6,
        },
        "cross_bench_vs_cv2024_kvasir": {
            "n_cv2024_kvasir": len(cv_rows),
            "joint_le6_pairs": len(cross_pairs),
            "joint_le6_examples": [
                {"hyperkvasir": a, "cv2024_kvasir": b, "joint": j}
                for a, b, j in cross_pairs[:30]
            ],
            "phash_exact_hyperkvasir_rows": hk_exact,
            "phash_exact_rate_per_hyperkvasir_image_pct": round(
                100 * hk_exact / max(len(rows), 1), 3
            ),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
