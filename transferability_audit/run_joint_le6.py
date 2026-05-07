#!/usr/bin/env python3
"""
ISIC 2019 joint pHash+dHash <= 6 cross-source audit.

Reads isic2019_phash_annotated.csv (per-image pHash + dHash + source) and
computes for each ordered cross-source pair (A < B) the count of joint
pHash+dHash Hamming-distance <= 6 collisions, using numpy-vectorized XOR
+ popcount over uint64.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def hex_to_u64(h: str) -> np.uint64:
    return np.uint64(int(h, 16))


def popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorized popcount on uint64 array — SWAR algorithm."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + (
        (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def cross_source_le6(rows_a, rows_b, threshold=6, chunk_size=512):
    """
    rows_*: list of dicts with 'image_id', 'phash', 'dhash'.
    Returns list of (a_id, b_id, joint_distance) tuples with joint <= threshold.
    """
    p_a = np.array([hex_to_u64(r["phash"]) for r in rows_a], dtype=np.uint64)
    d_a = np.array([hex_to_u64(r["dhash"]) for r in rows_a], dtype=np.uint64)
    p_b = np.array([hex_to_u64(r["phash"]) for r in rows_b], dtype=np.uint64)
    d_b = np.array([hex_to_u64(r["dhash"]) for r in rows_b], dtype=np.uint64)
    ids_a = [r["image_id"] for r in rows_a]
    ids_b = [r["image_id"] for r in rows_b]

    pairs = []
    n_a, n_b = len(rows_a), len(rows_b)
    for start in range(0, n_a, chunk_size):
        end = min(start + chunk_size, n_a)
        chunk_pa = p_a[start:end, None]  # (chunk, 1)
        chunk_da = d_a[start:end, None]
        # Broadcast: (chunk, 1) ^ (n_b,) -> (chunk, n_b)
        ph_xor = chunk_pa ^ p_b[None, :]
        dh_xor = chunk_da ^ d_b[None, :]
        joint = popcount64(ph_xor) + popcount64(dh_xor)
        # Find indices where joint <= threshold
        ai, bi = np.where(joint <= threshold)
        for i_local, j in zip(ai, bi):
            i = start + int(i_local)
            j = int(j)
            pairs.append((ids_a[i], ids_b[j], int(joint[i_local, j])))
        if (end // chunk_size) % 5 == 0:
            print(f"    {end}/{n_a}", file=sys.stderr, flush=True)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/isic2019_phash_annotated.csv")
    ap.add_argument("--out", default="results/isic2019_joint_le6.json")
    ap.add_argument("--threshold", type=int, default=6)
    args = ap.parse_args()

    rows = []
    with open(args.input) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    sources = sorted(by_source.keys())
    print(f"Sources: {[(s, len(by_source[s])) for s in sources]}", file=sys.stderr)

    cross_pair_counts = {}
    cross_pairs_examples = {}
    for i, sa in enumerate(sources):
        for sb in sources[i + 1 :]:
            print(
                f"  Cross-source: {sa} ({len(by_source[sa])}) x {sb} ({len(by_source[sb])})",
                file=sys.stderr,
            )
            pairs = cross_source_le6(
                by_source[sa], by_source[sb], threshold=args.threshold
            )
            key = f"{sa}__{sb}"
            cross_pair_counts[key] = len(pairs)
            cross_pairs_examples[key] = [
                {"a": a, "b": b, "joint": j} for a, b, j in pairs[:20]
            ]
            print(f"    joint<={args.threshold}: {len(pairs)} pairs", file=sys.stderr)

    # Intra-source joint <= 6 — full unordered i<j enumeration
    intra_pair_counts = {}
    intra_pairs_examples = {}
    for s in sources:
        rows_s = by_source[s]
        n = len(rows_s)
        print(f"  Intra-source: {s} (n={n}, full unordered i<j enumeration)",
              file=sys.stderr)
        # Build chunked pairwise mask, count i<j pairs with joint<=threshold.
        p_arr = np.array([hex_to_u64(r["phash"]) for r in rows_s], dtype=np.uint64)
        d_arr = np.array([hex_to_u64(r["dhash"]) for r in rows_s], dtype=np.uint64)
        chunk = 256
        unordered_pairs = []
        for s_idx in range(0, n, chunk):
            e_idx = min(s_idx + chunk, n)
            ph_xor = p_arr[s_idx:e_idx, None] ^ p_arr[None, :]
            dh_xor = d_arr[s_idx:e_idx, None] ^ d_arr[None, :]
            joint = popcount64(ph_xor) + popcount64(dh_xor)
            rows_idx = np.arange(s_idx, e_idx)[:, None]
            cols_idx = np.arange(n)[None, :]
            mask_lower = rows_idx < cols_idx
            le = (joint <= args.threshold) & mask_lower
            ai, bi = np.where(le)
            for i_local, j in zip(ai, bi):
                ig = s_idx + int(i_local)
                jg = int(j)
                unordered_pairs.append(
                    (rows_s[ig]["image_id"], rows_s[jg]["image_id"],
                     int(joint[i_local, j])))
        intra_pair_counts[s] = len(unordered_pairs)
        intra_pairs_examples[s] = [
            {"a": a, "b": b, "joint": j} for a, b, j in unordered_pairs[:20]
        ]
        print(f"    unordered joint<={args.threshold}: {len(unordered_pairs)} pairs",
              file=sys.stderr)

    n_total = sum(len(v) for v in by_source.values())
    cross_total = sum(cross_pair_counts.values())

    summary = {
        "benchmark": "ISIC 2019 (training)",
        "n_indexed": n_total,
        "threshold_joint": args.threshold,
        "source_distribution": {s: len(by_source[s]) for s in sources},
        "cross_source_joint_le6": {
            "pair_counts": cross_pair_counts,
            "total_pairs": cross_total,
            "rate_per_indexed_image_pct": round(100 * cross_total / n_total, 4),
            "examples": cross_pairs_examples,
        },
        "intra_source_joint_le6": {
            "pair_counts": intra_pair_counts,
            "examples": intra_pairs_examples,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
