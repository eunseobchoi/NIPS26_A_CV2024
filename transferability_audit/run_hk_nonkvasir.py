#!/usr/bin/env python3
"""
HyperKvasir x CV2024 non-KVASIR slices cross-bench.
Pre-registered threshold: joint <= 6 (same as our HK x KVASIR cross-bench).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np


def hex_to_u64(h):
    return np.uint64(int(h, 16))


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + (
        (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def cross_le6(a_phash, a_dhash, b_phash, b_dhash, threshold=6, chunk=512):
    pairs = []
    n_a = len(a_phash)
    for s in range(0, n_a, chunk):
        e = min(s + chunk, n_a)
        ph_xor = a_phash[s:e, None] ^ b_phash[None, :]
        dh_xor = a_dhash[s:e, None] ^ b_dhash[None, :]
        joint = popcount64(ph_xor) + popcount64(dh_xor)
        ai, bi = np.where(joint <= threshold)
        for i_local, j in zip(ai, bi):
            pairs.append((s + int(i_local), int(j), int(joint[i_local, j])))
    return pairs


def load_cv_csv(path, source_label):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "image_id": r["filename"],
                "source": source_label,
                "phash": r["phash"],
                "dhash": r["dhash"],
            })
    return rows


def main():
    print("Loading HyperKvasir hashes...", file=sys.stderr)
    hk = []
    with open("results/hyperkvasir_phash_annotated.csv") as f:
        for r in csv.DictReader(f):
            hk.append(r)
    hk_p = np.array([hex_to_u64(r["phash"]) for r in hk], dtype=np.uint64)
    hk_d = np.array([hex_to_u64(r["dhash"]) for r in hk], dtype=np.uint64)
    print(f"  HyperKvasir n={len(hk)}", file=sys.stderr)

    sources = {
        "SEE-AI": "/home/user/main/capsule_tta/results/cv2024_SEE-AI_phash_annotated.csv",
        "KID":    "/home/user/main/capsule_tta/results/cv2024_KID_phash_annotated.csv",
        "AIIMS":  "/home/user/main/capsule_tta/results/cv2024_AIIMS_phash_annotated.csv",
    }
    summary = {"hyperkvasir_n": len(hk), "threshold_joint": 6, "results": {}}
    for src, p in sources.items():
        cv = load_cv_csv(p, src)
        cv_p = np.array([hex_to_u64(r["phash"]) for r in cv], dtype=np.uint64)
        cv_d = np.array([hex_to_u64(r["dhash"]) for r in cv], dtype=np.uint64)
        print(f"  cross-bench HyperKvasir x CV2024-{src} ({len(cv)})...",
              file=sys.stderr)
        pairs = cross_le6(hk_p, hk_d, cv_p, cv_d)
        # phash-exact specifically
        cv_phash_set = set(r["phash"] for r in cv)
        hk_exact = sum(1 for r in hk if r["phash"] in cv_phash_set)
        summary["results"][src] = {
            "cv_source_n": len(cv),
            "joint_le6_pairs": len(pairs),
            "phash_exact_hk_rows": hk_exact,
            "examples": [
                {"hk": hk[a]["image_id"], f"cv_{src}": cv[b]["image_id"], "joint": j}
                for a, b, j in pairs[:10]
            ],
        }
        print(f"    joint<=6 pairs: {len(pairs)}, pHash-exact HK rows: {hk_exact}",
              file=sys.stderr)

    Path("results/hk_nonkvasir_xbench.json").write_text(
        json.dumps(summary, indent=2))
    print("Wrote results/hk_nonkvasir_xbench.json", file=sys.stderr)


if __name__ == "__main__":
    main()
