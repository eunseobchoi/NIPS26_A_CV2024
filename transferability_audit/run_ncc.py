#!/usr/bin/env python3
"""
NCC pixel-level confirmation pass on the pHash-joint <= 6 flagged pairs
from ISIC 2019. Reads results/isic2019_joint_le6.json, loads each
flagged pair, resizes to 256x256 grayscale, and computes
normalized cross-correlation. Outputs results/isic2019_ncc.json.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def load_gray_256(path: Path) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("L").resize((256, 256), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32)
        return a
    except Exception as e:
        return None


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-9:
        return 0.0
    return float((a * b).sum() / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", default="results/isic2019_joint_le6.json")
    ap.add_argument("--images", default="images/ISIC_2019_Training_Input")
    ap.add_argument("--out", default="results/isic2019_ncc.json")
    args = ap.parse_args()

    with open(args.joint) as f:
        joint = json.load(f)

    images_dir = Path(args.images)

    # Collect flagged pairs from cross-source + intra-source examples.
    # The audit JSON only stores up to 20 examples per pair-bucket, so we
    # rerun on the FULL flagged list. To get the full list, we re-run the
    # vectorized check here on the per-image hash CSV.
    import csv
    rows = []
    with open("results/isic2019_phash_annotated.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    sources = sorted(by_source.keys())

    def hex_to_u64(h):
        return np.uint64(int(h, 16))

    def popcount64(x):
        x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + (
            (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
        )
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

    def pairs_le6(rows_a, rows_b, threshold=6):
        p_a = np.array([hex_to_u64(r["phash"]) for r in rows_a], dtype=np.uint64)
        d_a = np.array([hex_to_u64(r["dhash"]) for r in rows_a], dtype=np.uint64)
        p_b = np.array([hex_to_u64(r["phash"]) for r in rows_b], dtype=np.uint64)
        d_b = np.array([hex_to_u64(r["dhash"]) for r in rows_b], dtype=np.uint64)
        ids_a = [r["image_id"] for r in rows_a]
        ids_b = [r["image_id"] for r in rows_b]
        out = []
        chunk = 512
        for s in range(0, len(rows_a), chunk):
            e = min(s + chunk, len(rows_a))
            ph_xor = p_a[s:e, None] ^ p_b[None, :]
            dh_xor = d_a[s:e, None] ^ d_b[None, :]
            joint = popcount64(ph_xor) + popcount64(dh_xor)
            ai, bi = np.where(joint <= threshold)
            for i_local, j in zip(ai, bi):
                i = s + int(i_local)
                if ids_a[i] == ids_b[int(j)]:
                    continue
                out.append((ids_a[i], ids_b[int(j)], int(joint[i_local, j])))
        return out

    # Build the full flagged-pair list
    print("Rebuilding full flagged-pair list...", file=sys.stderr)
    cross_pairs = {}
    intra_pairs = {}
    for i, sa in enumerate(sources):
        for sb in sources[i + 1:]:
            cross_pairs[f"{sa}__{sb}"] = pairs_le6(by_source[sa], by_source[sb])
    for s in sources:
        n = len(by_source[s])
        if n > 5000:
            half = n // 2
            intra_pairs[s] = pairs_le6(by_source[s][:half], by_source[s][half:])
        else:
            intra_pairs[s] = pairs_le6(by_source[s], by_source[s])

    # Cache loaded images
    cache: dict[str, np.ndarray] = {}

    def get_image(image_id: str):
        if image_id in cache:
            return cache[image_id]
        path = images_dir / f"{image_id}.jpg"
        arr = load_gray_256(path)
        cache[image_id] = arr
        return arr

    def ncc_pairs(pair_list, label):
        scores = []
        skipped = 0
        for a_id, b_id, joint_d in pair_list:
            ai = get_image(a_id)
            bi = get_image(b_id)
            if ai is None or bi is None:
                skipped += 1
                continue
            scores.append((a_id, b_id, joint_d, ncc(ai, bi)))
        if scores:
            ncc_vals = np.array([s[3] for s in scores])
            stats = {
                "n_pairs": len(scores),
                "skipped": skipped,
                "ncc_mean": float(ncc_vals.mean()),
                "ncc_max": float(ncc_vals.max()),
                "ncc_min": float(ncc_vals.min()),
                "n_ge_0_99": int((ncc_vals >= 0.99).sum()),
                "frac_ge_0_99": float((ncc_vals >= 0.99).mean()),
                "n_ge_0_95": int((ncc_vals >= 0.95).sum()),
                "frac_ge_0_95": float((ncc_vals >= 0.95).mean()),
                "examples": [
                    {"a": a, "b": b, "joint": j, "ncc": float(n)}
                    for a, b, j, n in sorted(scores, key=lambda x: -x[3])[:10]
                ],
            }
        else:
            stats = {"n_pairs": 0, "skipped": skipped}
        print(f"  {label}: {stats.get('n_pairs', 0)} pairs, "
              f"NCC>=0.99 = {stats.get('n_ge_0_99', 0)} "
              f"(frac {stats.get('frac_ge_0_99', 0):.4f})",
              file=sys.stderr)
        return stats

    print("Computing NCC on cross-source flagged pairs...", file=sys.stderr)
    cross_ncc = {k: ncc_pairs(v, f"cross {k}") for k, v in cross_pairs.items()}
    print("Computing NCC on intra-source flagged pairs...", file=sys.stderr)
    intra_ncc = {k: ncc_pairs(v, f"intra {k}") for k, v in intra_pairs.items()}

    summary = {
        "benchmark": "ISIC 2019 (training)",
        "method": "NCC on 256x256 grayscale, applied to pHash+dHash joint<=6 flagged pairs",
        "cross_source_ncc": cross_ncc,
        "intra_source_ncc": intra_ncc,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
