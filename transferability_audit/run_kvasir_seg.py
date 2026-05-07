#!/usr/bin/env python3
"""
Kvasir-SEG cross-bench against CV2024 (KVASIR + non-KVASIR slices).
Pre-registered threshold: joint <= 6.

Same lab (Simula) as Kvasir-Capsule but Kvasir-SEG is colonoscopy with
polyp masks (different procedure from VCE small bowel). We expect
near-zero overlap as another procedure-disjoint negative control.
"""
import csv
import json
import sys
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


def cross_le6(a_p, a_d, b_p, b_d, threshold=6, chunk=512):
    pairs = []
    for s in range(0, len(a_p), chunk):
        e = min(s + chunk, len(a_p))
        ph_xor = a_p[s:e, None] ^ b_p[None, :]
        dh_xor = a_d[s:e, None] ^ b_d[None, :]
        joint = popcount64(ph_xor) + popcount64(dh_xor)
        ai, bi = np.where(joint <= threshold)
        for i_local, j in zip(ai, bi):
            pairs.append((s + int(i_local), int(j), int(joint[i_local, j])))
    return pairs


def main():
    images_dir = Path("kvasir_seg/Kvasir-SEG/images")
    image_paths = sorted(images_dir.glob("*.jpg"))
    print(f"Hashing {len(image_paths)} Kvasir-SEG images...", file=sys.stderr)
    rows = []
    for idx, p in enumerate(image_paths, 1):
        try:
            img = Image.open(p)
            ph = imagehash.phash(img, hash_size=8)
            dh = imagehash.dhash(img, hash_size=8)
            rows.append({"image_id": p.name, "phash": str(ph), "dhash": str(dh)})
        except Exception:
            pass
        if idx % 500 == 0:
            print(f"  {idx}/{len(image_paths)}", file=sys.stderr)
    print(f"  hashed {len(rows)}", file=sys.stderr)
    Path("results/kvasir_seg_phash_annotated.csv").write_text(
        "image_id,phash,dhash\n" +
        "\n".join(f"{r['image_id']},{r['phash']},{r['dhash']}" for r in rows)
    )

    seg_p = np.array([hex_to_u64(r["phash"]) for r in rows], dtype=np.uint64)
    seg_d = np.array([hex_to_u64(r["dhash"]) for r in rows], dtype=np.uint64)

    sources = {
        "KVASIR":  "/home/user/main/capsule_tta/results/cv2024_KVASIR_phash_annotated.csv",
        "SEE-AI":  "/home/user/main/capsule_tta/results/cv2024_SEE-AI_phash_annotated.csv",
        "KID":     "/home/user/main/capsule_tta/results/cv2024_KID_phash_annotated.csv",
        "AIIMS":   "/home/user/main/capsule_tta/results/cv2024_AIIMS_phash_annotated.csv",
    }
    summary = {"kvasir_seg_n": len(rows), "threshold_joint": 6, "results": {}}
    for src, p in sources.items():
        cv = []
        with open(p) as f:
            for r in csv.DictReader(f):
                cv.append(r)
        cv_p = np.array([hex_to_u64(r["phash"]) for r in cv], dtype=np.uint64)
        cv_d = np.array([hex_to_u64(r["dhash"]) for r in cv], dtype=np.uint64)
        cv_phash_set = set(r["phash"] for r in cv)
        seg_exact = sum(1 for r in rows if r["phash"] in cv_phash_set)
        pairs = cross_le6(seg_p, seg_d, cv_p, cv_d)
        print(f"  Kvasir-SEG x CV2024-{src} ({len(cv)}): "
              f"joint<=6 pairs={len(pairs)}, pHash-exact={seg_exact}",
              file=sys.stderr)
        summary["results"][src] = {
            "cv_source_n": len(cv),
            "joint_le6_pairs": len(pairs),
            "phash_exact_seg_rows": seg_exact,
            "examples": [
                {"seg": rows[a]["image_id"], f"cv_{src}": cv[b]["filename"], "joint": j}
                for a, b, j in pairs[:10]
            ],
        }

    Path("results/kvasir_seg_audit.json").write_text(json.dumps(summary, indent=2))
    print("Wrote results/kvasir_seg_audit.json", file=sys.stderr)


if __name__ == "__main__":
    main()
