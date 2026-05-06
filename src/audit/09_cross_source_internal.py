"""Internal train→val pHash audit for CV2024 non-KVASIR sources (SEE-AI, KID, AIIMS).

Addresses r2 §5.4 missing analysis: is internal train→val leakage a
KVASIR-only phenomenon, or also present in SEE-AI / KID / AIIMS?

Same methodology as cv2024_internal_leak.py but per-source.
"""
import os
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("CAPSULE_ROOT", "."))
OUT = ROOT / "results"

_POP16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)


def popcount_u64_2d(arr):
    f = arr.view(np.uint16).reshape(*arr.shape, 4)
    return _POP16[f].astype(np.int16).sum(axis=-1)


def audit_source(cv, src):
    train = [x for x in cv if x.get("cv_split") == "training" and x.get("cv_dataset") == src and x.get("phash")]
    val = [x for x in cv if x.get("cv_split") == "validation" and x.get("cv_dataset") == src and x.get("phash")]
    if not train or not val:
        return {"source": src, "n_train": len(train), "n_val": len(val), "no_data": True}

    tr_ph = np.array([int(x["phash"], 16) for x in train], dtype=np.uint64)
    va_ph = np.array([int(x["phash"], 16) for x in val], dtype=np.uint64)
    tr_dh = np.array([int(x["dhash"], 16) for x in train], dtype=np.uint64)
    va_dh = np.array([int(x["dhash"], 16) for x in val], dtype=np.uint64)

    # Filename overlap
    fn_overlap = len(set(x["filename"] for x in train) & set(x["filename"] for x in val))

    # min pHash val -> train
    chunk = 256
    min_ph = np.full(len(va_ph), 64, dtype=np.int16)
    min_dh = np.full(len(va_ph), 64, dtype=np.int16)
    joint_flag = np.zeros(len(va_ph), dtype=bool)
    for i in range(0, len(va_ph), chunk):
        v_ph = va_ph[i:i+chunk]
        v_dh = va_dh[i:i+chunk]
        xor_ph = v_ph[:, None] ^ tr_ph[None, :]
        pc_ph = popcount_u64_2d(xor_ph)
        xor_dh = v_dh[:, None] ^ tr_dh[None, :]
        pc_dh = popcount_u64_2d(xor_dh)
        min_ph[i:i+chunk] = pc_ph.min(axis=1)
        min_dh[i:i+chunk] = pc_dh.min(axis=1)
        # joint: some single train file with both ≤ 6
        joint = (pc_ph <= 6) & (pc_dh <= 6)
        joint_flag[i:i+chunk] = joint.any(axis=1)

    return {
        "source": src,
        "n_train": len(train),
        "n_val": len(val),
        "filename_overlap": fn_overlap,
        "phash_hamming_median": float(np.median(min_ph)),
        "dhash_hamming_median": float(np.median(min_dh)),
        "independent_phash_le6": int((min_ph <= 6).sum()),
        "independent_phash_le6_frac": float((min_ph <= 6).mean()),
        "independent_phash_le0": int((min_ph == 0).sum()),
        "independent_phash_le0_frac": float((min_ph == 0).mean()),
        "joint_ph_and_dh_le6": int(joint_flag.sum()),
        "joint_ph_and_dh_le6_frac": float(joint_flag.mean()),
    }


def main():
    with open(OUT / "hashes_cv2024.json") as f:
        cv = json.load(f)

    results = {}
    for src in ("KVASIR", "SEE-AI", "KID", "AIIMS"):
        print(f"\n=== {src} ===")
        r = audit_source(cv, src)
        results[src] = r
        if r.get("no_data"):
            print(f"  n_train={r['n_train']} n_val={r['n_val']} — too sparse")
            continue
        print(f"  n_train={r['n_train']} n_val={r['n_val']}")
        print(f"  filename overlap: {r['filename_overlap']}")
        print(f"  median pHash Hamming: {r['phash_hamming_median']:.0f}")
        print(f"  independent pHash==0: {r['independent_phash_le0']}/{r['n_val']} ({100*r['independent_phash_le0_frac']:.2f}%)")
        print(f"  independent pHash<=6: {r['independent_phash_le6']}/{r['n_val']} ({100*r['independent_phash_le6_frac']:.2f}%)")
        print(f"  joint ph AND dh <=6 (same frame): {r['joint_ph_and_dh_le6']}/{r['n_val']} ({100*r['joint_ph_and_dh_le6_frac']:.2f}%)")

    with open(OUT / "cv2024_internal_cross_source.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT / 'cv2024_internal_cross_source.json'}")


if __name__ == "__main__":
    main()
