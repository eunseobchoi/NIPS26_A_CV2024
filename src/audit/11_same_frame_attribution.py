"""Same-frame joint flagging: recompute with stricter criterion.

Addresses r2 Q1: current audit uses independent min-pHash AND min-dHash,
which means the two hashes may flag different Kvasir frames. The stricter
"same-frame joint" criterion requires that at least one SINGLE Kvasir
frame satisfy both pHash<=6 AND dHash<=6.

Compute both counts and report the difference.
"""
import os
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("CAPSULE_ROOT", "."))
OUT = ROOT / "results"

_POP16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)


def popcount_u64(arr):
    f = arr.view(np.uint16).reshape(*arr.shape, 4)
    return _POP16[f].astype(np.int16).sum(axis=-1)


def hamming_mat(src_int, tgt_int):
    """Returns (n_src, n_tgt) int16 Hamming."""
    xor = src_int[:, None] ^ tgt_int[None, :]
    return popcount_u64(xor)


def main():
    with open(OUT / "hashes_kvasir.json") as f:
        kv = json.load(f)
    with open(OUT / "hashes_cv2024.json") as f:
        cv = json.load(f)

    # Filter to valid hashes only
    kv = [x for x in kv if x.get("phash") and x.get("dhash")]
    print(f"Kvasir: {len(kv)}")
    kv_ph = np.array([int(x["phash"], 16) for x in kv], dtype=np.uint64)
    kv_dh = np.array([int(x["dhash"], 16) for x in kv], dtype=np.uint64)

    report = {"threshold": 6, "sources": {}}
    for src in ("KVASIR", "SEE-AI", "KID", "AIIMS"):
        sub = [x for x in cv if x.get("cv_dataset") == src and x.get("phash") and x.get("dhash")]
        if not sub:
            continue
        src_ph = np.array([int(x["phash"], 16) for x in sub], dtype=np.uint64)
        src_dh = np.array([int(x["dhash"], 16) for x in sub], dtype=np.uint64)
        print(f"\n=== {src} n={len(sub)} ===")

        # Independent min on each hash
        n_src = len(sub)
        chunk = 128
        independent_flagged = 0
        samesrc_flagged = 0
        import time
        t0 = time.perf_counter()
        for i in range(0, n_src, chunk):
            ph_d = hamming_mat(src_ph[i:i+chunk], kv_ph)  # (c, N_kv)
            dh_d = hamming_mat(src_dh[i:i+chunk], kv_dh)  # (c, N_kv)
            # Independent: min_ph <= 6 AND min_dh <= 6
            indep = (ph_d.min(axis=1) <= 6) & (dh_d.min(axis=1) <= 6)
            # Same-frame: exists j such that ph_d[i,j] <= 6 AND dh_d[i,j] <= 6
            both = (ph_d <= 6) & (dh_d <= 6)  # (c, N_kv) bool
            same = both.any(axis=1)
            independent_flagged += int(indep.sum())
            samesrc_flagged += int(same.sum())
            if (i // chunk) % 50 == 0 and i > 0:
                dt = time.perf_counter() - t0
                eta = (n_src - i) * dt / i
                print(f"  {i}/{n_src}  elapsed {dt:.0f}s  ETA {eta:.0f}s", flush=True)
        dt = time.perf_counter() - t0
        print(f"  Total {dt:.0f}s")
        print(f"  Independent (min pHash ≤6 AND min dHash ≤6): {independent_flagged}/{n_src} = {100*independent_flagged/n_src:.2f}%")
        print(f"  Same-frame (some Kvasir frame has pHash ≤6 AND dHash ≤6): {samesrc_flagged}/{n_src} = {100*samesrc_flagged/n_src:.2f}%")
        report["sources"][src] = {
            "n": n_src,
            "independent_flagged": independent_flagged,
            "independent_frac": independent_flagged / n_src,
            "same_frame_flagged": samesrc_flagged,
            "same_frame_frac": samesrc_flagged / n_src,
        }

    with open(OUT / "same_frame_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {OUT / 'same_frame_audit.json'}")


if __name__ == "__main__":
    main()
