"""Video-prefix CV2024 internal train->val attribution (r2 Q6).

For each CV2024-KVASIR validation file that has a pHash collision with
a training file, determine: does the matching Kvasir video prefix
appear in BOTH train and val? This would show the 11.9% is caused by
same-video frames distributed across CV2024 train/val boundary
(rather than random uncorrelated frames or independent collisions).
"""
import os
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CAPSULE_ROOT", "."))
OUT = ROOT / "results"


_POP16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)

def popcount(arr):
    f = arr.view(np.uint16).reshape(*arr.shape, 4)
    return _POP16[f].astype(np.int16).sum(axis=-1)


def main():
    anno = pd.read_csv(OUT / "cv2024_KVASIR_phash_annotated.csv")
    # Each row has "nearest_kvasir_file" (Kvasir filename matched by pHash)
    # Extract Kvasir video ID from filename prefix
    anno["kvasir_video"] = anno["nearest_kvasir_file"].str.split("_").str[0]

    train = anno[anno["cv_split"] == "training"].copy()
    val = anno[anno["cv_split"] == "validation"].copy()
    print(f"CV2024-KVASIR: train={len(train)} val={len(val)}")

    # Distinct Kvasir videos per split
    tr_vids = set(train["kvasir_video"].unique())
    va_vids = set(val["kvasir_video"].unique())
    shared = tr_vids & va_vids
    print(f"Distinct Kvasir videos in CV2024-train: {len(tr_vids)}")
    print(f"Distinct Kvasir videos in CV2024-val:   {len(va_vids)}")
    print(f"Shared between train+val:                {len(shared)}")
    print(f"  (val-only: {len(va_vids - tr_vids)}, train-only: {len(tr_vids - va_vids)})")

    # For each val file, is its Kvasir video also in training?
    val["kvasir_vid_in_train"] = val["kvasir_video"].isin(tr_vids)
    print(f"\nVal files whose Kvasir video ALSO appears in CV2024 train: "
          f"{val['kvasir_vid_in_train'].sum()}/{len(val)} ({100*val['kvasir_vid_in_train'].mean():.1f}%)")

    # Now load the pHash-exact internal pairs and attribute to Kvasir videos
    import json as _j
    with open(OUT / "hashes_cv2024.json") as f:
        cv = _j.load(f)
    train_h = [x for x in cv if x.get("cv_split")=="training" and x.get("cv_dataset")=="KVASIR" and x.get("phash")]
    val_h = [x for x in cv if x.get("cv_split")=="validation" and x.get("cv_dataset")=="KVASIR" and x.get("phash")]

    tr_ph = np.array([int(x["phash"], 16) for x in train_h], dtype=np.uint64)
    va_ph = np.array([int(x["phash"], 16) for x in val_h], dtype=np.uint64)
    # Build {cv_filename: kvasir_video}
    fn_to_kv_vid = dict(zip(anno["filename"], anno["kvasir_video"]))

    # For each val file with pHash collision to some train file, check video correspondence
    chunk = 256
    same_video_count = 0
    different_video_count = 0
    phash_zero_count = 0
    for i in range(0, len(va_ph), chunk):
        vh = va_ph[i:i+chunk]
        xor = vh[:, None] ^ tr_ph[None, :]
        pc = popcount(xor)
        min_d = pc.min(axis=1)
        min_idx = pc.argmin(axis=1)
        for k in range(len(vh)):
            if min_d[k] == 0:
                phash_zero_count += 1
                v_fn = val_h[i+k]["filename"]
                t_fn = train_h[int(min_idx[k])]["filename"]
                v_vid = fn_to_kv_vid.get(v_fn, "?")
                t_vid = fn_to_kv_vid.get(t_fn, "?")
                if v_vid == t_vid:
                    same_video_count += 1
                else:
                    different_video_count += 1

    print(f"\n=== pHash-exact internal pairs (val→train, n={phash_zero_count}) ===")
    print(f"  Same Kvasir video:       {same_video_count} ({100*same_video_count/max(phash_zero_count,1):.1f}%)")
    print(f"  Different Kvasir video:  {different_video_count} ({100*different_video_count/max(phash_zero_count,1):.1f}%)")

    # Save
    summary = {
        "cv2024_train_kvasir_videos": len(tr_vids),
        "cv2024_val_kvasir_videos": len(va_vids),
        "shared_videos": len(shared),
        "val_only_videos": len(va_vids - tr_vids),
        "train_only_videos": len(tr_vids - va_vids),
        "val_files_with_kv_vid_also_in_train": int(val["kvasir_vid_in_train"].sum()),
        "val_files_frac_kv_vid_also_in_train": float(val["kvasir_vid_in_train"].mean()),
        "phash_exact_pairs": phash_zero_count,
        "phash_exact_same_kvasir_video": same_video_count,
        "phash_exact_different_kvasir_video": different_video_count,
    }
    with open(OUT / "cv2024_internal_per_patient.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT / 'cv2024_internal_per_patient.json'}")


if __name__ == "__main__":
    main()
