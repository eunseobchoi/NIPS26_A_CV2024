#!/usr/bin/env python3
"""Generate KVASIR-only class-count-preserving label-shuffle CSVs.

This reproduces the existing stratified shuffle files by permuting the
one-hot label matrix among KVASIR rows with numpy.default_rng(seed), leaving
row order, non-KVASIR labels, and all image paths unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = [
    "Angioectasia",
    "Bleeding",
    "Erosion",
    "Erythema",
    "Foreign Body",
    "Lymphangiectasia",
    "Normal",
    "Polyp",
    "Ulcer",
    "Worms",
]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--prefix", default="cv2024_training_kvasir_strat_shuffle_s")
    ap.add_argument("--verify-md5", type=Path, default=None)
    args = ap.parse_args()

    df = read_table(args.input)
    missing = [c for c in ["image_path", "Dataset", *CLASSES] if c not in df.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")

    kvasir = df["Dataset"].astype(str).str.upper().eq("KVASIR").to_numpy()
    if not kvasir.any():
        raise SystemExit("no KVASIR rows found")

    labels = df.loc[kvasir, CLASSES].to_numpy(copy=True)
    class_counts = dict(zip(CLASSES, labels.sum(axis=0).astype(int).tolist()))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    made = {}
    for seed in args.seeds:
        out = args.out_dir / f"{args.prefix}{seed}.csv"
        shuffled = df.copy()
        rng = np.random.default_rng(seed)
        permuted = labels[rng.permutation(len(labels))]
        shuffled.loc[kvasir, CLASSES] = permuted
        shuffled.to_csv(out, index=False)

        changed = int((labels.argmax(axis=1) != permuted.argmax(axis=1)).sum())
        made[str(seed)] = {
            "path": str(out),
            "md5": md5(out),
            "kvasir_rows": int(kvasir.sum()),
            "changed_kvasir_labels": changed,
            "unchanged_kvasir_labels": int(kvasir.sum()) - changed,
            "class_counts": class_counts,
        }

    if args.verify_md5 is not None:
        expected = {}
        with args.verify_md5.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                seed, digest = line.split()
                expected[seed] = digest
        mismatches = {
            seed: (expected[seed], made[seed]["md5"])
            for seed in expected
            if seed in made and expected[seed] != made[seed]["md5"]
        }
        if mismatches:
            raise SystemExit(f"MD5 mismatch: {mismatches}")

    print(json.dumps(made, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
