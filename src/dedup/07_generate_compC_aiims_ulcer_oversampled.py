"""Generate Comp-C for the Ulcer source/count control.

Comp-C starts from Comp-A, which removes the 66 KVASIR Ulcer rows and
adds the 66 AIIMS Ulcer rows. It then drops the same number of KVASIR
Normal rows as Comp-B and appends a second copy of the 66 AIIMS Ulcer
rows. Total size matches the other matched arms, Ulcer count matches
Comp-B, but KVASIR Ulcer content remains absent.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constants import (  # noqa: E402
    COMP_DOUBLED_ULCER_COUNT,
    COMP_NORMAL_DROP_SEED,
    COMP_SOURCE_ULCER_COUNT,
    LE6_TRAIN_TOTAL,
)

OUT_NAME = "cv2024_training_compC_aiims_ulcer_oversampled_s0.csv"


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_comp_c(comp_a_csv: Path) -> pd.DataFrame:
    comp_a = pd.read_csv(comp_a_csv)
    aiims_ulcer = comp_a[(comp_a["Dataset"] == "AIIMS") & (comp_a["Ulcer"] == 1)]
    if len(aiims_ulcer) != COMP_SOURCE_ULCER_COUNT:
        raise RuntimeError(f"Expected {COMP_SOURCE_ULCER_COUNT} AIIMS Ulcer rows, got {len(aiims_ulcer)}")
    if int(((comp_a["Dataset"] == "KVASIR") & (comp_a["Ulcer"] == 1)).sum()) != 0:
        raise RuntimeError("Comp-A input must contain zero KVASIR Ulcer rows")

    kv_normal = comp_a[(comp_a["Dataset"] == "KVASIR") & (comp_a["Normal"] == 1)]
    if len(kv_normal) < COMP_SOURCE_ULCER_COUNT:
        raise RuntimeError(f"Not enough KVASIR Normal rows to drop: {len(kv_normal)}")
    dropped = kv_normal.sample(n=COMP_SOURCE_ULCER_COUNT, random_state=COMP_NORMAL_DROP_SEED)

    out = pd.concat([comp_a.drop(dropped.index), aiims_ulcer], ignore_index=True)
    if len(out) != LE6_TRAIN_TOTAL:
        raise RuntimeError(f"Comp-C expected {LE6_TRAIN_TOTAL} rows, got {len(out)}")
    if int(out["Ulcer"].sum()) != COMP_DOUBLED_ULCER_COUNT:
        raise RuntimeError(f"Comp-C expected {COMP_DOUBLED_ULCER_COUNT} Ulcer rows, got {int(out['Ulcer'].sum())}")
    if int(((out["Dataset"] == "KVASIR") & (out["Ulcer"] == 1)).sum()) != 0:
        raise RuntimeError("Comp-C should contain zero KVASIR Ulcer rows")
    if int(((out["Dataset"] == "AIIMS") & (out["Ulcer"] == 1)).sum()) != COMP_DOUBLED_ULCER_COUNT:
        raise RuntimeError(
            f"Comp-C should contain {COMP_DOUBLED_ULCER_COUNT} AIIMS Ulcer rows including the intentional duplicate pass"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp-a-csv", default=Path("artifacts/csvs/cv2024_training_compA_ulcer_aligned_s0.csv"), type=Path)
    ap.add_argument("--out-dir", default=Path("artifacts/csvs"), type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / OUT_NAME
    df = build_comp_c(args.comp_a_csv)
    df.to_csv(out, index=False)
    print(
        f"Comp-C: wrote {out} rows={len(df)} "
        f"Normal={int(df['Normal'].sum())} Ulcer={int(df['Ulcer'].sum())} "
        f"AIIMS_Ulcer={int(((df['Dataset'] == 'AIIMS') & (df['Ulcer'] == 1)).sum())} "
        f"KVASIR_Ulcer={int(((df['Dataset'] == 'KVASIR') & (df['Ulcer'] == 1)).sum())} "
        f"md5={file_md5(out)}"
    )


if __name__ == "__main__":
    main()
