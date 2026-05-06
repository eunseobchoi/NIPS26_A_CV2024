"""Generate duplicate-KVASIR Comp-D for a symmetric Ulcer count control.

This companion to ``generate_compC_aiims_ulcer_oversampled.py`` starts from
the composition-matched contaminated arm, drops the fixed 66 KVASIR Normal
rows used by Comp-B/Comp-C, and appends a second copy of the same 66 KVASIR
Ulcer rows already present in the matched arm. It therefore matches Comp-C's
intentional duplicate-66 construction, but with KVASIR rather than AIIMS Ulcer.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constants import (  # noqa: E402
    COMP_DISPLACED_KVASIR_NORMAL_TOTAL,
    COMP_DOUBLED_ULCER_COUNT,
    COMP_NORMAL_DROP_SEED,
    COMP_SOURCE_ULCER_COUNT,
    LE6_TRAIN_TOTAL,
)

OUT_NAME = "cv2024_training_compD_kvasir_ulcer_duplicate_s0.csv"


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_comp_d_duplicate(compmatched_csv: Path) -> pd.DataFrame:
    compmatched = pd.read_csv(compmatched_csv)
    kv_ulcer = compmatched[(compmatched["Dataset"] == "KVASIR") & (compmatched["Ulcer"] == 1)]
    if len(kv_ulcer) != COMP_SOURCE_ULCER_COUNT:
        raise RuntimeError(f"Expected {COMP_SOURCE_ULCER_COUNT} KVASIR Ulcer rows, got {len(kv_ulcer)}")
    if int(((compmatched["Dataset"] == "AIIMS") & (compmatched["Ulcer"] == 1)).sum()) != 0:
        raise RuntimeError("Comp-matched input should contain zero AIIMS Ulcer rows")

    kv_normal = compmatched[(compmatched["Dataset"] == "KVASIR") & (compmatched["Normal"] == 1)]
    if len(kv_normal) < COMP_SOURCE_ULCER_COUNT:
        raise RuntimeError(f"Not enough KVASIR Normal rows to drop: {len(kv_normal)}")
    dropped = kv_normal.sample(n=COMP_SOURCE_ULCER_COUNT, random_state=COMP_NORMAL_DROP_SEED)

    out = pd.concat([compmatched.drop(dropped.index), kv_ulcer], ignore_index=True)
    if len(out) != LE6_TRAIN_TOTAL:
        raise RuntimeError(f"Comp-D duplicate expected {LE6_TRAIN_TOTAL} rows, got {len(out)}")
    if int(out["Ulcer"].sum()) != COMP_DOUBLED_ULCER_COUNT:
        raise RuntimeError(f"Expected {COMP_DOUBLED_ULCER_COUNT} Ulcer rows, got {int(out['Ulcer'].sum())}")
    if int(((out["Dataset"] == "KVASIR") & (out["Ulcer"] == 1)).sum()) != COMP_DOUBLED_ULCER_COUNT:
        raise RuntimeError(f"Comp-D duplicate should contain {COMP_DOUBLED_ULCER_COUNT} KVASIR Ulcer rows")
    if int(((out["Dataset"] == "AIIMS") & (out["Ulcer"] == 1)).sum()) != 0:
        raise RuntimeError("Comp-D duplicate should contain zero AIIMS Ulcer rows")
    if int(((out["Dataset"] == "KVASIR") & (out["Normal"] == 1)).sum()) != COMP_DISPLACED_KVASIR_NORMAL_TOTAL:
        raise RuntimeError(
            f"Comp-D duplicate should contain {COMP_DISPLACED_KVASIR_NORMAL_TOTAL} KVASIR Normal rows after fixed displacement"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compmatched-csv", type=Path, default=Path("artifacts/csvs/cv2024_training_compmatched_strict_s0.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/csvs"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / OUT_NAME
    df = build_comp_d_duplicate(args.compmatched_csv)
    df.to_csv(out_path, index=False)
    print(
        f"Comp-D duplicate: wrote {out_path} rows={len(df)} "
        f"Normal={int(df['Normal'].sum())} Ulcer={int(df['Ulcer'].sum())} "
        f"KVASIR_Ulcer={int(((df['Dataset'] == 'KVASIR') & (df['Ulcer'] == 1)).sum())} "
        f"AIIMS_Ulcer={int(((df['Dataset'] == 'AIIMS') & (df['Ulcer'] == 1)).sum())} "
        f"KVASIR_Normal={int(((df['Dataset'] == 'KVASIR') & (df['Normal'] == 1)).sum())} "
        f"md5={file_md5(out_path)}"
    )


if __name__ == "__main__":
    main()
