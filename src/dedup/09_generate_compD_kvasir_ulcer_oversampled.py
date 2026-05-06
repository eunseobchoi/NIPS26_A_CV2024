"""Generate Comp-D for the Ulcer source/count factorial control.

Comp-D completes the 2 x 2 Ulcer-source/count grid:

    KVASIR, 66  rows: composition-matched contaminated control
    AIIMS,  66  rows: Comp-A
    AIIMS,  132 rows: Comp-C
    KVASIR, 132 rows: Comp-D (this script)

The script starts from the composition-matched contaminated CSV, drops the
same 66 KVASIR Normal rows used by Comp-B/Comp-C to preserve total size, and
adds 66 additional KVASIR Ulcer rows sampled from the full CV2024 training
pool excluding the 66 KVASIR Ulcer rows already present in comp-matched.
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
    COMP_EXTRA_KVASIR_ULCER_SEED,
    COMP_NORMAL_DROP_SEED,
    COMP_SOURCE_ULCER_COUNT,
    LE6_TRAIN_TOTAL,
)

OUT_NAME = "cv2024_training_compD_kvasir_ulcer_oversampled_s0.csv"


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_comp_d(compmatched_csv: Path, train_xlsx: Path) -> pd.DataFrame:
    compmatched = pd.read_csv(compmatched_csv)
    full = pd.read_excel(train_xlsx) if train_xlsx.suffix in {".xlsx", ".xls"} else pd.read_csv(train_xlsx)

    base_kv_ulcer = compmatched[(compmatched["Dataset"] == "KVASIR") & (compmatched["Ulcer"] == 1)]
    if len(base_kv_ulcer) != COMP_SOURCE_ULCER_COUNT:
        raise RuntimeError(
            f"Expected {COMP_SOURCE_ULCER_COUNT} KVASIR Ulcer rows in comp-matched, got {len(base_kv_ulcer)}"
        )

    kv_normal = compmatched[(compmatched["Dataset"] == "KVASIR") & (compmatched["Normal"] == 1)]
    if len(kv_normal) < COMP_SOURCE_ULCER_COUNT:
        raise RuntimeError(f"Not enough KVASIR Normal rows to drop: {len(kv_normal)}")
    dropped = kv_normal.sample(n=COMP_SOURCE_ULCER_COUNT, random_state=COMP_NORMAL_DROP_SEED)

    existing_paths = set(base_kv_ulcer["image_path"].astype(str))
    kv_ulcer_pool = full[(full["Dataset"] == "KVASIR") & (full["Ulcer"] == 1)].copy()
    kv_ulcer_pool = kv_ulcer_pool[~kv_ulcer_pool["image_path"].astype(str).isin(existing_paths)]
    if len(kv_ulcer_pool) < COMP_SOURCE_ULCER_COUNT:
        raise RuntimeError(
            f"Need {COMP_SOURCE_ULCER_COUNT} extra KVASIR Ulcer rows, got {len(kv_ulcer_pool)} candidates"
        )
    extra_kv_ulcer = kv_ulcer_pool.sample(
        n=COMP_SOURCE_ULCER_COUNT,
        random_state=COMP_EXTRA_KVASIR_ULCER_SEED,
        replace=False,
    )[compmatched.columns.tolist()]

    out = pd.concat([compmatched.drop(dropped.index), extra_kv_ulcer], ignore_index=True)
    if len(out) != LE6_TRAIN_TOTAL:
        raise RuntimeError(f"Comp-D expected {LE6_TRAIN_TOTAL} rows, got {len(out)}")
    if int(out["Ulcer"].sum()) != COMP_DOUBLED_ULCER_COUNT:
        raise RuntimeError(
            f"Comp-D expected {COMP_DOUBLED_ULCER_COUNT} Ulcer rows, got {int(out['Ulcer'].sum())}"
        )
    if int(((out["Dataset"] == "KVASIR") & (out["Ulcer"] == 1)).sum()) != COMP_DOUBLED_ULCER_COUNT:
        raise RuntimeError(f"Comp-D should contain {COMP_DOUBLED_ULCER_COUNT} KVASIR Ulcer rows")
    if int(((out["Dataset"] == "AIIMS") & (out["Ulcer"] == 1)).sum()) != 0:
        raise RuntimeError("Comp-D should contain zero AIIMS Ulcer rows")
    if int(((out["Dataset"] == "KVASIR") & (out["Normal"] == 1)).sum()) != COMP_DISPLACED_KVASIR_NORMAL_TOTAL:
        raise RuntimeError(
            f"Comp-D should contain {COMP_DISPLACED_KVASIR_NORMAL_TOTAL} KVASIR Normal rows after the fixed displacement"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compmatched-csv", type=Path, default=Path("artifacts/csvs/cv2024_training_compmatched_strict_s0.csv"))
    ap.add_argument("--train-xlsx", type=Path, default=Path("data/cv2024/Dataset/training/training_data.xlsx"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/csvs"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / OUT_NAME
    df = build_comp_d(args.compmatched_csv, args.train_xlsx)
    df.to_csv(out_path, index=False)
    print(
        f"Comp-D: wrote {out_path} rows={len(df)} "
        f"Normal={int(df['Normal'].sum())} Ulcer={int(df['Ulcer'].sum())} "
        f"KVASIR_Ulcer={int(((df['Dataset'] == 'KVASIR') & (df['Ulcer'] == 1)).sum())} "
        f"AIIMS_Ulcer={int(((df['Dataset'] == 'AIIMS') & (df['Ulcer'] == 1)).sum())} "
        f"KVASIR_Normal={int(((df['Dataset'] == 'KVASIR') & (df['Normal'] == 1)).sum())} "
        f"md5={file_md5(out_path)}"
    )


if __name__ == "__main__":
    main()
