"""Generate Comp-A and Comp-B training CSVs for the Ulcer-control arms.

Both arms start from the seed-0 composition-matched CSV:

Comp-A:
    Remove the 66 KVASIR Ulcer rows and append the 66 AIIMS Ulcer rows.
    Total size and per-class counts stay fixed; Ulcer source changes from
    KVASIR to AIIMS.

Comp-B:
    Keep the 66 KVASIR Ulcer rows, drop 66 KVASIR Normal rows with seed 42,
    and append the 66 AIIMS Ulcer rows. Total size stays fixed while Ulcer
    doubles from 66 to 132.
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
    COMP_MATCHED_KVASIR_NORMAL_TOTAL,
    COMP_NORMAL_DROP_SEED,
    COMP_SOURCE_ULCER_COUNT,
    LE6_TRAIN_TOTAL,
)

EXPECTED_NAMES = {
    "A": "cv2024_training_compA_ulcer_aligned_s0.csv",
    "B": "cv2024_training_compB_ulcer_balanced_s0.csv",
}


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_aiims_ulcer(pool_path: Path, columns: list[str]) -> pd.DataFrame:
    pool = read_table(pool_path)
    aiims_ulcer = pool[(pool["Dataset"] == "AIIMS") & (pool["Ulcer"] == 1)]
    if len(aiims_ulcer) != COMP_SOURCE_ULCER_COUNT:
        raise RuntimeError(f"Expected {COMP_SOURCE_ULCER_COUNT} AIIMS Ulcer rows, got {len(aiims_ulcer)}")
    return aiims_ulcer[columns]


def build_comp_a(compmatched_csv: Path, train_xlsx: Path) -> pd.DataFrame:
    cm = pd.read_csv(compmatched_csv)
    kv_ulcer = (cm["Dataset"] == "KVASIR") & (cm["Ulcer"] == 1)
    n_kv_ulcer = int(kv_ulcer.sum())
    if n_kv_ulcer != COMP_SOURCE_ULCER_COUNT:
        raise RuntimeError(f"Expected {COMP_SOURCE_ULCER_COUNT} KVASIR Ulcer rows, got {n_kv_ulcer}")
    aiims_ulcer = load_aiims_ulcer(train_xlsx, cm.columns.tolist())
    out = pd.concat([cm[~kv_ulcer], aiims_ulcer], ignore_index=True)
    if len(out) != LE6_TRAIN_TOTAL:
        raise RuntimeError(f"Comp-A expected {LE6_TRAIN_TOTAL} rows, got {len(out)}")
    return out


def build_comp_b(compmatched_csv: Path, train_xlsx: Path) -> pd.DataFrame:
    cm = pd.read_csv(compmatched_csv)
    kv_normal = cm[(cm["Dataset"] == "KVASIR") & (cm["Normal"] == 1)]
    if len(kv_normal) != COMP_MATCHED_KVASIR_NORMAL_TOTAL:
        raise RuntimeError(f"Expected {COMP_MATCHED_KVASIR_NORMAL_TOTAL} KVASIR Normal rows, got {len(kv_normal)}")
    dropped = kv_normal.sample(n=COMP_SOURCE_ULCER_COUNT, random_state=COMP_NORMAL_DROP_SEED)
    aiims_ulcer = load_aiims_ulcer(train_xlsx, cm.columns.tolist())
    out = pd.concat([cm.drop(dropped.index), aiims_ulcer], ignore_index=True)
    if len(out) != LE6_TRAIN_TOTAL:
        raise RuntimeError(f"Comp-B expected {LE6_TRAIN_TOTAL} rows, got {len(out)}")
    if int(out["Ulcer"].sum()) != COMP_DOUBLED_ULCER_COUNT:
        raise RuntimeError(f"Comp-B expected {COMP_DOUBLED_ULCER_COUNT} Ulcer rows, got {int(out['Ulcer'].sum())}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_xlsx", required=True, type=Path)
    ap.add_argument("--compmatched_csv", required=True, type=Path)
    ap.add_argument("--out_dir", default=Path("artifacts/csvs"), type=Path)
    ap.add_argument("--arm", choices=("A", "B", "both"), default="both")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "A": build_comp_a,
        "B": build_comp_b,
    }
    arms = ("A", "B") if args.arm == "both" else (args.arm,)
    for arm in arms:
        df = builders[arm](args.compmatched_csv, args.train_xlsx)
        out = args.out_dir / EXPECTED_NAMES[arm]
        df.to_csv(out, index=False)
        print(
            f"Comp-{arm}: wrote {out} rows={len(df)} "
            f"Ulcer={int(df['Ulcer'].sum())} Normal={int(df['Normal'].sum())} "
            f"md5={file_md5(out)}"
        )


if __name__ == "__main__":
    main()
