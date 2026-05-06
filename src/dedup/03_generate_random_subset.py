"""Generate random-subset CSV of CV2024 training, size N (default: le6 size 10,596).

Purpose: for the "size-effect confound" counterfactual (r2 §4.3).
Training on this sample gives the baseline Δ attributable to
n-decrease alone; Δ(le6) − Δ(random-subset) isolates leakage.
"""
import os
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
CV = Path(os.environ.get("CV2024_ROOT", ROOT / "data/cv2024/Dataset"))
if (CV / "Dataset").is_dir():
    CV = CV / "Dataset"
OUT = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10596)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", default="cv2024_training_random10596_s42.csv")
    args = ap.parse_args()

    df = pd.read_excel(CV / "training" / "training_data.xlsx")
    print(f"Full CV2024 training: {len(df)} rows")
    sub = df.sample(n=args.n, random_state=args.seed)
    print(f"Random subset (seed={args.seed}): {len(sub)} rows")
    src_counts = sub["Dataset"].value_counts().to_dict()
    print(f"Source distribution: {src_counts}")
    sub.to_csv(OUT / args.out_csv, index=False)
    print(f"Saved {OUT / args.out_csv}")


if __name__ == "__main__":
    main()
