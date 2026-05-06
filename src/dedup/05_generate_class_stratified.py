"""Generate class-stratified random subset CSV that matches le6's (class, source) marginals.

This control isolates sample-size reduction plus class-composition shift,
separately from the existing random10596 control, which preserves the full
contaminated pool's 72% KVASIR prior and its class distribution.

Strategy: draw examples from the FULL contaminated pool (CV2024
training) such that the per-class count matches le6's training
distribution exactly.  Total matches le6 size (10,596).

le6 per-class (from Table 3 of paper):
  Angioectasia 548, Bleeding 522, Erosion 2,340, Erythema 580,
  Foreign Body 249, Lymphangiectasia 382, Normal 4,627, Polyp 1,124,
  Ulcer 66, Worms 158.  Total 10,596.

For Ulcer (AIIMS-only in le6, 66 rows) and Worms (AIIMS-only, 158 rows),
the contaminated pool has plenty more AIIMS + other source examples,
so stratification by class alone (not by source) is possible.
"""
import argparse
import pandas as pd
from pathlib import Path

CV2024_CLASSES = [
    "Angioectasia", "Bleeding", "Erosion", "Erythema", "Foreign Body",
    "Lymphangiectasia", "Normal", "Polyp", "Ulcer", "Worms"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_xlsx", required=True,
                    help="CV2024 training_data.xlsx (full contaminated pool)")
    ap.add_argument("--le6_csv", required=True,
                    help="cv2024_training_dedup_le6.csv (target class distribution)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load full pool
    df_full = pd.read_excel(args.train_xlsx) if args.train_xlsx.endswith((".xlsx", ".xls")) \
              else pd.read_csv(args.train_xlsx)
    # Load le6 target
    df_le6 = pd.read_csv(args.le6_csv)

    # Compute le6 class counts
    le6_counts = {c: int(df_le6[c].sum()) for c in CV2024_CLASSES if c in df_le6.columns}
    total_target = sum(le6_counts.values())
    print(f"le6 target: total={total_target}, per-class={le6_counts}")

    # Draw from full pool with per-class count = le6_counts[c]
    picks = []
    for c, n in le6_counts.items():
        pool = df_full[df_full[c] == 1]
        if len(pool) < n:
            raise ValueError(f"Pool of class {c} has {len(pool)} < target {n}")
        sel = pool.sample(n=n, random_state=args.seed, replace=False)
        picks.append(sel)
        print(f"  {c}: drew {n}/{len(pool)}")

    out = pd.concat(picks, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows -> {args.out}")

    # Sanity check class distribution
    for c in CV2024_CLASSES:
        if c in out.columns:
            print(f"  out {c}: {int(out[c].sum())}")

if __name__ == "__main__":
    main()
