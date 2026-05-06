"""Composition-matched contaminated control for fixed-list sensitivity analysis.

Design:
  - Match le6's CLASS distribution exactly (so class-composition shift is
    held constant between le6 and this control).
  - Within each class, PREFER KVASIR to preserve the KVASIR
    contamination channel (so leakage is present).
  - Fall back to SEE-AI/KID/AIIMS only when KVASIR doesn't cover the
    target count (or when the class is KVASIR-absent, e.g. Worms).

This isolates: residual_{composition-matched} - residual_{le6}
= effect of REMOVING KVASIR re-exposure at fixed class composition.

By contrast, the existing random10596 matches the contaminated pool's
prior (KVASIR ≈ 72%), which differs from le6's (0%) in BOTH source
composition AND class composition.
"""
import argparse
import pandas as pd
from pathlib import Path

CV2024_CLASSES = [
    "Angioectasia", "Bleeding", "Erosion", "Erythema", "Foreign Body",
    "Lymphangiectasia", "Normal", "Polyp", "Ulcer", "Worms"
]


def source_of(path):
    p = str(path).replace("\\", "/").lower()
    for src in ["kvasir", "see-ai", "seeai", "kid", "aiims"]:
        if f"/{src}/" in p:
            return {"kvasir":"KVASIR","see-ai":"SEE-AI","seeai":"SEE-AI",
                    "kid":"KID","aiims":"AIIMS"}[src]
    return "UNK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_xlsx", required=True)
    ap.add_argument("--le6_csv", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df_full = pd.read_excel(args.train_xlsx) if args.train_xlsx.endswith((".xlsx", ".xls")) \
              else pd.read_csv(args.train_xlsx)
    df_le6 = pd.read_csv(args.le6_csv)
    df_full["__source__"] = df_full["image_path"].apply(source_of)

    le6_counts = {c: int(df_le6[c].sum()) for c in CV2024_CLASSES if c in df_le6.columns}
    total_target = sum(le6_counts.values())
    print(f"le6 per-class target: total={total_target}")
    for c, n in le6_counts.items(): print(f"  {c:18s}: {n}")

    picks = []
    src_summary = {}
    for c, n in le6_counts.items():
        # Stage 1: draw from KVASIR as much as possible
        kv_pool = df_full[(df_full[c] == 1) & (df_full["__source__"] == "KVASIR")]
        take_kv = min(n, len(kv_pool))
        if take_kv > 0:
            sel_kv = kv_pool.sample(n=take_kv, random_state=args.seed, replace=False)
            picks.append(sel_kv)
        remaining = n - take_kv

        # Stage 2: fill remainder from SEE-AI/KID/AIIMS (non-KVASIR)
        sel_nk = None
        if remaining > 0:
            nk_pool = df_full[(df_full[c] == 1) & (df_full["__source__"] != "KVASIR")]
            if len(nk_pool) < remaining:
                print(f"  WARNING: {c} pool (non-KVASIR) has {len(nk_pool)} < {remaining}")
                remaining = len(nk_pool)
            sel_nk = nk_pool.sample(n=remaining, random_state=args.seed, replace=False)
            picks.append(sel_nk)

        src_summary[c] = (take_kv, remaining)
        print(f"  {c:18s}: {n} = {take_kv} KVASIR + {remaining} non-KVASIR")

    out = pd.concat(picks, ignore_index=True).drop(columns=["__source__"])
    out = out.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} rows -> {args.out}")

    # Verify contamination rate
    total_kv = sum(k for k, _ in src_summary.values())
    print(f"KVASIR fraction: {total_kv}/{len(out)} = {total_kv/len(out):.3f}")
    print(f"(le6 KVASIR fraction: 0/10596 = 0.000)")
    print(f"(random10596 KVASIR fraction: ~0.72 as full pool)")


if __name__ == "__main__":
    main()
