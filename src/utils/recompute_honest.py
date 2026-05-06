"""Legacy helper for recomputing summaries without test-fold tuning.

Historical exploratory tables sometimes used `best_epoch` selected on the
test fold. The current paper reports final-epoch metrics for its main
numbers; this script is retained only to reproduce the old best-vs-last
audit table.

We recompute results using:
(a) last-epoch (entire 30 epochs without early stopping) — pessimistic fair.
(b) fixed-epoch (epoch 10) — reference point.
(c) inner-val-proxy: mean of last 3 epochs (proxy for what a properly
    held-out inner val would select if available).

This lets us report an honest range without re-running all experiments.
"""
import os
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(os.environ.get("CAPSULE_ROOT", ".") + "/results")


def last_epoch_metrics(history):
    """bal_acc of the final logged epoch."""
    if not history:
        return None
    return history[-1]


def last_n_mean_metrics(history, n=3):
    """Mean bal_acc of the last n logged epochs."""
    if not history:
        return None
    tail = history[-n:] if len(history) >= n else history
    return {
        "bal_acc": float(np.mean([h["bal_acc"] for h in tail])),
        "acc": float(np.mean([h["acc"] for h in tail])),
        "null_acc": float(np.mean([h["null_acc"] for h in tail])),
        "f1_macro": float(np.mean([h["f1_macro"] for h in tail])),
        "epochs_used": [h["epoch"] for h in tail],
    }


def fixed_epoch_metrics(history, epoch=10):
    """bal_acc at a fixed epoch, if reached."""
    for h in history:
        if h["epoch"] == epoch:
            return h
    # If we stopped before that, return the last reached epoch
    return history[-1] if history else None


def summarize_file(path):
    with open(path) as f:
        d = json.load(f)
    runs = d.get("runs", [])
    # Group by (mode, lora_r, train_aug, scale, subset)
    groups = defaultdict(list)
    for r in runs:
        c = r.get("config", {})
        # Also pick up data-scaling or class-merging keys
        key = (
            c.get("mode", "?"),
            c.get("lora_r", r.get("scale") or r.get("subset", "?")),
            c.get("train_aug", False),
        )
        groups[key].append(r)

    out = {}
    for key, rows in groups.items():
        best_bal = []
        last_bal = []
        last3_bal = []
        fix10_bal = []
        nulls = []
        for r in rows:
            best_bal.append(r.get("bal_acc") or r.get("best_metrics", {}).get("bal_acc"))
            nulls.append(r.get("null_acc") or r.get("best_metrics", {}).get("null_acc"))
            h = r.get("history", [])
            if h:
                last_bal.append(h[-1]["bal_acc"])
                tail = h[-3:] if len(h) >= 3 else h
                last3_bal.append(float(np.mean([x["bal_acc"] for x in tail])))
                fix = next((x for x in h if x["epoch"] == 10), h[-1])
                fix10_bal.append(fix["bal_acc"])
        out[str(key)] = {
            "n": len(rows),
            "best_bal_mean": float(np.mean(best_bal)) if best_bal and all(b is not None for b in best_bal) else None,
            "best_bal_std": float(np.std(best_bal)) if best_bal and all(b is not None for b in best_bal) else None,
            "last_bal_mean": float(np.mean(last_bal)) if last_bal else None,
            "last_bal_std": float(np.std(last_bal)) if last_bal else None,
            "last3_bal_mean": float(np.mean(last3_bal)) if last3_bal else None,
            "fix10_bal_mean": float(np.mean(fix10_bal)) if fix10_bal else None,
            "null_acc_mean": float(np.mean(nulls)) if nulls and all(n is not None for n in nulls) else None,
        }
    return out


def main():
    print(f"{'='*90}")
    print("  HONEST RECOMPUTE: best (test-tuned) vs last vs last3 vs fixed-epoch 10")
    print(f"{'='*90}")
    print(f"{'File':<40} {'Key':<40} {'best':>6} {'last':>6} {'last3':>6} {'ep10':>6} {'null':>6}")
    print("-"*100)
    files_to_check = [
        "strong_baseline.json",
        "strong_lora_rank.json",
        "strong_full_ft.json",
        "strong_aug.json",
        "side_data_scaling.json",
        "side_5fold_video.json",
        "extra_class_merging.json",
        "combined_kvasir_only.json",
        "combined_multisrc.json",
        "cv_to_kvasir.json",
        "cv2024_pooled.json",
        "cv2024_lso.json",
    ]
    for fn in files_to_check:
        path = RESULTS / fn
        if not path.exists():
            continue
        s = summarize_file(path)
        for k, v in s.items():
            fn_str = fn[:38]
            key_str = k[:38]
            print(f"{fn_str:<40} {key_str:<40} "
                  f"{v['best_bal_mean']:>6.3f} "
                  f"{v['last_bal_mean']:>6.3f} "
                  f"{v['last3_bal_mean']:>6.3f} "
                  f"{v['fix10_bal_mean']:>6.3f} "
                  f"{v['null_acc_mean'] or 0:>6.3f}")


if __name__ == "__main__":
    main()
