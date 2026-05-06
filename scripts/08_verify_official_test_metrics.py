#!/usr/bin/env python3
"""Replay official-test aggregate metric definitions from packaged JSONs.

The CV2024 organizer test script (`gen_metrics_test.py`) reports mean AUC and
balanced accuracy from probability files. The packaged official-test JSONs do
not redistribute per-image prediction XLSX files, but they do store the final
confusion matrix and per-class AUC values. This verifier recomputes the two
leaderboard aggregate quantities from those stored sufficient statistics and
checks that the reported combined metric is exactly the organizer definition:

    combined = (mean_auc + balanced_accuracy) / 2.

It is intentionally dependency-free so reviewers can run it as part of the
CPU-only smoke test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def final_metrics(result: dict) -> list[tuple[int, dict]]:
    out = []
    for run in result.get("runs", []):
        metric = run.get("last", {}).get("official_test")
        if metric:
            out.append((int(run.get("seed", -1)), metric))
    return out


def replay(metric: dict) -> dict:
    cm = metric["confusion_matrix"]
    recalls = []
    for i, row in enumerate(cm):
        denom = sum(row)
        if denom <= 0:
            raise ValueError(f"class {i} has no examples in confusion matrix")
        recalls.append(row[i] / denom)
    bal_acc = sum(recalls) / len(recalls)

    aucs = []
    for cls in CLASSES:
        auc = metric["per_class"][cls]["auc"]
        if auc is None:
            raise ValueError(f"class {cls} has no AUC")
        aucs.append(float(auc))
    mean_auc = sum(aucs) / len(aucs)
    combined = (bal_acc + mean_auc) / 2.0
    n_test = sum(sum(row) for row in cm)
    return {
        "balanced_accuracy": bal_acc,
        "mean_auc": mean_auc,
        "combined": combined,
        "n_test": n_test,
    }


def close(a: float, b: float, tol: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Artifact root")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--write-json", default="", help="Optional output JSON path")
    args = ap.parse_args()

    root = Path(args.root)
    metric_script = root / "external/cv2024_repo/Results/gen_metrics_test.py"
    result_dir = root / "results/official_test"
    paths = sorted(
        p
        for p in result_dir.glob("*_official_test_s*.json")
        if p.name.startswith(("baseline_", "random10596_s0_", "le6_"))
    )
    if len(paths) != 30:
        raise SystemExit(f"expected 30 canonical official-test JSONs, found {len(paths)}")

    checks = []
    for path in paths:
        data = json.loads(path.read_text())
        for seed, metric in final_metrics(data):
            got = replay(metric)
            ok = (
                close(got["balanced_accuracy"], metric["bal_acc"], args.tol)
                and close(got["mean_auc"], metric["mean_auc"], args.tol)
                and close(got["combined"], metric["combined"], args.tol)
                and got["n_test"] == metric["n_test"] == 4385
            )
            checks.append(
                {
                    "path": str(path.relative_to(root)),
                    "seed": seed,
                    "ok": ok,
                    "reported": {
                        "balanced_accuracy": metric["bal_acc"],
                        "mean_auc": metric["mean_auc"],
                        "combined": metric["combined"],
                        "n_test": metric["n_test"],
                    },
                    "replayed": got,
                }
            )
    failed = [c for c in checks if not c["ok"]]
    report = {
        "description": "Dependency-free replay of CV2024 official-test aggregate metric definitions from packaged sufficient statistics.",
        "official_script": str(metric_script.relative_to(root)),
        "official_script_sha256": sha256(metric_script),
        "n_files": len(paths),
        "n_seed_checks": len(checks),
        "tolerance": args.tol,
        "all_ok": not failed,
        "checks": checks,
    }
    if args.write_json:
        out = root / args.write_json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        print(f"FAIL: {len(failed)} official-test metric replays failed")
        for c in failed[:5]:
            print(f"  {c['path']} seed={c['seed']}")
        raise SystemExit(2)
    print(
        f"PASS: {len(checks)} official-test aggregate metric replays match "
        f"{metric_script.relative_to(root)}"
    )


if __name__ == "__main__":
    main()
