#!/usr/bin/env python3
"""Preflight the CV2024 organizer validation assets needed for M7 analyses."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def has_required_assets(root: Path) -> bool:
    return (
        (root / "gen_metrics_report_val_train.py").exists()
        and (root / "training_data.xlsx").exists()
        and (root / "validation_data.xlsx").exists()
        and (root / "submitted_excel_files" / "validation").is_dir()
        and (root / "metrics_reports" / "metrics_reports_val").is_dir()
    )


def default_results_dir() -> str:
    candidates = []
    env = os.environ.get("CV2024_RESULTS_DIR")
    if env:
        candidates.append(Path(env))
    candidates.extend([
        Path("external/cv2024_repo/Results"),
        Path("../external/cv2024_repo/Results"),
    ])
    for candidate in candidates:
        if has_required_assets(candidate):
            return str(candidate)
    return str(candidates[0] if candidates else Path("external/cv2024_repo/Results"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=default_results_dir(),
        help="CV2024 organizer Results directory.",
    )
    parser.add_argument(
        "--write-json",
        default=None,
        help="Optional path for a JSON preflight report.",
    )
    args = parser.parse_args()

    root = Path(args.results_dir)
    checks = {
        "metric_script": root / "gen_metrics_report_val_train.py",
        "training_labels": root / "training_data.xlsx",
        "validation_labels": root / "validation_data.xlsx",
        "prediction_dir": root / "submitted_excel_files" / "validation",
        "organizer_metric_dir": root / "metrics_reports" / "metrics_reports_val",
    }
    report = {
        "results_dir": "<CV2024_RESULTS_DIR>",
        "checks": {
            k: {
                "path": str(v.relative_to(root)) if v.is_relative_to(root) else v.name,
                "exists": v.exists(),
            }
            for k, v in checks.items()
        },
        "prediction_xlsx_count": 0,
        "organizer_json_count": 0,
        "ok": False,
    }

    pred_dir = checks["prediction_dir"]
    metric_dir = checks["organizer_metric_dir"]
    if pred_dir.exists():
        report["prediction_xlsx_count"] = len(list(pred_dir.glob("*.xlsx")))
    if metric_dir.exists():
        report["organizer_json_count"] = len(list(metric_dir.glob("*.json")))

    report["ok"] = (
        all(item["exists"] for item in report["checks"].values())
        and report["prediction_xlsx_count"] >= 25
        and report["organizer_json_count"] == 27
    )

    if args.write_json:
        out = Path(args.write_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
