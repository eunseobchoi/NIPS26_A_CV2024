"""Trivial public-validation leakage baselines for CV2024.

This diagnostic uses only training labels plus identity/hash/video-prefix
signals to produce CV2024-format validation predictions. It is intentionally
not an official-test baseline and must not parse validation folder labels as a
prediction signal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


CV2024_CLASSES = [
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


def norm_path(path: str) -> str:
    return str(path).replace("\\", "/")


def cv_rel_key(path: str) -> str:
    rel = norm_path(path)
    for marker in ("training/", "validation/"):
        idx = rel.find(marker)
        if idx >= 0:
            return rel[idx:]
    return rel.lstrip("./")


def filename(path: str) -> str:
    return Path(norm_path(path)).name


CLASS_TOKENS = {
    re.sub(r"[^a-z0-9]+", "", c.lower())
    for c in CV2024_CLASSES
}


def source_prefix(name: str, dataset: str) -> str:
    stem = Path(name).stem
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
    elif "-" in stem:
        prefix = stem.split("-", 1)[0]
    else:
        prefix = stem
    token = re.sub(r"[^a-z0-9]+", "", prefix.lower())
    if token in CLASS_TOKENS:
        return ""
    # Treat only Kvasir's 16-hex filename prefix as a verified video ID.
    if dataset == "KVASIR" and re.fullmatch(r"[0-9a-fA-F]{16}", prefix):
        return prefix.lower()
    return ""


def label_from_row(row: pd.Series) -> str | None:
    vals = [int(row.get(c, 0) or 0) for c in CV2024_CLASSES]
    if sum(vals) == 0:
        return None
    return CV2024_CLASSES[int(np.argmax(vals))]


def onehot(label: str) -> dict[str, float]:
    return {c: float(c == label) for c in CV2024_CLASSES}


def vote(counter: Counter[str]) -> str:
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def load_annotations(root: Path) -> tuple[dict[str, dict], dict]:
    anns: dict[str, dict] = {}
    duplicate_keys = 0
    for csv_path in sorted((root / "artifacts" / "annotations").glob("cv2024_*_phash_annotated.csv")):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            key = cv_rel_key(str(row.get("path", row["filename"])))
            duplicate_keys += int(key in anns)
            anns[key] = {
                "cv_dataset": str(row["cv_dataset"]),
                "phash": str(row["phash"]),
                "dhash": str(row["dhash"]),
                "nearest_kvasir_file": str(row.get("nearest_kvasir_file", "")),
            }
    return anns, {
        "annotation_key": "normalized CV2024 relative image_path",
        "annotation_rows": int(len(anns) + duplicate_keys),
        "annotation_duplicate_path_keys": int(duplicate_keys),
    }


def load_official_metric(results_dir: Path):
    script = results_dir / "gen_metrics_report_val_train.py"
    if not script.exists():
        raise SystemExit(f"Missing organizer metric script: {script}")
    sys.path.insert(0, str(results_dir))
    from gen_metrics_report_val_train import generate_metrics_report, sanity_check  # noqa: PLC0415

    return generate_metrics_report, sanity_check


def build_predictions(root: Path, train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    annotations, annotation_meta = load_annotations(root)
    train_label_by_filename: dict[str, Counter[str]] = defaultdict(Counter)
    train_hash_votes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    train_kvasir_votes: dict[str, Counter[str]] = defaultdict(Counter)
    train_prefix_votes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    train_prior: Counter[str] = Counter()
    annotation_misses = Counter()

    for _, row in train_df.iterrows():
        label = label_from_row(row)
        if label is None:
            continue
        img = str(row["image_path"])
        name = filename(img)
        dataset = str(row.get("Dataset", ""))
        ann = annotations.get(cv_rel_key(img), {})
        annotation_misses["training"] += int(not ann)
        train_label_by_filename[name][label] += 1
        train_prior[label] += 1
        if ann.get("phash") and ann.get("dhash"):
            train_hash_votes[(ann["phash"], ann["dhash"])][label] += 1
        if ann.get("nearest_kvasir_file"):
            train_kvasir_votes[ann["nearest_kvasir_file"]][label] += 1
        prefix = source_prefix(name, dataset)
        if prefix:
            train_prefix_votes[(dataset, prefix)][label] += 1

    total = sum(train_prior.values())
    prior_probs = {c: train_prior[c] / total if total else 0.0 for c in CV2024_CLASSES}

    rows = []
    coverage: Counter[str] = Counter()
    for _, row in val_df.iterrows():
        img = str(row["image_path"])
        name = filename(img)
        dataset = str(row.get("Dataset", ""))
        ann = annotations.get(cv_rel_key(img), {})
        annotation_misses["validation"] += int(not ann)
        prefix = source_prefix(name, dataset)

        rule = "fallback_train_prior"
        probs = prior_probs.copy()
        if name in train_label_by_filename:
            rule = "exact_filename_train_match"
            probs = onehot(vote(train_label_by_filename[name]))
        elif (ann.get("phash"), ann.get("dhash")) in train_hash_votes:
            rule = "exact_phash_dhash_train_match"
            probs = onehot(vote(train_hash_votes[(ann["phash"], ann["dhash"])]))
        elif ann.get("nearest_kvasir_file") in train_kvasir_votes:
            rule = "same_nearest_kvasir_frame"
            probs = onehot(vote(train_kvasir_votes[ann["nearest_kvasir_file"]]))
        elif prefix and (dataset, prefix) in train_prefix_votes:
            rule = "same_source_prefix"
            probs = onehot(vote(train_prefix_votes[(dataset, prefix)]))

        coverage[rule] += 1
        out = {"image_path": img, "rule": rule}
        out.update(probs)
        rows.append(out)

    return pd.DataFrame(rows), {
        "n_train": int(len(train_df)),
        "n_validation": int(len(val_df)),
        "coverage": dict(coverage),
        "train_prior": prior_probs,
        "annotation_lookup": {
            **annotation_meta,
            "annotation_misses": dict(annotation_misses),
        },
        "source_prefix_rule": "Only Kvasir 16-hex filename prefixes are used; class-name tokens and unverified source filename stems are denied.",
        "rules_in_priority_order": [
            "exact_filename_train_match",
            "exact_phash_dhash_train_match",
            "same_nearest_kvasir_frame",
            "same_source_prefix",
            "fallback_train_prior",
        ],
    }


def score_subset(gt_df: pd.DataFrame, pred_df: pd.DataFrame, generate_metrics_report, sanity_check) -> dict:
    pred_sub = pred_df[pred_df["image_path"].isin(set(gt_df["image_path"]))].copy()
    pred_sub = pred_sub[["image_path", *CV2024_CLASSES]]
    ok, aligned = sanity_check(gt_df.reset_index(drop=True), pred_sub.reset_index(drop=True))
    if not ok:
        raise RuntimeError("Organizer sanity_check failed for trivial leakage predictions")
    y_true = gt_df[CV2024_CLASSES].to_numpy()
    y_pred = aligned[CV2024_CLASSES].to_numpy()
    metrics = generate_metrics_report(y_true, y_pred)
    return {
        "n": int(len(gt_df)),
        "mean_auc": float(metrics["mean_auc"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "combined": float((metrics["mean_auc"] + metrics["balanced_accuracy"]) / 2),
    }


def markdown_summary(result: dict) -> str:
    lines = [
        "# Trivial Leakage Baseline",
        "",
        "Diagnostic only: this uses training labels plus identity/hash/verified Kvasir video-prefix signals on the public validation pool.",
        "It is not an official-test baseline or replacement leaderboard.",
        "",
        f"Source-prefix rule: {result['source_prefix_rule']}",
        "",
        "| Subset | n | mean AUC | bal. acc. | combined |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["scores"].items():
        lines.append(
            f"| {name} | {row['n']} | {row['mean_auc']:.4f} | {row['balanced_accuracy']:.4f} | {row['combined']:.4f} |"
        )
    lines += ["", "## Coverage", "", "| Rule | n |", "| --- | ---: |"]
    for rule, count in result["coverage"].items():
        lines.append(f"| {rule} | {count} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("CAPSULE_ARTIFACT_ROOT", "."))
    parser.add_argument("--cv2024-results-dir", default=os.environ.get("CV2024_RESULTS_DIR", "external/cv2024_repo/Results"))
    parser.add_argument("--write-pred-xlsx", default="results/cv2024_trivial_leakage_baseline_predictions.xlsx")
    parser.add_argument("--write-json", default="results/cv2024_trivial_leakage_baseline.json")
    parser.add_argument("--write-md", default="results/cv2024_trivial_leakage_baseline.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    results_dir = Path(args.cv2024_results_dir)
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    generate_metrics_report, sanity_check = load_official_metric(results_dir)

    train_df = pd.read_excel(results_dir / "training_data.xlsx")
    val_df = pd.read_excel(results_dir / "validation_data.xlsx")
    pred_df, meta = build_predictions(root, train_df, val_df)

    subsets = {"orig_public_val": val_df}
    for name, rel in {
        "le6": "artifacts/csvs/cv2024_validation_dedup_le6.csv",
        "le6_plus_internal": "artifacts/csvs/cv2024_validation_le6_plus_internal.csv",
    }.items():
        paths = set(pd.read_csv(root / rel)["image_path"])
        subsets[name] = val_df[val_df["image_path"].isin(paths)].copy()

    scores = {
        name: score_subset(df, pred_df, generate_metrics_report, sanity_check)
        for name, df in subsets.items()
    }

    out = {
        "description": "Trivial public-validation leakage baseline; not an official-test or replacement leaderboard.",
        "implementation_constraints": [
            "Uses training labels and train/validation identity/hash/video-prefix metadata only.",
            "Does not parse validation class folder names as a prediction signal.",
            "Scores with CV2024 organizer public-validation metric implementation unchanged.",
        ],
        **meta,
        "scores": scores,
    }

    for path in [args.write_pred_xlsx, args.write_json, args.write_md]:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
    pred_df[["image_path", *CV2024_CLASSES]].to_excel(root / args.write_pred_xlsx, index=False)
    (root / args.write_json).write_text(json.dumps(out, indent=2) + "\n")
    (root / args.write_md).write_text(markdown_summary(out))
    print(json.dumps(out["scores"], indent=2))


if __name__ == "__main__":
    main()
