#!/usr/bin/env python3
"""Verify selected paper-facing numeric claims against packaged artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def load_json(root: Path, rel: str):
    return json.loads((root / rel).read_text())


def check_close(name: str, observed: float, expected: float, tol: float, source: str) -> dict:
    return {
        "claim": name,
        "source": source,
        "observed": observed,
        "expected": expected,
        "tolerance": tol,
        "ok": math.isfinite(observed) and abs(observed - expected) <= tol,
    }


def check_equal(name: str, observed, expected, source: str) -> dict:
    return {
        "claim": name,
        "source": source,
        "observed": observed,
        "expected": expected,
        "ok": observed == expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--write-json", default=None)
    args = parser.parse_args()
    root = Path(args.root).resolve()

    checks = []

    base_src = "results/baseline/phase5_v5_baseline_n10.json"
    le6_src = "results/baseline/phase5_v5_le6_n10.json"
    base = load_json(root, base_src)
    le6 = load_json(root, le6_src)
    base_vals = [r["last"]["orig_val"]["bal_acc"] for r in base["runs"]]
    le6_vals = [r["last"]["orig_val"]["bal_acc"] for r in le6["runs"]]
    checks.append(check_equal("baseline seed count", len(base_vals), 10, base_src))
    checks.append(check_equal("le6 seed count", len(le6_vals), 10, le6_src))
    checks.append(check_close("Delta_le6 original public-val balanced accuracy", mean(le6_vals) - mean(base_vals), -0.2131, 0.0005, f"{base_src}; {le6_src}"))

    for source, expected in {
        "KVASIR": -0.4728,
        "SEE-AI": 0.0131,
        "KID": -0.0443,
        "AIIMS": 0.0215,
    }.items():
        source_delta = mean(
            le6_run["last"]["orig_val"]["per_source"][source]["bal_acc"]
            - base_run["last"]["orig_val"]["per_source"][source]["bal_acc"]
            for base_run, le6_run in zip(base["runs"], le6["runs"])
        )
        checks.append(
            check_close(
                f"evidence-and-scope table per-source delta {source}",
                source_delta,
                expected,
                0.0006,
                f"{base_src}; {le6_src}",
            )
        )

    classes = list(base["runs"][0]["last"]["orig_val"]["per_class"])
    for label, excluded, expected in [
        ("exclude Ulcer", {"Ulcer"}, -0.1390),
        ("exclude Ulcer/Worms/Normal", {"Ulcer", "Worms", "Normal"}, -0.1207),
    ]:
        kept = [c for c in classes if c not in excluded]
        subset_delta = mean(
            mean(
                le6_run["last"]["orig_val"]["per_class"][c]["recall"]
                - base_run["last"]["orig_val"]["per_class"][c]["recall"]
                for c in kept
            )
            for base_run, le6_run in zip(base["runs"], le6["runs"])
        )
        checks.append(
            check_close(
                f"evidence-and-scope table class-subset delta {label}",
                subset_delta,
                expected,
                0.0006,
                f"{base_src}; {le6_src}",
            )
        )

    phash_src = "artifacts/annotations/cv2024_KVASIR_phash_annotated.csv"
    n = flagged = 0
    with (root / phash_src).open(newline="") as f:
        for row in csv.DictReader(f):
            if row["cv_dataset"] != "KVASIR":
                continue
            n += 1
            if int(row["min_phash_dist_to_kvasir"]) <= 6 and int(row["min_dhash_dist_to_kvasir"]) <= 6:
                flagged += 1
    checks.append(check_equal("CV2024-KVASIR rows", n, 38592, phash_src))
    checks.append(check_equal("CV2024-KVASIR pHash+dHash<=6 flagged rows", flagged, 38592, phash_src))

    table5_sources = {
        "DINOv2-L": {
            "baseline": "results/baseline/phase5_v5_baseline_n10.json",
            "random": "results/acceptance_lift/phase5_v4_random_s0_n10.json",
            "le6": "results/baseline/phase5_v5_le6_n10.json",
            "expected": {"baseline": 0.825, "random": 0.779, "le6": 0.612, "residual": -0.167},
        },
        "DINOv2-B": {
            "baseline": "results/crossmodel/phase5_crossmodel_dinov2B_baseline_n10.json",
            "random": "results/crossmodel/phase5_crossmodel_dinov2B_random_n10.json",
            "le6": "results/crossmodel/phase5_crossmodel_dinov2B_le6_n10.json",
            "expected": {"baseline": 0.841, "random": 0.797, "le6": 0.641, "residual": -0.156},
        },
        "DINOv2-S": {
            "baseline": "results/crossmodel/phase5_crossmodel_dinov2S_baseline_n10.json",
            "random": "results/crossmodel/phase5_crossmodel_dinov2S_random_n10.json",
            "le6": "results/crossmodel/phase5_crossmodel_dinov2S_le6_n10.json",
            "expected": {"baseline": 0.823, "random": 0.772, "le6": 0.624, "residual": -0.148},
        },
        "ResNet-50": {
            "baseline": "results/crossmodel/phase5_crossmodel_resnet50_baseline_n10.json",
            "random": "results/crossmodel/phase5_crossmodel_resnet50_random_n10.json",
            "le6": "results/crossmodel/phase5_crossmodel_resnet50_le6_n10.json",
            "expected": {"baseline": 0.630, "random": 0.586, "le6": 0.430, "residual": -0.156},
        },
        "ConvNeXt-Tiny": {
            "baseline": "results/crossmodel/phase5_crossmodel_convnextT_baseline_n10.json",
            "random": "results/crossmodel/phase5_crossmodel_convnextT_random_n10.json",
            "le6": "results/crossmodel/phase5_crossmodel_convnextT_le6_n10.json",
            "expected": {"baseline": 0.691, "random": 0.646, "le6": 0.438, "residual": -0.208},
        },
    }
    for name, cfg in table5_sources.items():
        vals = {}
        source_list = []
        for pool in ("baseline", "random", "le6"):
            src = cfg[pool]
            source_list.append(src)
            obj = load_json(root, src)
            runs = obj["runs"]
            checks.append(check_equal(f"Cross-model robustness {name} {pool} seed count", len(runs), 10, src))
            vals[pool] = mean(r["last"]["orig_val"]["bal_acc"] for r in runs)
            checks.append(check_close(f"Cross-model robustness {name} {pool} mean", vals[pool], cfg["expected"][pool], 0.0005, src))
        residual = (vals["le6"] - vals["baseline"]) - (vals["random"] - vals["baseline"])
        checks.append(check_close(f"Cross-model robustness {name} size-adjusted residual", residual, cfg["expected"]["residual"], 0.0005, "; ".join(source_list)))

    m7_src = "results/cv2024_m7_inference.json"
    m7 = load_json(root, m7_src)
    checks.append(check_close("trained-19 le6 public re-score mean delta combined", m7["le6__trained_19"]["observed_mean"], -0.062, 0.001, m7_src))
    checks.append(check_close("trained-19 le6 public re-score CI lower", m7["le6__trained_19"]["paired_bootstrap_ci95_lo"], -0.079, 0.0015, m7_src))
    checks.append(check_close("trained-19 le6 public re-score CI upper", m7["le6__trained_19"]["paired_bootstrap_ci95_hi"], -0.045, 0.0015, m7_src))

    sb_src = "results/cv2024_source_balanced_rescore.json"
    sb = load_json(root, sb_src)
    sb_train = sb["subsets"]["le6"]["aggregates"]
    checks.append(check_close("trained-19 le6 source-balanced balanced-accuracy delta", sb_train["source_balanced_balanced_accuracy"]["trained_19"]["mean_delta"], -0.037, 0.0015, sb_src))
    checks.append(check_close("trained-19 le6 source-balanced present-class combined delta", sb_train["source_balanced_combined_present_auc"]["trained_19"]["mean_delta"], -0.022, 0.0015, sb_src))
    checks.append(check_equal("trained-19 le6 source-balanced BA rank changes", sb_train["source_balanced_balanced_accuracy"]["trained_19"]["n_rank_changed"], 9, sb_src))

    off_src = "results/official_test/official_test_direct_eval_summary.json"
    off = load_json(root, off_src)
    for arm, expected in {
        "baseline": 0.4991,
        "random10596_s0": 0.5069,
        "le6": 0.5110,
    }.items():
        checks.append(check_close(f"official AIIMS-test combined mean {arm}", off["arms"][arm]["metrics"]["combined"]["mean"], expected, 0.0001, off_src))

    auc_base_src = "results/phase5_v5_auc_baseline_n10.json"
    auc_le6_src = "results/phase5_v5_auc_le6_n10.json"
    auc_base = load_json(root, auc_base_src)
    auc_le6 = load_json(root, auc_le6_src)
    checks.append(check_equal("AUC audit baseline seed count", len(auc_base["runs"]), 10, auc_base_src))
    checks.append(check_equal("AUC audit le6 seed count", len(auc_le6["runs"]), 10, auc_le6_src))
    for metric, (expected, tol) in {
        "mean_auc_ovr": (-0.102, 0.001),
        "combined_ovr": (-0.158, 0.001),
    }.items():
        delta = mean(r["last"]["orig_val"][metric] for r in auc_le6["runs"]) - mean(r["last"]["orig_val"][metric] for r in auc_base["runs"])
        checks.append(check_close(f"AUC audit le6-baseline delta {metric}", delta, expected, tol, f"{auc_base_src}; {auc_le6_src}"))

    for source, rel, expected_gap, expected_pooled in [
        ("SEE-AI", "results/loso/phase5_v5_loso_seeai_out_n10.json", 0.458, 0.604),
        ("KID", "results/loso/phase5_v5_loso_kid_out_n10.json", 0.317, 0.822),
    ]:
        obj = load_json(root, rel)
        checks.append(check_equal(f"LOSO {source} seed count", len(obj["runs"]), 10, rel))
        loso_source = mean(r["last"]["orig_val"]["per_source"][source]["bal_acc"] for r in obj["runs"])
        baseline_source = mean(r["last"]["orig_val"]["per_source"][source]["bal_acc"] for r in auc_base["runs"])
        pooled = mean(r["last"]["orig_val"]["bal_acc"] for r in obj["runs"])
        checks.append(check_close(f"LOSO {source} held-out source gap", baseline_source - loso_source, expected_gap, 0.001, f"{auc_base_src}; {rel}"))
        checks.append(check_close(f"LOSO {source} pooled val", pooled, expected_pooled, 0.001, rel))

    strat_src = "results/mechanism_probes/phase9_kvasir_strat_shuffle_n10.json"
    strat = load_json(root, strat_src)
    checks.append(check_equal("KVASIR stratified shuffle seed count", len(strat["runs"]), 10, strat_src))
    checks.append(check_close("KVASIR stratified shuffle orig-val mean", mean(r["last"]["orig_val"]["bal_acc"] for r in strat["runs"]), 0.504, 0.001, strat_src))
    checks.append(check_close("KVASIR stratified shuffle split_1 mean", mean(r["last"]["kvasir_s1"]["bal_acc"] for r in strat["runs"]), 0.134, 0.001, strat_src))

    missing_hashes = []
    script_hashes = {}
    for path in (root / "src").rglob("*.py"):
        import hashlib

        script_hashes[hashlib.sha256(path.read_bytes()).hexdigest()] = str(path.relative_to(root))
    historical_to_packaged = {
        "be3fd06fab1a04d7311f70ab9f1eb0563a7c8f1db5b2c6081987e04e68b90edf": (
            "src/provenance/phase5_counterfactual_v5_be3fd06_exact.py",
            "0199652d00c49a5d83d12728bac0e1c42bb44c236917622b4231fd96408ff71e",
        ),
        "72047d35682f487375235458375f6359ef9d4ba00c94d9fb5aaf0c7fd1237e5c": (
            "src/provenance/phase5_counterfactual_v4_72047d35_exact.py",
            "46a6cc2c82f5db668b9bd79c78c1acdad58bcbd68de56e834978b2b9d82f9d88",
        ),
        "97312ac3f5f87816a5f207b57ed98f82d041defff85dbbae97bfd8224eb77af9": (
            "src/provenance/phase5_exp3_mi_probe_97312ac3_exact.py",
            "c084c9190ec2ade08e44b3023383aee87de52721afe16ccbdb4e08195bc112eb",
        ),
        "c06c5693d2d09ed65e6419f6a29107c6f1ef159638d4d291f40ed1c9e9155137": (
            "src/provenance/phase5_exp2b_split0_only_c06c5693_exact.py",
            "70bd6378938d604fcfd4dabb4ee41bbeb671f688ae484cce0e09c09bb1e3011f",
        ),
    }
    for rel in [
        "results/baseline/phase5_v5_baseline_n10.json",
        "results/baseline/phase5_v5_le6_n10.json",
        "results/mechanism_probes/phase5_exp1_le6_kvfree_s1_n10.json",
        "results/phase5_v5_auc_baseline_n10.json",
        "results/phase5_v5_auc_le6_n10.json",
        "results/loso/phase5_v5_loso_seeai_out_n10.json",
        "results/loso/phase5_v5_loso_kid_out_n10.json",
        "results/mechanism_probes/phase9_kvasir_strat_shuffle_n10.json",
    ]:
        obj = load_json(root, rel)
        sha = obj.get("meta", {}).get("script_sha256")
        if sha and sha not in script_hashes:
            mapped = historical_to_packaged.get(sha)
            if mapped:
                mapped_rel, expected_packaged_sha = mapped
                mapped_path = root / mapped_rel
                if mapped_path.exists() and hashlib.sha256(mapped_path.read_bytes()).hexdigest() == expected_packaged_sha:
                    continue
            missing_hashes.append({"result": rel, "script_sha256": sha})
    checks.append({
        "claim": "selected headline result script_sha256 values trace to packaged scripts or documented path-scrubbed snapshots",
        "source": "src/**/*.py plus selected result JSON meta and provenance hash map",
        "observed": missing_hashes,
        "expected": [],
        "ok": not missing_hashes,
    })

    ok = all(c["ok"] for c in checks)
    out = {"ok": ok, "checks": checks}
    if args.write_json:
        out_path = Path(args.write_json)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
