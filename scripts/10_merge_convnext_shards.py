#!/usr/bin/env python3
"""Merge ConvNeXt-Tiny shard JSONs into per-pool n=10 result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


POOLS = ("baseline", "le6", "random")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    shard_dir = root / "results" / "crossmodel" / "convnext_shards"
    out_dir = root / "results" / "crossmodel"
    summary = {"ok": True, "pools": {}}

    for pool in POOLS:
        runs = []
        missing = []
        metas = []
        for seed in args.seeds:
            path = shard_dir / f"phase5_crossmodel_convnextT_{pool}_s{seed}.json"
            if not path.exists():
                missing.append(seed)
                continue
            obj = load(path)
            metas.append(obj.get("meta", {}))
            for run in obj.get("runs", []):
                if int(run.get("seed")) == seed:
                    runs.append(run)
        runs.sort(key=lambda r: int(r["seed"]))
        existing = out_dir / f"phase5_crossmodel_convnextT_{pool}_n{len(args.seeds)}.json"
        if missing and existing.exists():
            obj = load(existing)
            existing_runs = sorted(obj.get("runs", []), key=lambda r: int(r["seed"]))
            existing_seeds = [int(r["seed"]) for r in existing_runs]
            if existing_seeds == args.seeds:
                summary["pools"][pool] = {
                    "n": len(existing_runs),
                    "seeds": existing_seeds,
                    "missing": [],
                    "complete": True,
                    "source": "existing_n10",
                }
                continue
        complete = not missing and [int(r["seed"]) for r in runs] == args.seeds
        summary["pools"][pool] = {
            "n": len(runs),
            "seeds": [int(r["seed"]) for r in runs],
            "missing": missing,
            "complete": complete,
            "source": "shards",
        }
        if missing and not args.allow_missing:
            summary["ok"] = False
            continue
        if runs:
            first_meta = metas[0] if metas else {}
            merged = {
                "args": {
                    "backbone": "convnext_tiny",
                    "pool": pool,
                    "seeds": [int(r["seed"]) for r in runs],
                    "source": "merged from per-seed shards",
                },
                "meta": {
                    **first_meta,
                    "merge_source": "scripts/10_merge_convnext_shards.py",
                    "shard_count": len(runs),
                    "missing_seeds": missing,
                },
                "runs": runs,
            }
            out = out_dir / f"phase5_crossmodel_convnextT_{pool}_n{len(runs)}.json"
            out.write_text(json.dumps(merged, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
