#!/usr/bin/env python3
"""Scrub machine-local paths from packaged result JSON files.

This preserves numeric results and relative artifact provenance while replacing
cluster-local absolute prefixes with portable placeholders.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REPLACEMENTS = [
    (re.compile(r"/home/[^/]+/CapsuleMamba/data/cv2024/train_val/Dataset"), "${CV2024_ROOT}"),
    (re.compile(r"/home/[^/]+/capsule_tta(?:_repro)?"), "${CAPSULE_ROOT}"),
    (re.compile(r"/var/tmp/pbs\.[^/\s\"]+"), "${PBS_TMPDIR}"),
]


def scrub_obj(obj):
    if isinstance(obj, dict):
        return {k: scrub_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_obj(v) for v in obj]
    if isinstance(obj, str):
        out = obj
        for pattern, repl in REPLACEMENTS:
            out = pattern.sub(repl, out)
        out = out.replace("submission_FINAL/results/", "results/")
        return out
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        data = json.loads(path.read_text())
        scrubbed = scrub_obj(data)
        path.write_text(json.dumps(scrubbed, indent=2, default=str) + "\n")
        print(f"scrubbed {path}")


if __name__ == "__main__":
    main()
