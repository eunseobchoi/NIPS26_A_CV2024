#!/usr/bin/env python3
"""Verify Croissant distribution entries against checksums.txt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SELF_EXCEPTIONS = {"checksums.txt", "croissant.json"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--write-json", default=None)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    croissant = json.loads((root / "croissant.json").read_text())
    checksums = {}
    for line in (root / "checksums.txt").read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        checksums[rel.removeprefix("./")] = digest

    distribution = {
        entry["contentUrl"]: entry
        for entry in croissant.get("distribution", [])
        if entry.get("@type") == "cr:FileObject" and entry.get("contentUrl")
    }
    files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
    }
    # Generated verifier outputs are allowed to be absent from an older manifest
    # only if the caller has not regenerated Croissant yet; after release this
    # should be empty.
    missing_distribution = sorted(files - set(distribution) - SELF_EXCEPTIONS)
    checksum_mismatches = []
    missing_checksums = []
    for rel, digest in checksums.items():
        p = root / rel
        if not p.exists():
            missing_checksums.append(rel)
        elif sha256(p) != digest:
            checksum_mismatches.append(rel)

    croissant_hash_gaps = []
    for rel, entry in distribution.items():
        if rel in SELF_EXCEPTIONS:
            continue
        if entry.get("sha256") != checksums.get(rel):
            croissant_hash_gaps.append(rel)

    out = {
        "ok": not (missing_distribution or missing_checksums or checksum_mismatches or croissant_hash_gaps),
        "n_files": len(files),
        "n_distribution": len(distribution),
        "n_checksums": len(checksums),
        "self_exceptions": sorted(SELF_EXCEPTIONS),
        "missing_distribution": missing_distribution,
        "missing_checksums": missing_checksums,
        "checksum_mismatches": checksum_mismatches,
        "croissant_hash_gaps": croissant_hash_gaps,
    }
    if args.write_json:
        out_path = Path(args.write_json)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    if not out["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
