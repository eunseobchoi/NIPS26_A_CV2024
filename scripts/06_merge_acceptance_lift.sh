#!/usr/bin/env bash
# Optional Stage 6b: regenerate the shipped acceptance-lift aggregate JSONs
# from existing single-seed shard outputs.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-python3}"
"$PY" scripts/merge_acceptance_lift.py --root "$PWD"
