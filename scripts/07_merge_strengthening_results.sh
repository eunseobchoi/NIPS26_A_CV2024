#!/usr/bin/env bash
# Optional Stage 7: regenerate strengthening-experiment aggregate JSONs
# from shipped single-seed shard outputs.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-python3}"
"$PY" scripts/merge_strengthening_results.py --root "$PWD"
