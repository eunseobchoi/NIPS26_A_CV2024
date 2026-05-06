#!/usr/bin/env bash
# Verify every released file listed in checksums.txt against its recorded
# SHA-256. The verifier is intentionally read-only.
set -euo pipefail

if [ ! -f checksums.txt ]; then
  echo "ERROR: checksums.txt not found." >&2
  echo "This verifier is read-only; regenerate the manifest explicitly during release packaging." >&2
  exit 2
fi

echo "Verifying $(wc -l < checksums.txt) checksums..."
sha256sum --check checksums.txt
echo "All checksums OK."
