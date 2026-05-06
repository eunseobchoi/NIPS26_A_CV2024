"""Public entry point for fixed-list counterfactual training.

The executable implementation used for the released result JSONs is kept in
``phase5_counterfactual_v5.py`` because its file bytes are part of the recorded
provenance hashes. This wrapper gives the paper and README a stable, descriptive
command path without changing the historical training script.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("phase5_counterfactual_v5.py")),
        run_name="__main__",
    )
