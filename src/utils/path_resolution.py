"""Path normalization helpers for released CSV variants."""

from __future__ import annotations

from pathlib import Path


def resolve_cv2024_or_kvasir_path(
    rel_path: str,
    *,
    cv2024_root: Path,
    kvasir_data_root: Path,
) -> Path:
    """Resolve historical CV2024/Kvasir CSV path formats.

    The released CSVs accumulated a few path prefixes over the audit:
    original CV2024 rows, Kvasir-Capsule add-back rows, and historical
    labelled_images-relative rows.  This helper keeps those compatibility
    rules explicit and unit-testable.
    """
    rel = rel_path.replace("\\", "/")
    raw_path = Path(rel)
    if raw_path.is_absolute():
        return raw_path
    if rel.startswith("kvasir_capsule_split_1/"):
        parts = rel.split("/")
        if len(parts) < 2:
            raise ValueError(f"Malformed Kvasir split_1 path: {rel_path}")
        return kvasir_data_root / parts[-2] / parts[-1]
    if rel.startswith("kvasir_capsule/labelled_images/"):
        return kvasir_data_root / rel.split("kvasir_capsule/labelled_images/", 1)[1]
    if rel.startswith("labelled_images/"):
        return kvasir_data_root / rel.split("labelled_images/", 1)[1]
    return cv2024_root / rel
