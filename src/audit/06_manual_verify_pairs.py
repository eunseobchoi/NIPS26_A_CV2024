"""Manual verification of flagged near-duplicate pairs (ciFAIR protocol).

Draws a random sample of 200 flagged CV2024-KVASIR ↔ Kvasir pairs and
saves side-by-side comparison images for manual inspection. Also saves
the sample metadata for paper's "precision estimate" reporting.
"""
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get("CAPSULE_ROOT", "."))
OUT = ROOT / "results"
VERIFY_DIR = OUT / "manual_verify_pairs"
VERIFY_DIR.mkdir(exist_ok=True)


def load_kvasir_index():
    idx = {}
    kvasir_root = Path(os.environ.get("KVASIR_ROOT", ROOT / "data/kvasir_capsule/labelled_images"))
    if (kvasir_root / "labelled_images").is_dir():
        kvasir_root = kvasir_root / "labelled_images"
    for folder in kvasir_root.iterdir():
        if folder.is_dir():
            for fn in folder.glob("*.jpg"):
                idx[fn.name] = str(fn)
    return idx


def main():
    # Load annotated KVASIR-source pairs with minimum distance
    df = pd.read_csv(OUT / "cv2024_KVASIR_phash_annotated.csv")
    # Flag = pHash≤6 ∧ dHash≤6
    flagged = df[(df["min_phash_dist_to_kvasir"] <= 6) &
                 (df["min_dhash_dist_to_kvasir"] <= 6)]
    print(f"Total flagged: {len(flagged)} / {len(df)}")

    rng = random.Random(42)
    sample = flagged.sample(min(200, len(flagged)), random_state=42)
    kv_idx = load_kvasir_index()

    out_rows = []
    for i, (_, row) in enumerate(sample.iterrows()):
        cv_path = row["path"]
        kv_name = row["nearest_kvasir_file"]
        if kv_name not in kv_idx:
            continue
        kv_path = kv_idx[kv_name]
        try:
            cv_img = Image.open(cv_path).convert("RGB")
            kv_img = Image.open(kv_path).convert("RGB")
            w, h = 224, 224
            cv_img = cv_img.resize((w, h))
            kv_img = kv_img.resize((w, h))
            combo = Image.new("RGB", (2 * w + 10, h), (255, 255, 255))
            combo.paste(kv_img, (0, 0))
            combo.paste(cv_img, (w + 10, 0))
            combo.save(VERIFY_DIR / f"pair_{i:03d}_kv_{kv_name}.jpg")
            out_rows.append({
                "idx": i,
                "kvasir_file": kv_name,
                "cv2024_file": row["filename"],
                "cv2024_split": row["cv_split"],
                "phash_dist": int(row["min_phash_dist_to_kvasir"]),
                "dhash_dist": int(row["min_dhash_dist_to_kvasir"]),
            })
        except Exception as e:
            print(f"  {i}: error {e}")

    pd.DataFrame(out_rows).to_csv(VERIFY_DIR / "pair_metadata.csv", index=False)
    print(f"Saved {len(out_rows)} pair images to {VERIFY_DIR}")
    print(f"Summary CSV: {VERIFY_DIR}/pair_metadata.csv")
    print("\nManual inspection protocol:")
    print("  1. For each pair_*.jpg, confirm visually that the two images")
    print("     are the same frame (possibly re-encoded/cropped).")
    print("  2. Mark any disagreements in pair_metadata.csv as 'false_positive'.")
    print("  3. Report precision = (n - fp) / n in the paper.")


if __name__ == "__main__":
    main()
