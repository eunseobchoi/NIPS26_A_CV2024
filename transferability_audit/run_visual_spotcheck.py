#!/usr/bin/env python3
"""
Visual spot-check of confirmed near-duplicate pairs:
  - 2 ISIC NCC-confirmed cross-source pairs (MSK ↔ ISIC-archive)
  - 5 random sample of CV2024 within-split pHash-exact pairs
  - 5 random sample of HyperKvasir intra-set pHash-exact pairs

Outputs results/visual_spotcheck.json (image_id, computed pixel hash,
NCC, file size, image dimensions) and a montage PNG for manual review.
"""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image


CV_PAIRS_CSV = ("/home/user/main/capsule_tta/submission_FINAL/artifacts/"
                "annotations/cv2024_KVASIR_internal_train_val_phash_exact_pairs.csv")
DATA_ROOT = "/home/user/main/capsule_tta/data/cv2024"  # CSV path includes /Dataset/ already
ISIC_DIR = Path("images/ISIC_2019_Training_Input")
HK_DIR = Path("hyperkvasir/labeled-images")
OUT_JSON = Path("results/visual_spotcheck.json")
OUT_PNG = Path("results/visual_spotcheck.png")


def load_g_256(path):
    img = Image.open(path).convert("L").resize((256, 256), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def load_thumb(path, size=128):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-9 else 0.0


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def main():
    rng = random.Random(42)
    pairs_to_check = []

    # ISIC 2 confirmed cross-source pairs
    isic_ncc = json.load(open("results/isic2019_ncc.json"))
    for ex in isic_ncc["cross_source_ncc"]["ISIC_archive__MSK"]["examples"]:
        pairs_to_check.append({
            "label": "ISIC cross-source NCC-confirmed",
            "a_path": ISIC_DIR / f"{ex['a']}.jpg",
            "b_path": ISIC_DIR / f"{ex['b']}.jpg",
            "claimed_joint": ex["joint"],
            "claimed_ncc": ex.get("ncc"),
        })

    # CV2024 5 random pHash-exact pairs
    with open(CV_PAIRS_CSV) as f:
        cv_rows = list(csv.DictReader(f))
    sampled = rng.sample(cv_rows, 5)
    for r in sampled:
        pairs_to_check.append({
            "label": "CV2024 within-split pHash-exact",
            "a_path": Path(r["val_path"].replace("<CV2024_ROOT>", DATA_ROOT)),
            "b_path": Path(r["train_path"].replace("<CV2024_ROOT>", DATA_ROOT)),
            "claimed_joint": 0,
            "claimed_ncc": None,
        })

    # HyperKvasir 5 random pHash-exact pairs (from existing audit json)
    hk_audit = json.load(open("results/hyperkvasir_audit.json"))
    # We don't have exact pairs in JSON, recompute from the CSV
    hk_rows = []
    with open("results/hyperkvasir_phash_annotated.csv") as f:
        for r in csv.DictReader(f):
            hk_rows.append(r)
    phash_to_ids = {}
    for r in hk_rows:
        phash_to_ids.setdefault(r["phash"], []).append(r["image_id"])
    exact_groups = [ids for h, ids in phash_to_ids.items() if len(ids) > 1]
    rng.shuffle(exact_groups)
    for ids in exact_groups[:5]:
        a, b = ids[0], ids[1]
        pairs_to_check.append({
            "label": "HyperKvasir intra-set pHash-exact",
            "a_path": HK_DIR / a,
            "b_path": HK_DIR / b,
            "claimed_joint": 0,
            "claimed_ncc": None,
        })

    # Verify each pair
    results = []
    thumbs = []
    for p in pairs_to_check:
        ap, bp = p["a_path"], p["b_path"]
        if not ap.exists() or not bp.exists():
            results.append({**p, "skipped": "missing image"})
            continue
        a_arr = load_g_256(ap)
        b_arr = load_g_256(bp)
        n = ncc(a_arr, b_arr)
        a_md5 = md5(ap)
        b_md5 = md5(bp)
        a_thumb = load_thumb(ap, 96)
        b_thumb = load_thumb(bp, 96)
        thumbs.append(np.concatenate([a_thumb, b_thumb], axis=1))
        results.append({
            "label": p["label"],
            "a": str(ap.name),
            "b": str(bp.name),
            "claimed_joint": p["claimed_joint"],
            "claimed_ncc": p["claimed_ncc"],
            "computed_ncc_256gray": round(n, 6),
            "a_md5": a_md5,
            "b_md5": b_md5,
            "byte_identical": a_md5 == b_md5,
            "a_size": ap.stat().st_size,
            "b_size": bp.stat().st_size,
        })

    # Save montage (one row per pair, A on left, B on right)
    if thumbs:
        h, w = thumbs[0].shape[:2]
        montage = np.full((h * len(thumbs) + (len(thumbs)-1)*4, w, 3),
                          255, dtype=np.uint8)
        for i, t in enumerate(thumbs):
            y = i * (h + 4)
            montage[y:y+h] = t
        Image.fromarray(montage).save(OUT_PNG)

    # Coerce any remaining Path objects to str
    def _conv(o):
        if hasattr(o, "as_posix"):
            return o.as_posix()
        return str(o)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"n_pairs_checked": len(results), "pairs": results},
                  f, indent=2, default=_conv)

    print(f"\nSummary:", file=sys.stderr)
    print(f"  pairs checked: {len(results)}", file=sys.stderr)
    bi_count = sum(1 for r in results if r.get("byte_identical"))
    ge99 = sum(1 for r in results if r.get("computed_ncc_256gray", 0) >= 0.99)
    print(f"  byte-identical: {bi_count}", file=sys.stderr)
    print(f"  NCC>=0.99: {ge99}", file=sys.stderr)
    print(f"  Wrote {OUT_JSON} and {OUT_PNG}", file=sys.stderr)


if __name__ == "__main__":
    main()
