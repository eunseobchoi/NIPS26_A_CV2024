"""Geometric robustness test for pHash threshold.

Applies synthetic transformations to 50 random Kvasir frames:
  - Center crop to 90%, 75%, 60%
  - Rotation by 2°, 5°, 10°
  - JPEG recompression at quality 50, 70, 90
  - Resize 128×128, 384×384 (up/down sample)

For each (frame × transform), computes pHash distance from the original.
Reports what fraction of transformed frames still fall within the
community threshold (≤6) — defends that our pHash audit catches
real-world re-encoded duplicates.
"""
import os
import io
import json
import random
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
KVASIR_ROOT = Path(os.environ.get("KVASIR_ROOT", ROOT / "data/kvasir_capsule/labelled_images"))
if (KVASIR_ROOT / "labelled_images").is_dir():
    KVASIR_ROOT = KVASIR_ROOT / "labelled_images"
OUT = ROOT / "results"


def center_crop(img, frac):
    w, h = img.size
    nw, nh = int(w * frac), int(h * frac)
    lx, ly = (w - nw) // 2, (h - nh) // 2
    return img.crop((lx, ly, lx + nw, ly + nh)).resize((w, h))


def rotate(img, deg):
    return img.rotate(deg, fillcolor=(0, 0, 0))


def jpeg_recompress(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize(img, size):
    return img.resize((size, size)).resize(img.size)


def hash_dist(im_a, im_b):
    return imagehash.phash(im_a) - imagehash.phash(im_b)


def dhash_dist(im_a, im_b):
    return imagehash.dhash(im_a) - imagehash.dhash(im_b)


def main():
    rng = random.Random(0)
    all_frames = list(KVASIR_ROOT.glob("*/*.jpg"))
    sample = rng.sample(all_frames, 50)
    print(f"Testing on 50 random Kvasir frames.")

    transforms = {
        "crop90": lambda im: center_crop(im, 0.90),
        "crop75": lambda im: center_crop(im, 0.75),
        "crop60": lambda im: center_crop(im, 0.60),
        "rot2":   lambda im: rotate(im, 2),
        "rot5":   lambda im: rotate(im, 5),
        "rot10":  lambda im: rotate(im, 10),
        "jpeg90": lambda im: jpeg_recompress(im, 90),
        "jpeg70": lambda im: jpeg_recompress(im, 70),
        "jpeg50": lambda im: jpeg_recompress(im, 50),
        "rsz128": lambda im: resize(im, 128),
        "rsz384": lambda im: resize(im, 384),
    }

    results = {}
    for name, fn in transforms.items():
        ph_dists = []
        dh_dists = []
        for path in sample:
            try:
                orig = Image.open(path).convert("RGB")
                trans = fn(orig)
                ph_dists.append(hash_dist(orig, trans))
                dh_dists.append(dhash_dist(orig, trans))
            except Exception:
                pass
        ph = np.array(ph_dists)
        dh = np.array(dh_dists)
        # Fraction within threshold ≤6
        frac_le6 = ((ph <= 6) & (dh <= 6)).mean() if len(ph) else 0
        frac_le2 = ((ph <= 2) & (dh <= 2)).mean() if len(ph) else 0
        results[name] = {
            "n": len(ph),
            "phash_mean": float(ph.mean()) if len(ph) else None,
            "phash_median": float(np.median(ph)) if len(ph) else None,
            "phash_max": int(ph.max()) if len(ph) else None,
            "dhash_mean": float(dh.mean()) if len(dh) else None,
            "dhash_median": float(np.median(dh)) if len(dh) else None,
            "dhash_max": int(dh.max()) if len(dh) else None,
            "frac_within_le6": float(frac_le6),
            "frac_within_le2": float(frac_le2),
        }
        print(f"  {name:<8}: pHash mean={ph.mean():.1f} median={np.median(ph):.0f} max={ph.max()} | "
              f"dHash mean={dh.mean():.1f} | caught(≤6)={100*frac_le6:.0f}% caught(≤2)={100*frac_le2:.0f}%")

    out = OUT / "phash_geometric_robustness.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
