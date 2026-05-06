"""DINOv2 semantic cosine-similarity audit for CV2024 <-> Kvasir-Capsule pairs.

Rationale:
  pHash/dHash are perceptual but low-dimensional. To ground the near-duplicate
  claim in semantic feature space we also compute cosine similarity of
  DINOv2 ViT-L/14 CLS embeddings for every flagged (CV2024, Kvasir) pair.

Inputs (paths relative to the repo root unless absolute):
  --cv-kvasir-ann   CSV of CV2024-KVASIR files with pHash attribution
                    (produced by 01_phash_dhash_audit.py). Must contain
                    columns ``path``, ``nearest_kvasir_file``,
                    ``min_phash_dist_to_kvasir`` and
                    ``min_dhash_dist_to_kvasir``.
  --cv-nonk-anns    Same schema, one per non-KVASIR CV2024 source
                    (SEE-AI / KID / AIIMS). May be empty.
  --kvasir-root     Root directory of Kvasir-Capsule labelled_images (14
                    class subfolders containing the referenced jpg files).
  --cv2024-root     Root directory of the CV2024 Dataset. Released annotation
                    CSVs use ``<CV2024_ROOT>/Dataset/...`` placeholders;
                    this argument resolves those placeholders at runtime.

Selection rule (joint Hamming band, matches sibling audits 01/03):
    flagged <=> min_phash_dist_to_kvasir <= --phash-max (default 6)
              AND min_dhash_dist_to_kvasir <= --dhash-max (default 6)
              AND nearest_kvasir_file is not null.

Feature model:
  facebookresearch/dinov2 ViT-L/14 (via torch.hub), CLS token, 1024-D.
  Preprocessing matches Meta's official recipe: Resize((224,224)) +
  ImageNet mean/std normalization. Features are L2-normalized; pairwise
  similarity is the dot product of the L2-normalized embeddings
  (equivalent to cosine similarity).

Random non-matching baseline:
  ``--random-neg`` pairs of (CV2024-KVASIR file, randomly drawn Kvasir
  file where the drawn file is NOT the pHash-matched neighbor). Seed
  fixed to ``--seed`` (default 42) for determinism.

Output schema (matches results/dinov2_cossim_audit.json):
  {
    "args": {...},
    "kvasir": { "n", "mean", "median", "p25", "p75", "min",
                "pct_ge_099", "pct_ge_095", "pct_ge_090",
                "by_phash_band": { "0-0": {...}, "1-2": {...}, "3-4": {...} } },
    "nonk":   { "<source>": { "n", "values": [...], "mean", "max" } },
    "random_neg": { "n", "mean", "median", "p95", "values": [...] }
  }

Determinism:
  Given identical inputs, the same ``--seed``, identical DINOv2 weights and
  identical ``--batch``, the output JSON is stable up to floating-point
  noise from CUDA kernel non-determinism. The script seeds ``random``,
  ``numpy`` and ``torch`` and disables cuDNN benchmarking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(os.environ.get("CAPSULE_ROOT", "."))

# DINOv2 / ImageNet preprocessing (matches facebookresearch/dinov2 official recipe).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PathDataset(Dataset):
    """Loads RGB jpg images from a list of filesystem paths."""

    def __init__(self, paths: List[str]) -> None:
        self.paths = list(paths)
        self.transform = EVAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), idx


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(
    paths: List[str],
    backbone: torch.nn.Module,
    device: str,
    batch: int,
    num_workers: int,
    tag: str,
) -> torch.Tensor:
    """Return L2-normalized (N, 1024) CLS embeddings for paths (CPU tensor)."""
    ds = PathDataset(paths)
    loader = DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    feats = torch.empty(len(paths), 1024, dtype=torch.float32)
    backbone.eval()
    t0 = time.perf_counter()
    seen = 0
    for imgs, idxs in loader:
        imgs = imgs.to(device, non_blocking=True)
        f = backbone(imgs)
        f = F.normalize(f, dim=-1)
        feats[idxs] = f.cpu().float()
        seen += imgs.size(0)
        if seen % (batch * 20) == 0 or seen == len(paths):
            dt = time.perf_counter() - t0
            print(
                f"  [{tag}] {seen}/{len(paths)}  {seen/max(dt,1e-6):.0f} img/s",
                flush=True,
            )
    return feats


def build_kvasir_index(kvasir_root: Path) -> Dict[str, str]:
    """Map basename (unique across labelled_images/) -> absolute path."""
    idx: Dict[str, str] = {}
    for cls_dir in sorted(p for p in kvasir_root.iterdir() if p.is_dir()):
        for fn in sorted(cls_dir.glob("*.jpg")):
            idx.setdefault(fn.name, str(fn))
    return idx


def file_sha256(path: Path) -> str | None:
    """Return SHA-256 for a file, or None when an optional input is absent."""
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_cv2024_root(path: Path) -> Path:
    """Return the concrete CV2024 Dataset root regardless of parent/root input."""
    if (path / "Dataset").is_dir():
        return path / "Dataset"
    return path


def resolve_cv2024_image_path(raw: object, cv2024_root: Path) -> str:
    """Resolve released CSV path placeholders to a local CV2024 image path."""
    s = str(raw).replace("\\", "/")
    marker = "<CV2024_ROOT>"
    if s.startswith(marker):
        rel = s[len(marker):].lstrip("/")
        if rel.startswith("Dataset/"):
            rel = rel[len("Dataset/"):]
        return str(cv2024_root / rel)
    p = Path(s)
    return str(p if p.is_absolute() else ROOT / p)


def provenance_meta(
    args: argparse.Namespace,
    cv_kvasir_csv: Path,
    non_k_csvs: List[Path],
    cv2024_root: Path,
    kvasir_root: Path,
) -> Dict[str, object]:
    script_rel = "src/audit/dinov2_cossim_audit.py"
    script_path = ROOT / script_rel
    input_csvs = [cv_kvasir_csv] + non_k_csvs
    return {
        "script_path": script_rel,
        "script_sha256": file_sha256(script_path),
        "input_sha256": {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): file_sha256(p)
                         for p in input_csvs},
        "resolved_cv2024_root": str(cv2024_root),
        "resolved_kvasir_root": str(kvasir_root),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(args.device),
        "gpu_name": (
            torch.cuda.get_device_name(args.device)
            if torch.cuda.is_available() and str(args.device).startswith("cuda")
            else "cpu"
        ),
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": None, "median": None, "p25": None, "p75": None, "min": None,
            "pct_ge_099": None, "pct_ge_095": None, "pct_ge_090": None,
        }
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "min": float(values.min()),
        "pct_ge_099": float((values >= 0.99).mean()),
        "pct_ge_095": float((values >= 0.95).mean()),
        "pct_ge_090": float((values >= 0.90).mean()),
    }


def band_summary(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"n": 0, "mean": None, "pct_ge_095": None, "pct_ge_099": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "pct_ge_095": float((values >= 0.95).mean()),
        "pct_ge_099": float((values >= 0.99).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cv-kvasir-ann",
                   default="artifacts/annotations/cv2024_KVASIR_phash_annotated.csv")
    p.add_argument("--kvasir-root",
                   default="data/kvasir_capsule/labelled_images")
    p.add_argument("--cv2024-root",
                   default=os.environ.get("CV2024_ROOT", "data/cv2024/Dataset"))
    p.add_argument("--cv-nonk-anns", nargs="*",
                   default=[
                       "artifacts/annotations/cv2024_SEE-AI_phash_annotated.csv",
                       "artifacts/annotations/cv2024_KID_phash_annotated.csv",
                       "artifacts/annotations/cv2024_AIIMS_phash_annotated.csv",
                   ])
    p.add_argument("--output", default="results/dinov2_cossim_audit.json")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--random-neg", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phash-max", type=int, default=6)
    p.add_argument("--dhash-max", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs and counts only; do not load DINOv2.")
    return p.parse_args()


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> int:
    args = parse_args()
    seed_all(args.seed)

    cv_kvasir_csv = resolve(args.cv_kvasir_ann)
    cv2024_root = normalize_cv2024_root(resolve(args.cv2024_root))
    kvasir_root = resolve(args.kvasir_root)
    output_path = resolve(args.output)
    non_k_csvs = [resolve(c) for c in args.cv_nonk_anns]
    meta = provenance_meta(args, cv_kvasir_csv, non_k_csvs, cv2024_root, kvasir_root)

    if not cv_kvasir_csv.exists():
        print(f"[error] missing CV2024-KVASIR annotation CSV: {cv_kvasir_csv}")
        return 1
    if not kvasir_root.exists():
        print(f"[error] missing Kvasir root: {kvasir_root}")
        return 1

    print(f"Loading Kvasir file index from {kvasir_root} ...")
    kv_idx = build_kvasir_index(kvasir_root)
    print(f"  {len(kv_idx)} Kvasir-Capsule labelled images indexed.")

    # Flagged KVASIR pairs ---------------------------------------------------
    df_k = pd.read_csv(cv_kvasir_csv)
    print(f"CV2024-KVASIR annotations: {len(df_k)} rows")
    mask_k = (
        (df_k["min_phash_dist_to_kvasir"] <= args.phash_max)
        & (df_k["min_dhash_dist_to_kvasir"] <= args.dhash_max)
        & df_k["nearest_kvasir_file"].notna()
    )
    flagged = df_k[mask_k].reset_index(drop=True)
    print(
        f"  flagged pairs (pHash<={args.phash_max} AND dHash<={args.dhash_max}): "
        f"{len(flagged)}"
    )
    kv_paths_all = [kv_idx.get(fn) for fn in flagged["nearest_kvasir_file"]]
    keep = [p is not None for p in kv_paths_all]
    if not all(keep):
        print(f"  WARNING: dropping {sum(1 for k in keep if not k)} rows with "
              "missing Kvasir file.")
    flagged = flagged.loc[keep].reset_index(drop=True)
    kv_paths = [p for p in kv_paths_all if p is not None]
    cv_paths = [resolve_cv2024_image_path(p, cv2024_root)
                for p in flagged["path"].tolist()]

    bands_arr = flagged["min_phash_dist_to_kvasir"].to_numpy()
    band_defs = [("0-0", (0, 0)), ("1-2", (1, 2)), ("3-4", (3, 4))]
    for label, (lo, hi) in band_defs:
        sel = (bands_arr >= lo) & (bands_arr <= hi)
        print(f"    band {label}: n={int(sel.sum())}")

    # Flagged non-KVASIR pairs ----------------------------------------------
    nonk_records: Dict[str, Dict[str, list]] = {}
    for csv_path in non_k_csvs:
        if not csv_path.exists():
            print(f"[warn] missing non-KVASIR CSV: {csv_path}")
            continue
        df_n = pd.read_csv(csv_path)
        if "cv_dataset" in df_n.columns and df_n["cv_dataset"].notna().any():
            source_name = str(df_n["cv_dataset"].dropna().iloc[0])
        else:
            source_name = csv_path.stem.replace("cv2024_", "").replace(
                "_phash_annotated", ""
            )
        mask_n = (
            (df_n["min_phash_dist_to_kvasir"] <= args.phash_max)
            & (df_n["min_dhash_dist_to_kvasir"] <= args.dhash_max)
            & df_n["nearest_kvasir_file"].notna()
        )
        fl_n = df_n[mask_n].reset_index(drop=True)
        kv_n = [kv_idx.get(fn) for fn in fl_n["nearest_kvasir_file"]]
        keep_n = [p is not None for p in kv_n]
        fl_n = fl_n.loc[keep_n].reset_index(drop=True)
        kv_n = [p for p in kv_n if p is not None]
        nonk_records[source_name] = {
            "cv_paths": [resolve_cv2024_image_path(p, cv2024_root)
                         for p in fl_n["path"].tolist()],
            "kv_paths": kv_n,
        }
        print(f"  {source_name}: {len(kv_n)} flagged pairs")

    # Random non-matching baseline sampling plan ----------------------------
    rng = random.Random(args.seed)
    all_kv_names = sorted(kv_idx.keys())
    if len(flagged) == 0:
        print("[error] no flagged pairs; cannot form random negatives.")
        return 1
    if len(flagged) < args.random_neg:
        print(f"[warn] random_neg ({args.random_neg}) > flagged "
              f"({len(flagged)}); sampling CV indices with replacement.")
    cv_indices = list(range(len(flagged)))
    neg_cv: List[str] = []
    neg_kv: List[str] = []
    for _ in range(args.random_neg):
        i = rng.choice(cv_indices)
        cv_path = cv_paths[i]
        matched_name = flagged["nearest_kvasir_file"].iloc[i]
        while True:
            name = rng.choice(all_kv_names)
            if name != matched_name:
                break
        neg_cv.append(cv_path)
        neg_kv.append(kv_idx[name])

    # Dry-run short-circuit --------------------------------------------------
    if args.dry_run:
        print("\n[dry-run] skipping DINOv2 feature extraction; counts only.")
        dry = {
            "args": {
                "cv_kvasir_ann": str(args.cv_kvasir_ann),
                "cv2024_root": str(args.cv2024_root),
                "kvasir_root": str(args.kvasir_root),
                "cv_nonk_anns": list(args.cv_nonk_anns),
                "output": str(args.output),
                "batch": int(args.batch),
                "random_neg": int(args.random_neg),
                "seed": int(args.seed),
                "phash_max": int(args.phash_max),
                "dhash_max": int(args.dhash_max),
                "model": "facebookresearch/dinov2:dinov2_vitl14",
                "preprocess": "Resize((224,224)) + ImageNet mean/std",
                "dry_run": True,
            },
            "meta": {**meta, "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
            "counts": {
                "kvasir_flagged": int(len(flagged)),
                "kvasir_band_0_0": int(((bands_arr >= 0) & (bands_arr <= 0)).sum()),
                "kvasir_band_1_2": int(((bands_arr >= 1) & (bands_arr <= 2)).sum()),
                "kvasir_band_3_4": int(((bands_arr >= 3) & (bands_arr <= 4)).sum()),
                "nonk": {src: len(rec["kv_paths"])
                         for src, rec in nonk_records.items()},
                "random_neg_planned": int(args.random_neg),
            },
            "note": (
                "Dry-run output. No cosine similarities computed. Run without "
                "--dry-run on a CUDA host with DINOv2 weights available."
            ),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dry_out = output_path.with_suffix(".dryrun.json")
        with open(dry_out, "w") as f:
            json.dump(dry, f, indent=2)
        print(f"Wrote {dry_out}")
        return 0

    # Full run: load DINOv2 and extract features ----------------------------
    print("\nLoading DINOv2 backbone facebookresearch/dinov2:dinov2_vitl14 ...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    backbone = backbone.to(args.device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    cv_needed = list(cv_paths)
    for rec in nonk_records.values():
        cv_needed.extend(rec["cv_paths"])
    cv_needed.extend(neg_cv)
    uniq_cv = sorted(set(cv_needed))
    cv_to_row = {p: i for i, p in enumerate(uniq_cv)}

    kv_needed = list(kv_paths)
    for rec in nonk_records.values():
        kv_needed.extend(rec["kv_paths"])
    kv_needed.extend(neg_kv)
    uniq_kv = sorted(set(kv_needed))
    kv_to_row = {p: i for i, p in enumerate(uniq_kv)}

    print(f"Unique CV images to encode: {len(uniq_cv)}")
    print(f"Unique Kvasir images to encode: {len(uniq_kv)}")

    feats_cv = extract_features(
        uniq_cv, backbone, args.device, args.batch, args.num_workers, "cv"
    )
    feats_kv = extract_features(
        uniq_kv, backbone, args.device, args.batch, args.num_workers, "kv"
    )

    def cos_for(cv_list: List[str], kv_list: List[str]) -> np.ndarray:
        if not cv_list:
            return np.array([], dtype=np.float32)
        a = feats_cv[[cv_to_row[p] for p in cv_list]]
        b = feats_kv[[kv_to_row[p] for p in kv_list]]
        return (a * b).sum(dim=-1).numpy().astype(np.float32)

    k_sims = cos_for(cv_paths, kv_paths)
    by_band: Dict[str, Dict[str, float]] = {}
    for label, (lo, hi) in band_defs:
        sel = (bands_arr >= lo) & (bands_arr <= hi)
        by_band[label] = band_summary(k_sims[sel])
    kv_summary = summarize(k_sims)
    kv_summary["by_phash_band"] = by_band

    nonk_out: Dict[str, Dict[str, object]] = {}
    for src, rec in nonk_records.items():
        vals = cos_for(rec["cv_paths"], rec["kv_paths"])
        nonk_out[src] = {
            "n": int(vals.size),
            "values": [float(v) for v in vals.tolist()],
            "mean": float(vals.mean()) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
        }

    neg_vals = cos_for(neg_cv, neg_kv)
    random_neg_out = {
        "n": int(neg_vals.size),
        "mean": float(neg_vals.mean()),
        "median": float(np.median(neg_vals)),
        "p95": float(np.quantile(neg_vals, 0.95)),
        "values": [float(v) for v in neg_vals.tolist()],
    }

    result = {
        "args": {
            "cv_kvasir_ann": str(args.cv_kvasir_ann),
            "cv2024_root": str(args.cv2024_root),
            "kvasir_root": str(args.kvasir_root),
            "cv_nonk_anns": list(args.cv_nonk_anns),
            "output": str(args.output),
            "batch": int(args.batch),
            "random_neg": int(args.random_neg),
            "seed": int(args.seed),
            "phash_max": int(args.phash_max),
            "dhash_max": int(args.dhash_max),
            "model": "facebookresearch/dinov2:dinov2_vitl14",
            "preprocess": "Resize((224,224)) + ImageNet mean/std",
        },
        "meta": {**meta, "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        "kvasir": kv_summary,
        "nonk": nonk_out,
        "random_neg": random_neg_out,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {output_path}")

    print("\n=== KVASIR pairs ===")
    print(f"  n={kv_summary['n']}  mean={kv_summary['mean']:.4f}  "
          f"pct>=0.90={100*kv_summary['pct_ge_090']:.2f}%")
    print("\n=== Random non-matching baseline ===")
    print(f"  n={random_neg_out['n']}  mean={random_neg_out['mean']:.4f}  "
          f"pct>=0.90={100*((neg_vals>=0.9).mean()):.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
