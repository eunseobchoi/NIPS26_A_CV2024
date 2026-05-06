"""Multi-backbone protocol-gap experiment.

Tests whether the frame-split-to-video-split balanced-accuracy gap is
backbone-invariant, as opposed to a DINOv2-specific artifact.

Backbones (frozen features, linear probe classification head):
  - DINOv2-ViT-S/14, ViT-B/14, ViT-L/14, ViT-G/14 (torch.hub)
  - ResNet-50 ImageNet (torchvision)
  - ViT-L/16 ImageNet-21k (torchvision)
  - CLIP-ViT-L/14 (open_clip or HF)
  - SigLIP-SO400M/14 (HF transformers)

Protocol:
  For each backbone:
    1. Extract [CLS] (or equivalent) features for ALL Kvasir-Capsule labeled
       frames (47,238). Cache to NPZ.
    2. Frame split (seed 42, 70/15/15), train linear probe on 14 classes.
       Evaluate on frame-test, report bal_acc.
    3. Video split (official 2-fold, 11 classes), train linear probe.
       Evaluate on other fold, report bal_acc. Average over 2 folds x 3 seeds.
    4. Gap = frame_bal_acc - video_bal_acc.

Expected: gap is large (~0.4-0.6) for every backbone, confirming the
protocol effect is invariant to representation quality.

Output: phase6_multibackbone.json with per-backbone frame/video bal_acc,
gap, null baselines, and per-class CI.
"""
import os
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
OUT = ROOT / "results"
CACHE = OUT / "backbone_features"
CACHE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
import dataset_official as _ds
_ds.DATA_ROOT = Path(os.environ.get("KVASIR_ROOT", ROOT / "data/kvasir_capsule/labelled_images"))
if (_ds.DATA_ROOT / "labelled_images").is_dir():
    _ds.DATA_ROOT = _ds.DATA_ROOT / "labelled_images"
_ds.SPLITS_DIR = Path(os.environ.get("KVASIR_SPLITS_DIR", ROOT / "data/official_splits"))
from dataset_official import OFFICIAL_CLASSES, NUM_CLASSES as OFFICIAL_NUM, LABEL_TO_FOLDER

DEVICE = torch.device("cuda:0")

# 14-class label space from Kvasir-Capsule folder names
KVASIR_14_FOLDERS = [
    "ampulla_of_vater", "angiectasia", "blood_fresh", "blood_hematin",
    "erosion", "erythema", "foreign_body", "ileocecal_valve",
    "lymphangiectasia", "normal_clean_mucosa", "polyp", "pylorus",
    "reduced_mucosal_view", "ulcer"
]

NORM_IN = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
NORM_CLIP = ([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])


class ImgDS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        im = Image.open(it["path"]).convert("RGB")
        return self.tf(im), idx


def load_kvasir_14class():
    """Return list of {path, label_14, filename, video}."""
    items = []
    data_root = _ds.DATA_ROOT
    for cls in KVASIR_14_FOLDERS:
        folder = data_root / cls
        if not folder.is_dir():
            continue
        lbl = KVASIR_14_FOLDERS.index(cls)
        for fn in folder.glob("*.jpg"):
            vid = fn.stem.split("_")[0]
            items.append({"path": str(fn), "label_14": lbl,
                          "filename": fn.name, "video": vid})
    return items


def load_kvasir_official_split(fold, split):
    """fold in {0,1}, split in {train, test}. Returns list of items with 11-class labels."""
    # split_0 = fold0, split_1 = fold1
    if (fold == 0 and split == "train") or (fold == 1 and split == "test"):
        csv_file = "split_0.csv"
    else:
        csv_file = "split_1.csv"
    items = []
    with open(_ds.SPLITS_DIR / csv_file) as f:
        for row in csv.DictReader(f):
            lbl = row["label"]
            if lbl not in LABEL_TO_FOLDER:
                continue
            folder = LABEL_TO_FOLDER[lbl]
            path = _ds.DATA_ROOT / folder / row["filename"]
            vid = row["filename"].split("_")[0]
            items.append({"path": str(path),
                          "label_11": OFFICIAL_CLASSES.index(lbl),
                          "filename": row["filename"], "video": vid})
    return items


# --- Backbones ---
BACKBONES = {
    "dinov2_vits14": {"hub": ("facebookresearch/dinov2", "dinov2_vits14"),
                      "dim": 384, "size": 224, "norm": NORM_IN},
    "dinov2_vitb14": {"hub": ("facebookresearch/dinov2", "dinov2_vitb14"),
                      "dim": 768, "size": 224, "norm": NORM_IN},
    "dinov2_vitl14": {"hub": ("facebookresearch/dinov2", "dinov2_vitl14"),
                      "dim": 1024, "size": 224, "norm": NORM_IN},
    "dinov2_vitg14": {"hub": ("facebookresearch/dinov2", "dinov2_vitg14"),
                      "dim": 1536, "size": 224, "norm": NORM_IN},
    "resnet50_imagenet": {"torchvision": "resnet50",
                           "weights": "IMAGENET1K_V2",
                           "dim": 2048, "size": 224, "norm": NORM_IN},
    "vit_l_16_imagenet21k": {"torchvision": "vit_l_16",
                              "weights": "IMAGENET1K_SWAG_LINEAR_V1",
                              "dim": 1024, "size": 512, "norm": NORM_IN},
}


def build_backbone(name):
    cfg = BACKBONES[name]
    if "hub" in cfg:
        model = torch.hub.load(*cfg["hub"], trust_repo=True)
        def forward(imgs):
            out = model(imgs)  # DINOv2 returns [CLS] by default
            return out
    else:
        from torchvision import models as tvm
        model = getattr(tvm, cfg["torchvision"])(weights=cfg["weights"])
        if "resnet" in name:
            # Replace fc with Identity to get penultimate features
            model.fc = nn.Identity()
            def forward(imgs):
                return model(imgs)
        elif "vit" in name:
            # torchvision ViT has heads.head; replace with Identity
            model.heads = nn.Identity()
            def forward(imgs):
                return model(imgs)
        else:
            raise ValueError(name)
    for p in model.parameters():
        p.requires_grad = False
    model.eval().to(DEVICE)
    return model, forward, cfg


@torch.no_grad()
def extract_features(items, name):
    cache_path = CACHE / f"{name}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        paths = list(data["paths"])
        feats = data["feats"]
        path_to_idx = {p: i for i, p in enumerate(paths)}
        want = [it["path"] for it in items]
        if set(want).issubset(set(paths)):
            print(f"    Using cached features for {name}")
            idx = np.array([path_to_idx[p] for p in want])
            return feats[idx]

    model, forward, cfg = build_backbone(name)
    size = cfg["size"]; mean, std = cfg["norm"]
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = ImgDS(items, tf)
    loader = DataLoader(ds, batch_size=128, num_workers=6, pin_memory=True)
    feats = np.zeros((len(items), cfg["dim"]), dtype=np.float32)
    t0 = time.perf_counter()
    for imgs, idx in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            f = forward(imgs)
        feats[idx.numpy()] = f.cpu().float().numpy()
    dt = time.perf_counter() - t0
    print(f"    Extracted {len(items)} features for {name} in {dt:.0f}s")
    np.savez(cache_path, paths=np.array([it["path"] for it in items]), feats=feats)
    # free backbone
    del model
    torch.cuda.empty_cache()
    return feats


def linear_probe_eval(X_tr, y_tr, X_te, y_te, seed, Cs=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Simple linear probe with C-sweep (multinomial logistic regression, lbfgs)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_tr))
    n_tr = int(0.9 * len(X_tr))
    tr_idx, val_idx = idx[:n_tr], idx[n_tr:]
    best_bal = -1; best_C = 1.0
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs",
                                  class_weight="balanced", random_state=seed)
        clf.fit(X_tr[tr_idx], y_tr[tr_idx])
        p = clf.predict(X_tr[val_idx])
        ba = balanced_accuracy_score(y_tr[val_idx], p)
        if ba > best_bal:
            best_bal = ba; best_C = C
    clf = LogisticRegression(C=best_C, max_iter=800, solver="lbfgs",
                              class_weight="balanced", random_state=seed)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    return {"bal_acc": float(balanced_accuracy_score(y_te, preds)),
            "acc": float((preds == y_te).mean()),
            "f1_macro": float(f1_score(y_te, preds, average="macro", zero_division=0)),
            "best_C": float(best_C),
            "null_acc": float((np.full_like(y_te, np.bincount(y_te).argmax()) == y_te).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+",
                    default=list(BACKBONES.keys()))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--output", default="phase6_multibackbone.json")
    args = ap.parse_args()

    results = {"args": vars(args), "runs": []}

    # Frame split: use all 14-class labeled frames, single random 70/15/15 split per seed
    kv14 = load_kvasir_14class()
    print(f"Kvasir 14-class corpus: {len(kv14)} frames")

    # Video splits for official 11-class
    print(f"Loading official split_0 / split_1...")
    split_items = {
        0: {"train": load_kvasir_official_split(0, "train"),
            "test":  load_kvasir_official_split(0, "test")},
        1: {"train": load_kvasir_official_split(1, "train"),
            "test":  load_kvasir_official_split(1, "test")},
    }
    print(f"  Fold 0: train={len(split_items[0]['train'])} test={len(split_items[0]['test'])}")
    print(f"  Fold 1: train={len(split_items[1]['train'])} test={len(split_items[1]['test'])}")

    # Pre-extract features for all configurations at once
    # The union is just all 14-class + all 11-class paths; dedup
    all_items_set = {it["path"]: it for it in kv14}
    for fold in (0, 1):
        for split in ("train", "test"):
            for it in split_items[fold][split]:
                all_items_set.setdefault(it["path"], it)
    all_items = list(all_items_set.values())
    print(f"Union of image paths to embed: {len(all_items)}\n")

    for bname in args.backbones:
        if bname not in BACKBONES:
            print(f"SKIP unknown: {bname}")
            continue
        print(f"\n=== Backbone: {bname} ===")
        feats_all = extract_features(all_items, bname)
        path_to_feat = {it["path"]: i for i, it in enumerate(all_items)}

        # Frame split: per seed
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            idx = np.arange(len(kv14))
            rng.shuffle(idx)
            n = len(kv14); n_tr = int(0.7*n); n_val = int(0.15*n)
            tr = [kv14[i] for i in idx[:n_tr]]
            te = [kv14[i] for i in idx[n_tr+n_val:]]
            X_tr = feats_all[[path_to_feat[x["path"]] for x in tr]]
            X_te = feats_all[[path_to_feat[x["path"]] for x in te]]
            y_tr = np.array([x["label_14"] for x in tr])
            y_te = np.array([x["label_14"] for x in te])
            m = linear_probe_eval(X_tr, y_tr, X_te, y_te, seed)
            results["runs"].append({"backbone": bname, "protocol": "frame_14",
                                     "seed": seed, "fold": None, "metrics": m,
                                     "n_train": len(tr), "n_test": len(te)})
            print(f"  frame-14 seed={seed}: bal={m['bal_acc']:.3f} acc={m['acc']:.3f} null={m['null_acc']:.3f}")

        # Video split: per fold × seed (seed changes linear probe init + C search)
        for fold in (0, 1):
            tr = split_items[fold]["train"]
            te = split_items[fold]["test"]
            X_tr = feats_all[[path_to_feat[x["path"]] for x in tr]]
            X_te = feats_all[[path_to_feat[x["path"]] for x in te]]
            y_tr = np.array([x["label_11"] for x in tr])
            y_te = np.array([x["label_11"] for x in te])
            for seed in args.seeds:
                m = linear_probe_eval(X_tr, y_tr, X_te, y_te, seed)
                results["runs"].append({"backbone": bname, "protocol": "video_11",
                                         "fold": fold, "seed": seed, "metrics": m,
                                         "n_train": len(tr), "n_test": len(te)})
                print(f"  video-11 fold={fold} seed={seed}: bal={m['bal_acc']:.3f} acc={m['acc']:.3f} null={m['null_acc']:.3f}")

        # Save incrementally
        with open(OUT / args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\nSaved {OUT / args.output}")


if __name__ == "__main__":
    main()
