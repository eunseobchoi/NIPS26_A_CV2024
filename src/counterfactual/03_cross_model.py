"""Cross-model counterfactual retraining.

Supports frozen-backbone robustness checks:
  - dinov2_vitl14 (baseline; already in main paper)
  - dinov2_vits14 (small, 21M params, 384-dim)
  - dinov2_vitb14 (smaller, 86M params, 768-dim)
  - resnet50     (ImageNet-supervised, 2048-dim)
  - convnext_tiny (ImageNet-supervised, 768-dim)

All frozen backbones; train a classification head (LoRA-r=8 on
DINOv2 attn blocks; linear head for ResNet-50). The split construction,
optimizer schedule, and final-epoch balanced-accuracy estimand match the
phase5 counterfactual path. Preprocessing is backbone-standardized
resize/crop rather than byte-identical to the DINOv2-L direct-resize
training path, so these rows are robustness/accounting checks.

Usage:
  python phase5_cross_model.py --backbone dinov2_vitb14 \
      --train_csv results/cv2024_training_dedup_le6.csv \
      --dedup_val_csv results/cv2024_validation_dedup_le6.csv \
      --seeds 0 1 --epochs 10 --batch 128 \
      --output phase5_counterfactual_dinov2B_le6_seeds01.json
"""
import argparse
import csv
import hashlib
import json
import os
import random as py_random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

# Resolve ROOT: env var > script's parent containing 'src/' > script's parent
def _resolve_root():
    if "CAPSULE_ROOT" in os.environ:
        return Path(os.environ["CAPSULE_ROOT"]).resolve()
    if "CAPSULE_TTA_ROOT" in os.environ:
        return Path(os.environ["CAPSULE_TTA_ROOT"]).resolve()
    here = Path(__file__).resolve()
    for cand in (here.parent, *here.parents):
        if (cand / "src" / "dataset_official.py").exists() and (cand / "scripts").is_dir():
            return cand
    return Path.cwd()
ROOT = _resolve_root()
ARTIFACT_ROOT = Path(os.environ.get("CAPSULE_ARTIFACT_ROOT", ROOT))
OUT = ARTIFACT_ROOT / "results"
CV2024_ROOT = Path(os.environ.get("CV2024_ROOT", ROOT / "data/cv2024/Dataset"))
if (CV2024_ROOT / "Dataset").is_dir():
    CV2024_ROOT = CV2024_ROOT / "Dataset"

# dataset_official may live in ROOT/src/ or ROOT/
for _p in (ROOT, ROOT / "src"):
    if (_p / "dataset_official.py").exists():
        sys.path.insert(0, str(_p))
        break
import dataset_official as _ds
_ds.DATA_ROOT = Path(os.environ.get("KVASIR_ROOT", ROOT / "data/kvasir_capsule/labelled_images"))
if (_ds.DATA_ROOT / "labelled_images").is_dir():
    _ds.DATA_ROOT = _ds.DATA_ROOT / "labelled_images"
_ds.SPLITS_DIR = Path(os.environ.get("KVASIR_SPLITS_DIR", ROOT / "data/official_splits"))
from dataset_official import (
    KvasirCapsuleOfficial, OFFICIAL_CLASSES, NUM_CLASSES, LABEL_TO_FOLDER,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CV2024_CLASSES = [
    "Angioectasia", "Bleeding", "Erosion", "Erythema",
    "Foreign Body", "Lymphangiectasia", "Normal", "Polyp", "Ulcer", "Worms",
]


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_input_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (ROOT / p).resolve()


def scrub_path(value):
    if not isinstance(value, str):
        return value
    out = norm_path(value) if "norm_path" in globals() else value.replace("\\", "/")
    replacements = [
        ("<CAPSULE_ARTIFACT_ROOT>", ARTIFACT_ROOT),
        ("<CAPSULE_ROOT>", ROOT),
        ("<CV2024_ROOT>", CV2024_ROOT),
        ("<KVASIR_ROOT>", _ds.DATA_ROOT),
        ("<KVASIR_SPLITS_DIR>", _ds.SPLITS_DIR),
    ]
    for token, path in sorted(replacements, key=lambda kv: len(str(kv[1])), reverse=True):
        path_s = str(path.resolve()).replace("\\", "/")
        out = out.replace(path_s, token)
    return out


def scrub_obj(obj):
    if isinstance(obj, dict):
        return {k: scrub_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return [scrub_obj(v) for v in obj]
    if isinstance(obj, str):
        return scrub_path(obj)
    return obj


def set_determinism(seed: int) -> None:
    py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"  [warn] deterministic algorithms unavailable: {e}", flush=True)


def worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    py_random.seed(seed)

# === Preprocessing ===
TF_DINOV2 = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
TF_RESNET = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_dataset_class(tf):
    """Return a Dataset class using given transform."""
    class CV2024DS(torch.utils.data.Dataset):
        def __init__(self, csv_or_xlsx):
            p = Path(csv_or_xlsx) if not Path(csv_or_xlsx).is_absolute() else Path(csv_or_xlsx)
            if not p.exists():
                p = ROOT / csv_or_xlsx
            if p.suffix in (".xlsx", ".xls"):
                df = pd.read_excel(p)
            else:
                df = pd.read_csv(p)
            self.items = []
            for _, row in df.iterrows():
                img_rel = row.get("image_path", row.get("filename"))
                if pd.isna(img_rel): continue
                rel = str(img_rel).replace("\\", "/")
                # Find label: one-hot among CV2024_CLASSES or Dataset
                onehot = [int(row.get(c, 0) or 0) for c in CV2024_CLASSES]
                if sum(onehot) == 0: continue
                lbl = onehot.index(max(onehot))
                path = CV2024_ROOT / rel if not rel.startswith("/") else Path(rel)
                # Try both capitalizations
                if not path.exists():
                    alt = CV2024_ROOT / rel.replace("training\\", "training/").replace("validation\\", "validation/")
                    if alt.exists(): path = alt
                self.items.append((str(path), lbl))

        def __len__(self): return len(self.items)

        def __getitem__(self, idx):
            path, lbl = self.items[idx]
            return tf(Image.open(path).convert("RGB")), lbl, idx
    return CV2024DS


class KvasirCapsuleAdapter(torch.utils.data.Dataset):
    """Wraps Kvasir-Capsule split_1, remaps labels to CV2024 10-class."""
    # Map 11-class Kvasir official labels → 10-class CV2024 labels
    KV_TO_CV = {
        "Normal": "Normal",
        "Blood": "Bleeding",
        "Angiectasia": "Angioectasia",
        "Erosion": "Erosion",
        "Erythematous": "Erythema",
        "Foreign Bodies": "Foreign Body",
        "Lymphangiectasia": "Lymphangiectasia",
        "Ulcer": "Ulcer",
        # Ileo-cecal valve, Pylorus, Reduced Mucosal View → no CV2024 equivalent; skip
    }

    def __init__(self, split_idx, tf):
        self.inner = KvasirCapsuleOfficial(split=f"split_{split_idx}")
        self.tf = tf
        self.items = []
        # inner.items: list of (path, class_idx, label_name)
        for path, _cls, kv_name in self.inner.items:
            if kv_name not in self.KV_TO_CV: continue
            cv_lbl = CV2024_CLASSES.index(self.KV_TO_CV[kv_name])
            self.items.append((str(path), cv_lbl))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, lbl = self.items[idx]
        return self.tf(Image.open(path).convert("RGB")), lbl, idx


# === Backbones ===
class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16.0):
        super().__init__()
        self.base = base
        for p in base.parameters(): p.requires_grad = False
        self.lora_a = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


class DINOv2LoRA(nn.Module):
    def __init__(self, backbone_name, n_cls=10, r=8):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        feat_dim = {"dinov2_vits14": 384, "dinov2_vitb14": 768,
                    "dinov2_vitl14": 1024, "dinov2_vitg14": 1536}[backbone_name]
        for p in self.backbone.parameters(): p.requires_grad = False
        self.lora_wrappers = nn.ModuleList()
        for blk in self.backbone.blocks[-4:]:
            w = LoRALinear(blk.attn.qkv, r=r); blk.attn.qkv = w
            self.lora_wrappers.append(w)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 256),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_cls))

    def forward(self, x): return self.head(self.backbone(x))


class ResNet50Frozen(nn.Module):
    """ImageNet-supervised ResNet-50, frozen backbone, linear-probe head."""
    def __init__(self, n_cls=10):
        super().__init__()
        net = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        # Replace final FC with head
        for p in net.parameters(): p.requires_grad = False
        net.fc = nn.Identity()
        self.backbone = net  # outputs 2048
        self.head = nn.Sequential(
            nn.LayerNorm(2048), nn.Linear(2048, 256),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_cls))

    def forward(self, x): return self.head(self.backbone(x))


class ConvNeXtTinyFrozen(nn.Module):
    """ImageNet-supervised ConvNeXt-Tiny, frozen backbone, linear-probe head."""
    def __init__(self, n_cls=10):
        super().__init__()
        net = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        for p in net.parameters():
            p.requires_grad = False
        net.classifier = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 256),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_cls))

    def forward(self, x):
        feat = self.backbone(x)
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        return self.head(feat)


def build_model(backbone):
    if backbone.startswith("dinov2"):
        return DINOv2LoRA(backbone_name=backbone), TF_DINOV2
    elif backbone == "resnet50":
        return ResNet50Frozen(), TF_RESNET
    elif backbone == "convnext_tiny":
        return ConvNeXtTinyFrozen(), TF_RESNET
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


# === Training ===
def cbw(labels, n, beta=0.999):
    c = torch.bincount(labels, minlength=n).float()
    e = 1.0 - torch.pow(beta, c); e[c == 0] = 1.0
    w = (1.0 - beta) / e; w = w / w.sum() * n; w[c == 0] = 0
    return w


def eval_one(model, loader, n_cls):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(DEVICE, non_blocking=True)
            logits = model(x)
            preds.append(logits.argmax(-1).cpu())
            tgts.append(y)
    preds = torch.cat(preds).numpy()
    tgts = torch.cat(tgts).numpy()
    return {
        "acc": float((preds == tgts).mean()),
        "bal_acc": float(balanced_accuracy_score(tgts, preds)),
        "f1_macro": float(f1_score(tgts, preds, average="macro", zero_division=0)),
        "null_acc": float((tgts == tgts.mode() if hasattr(tgts, "mode") else tgts == np.bincount(tgts).argmax()).mean()),
        "n_test": len(tgts),
    }


def train_one_seed(backbone, train_csv, seed, args, orig_val_csv, dedup_val_csv):
    set_determinism(seed)
    model, tf = build_model(backbone)
    model = model.to(DEVICE)
    DS = build_dataset_class(tf)
    train = DS(train_csv)
    orig_val = DS(orig_val_csv)
    dedup_val = DS(dedup_val_csv)
    kv_test = KvasirCapsuleAdapter(split_idx=1, tf=tf)
    print(f"  Train={len(train)} OrigVal={len(orig_val)} DedupVal={len(dedup_val)} KvasirTest={len(kv_test)}", flush=True)

    g = torch.Generator()
    g.manual_seed(seed)
    l_tr = DataLoader(
        train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    l_ov = DataLoader(orig_val, batch_size=args.batch * 2, shuffle=False, num_workers=6)
    l_dv = DataLoader(dedup_val, batch_size=args.batch * 2, shuffle=False, num_workers=6)
    l_kt = DataLoader(kv_test, batch_size=args.batch * 2, shuffle=False, num_workers=6)

    # Class-balanced weights
    train_labels = torch.tensor([t[1] for t in train.items], dtype=torch.long)
    weights = cbw(train_labels, len(CV2024_CLASSES), beta=0.999).to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(l_tr))
    scaler = torch.cuda.amp.GradScaler()

    history = []
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        for x, y, _ in l_tr:
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16 if backbone == "resnet50" else torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits, y, weight=weights, label_smoothing=0.05)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()

        met_ov = eval_one(model, l_ov, len(CV2024_CLASSES))
        met_dv = eval_one(model, l_dv, len(CV2024_CLASSES))
        met_kt = eval_one(model, l_kt, len(CV2024_CLASSES))
        dt = time.perf_counter() - t0; t0 = time.perf_counter()
        print(f"  [bb={backbone} seed={seed} ep={ep+1}] orig_val bal={met_ov['bal_acc']:.4f} "
              f"| dedup_val bal={met_dv['bal_acc']:.4f} "
              f"| kvasir_s1 bal={met_kt['bal_acc']:.4f} [{dt:.0f}s]", flush=True)
        history.append({"epoch": ep + 1, "loss": float(loss.item()),
                        "orig_val": met_ov, "dedup_val": met_dv, "kvasir_s1": met_kt,
                        "train_s": dt})
    return {"seed": seed, "backbone": backbone, "history": history, "last": history[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True,
                    choices=[
                        "dinov2_vits14",
                        "dinov2_vitb14",
                        "dinov2_vitl14",
                        "dinov2_vitg14",
                        "resnet50",
                        "convnext_tiny",
                    ])
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--orig_val_csv", default=str(CV2024_ROOT / "validation" / "validation_data.xlsx"))
    ap.add_argument("--dedup_val_csv", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    print(f"Args: {vars(args)}")
    input_paths = {
        "train_csv": resolve_input_path(args.train_csv),
        "orig_val_csv": resolve_input_path(args.orig_val_csv),
        "dedup_val_csv": resolve_input_path(args.dedup_val_csv),
    }
    script_path = Path(__file__).resolve()
    meta = {
        "description": "Frozen-backbone robustness diagnostic; reports final-epoch balanced accuracy, not an official CV2024 leaderboard metric.",
        "script_path": scrub_path(str(script_path)),
        "script_sha256": file_sha256(script_path),
        "script_version": "cross-model-v2-provenance",
        "torch_version": torch.__version__,
        "torchvision_version": getattr(tvm, "__version__", None),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "determinism_enabled": True,
        "input_files": {
            key: {
                "path": scrub_path(str(path)),
                "sha256": file_sha256(path),
            }
            for key, path in input_paths.items()
        },
    }

    runs = []
    results = {"args": scrub_obj(vars(args)), "meta": meta, "runs": runs}
    for seed in args.seeds:
        print(f"\n=== {args.backbone} seed={seed} ===")
        r = train_one_seed(args.backbone, args.train_csv, seed, args,
                           args.orig_val_csv, args.dedup_val_csv)
        runs.append(r)
        # Save after each seed
        with open(OUT / args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {OUT / args.output}")
    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(OUT / args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
