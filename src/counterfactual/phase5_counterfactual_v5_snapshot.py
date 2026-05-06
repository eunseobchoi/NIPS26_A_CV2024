"""phase5_counterfactual_v5.py — v4 + per-source val stratification.

ADDED vs v4:
 - eval function returns per-source balanced accuracy (partition by Dataset column)
 - per-source per-class recall (nested dict)
 - Source partition enables LOSO interpretation (canonical: test on held-out-source-val)

UNCHANGED vs v4 (MD5 5deff74c... ca6b7cfa...):
 - AdamW persistence across epochs, param group at ep=3
 - Full seed determinism (random/np/torch/cuda/cudnn + worker_init)
 - backbone.eval() after m.train()
 - NaN guard
 - Provenance (script SHA256, CSV MD5)
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
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(os.environ.get("CAPSULE_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "results"
CV2024_ROOT = ROOT / "data/cv2024/Dataset"

sys.path.insert(0, str(ROOT / "src"))
import dataset_official as _ds
_ds.DATA_ROOT = ROOT / "data/kvasir_capsule/labelled_images"
_ds.SPLITS_DIR = ROOT / "data/official_splits"
from dataset_official import (
    KvasirCapsuleOfficial, OFFICIAL_CLASSES, NUM_CLASSES, LABEL_TO_FOLDER,
)

DEVICE = torch.device("cuda:0")

CV2024_CLASSES = [
    "Angioectasia", "Bleeding", "Erosion", "Erythema", "Foreign Body",
    "Lymphangiectasia", "Normal", "Polyp", "Ulcer", "Worms"
]
CV2024_SOURCES = ["KVASIR", "SEE-AI", "KID", "AIIMS"]

TF_EVAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
TF_TRAIN = TF_EVAL


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def set_determinism(seed):
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
        print(f"  [warn] use_deterministic_algorithms failed: {e}", flush=True)


def worker_init_fn(worker_id):
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    py_random.seed(seed)


class CV2024DS(torch.utils.data.Dataset):
    """CSV-based dataset with optional Dataset-source tracking."""
    def __init__(self, csv_path, classes=CV2024_CLASSES, tf=TF_TRAIN):
        df = pd.read_csv(csv_path) if str(csv_path).endswith(".csv") \
             else pd.read_excel(csv_path)
        label_cols = [c for c in df.columns if c in classes]
        if not label_cols:
            raise ValueError(f"No class columns in {csv_path}")
        df = df[df[label_cols].sum(axis=1) > 0].reset_index(drop=True)
        self.df = df
        self.classes = classes
        self.label_cols = label_cols
        self.tf = tf
        # Track Dataset column if present (for per-source eval)
        if "Dataset" in df.columns:
            self.sources = df["Dataset"].values
        else:
            self.sources = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = row["image_path"].replace("\\", "/")
        path = CV2024_ROOT / rel
        img = Image.open(path).convert("RGB")
        for i, c in enumerate(self.label_cols):
            if row[c] == 1:
                label = self.classes.index(c)
                break
        else:
            label = 0
        return self.tf(img), label, idx


class KvasirEvalDS(torch.utils.data.Dataset):
    KVASIR_TO_CV = {
        "Angiectasia": "Angioectasia",
        "Blood": "Bleeding",
        "Erosion": "Erosion",
        "Erythematous": "Erythema",
        "Foreign Bodies": "Foreign Body",
        "Lymphangiectasia": "Lymphangiectasia",
        "Normal": "Normal",
        "Pylorus": None,
        "Ileo-cecal valve": None,
        "Reduced Mucosal View": None,
        "Ulcer": "Ulcer",
    }

    def __init__(self):
        self.items = []
        with open(_ds.SPLITS_DIR / "split_1.csv") as f:
            for row in csv.DictReader(f):
                kv_lbl = row["label"]
                cv_lbl = self.KVASIR_TO_CV.get(kv_lbl)
                if cv_lbl is None or cv_lbl not in CV2024_CLASSES:
                    continue
                if kv_lbl not in LABEL_TO_FOLDER:
                    continue
                folder = LABEL_TO_FOLDER[kv_lbl]
                path = _ds.DATA_ROOT / folder / row["filename"]
                self.items.append((str(path), CV2024_CLASSES.index(cv_lbl)))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, lbl = self.items[idx]
        return TF_EVAL(Image.open(path).convert("RGB")), lbl, idx


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.lora_a = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


class DINOv2LoRA(nn.Module):
    def __init__(self, n_cls=10, r=8):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.lora_wrappers = nn.ModuleList()
        for blk in self.backbone.blocks[-4:]:
            w = LoRALinear(blk.attn.qkv, r=r)
            blk.attn.qkv = w
            self.lora_wrappers.append(w)
        self.head = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, 256),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_cls))

    def forward(self, x):
        return self.head(self.backbone(x))


def cbw(labels, n, beta=0.999):
    c = torch.bincount(labels, minlength=n).float()
    e = 1.0 - torch.pow(beta, c)
    e[c == 0] = 1.0
    w = (1.0 - beta) / e
    w = w / w.sum() * n
    w[c == 0] = 0
    return w


@torch.no_grad()
def ev(m, loader, classes=CV2024_CLASSES, dataset=None):
    """v5: returns also per-source metrics if dataset.sources is available."""
    m.eval()
    P, L, IDX = [], [], []
    for img, l, idx in loader:
        img = img.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lg = m(img)
        P.extend(lg.argmax(1).cpu().tolist())
        L.extend(l.tolist())
        IDX.extend(idx.tolist() if hasattr(idx, 'tolist') else idx)
    P, L = np.array(P), np.array(L)
    IDX = np.array(IDX)
    mj = int(np.bincount(L).argmax()) if len(L) else 0
    per_class = {}
    for i, c in enumerate(classes):
        mask = (L == i)
        if mask.sum() > 0:
            per_class[c] = {
                "recall": float((P[mask] == i).mean()),
                "n": int(mask.sum()),
            }
        else:
            per_class[c] = {"recall": None, "n": 0}
    out = {
        "acc": float((P == L).mean()),
        "bal_acc": float(balanced_accuracy_score(L, P)),
        "f1_macro": float(f1_score(L, P, average="macro", zero_division=0)),
        "null_acc": float((np.full_like(L, mj) == L).mean()),
        "n_test": len(L),
        "per_class": per_class,
    }
    # NEW in v5: per-source metrics if dataset has source info
    if dataset is not None and dataset.sources is not None:
        per_source = {}
        for src in CV2024_SOURCES:
            src_mask = np.array([s == src for s in dataset.sources[IDX]])
            if src_mask.sum() == 0:
                per_source[src] = {"n": 0, "bal_acc": None, "acc": None, "per_class": {}}
                continue
            P_s = P[src_mask]
            L_s = L[src_mask]
            if len(set(L_s)) < 2:
                # balanced_accuracy_score requires >=2 classes present
                ba = float((P_s == L_s).mean())
            else:
                ba = float(balanced_accuracy_score(L_s, P_s))
            pc_s = {}
            for i, c in enumerate(classes):
                cm = (L_s == i)
                if cm.sum() > 0:
                    pc_s[c] = {"recall": float((P_s[cm] == i).mean()), "n": int(cm.sum())}
            per_source[src] = {
                "n": int(src_mask.sum()),
                "bal_acc": ba,
                "acc": float((P_s == L_s).mean()),
                "per_class": pc_s,
            }
        out["per_source"] = per_source
    return out


def train_one_seed(train_csv, seed, args, orig_val_csv, dedup_val_csv):
    set_determinism(seed)
    tr = CV2024DS(train_csv, tf=TF_TRAIN)
    n = len(tr)
    labels = torch.tensor([tr.classes.index(
        [c for c in tr.label_cols if tr.df.iloc[i][c] == 1][0])
        for i in range(n)])
    w = cbw(labels, len(CV2024_CLASSES)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    g = torch.Generator()
    g.manual_seed(seed)
    ltr = DataLoader(tr, batch_size=args.batch, shuffle=True,
                     num_workers=6, pin_memory=True,
                     persistent_workers=True, drop_last=True,
                     worker_init_fn=worker_init_fn, generator=g)
    # v5: track sources for eval datasets
    orig_val = CV2024DS(orig_val_csv, tf=TF_EVAL)
    dedup_val = CV2024DS(dedup_val_csv, tf=TF_EVAL)
    kv_test = KvasirEvalDS()
    l_ov = DataLoader(orig_val, batch_size=args.batch*2, shuffle=False, num_workers=6)
    l_dv = DataLoader(dedup_val, batch_size=args.batch*2, shuffle=False, num_workers=6)
    l_kt = DataLoader(kv_test, batch_size=args.batch*2, shuffle=False, num_workers=6)
    print(f"  Train={n} OrigVal={len(orig_val)} DedupVal={len(dedup_val)} KvasirTest={len(kv_test)}",
          flush=True)

    model = DINOv2LoRA(n_cls=len(CV2024_CLASSES)).to(DEVICE)
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(model)
    else:
        m = model

    lora = [p for wr in model.lora_wrappers for p in [wr.lora_a, wr.lora_b]]
    head_params = list(model.head.parameters())

    opt = torch.optim.AdamW(
        [{"params": head_params, "lr": 3e-4}],
        weight_decay=1e-4,
    )
    lora_added = False

    scaler = torch.amp.GradScaler("cuda")
    history = []

    for ep in range(args.epochs):
        if ep >= 3 and not lora_added:
            opt.add_param_group({"params": lora, "lr": 1e-4, "weight_decay": 1e-4})
            lora_added = True
            print(f"  [seed={seed} ep={ep+1}] LoRA params added to optimizer", flush=True)

        m.train()
        model.backbone.eval()

        losses = []
        t0 = time.perf_counter()
        for step, (img, l, _) in enumerate(ltr):
            img = img.to(DEVICE, non_blocking=True)
            l = l.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lg = m(img)
                loss = loss_fn(lg, l)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at seed={seed} ep={ep+1} step={step}"
                )
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())

        met_ov = ev(m, l_ov, dataset=orig_val)
        met_dv = ev(m, l_dv, dataset=dedup_val)
        met_kt = ev(m, l_kt)
        dt = time.perf_counter() - t0
        print(f"  [seed={seed} ep={ep+1}] orig_val bal={met_ov['bal_acc']:.4f} "
              f"| dedup_val bal={met_dv['bal_acc']:.4f} "
              f"| kvasir_s1 bal={met_kt['bal_acc']:.4f} [{dt:.0f}s]",
              flush=True)
        history.append({
            "epoch": ep + 1,
            "loss": float(np.mean(losses)),
            "orig_val": met_ov,
            "dedup_val": met_dv,
            "kvasir_s1": met_kt,
            "train_s": float(dt),
        })
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--orig_val_csv",
                    default=str(CV2024_ROOT / "validation/validation_data.xlsx"))
    ap.add_argument("--dedup_val_csv", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no_batch128_assert", action="store_true")
    args = ap.parse_args()

    if not args.no_batch128_assert:
        assert args.batch == 128, (
            f"Expected batch=128 (canonical), got {args.batch}"
        )
    print(f"Args: {vars(args)}", flush=True)

    script_path = os.path.abspath(__file__)
    meta = {
        "script_path": script_path,
        "script_sha256": file_sha256(script_path),
        "script_md5": file_md5(script_path),
        "script_version": "v5 (per-source val)",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_csv_md5": file_md5(args.train_csv),
        "orig_val_path": args.orig_val_csv,
        "orig_val_md5": (
            file_md5(args.orig_val_csv) if os.path.exists(args.orig_val_csv) else None
        ),
        "dedup_val_csv": args.dedup_val_csv,
        "dedup_val_csv_md5": (
            file_md5(args.dedup_val_csv) if os.path.exists(args.dedup_val_csv) else None
        ),
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "determinism_enabled": True,
        "batch": args.batch,
    }
    print(f"Provenance: script_sha256={meta['script_sha256'][:16]}...", flush=True)

    results = {"args": vars(args), "meta": meta, "runs": []}
    for seed in args.seeds:
        print(f"\n=== seed={seed} ===", flush=True)
        history = train_one_seed(args.train_csv, seed, args,
                                 args.orig_val_csv, args.dedup_val_csv)
        results["runs"].append({
            "seed": seed,
            "history": history,
            "last": history[-1],
        })
        with open(OUT / args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)

    meta["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(OUT / args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {OUT / args.output}", flush=True)


if __name__ == "__main__":
    main()
