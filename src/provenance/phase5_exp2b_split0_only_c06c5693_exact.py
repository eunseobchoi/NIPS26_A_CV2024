"""Exp 2b — Smedsrud split_0 -> split_1 oracle (no CV2024).

Purpose (Path B Exp 2b):
  Establish an "oracle" video-level baseline by training exclusively on
  Kvasir-Capsule split_0 (community-standard video-level split) and
  evaluating on split_1. This isolates "video-level VCE ceiling" from
  CV2024-specific artifacts. Combined with Exp 2a (le6 baseline ->
  split_1), it shows the low AIIMS/split_1 ceiling is a general
  property of video-level VCE rather than a CV2024 training artifact.

Training:
  - DINOv2-ViT-L/14 + LoRA r=8 (same architecture as Exp 1/2a)
  - 10 classes mapped to CV2024 label space (Pylorus/Ileo-cecal/
    Reduced-Mucosal-View rows dropped -- out-of-schema)
  - 10 epochs, batch=128, AdamW head LR 3e-4, LoRA LR 1e-4 from epoch 4
  - class-balanced weights (cbw beta=0.999), label_smoothing 0.05

Eval:
  - Kvasir-Capsule split_1 (filtered to 10 CV2024 classes)
  - Report bal_acc, macro-F1, per-class recall
"""
import argparse
import csv
import hashlib
import json
import os
import random as py_random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(os.environ.get("CAPSULE_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "results"

sys.path.insert(0, str(ROOT / "src"))
import dataset_official as _ds
_ds.DATA_ROOT = ROOT / "data/kvasir_capsule/labelled_images"
_ds.SPLITS_DIR = ROOT / "data/official_splits"
from dataset_official import LABEL_TO_FOLDER

DEVICE = torch.device("cuda:0")

CV2024_CLASSES = [
    "Angioectasia", "Bleeding", "Erosion", "Erythema", "Foreign Body",
    "Lymphangiectasia", "Normal", "Polyp", "Ulcer", "Worms"
]
NCV = len(CV2024_CLASSES)

KVASIR_TO_CV = {
    "Angiectasia": "Angioectasia",
    "Blood": "Bleeding",
    "Erosion": "Erosion",
    "Erythematous": "Erythema",
    "Foreign Bodies": "Foreign Body",
    "Lymphangiectasia": "Lymphangiectasia",
    "Normal": "Normal",
    "Ulcer": "Ulcer",
    "Pylorus": None,
    "Ileo-cecal valve": None,
    "Reduced Mucosal View": None,
}

TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
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


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    py_random.seed(seed)


def load_split_items(split_name):
    items = []
    with open(_ds.SPLITS_DIR / f"{split_name}.csv") as f:
        for row in csv.DictReader(f):
            kv_lbl = row["label"]
            cv_lbl = KVASIR_TO_CV.get(kv_lbl)
            if cv_lbl is None or cv_lbl not in CV2024_CLASSES:
                continue
            if kv_lbl not in LABEL_TO_FOLDER:
                continue
            folder = LABEL_TO_FOLDER[kv_lbl]
            path = _ds.DATA_ROOT / folder / row["filename"]
            items.append((str(path), CV2024_CLASSES.index(cv_lbl)))
    return items


class ItemDS(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, lbl = self.items[idx]
        return TF(Image.open(path).convert("RGB")), lbl, idx


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
    def __init__(self, n_cls=NCV, r=8):
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
def ev(m, loader, n_cls=NCV):
    m.eval()
    P, L = [], []
    for img, l, _ in loader:
        img = img.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lg = m(img)
        P.extend(lg.argmax(1).cpu().tolist())
        L.extend(l.tolist())
    P, L = np.array(P), np.array(L)
    mj = int(np.bincount(L, minlength=n_cls).argmax())
    per_class = {}
    for i, c in enumerate(CV2024_CLASSES):
        mask = L == i
        if mask.sum() > 0:
            per_class[c] = {"recall": float((P[mask] == i).mean()),
                            "n": int(mask.sum())}
    return {
        "acc": float((P == L).mean()),
        "bal_acc": float(balanced_accuracy_score(L, P)),
        "f1_macro": float(f1_score(L, P, average="macro", zero_division=0)),
        "null_acc": float((np.full_like(L, mj) == L).mean()),
        "n_test": len(L),
        "per_class": per_class,
    }


def train_one_seed(seed, args, train_items, test_items):
    set_determinism(seed)
    tr_ds = ItemDS(train_items)
    te_ds = ItemDS(test_items)

    labels_t = torch.tensor([it[1] for it in train_items])
    w = cbw(labels_t, NCV).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    g = torch.Generator()
    g.manual_seed(seed)
    ltr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                     num_workers=6, pin_memory=True,
                     persistent_workers=True, drop_last=True,
                     worker_init_fn=worker_init_fn, generator=g)
    lte = DataLoader(te_ds, batch_size=args.batch * 2, shuffle=False,
                     num_workers=6, pin_memory=True, persistent_workers=True)
    print(f"  Train={len(tr_ds)} Test={len(te_ds)}", flush=True)

    model = DINOv2LoRA().to(DEVICE)
    lora = [p for wr in model.lora_wrappers for p in [wr.lora_a, wr.lora_b]]
    head = list(model.head.parameters())
    scaler = torch.amp.GradScaler("cuda")

    groups = [{"params": head, "lr": 3e-4}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)

    history = []
    for ep in range(args.epochs):
        if ep == 3:
            opt.add_param_group({"params": lora, "lr": 1e-4, "weight_decay": 1e-4})
        model.train()
        model.backbone.eval()
        losses = []
        t0 = time.perf_counter()
        for img, l, _ in ltr:
            img = img.to(DEVICE, non_blocking=True)
            l = l.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lg = model(img)
                loss = loss_fn(lg, l)
            if not torch.isfinite(loss):
                print(f"  [seed={seed}] WARN non-finite loss at ep {ep+1}", flush=True)
                continue
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        metrics = ev(model, lte)
        dt = time.perf_counter() - t0
        print(f"  [seed={seed}] Ep {ep+1} split1_bal={metrics['bal_acc']:.4f} "
              f"null={metrics['null_acc']:.4f} [{dt:.0f}s]", flush=True)
        history.append({"epoch": ep + 1,
                        "loss": float(np.mean(losses)) if losses else float("nan"),
                        "split_1": metrics,
                        "train_s": float(dt)})
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--output", default="phase5_exp2b_split0_only_n4.json")
    args = ap.parse_args()

    train_items = load_split_items("split_0")
    test_items = load_split_items("split_1")
    print(f"split_0 (train) = {len(train_items)} frames; "
          f"split_1 (test) = {len(test_items)} frames", flush=True)

    results = {
        "args": vars(args),
        "meta": {
            "task": "exp2b_split0_oracle",
            "script_sha256": file_sha256(__file__),
            "n_train": len(train_items),
            "n_test": len(test_items),
            "classes": CV2024_CLASSES,
        },
        "runs": [],
    }
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        history = train_one_seed(seed, args, train_items, test_items)
        results["runs"].append({"seed": seed, "history": history, "last": history[-1]})
        with open(OUT / args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {OUT / args.output}", flush=True)


if __name__ == "__main__":
    main()
