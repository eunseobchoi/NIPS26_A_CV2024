"""Matched 11-class frame-split baseline for protocol gap isolation.

Splits Kvasir-Capsule frames at 70/15/15 (same as historical frame-split
baseline) but uses the 11-class label space (same as official two-fold
split). Trained with the same pipeline. This is the "clean" control that
separates protocol-gap effect (frame-split vs video-split) from
label-space effect (14 vs 11 classes).
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
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
OUT = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
import dataset_official as _ds
_ds.DATA_ROOT = Path(os.environ.get("KVASIR_ROOT", ROOT / "data/kvasir_capsule/labelled_images"))
if (_ds.DATA_ROOT / "labelled_images").is_dir():
    _ds.DATA_ROOT = _ds.DATA_ROOT / "labelled_images"
_ds.SPLITS_DIR = Path(os.environ.get("KVASIR_SPLITS_DIR", ROOT / "data/official_splits"))
from dataset_official import (
    KvasirCapsuleOfficial, OFFICIAL_CLASSES, NUM_CLASSES, LABEL_TO_FOLDER,
)

DEVICE = torch.device("cuda:0")
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_all_items():
    items = []
    for split in ("split_0", "split_1"):
        with open(_ds.SPLITS_DIR / f"{split}.csv") as f:
            for row in csv.DictReader(f):
                lbl = row["label"]
                if lbl not in LABEL_TO_FOLDER:
                    continue
                folder = LABEL_TO_FOLDER[lbl]
                path = _ds.DATA_ROOT / folder / row["filename"]
                vid = row["filename"].split("_")[0]
                items.append((str(path), OFFICIAL_CLASSES.index(lbl), vid))
    return items


class ItemDS(torch.utils.data.Dataset):
    def __init__(self, items, transform=TF):
        self.items, self.tf = items, transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, lbl, _ = self.items[idx]
        return self.tf(Image.open(path).convert("RGB")), lbl, idx


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
    def __init__(self, n_cls=NUM_CLASSES, r=8):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        for p in self.backbone.parameters(): p.requires_grad = False
        self.lora_wrappers = nn.ModuleList()
        for blk in self.backbone.blocks[-4:]:
            w = LoRALinear(blk.attn.qkv, r=r); blk.attn.qkv = w
            self.lora_wrappers.append(w)
        self.head = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, 256),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_cls))
    def forward(self, x): return self.head(self.backbone(x))


def cbw(labels, n, beta=0.999):
    c = torch.bincount(labels, minlength=n).float()
    e = 1.0 - torch.pow(beta, c); e[c == 0] = 1.0
    w = (1.0 - beta) / e; w = w / w.sum() * n; w[c == 0] = 0
    return w


@torch.no_grad()
def ev(m, loader):
    m.eval()
    P, L = [], []
    for img, l, _ in loader:
        img = img.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lg = m(img)
        P.extend(lg.argmax(1).cpu().tolist()); L.extend(l.tolist())
    P, L = np.array(P), np.array(L)
    mj = int(np.bincount(L).argmax())
    return {"acc": float((P == L).mean()),
            "bal_acc": float(balanced_accuracy_score(L, P)),
            "f1_macro": float(f1_score(L, P, average="macro", zero_division=0)),
            "null_acc": float((np.full_like(L, mj) == L).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--output", default="phase4_matched_11class_frame_split.json")
    args = ap.parse_args()

    items = load_all_items()
    print(f"Total 11-class items: {len(items)} frames", flush=True)

    results = {"args": vars(args), "runs": []}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(items)); rng.shuffle(idx)
        n = len(items); n_tr = int(0.70 * n); n_val = int(0.15 * n)
        tr = [items[i] for i in idx[:n_tr]]
        te = [items[i] for i in idx[n_tr + n_val:]]
        torch.manual_seed(seed); np.random.seed(seed)
        tr_ds, te_ds = ItemDS(tr), ItemDS(te)
        ltr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=6,
                         pin_memory=True, persistent_workers=True, drop_last=True)
        lte = DataLoader(te_ds, batch_size=256, shuffle=False, num_workers=6,
                         pin_memory=True, persistent_workers=True)
        model = DINOv2LoRA().to(DEVICE)
        if torch.cuda.device_count() > 1:
            m = nn.DataParallel(model)
        else:
            m = model
        labels_t = torch.tensor([it[1] for it in tr])
        w = cbw(labels_t, NUM_CLASSES).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
        lora = [p for wr in model.lora_wrappers for p in [wr.lora_a, wr.lora_b]]
        head = list(model.head.parameters())
        scaler = torch.amp.GradScaler("cuda")
        history = []
        for ep in range(args.epochs):
            inc = ep >= 3
            groups = [{"params": head, "lr": 3e-4}]
            if inc: groups.append({"params": lora, "lr": 1e-4})
            opt = torch.optim.AdamW(groups, weight_decay=1e-4)
            m.train()
            losses = []
            t0 = time.perf_counter()
            for img, l, _ in ltr:
                img, l = img.to(DEVICE, non_blocking=True), l.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    lg = m(img); loss = loss_fn(lg, l)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                losses.append(loss.item())
            metrics = ev(m, lte)
            dt = time.perf_counter() - t0
            print(f"  [seed={seed}] Ep {ep+1} bal={metrics['bal_acc']:.4f} "
                  f"null={metrics['null_acc']:.4f} [{dt:.0f}s]", flush=True)
            history.append({"epoch": ep+1, "loss": float(np.mean(losses)),
                            **metrics, "train_s": float(dt)})
        results["runs"].append({"seed": seed, "n_train": len(tr), "n_test": len(te),
                                 "last_metrics": history[-1], "history": history})
        with open(OUT / args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"Saved {OUT / args.output}", flush=True)


if __name__ == "__main__":
    main()
