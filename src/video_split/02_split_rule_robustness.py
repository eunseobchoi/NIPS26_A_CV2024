"""Split-rule robustness: does protocol gap depend on split choice?

Tests:
  (A) Frame-split at varying ratios: 60/40, 70/30, 80/20, 90/10 (seed 42)
  (B) Leave-one-video-out (LOVO): per patient-video, 43 separate models
      but we subsample 10 LOVO folds to keep time tractable
  (C) Random 5-fold video-level GroupKFold (already done: 0.380)
  (D) Official 2-fold (already done: 0.250)

Output: phase7_split_robustness.json

This supports the protocol-gap robustness check by testing whether the
effect depends on the specific 2-fold official split.
"""
import os
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
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
from dataset_official import OFFICIAL_CLASSES, NUM_CLASSES, LABEL_TO_FOLDER

DEVICE = torch.device("cuda:0")
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_all_11class():
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
                items.append({"path": str(path),
                              "label": OFFICIAL_CLASSES.index(lbl),
                              "video": vid,
                              "filename": row["filename"]})
    return items


class ItemDS(torch.utils.data.Dataset):
    def __init__(self, items, tf=TF):
        self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        return self.tf(Image.open(it["path"]).convert("RGB")), it["label"], idx


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
    m.eval(); P, L = [], []
    for im, l, _ in loader:
        im = im.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lg = m(im)
        P.extend(lg.argmax(1).cpu().tolist()); L.extend(l.tolist())
    P, L = np.array(P), np.array(L)
    mj = int(np.bincount(L).argmax()) if len(L) else 0
    return {"acc": float((P == L).mean()),
            "bal_acc": float(balanced_accuracy_score(L, P)),
            "f1_macro": float(f1_score(L, P, average="macro", zero_division=0)),
            "null_acc": float((np.full_like(L, mj) == L).mean()),
            "n_test": len(L)}


def train_once(tr_items, te_items, epochs=15, seed=0, batch=128):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_ds, te_ds = ItemDS(tr_items), ItemDS(te_items)
    ltr = DataLoader(tr_ds, batch_size=batch, shuffle=True, num_workers=6,
                      pin_memory=True, persistent_workers=True, drop_last=True)
    lte = DataLoader(te_ds, batch_size=batch*2, shuffle=False, num_workers=6,
                      pin_memory=True, persistent_workers=True)
    model = DINOv2LoRA().to(DEVICE)
    m = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    labels_t = torch.tensor([x["label"] for x in tr_items])
    w = cbw(labels_t, NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    lora = [p for wr in model.lora_wrappers for p in [wr.lora_a, wr.lora_b]]
    head = list(model.head.parameters())
    scaler = torch.amp.GradScaler("cuda")
    for ep in range(epochs):
        groups = [{"params": head, "lr": 3e-4}]
        if ep >= 3: groups.append({"params": lora, "lr": 1e-4})
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        m.train()
        for im, l, _ in ltr:
            im = im.to(DEVICE, non_blocking=True); l = l.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lg = m(im); loss = loss_fn(lg, l)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    return ev(m, lte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n_lovo", type=int, default=10,
                    help="Number of leave-one-video-out folds to run")
    ap.add_argument("--output", default="phase7_split_robustness.json")
    args = ap.parse_args()

    items = load_all_11class()
    vids = sorted({it["video"] for it in items})
    print(f"Total 11-class labeled frames: {len(items)}  videos: {len(vids)}", flush=True)

    results = {"args": vars(args), "runs": []}

    # (A) Frame-split at varying ratios (14-class not available here, use 11-class + random)
    for ratio in (0.6, 0.7, 0.8, 0.9):
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            idx = np.arange(len(items)); rng.shuffle(idx)
            n_tr = int(ratio * len(items))
            tr = [items[i] for i in idx[:n_tr]]; te = [items[i] for i in idx[n_tr:]]
            print(f"\n[frame-split ratio={ratio} seed={seed}] n_tr={len(tr)} n_te={len(te)}", flush=True)
            m = train_once(tr, te, epochs=args.epochs, seed=seed)
            results["runs"].append({"protocol": "frame_split",
                                     "ratio": ratio, "seed": seed, "metrics": m,
                                     "n_train": len(tr), "n_test": len(te)})
            print(f"  → bal={m['bal_acc']:.3f} acc={m['acc']:.3f} null={m['null_acc']:.3f}")
            with open(OUT / args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)

    # (B) Leave-one-video-out (LOVO) — subsample of folds
    rng = np.random.default_rng(0)
    lovo_vids = list(rng.choice(vids, min(args.n_lovo, len(vids)), replace=False))
    for held_vid in lovo_vids:
        tr = [it for it in items if it["video"] != held_vid]
        te = [it for it in items if it["video"] == held_vid]
        if len(te) == 0 or len(tr) == 0: continue
        print(f"\n[LOVO held={held_vid[:14]}] n_tr={len(tr)} n_te={len(te)}", flush=True)
        m = train_once(tr, te, epochs=args.epochs, seed=0)
        results["runs"].append({"protocol": "LOVO",
                                 "held_video": held_vid, "seed": 0, "metrics": m,
                                 "n_train": len(tr), "n_test": len(te)})
        print(f"  → bal={m['bal_acc']:.3f} acc={m['acc']:.3f} null={m['null_acc']:.3f}")
        with open(OUT / args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\nSaved {OUT / args.output}")


if __name__ == "__main__":
    main()
