"""Cross-source generalization on CV2024 (Capsule Vision 2024 Challenge).

Sources: KVASIR, SEE-AI, KID, AIIMS (AIIMS very small, only for eval).
Experiments:
(A) pooled: train on pooled training split, test on pooled validation split (reference).
(B) leave-source-out: train on all sources except X, test on X.
    X ∈ {KVASIR, SEE-AI, KID}.

Metric: balanced accuracy, per-class, with null baseline (majority).
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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
ARTIFACT_ROOT = Path(os.environ.get("CAPSULE_ARTIFACT_ROOT", ROOT))
DATA = Path(os.environ.get("CV2024_ROOT", ROOT / "data/cv2024/Dataset"))
if (DATA / "Dataset").is_dir():
    DATA = DATA / "Dataset"
OUT = ARTIFACT_ROOT / "results"
OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda:0")

CV2024_CLASSES = [
    "Angioectasia", "Bleeding", "Erosion", "Erythema", "Foreign Body",
    "Lymphangiectasia", "Normal", "Polyp", "Ulcer", "Worms",
]
CLS2IDX = {c: i for i, c in enumerate(CV2024_CLASSES)}

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_items(split: str):
    """Return list of (path, class_idx, source) triples.  split ∈ {training, validation}."""
    xlsx = DATA / split / f"{split}_data.xlsx"
    df = pd.read_excel(xlsx)
    items = []
    for _, row in df.iterrows():
        path = DATA / row["image_path"].replace("\\", "/")
        if not path.exists():
            continue
        # Single-label: argmax over the 10 class columns
        cls_vals = [row[c] for c in CV2024_CLASSES]
        lbl_idx = int(np.argmax(cls_vals))
        items.append((str(path), lbl_idx, row["Dataset"]))
    return items


class ItemDS(torch.utils.data.Dataset):
    def __init__(self, items, transform=EVAL_TF):
        self.items = items
        self.tf = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, lbl, _ = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), lbl, idx


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        d_in, d_out = base.in_features, base.out_features
        self.lora_a = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(d_out, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


class DINOv2LoRA(nn.Module):
    def __init__(self, n_cls, lora_r=8, lora_blocks=4):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.lora_wrappers = nn.ModuleList()
        for blk in self.backbone.blocks[-lora_blocks:]:
            attn = blk.attn
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                w = LoRALinear(attn.qkv, r=lora_r)
                attn.qkv = w
                self.lora_wrappers.append(w)
        self.head = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, 256),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_cls),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def cbw(labels, n_cls, beta=0.999):
    counts = torch.bincount(labels, minlength=n_cls).float()
    eff = 1.0 - torch.pow(beta, counts)
    eff[counts == 0] = 1.0
    w = (1.0 - beta) / eff
    w = w / w.sum() * n_cls
    w[counts == 0] = 0
    return w


@torch.no_grad()
def evaluate(model, loader, n_cls):
    model.eval()
    preds, labs = [], []
    for imgs, lab, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(imgs)
        preds.extend(logits.argmax(1).cpu().tolist())
        labs.extend(lab.tolist())
    preds, labs = np.array(preds), np.array(labs)
    acc = float((preds == labs).mean())
    bal = float(balanced_accuracy_score(labs, preds))
    f1m = float(f1_score(labs, preds, average="macro", zero_division=0))
    majority = int(np.bincount(labs).argmax()) if len(labs) > 0 else 0
    null_acc = float((np.full_like(labs, majority) == labs).mean())
    cm = confusion_matrix(labs, preds, labels=list(range(n_cls)))
    per_cls = {
        CV2024_CLASSES[i]: {"n": int(cm[i].sum()),
                            "acc": float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else None}
        for i in range(n_cls)
    }
    return {"acc": acc, "bal_acc": bal, "f1_macro": f1m,
            "null_acc": null_acc, "per_class": per_cls}


def train_one(train_items, test_items, *, epochs=12, batch=128, head_warmup=2, patience=4, seed=0, tag=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = ItemDS(train_items)
    te = ItemDS(test_items)
    ltr = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=6,
                     pin_memory=True, drop_last=True, persistent_workers=True)
    lte = DataLoader(te, batch_size=batch*2, shuffle=False, num_workers=6,
                     pin_memory=True, persistent_workers=True)
    model = DINOv2LoRA(n_cls=len(CV2024_CLASSES), lora_r=8).to(DEVICE)
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    else:
        m = model
    labels_t = torch.tensor([i[1] for i in train_items])
    weights = cbw(labels_t, len(CV2024_CLASSES)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    lora = [p for w in model.lora_wrappers for p in [w.lora_a, w.lora_b]]
    head = list(model.head.parameters())
    scaler = torch.amp.GradScaler("cuda")
    best, best_m, pc = -1, None, 0
    history = []
    for ep in range(epochs):
        include = ep >= head_warmup
        groups = []
        if include:
            groups.append({"params": lora, "lr": 1e-4})
        groups.append({"params": head, "lr": 3e-4})
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        m.train()
        losses = []
        t0 = time.perf_counter()
        for imgs, labs, _ in ltr:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labs = labs.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = m(imgs)
                loss = loss_fn(logits, labs)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        dt = time.perf_counter() - t0
        mm = evaluate(m, lte, len(CV2024_CLASSES))
        history.append({"epoch": ep+1, "loss": float(np.mean(losses)),
                        **{k: v for k, v in mm.items() if k != "per_class"}})
        print(f"  [{tag}] Ep {ep+1} bal={mm['bal_acc']:.4f} null={mm['null_acc']:.4f} [{dt:.0f}s]", flush=True)
        if mm["bal_acc"] > best:
            best = mm["bal_acc"]
            best_m = mm
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break
    return best_m, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=["pooled", "leave_source_out"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print(f"Args: {vars(args)}", flush=True)
    tr_items = load_items("training")
    va_items = load_items("validation")
    print(f"Train={len(tr_items)}, Val={len(va_items)}", flush=True)

    # Source distribution
    src_tr = defaultdict(int)
    src_va = defaultdict(int)
    for _, _, s in tr_items:
        src_tr[s] += 1
    for _, _, s in va_items:
        src_va[s] += 1
    print(f"Train src: {dict(src_tr)}", flush=True)
    print(f"Val src: {dict(src_va)}", flush=True)

    results = {"args": vars(args), "runs": []}
    if args.experiment == "pooled":
        for seed in args.seeds:
            m, h = train_one(tr_items, va_items, seed=seed,
                             epochs=args.epochs, tag=f"pooled seed={seed}")
            results["runs"].append({
                "experiment": "pooled", "seed": seed,
                "n_train": len(tr_items), "n_test": len(va_items),
                "best_metrics": m, "history": h,
            })
    elif args.experiment == "leave_source_out":
        for left in ("KVASIR", "SEE-AI", "KID"):
            # Train on all except `left`
            tr_sub = [i for i in tr_items if i[2] != left]
            # Test on `left` (validation split of that source)
            te_sub = [i for i in va_items if i[2] == left]
            if not te_sub:
                print(f"  Skipping {left} — no validation items", flush=True)
                continue
            print(f"\n=== Leave-{left}-out: train {len(tr_sub)}, test {len(te_sub)} ===", flush=True)
            for seed in args.seeds:
                m, h = train_one(tr_sub, te_sub, seed=seed, epochs=args.epochs,
                                 tag=f"leave={left} seed={seed}")
                results["runs"].append({
                    "experiment": "leave_source_out", "source_left_out": left,
                    "seed": seed, "n_train": len(tr_sub), "n_test": len(te_sub),
                    "best_metrics": m, "history": h,
                })
                with open(OUT / args.output, "w") as f:
                    json.dump(results, f, indent=2, default=str)

    with open(OUT / args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved → {OUT / args.output}", flush=True)


if __name__ == "__main__":
    main()
