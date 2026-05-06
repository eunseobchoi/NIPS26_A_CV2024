"""Train baseline head on Kvasir-Capsule's official two-fold CSVs.

2-fold CV: train on split_0 → test split_1, then train split_1 → test split_0.
Uses feature caching for speed (DINOv2 features computed once per split).
The official CSVs are frame-list folds, not video-disjoint folds under the
filename-prefix audit reported in the paper.
"""
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_official import (
    KvasirCapsuleOfficial,
    OFFICIAL_CLASSES,
    NUM_CLASSES,
)

DEVICE = "cuda:0"
OUT = Path(os.environ.get("CAPSULE_ROOT", ".") + "/results")
OUT.mkdir(exist_ok=True)


class Head(nn.Module):
    def __init__(self, n_cls: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_cls),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def extract_features(split: str, backbone, loader):
    feats, labels = [], []
    backbone.eval()
    t0 = time.perf_counter()
    for i, (imgs, labs, _) in enumerate(loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        f = backbone(imgs)
        feats.append(f.cpu())
        labels.append(labs)
        if (i + 1) % 20 == 0:
            done = (i + 1) * loader.batch_size
            dt = time.perf_counter() - t0
            print(f"  [{split}] {done} images, {done/dt:.0f} img/s")
    return torch.cat(feats), torch.cat(labels)


def train_head(feats_train, labels_train, n_epochs=30, lr=3e-4, batch=512):
    head = Head().to(DEVICE)
    # Class-balanced CE
    class_counts = torch.bincount(labels_train, minlength=NUM_CLASSES).float()
    weights = (class_counts.sum() / (NUM_CLASSES * class_counts.clamp(min=1))).to(DEVICE)
    weights[class_counts == 0] = 0
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    feats_train_gpu = feats_train.to(DEVICE)
    labels_train_gpu = labels_train.to(DEVICE)
    n = len(feats_train_gpu)

    for epoch in range(n_epochs):
        head.train()
        perm = torch.randperm(n)
        losses = []
        for start in range(0, n, batch):
            idx = perm[start:start+batch]
            logits = head(feats_train_gpu[idx])
            loss = loss_fn(logits, labels_train_gpu[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} loss={np.mean(losses):.4f}")
    return head


@torch.no_grad()
def evaluate(head, feats, labels, n_classes):
    head.eval()
    feats_gpu = feats.to(DEVICE)
    logits = head(feats_gpu)
    preds = logits.argmax(1).cpu().numpy()
    labs = labels.numpy()

    acc = float((preds == labs).mean())
    bal = float(balanced_accuracy_score(labs, preds))
    f1m = float(f1_score(labs, preds, average="macro", zero_division=0))
    f1w = float(f1_score(labs, preds, average="weighted", zero_division=0))

    # Majority class null baseline
    majority = int(np.bincount(labs).argmax())
    null_acc = float((np.full_like(labs, majority) == labs).mean())

    # Per-class accuracy
    per_class = {}
    cm = confusion_matrix(labs, preds, labels=list(range(n_classes)))
    for i, cls in enumerate(OFFICIAL_CLASSES):
        support = cm[i].sum()
        if support > 0:
            per_class[cls] = {
                "acc": float(cm[i, i] / support),
                "n": int(support),
            }
        else:
            per_class[cls] = {"acc": None, "n": 0}

    return {
        "acc": acc,
        "bal_acc": bal,
        "f1_macro": f1m,
        "f1_weighted": f1w,
        "null_acc": null_acc,
        "per_class": per_class,
    }


def main():
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print("usage: 05_train_official_frozen.py")
        print("Frozen DINOv2 feature baseline on official split_0/split_1.")
        return
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading DINOv2 backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    backbone = backbone.to(DEVICE).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Extract features for both splits
    features = {}
    for s in ("split_0", "split_1"):
        cache = OUT / f"features_official_{s}.pt"
        if cache.exists():
            print(f"Loading cached {s} features")
            d = torch.load(cache, map_location="cpu", weights_only=True)
            features[s] = (d["feats"], d["labels"])
            continue
        print(f"\nExtracting {s} features...")
        ds = KvasirCapsuleOfficial(s)
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=6, pin_memory=True)
        feats, labels = extract_features(s, backbone, loader)
        torch.save({"feats": feats, "labels": labels}, cache)
        features[s] = (feats, labels)
        print(f"  {s}: {len(feats)} features cached to {cache.name}")

    # 2-fold CV
    results = {}
    for fold in (0, 1):
        train_key = f"split_{fold}"
        test_key = f"split_{1-fold}"
        print(f"\n{'='*60}\n  Fold {fold}: train={train_key}, test={test_key}\n{'='*60}")

        ft, lt = features[train_key]
        fv, lv = features[test_key]
        print(f"  Train: {len(ft)} feats | Test: {len(fv)} feats")

        head = train_head(ft, lt, n_epochs=30)

        metrics = evaluate(head, fv, lv, NUM_CLASSES)
        print(f"  acc={metrics['acc']:.4f}  bal_acc={metrics['bal_acc']:.4f}  "
              f"f1_macro={metrics['f1_macro']:.4f}  null={metrics['null_acc']:.4f}")
        print(f"  Per-class:")
        for cls, m in metrics["per_class"].items():
            mark = "  " if m["n"] >= 30 else " *"
            print(f"    {cls:<25} {mark} acc={m['acc'] if m['acc'] is not None else 'N/A':.4f} (n={m['n']})" if m['acc'] is not None else f"    {cls:<25} {mark} N/A (n={m['n']})")

        torch.save(head.state_dict(), OUT / f"head_official_fold{fold}.pth")
        results[f"fold_{fold}"] = {
            "train_split": train_key,
            "test_split": test_key,
            "metrics": metrics,
        }

    # Aggregate across folds
    avg = {k: float(np.mean([results[f"fold_{i}"]["metrics"][k]
                             for i in (0, 1)]))
           for k in ("acc", "bal_acc", "f1_macro", "null_acc")}
    print(f"\n{'='*60}\n  2-fold CV AVERAGE\n{'='*60}")
    for k, v in avg.items():
        print(f"  {k:<12} = {v:.4f}")
    results["cv_average"] = avg

    with open(OUT / "kvasir_official_baseline.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved → {OUT}/kvasir_official_baseline.json")


if __name__ == "__main__":
    main()
