"""DINOv2 + LoRA on last 4 attn blocks, trained on official two-fold CSVs.

Goal: get above null baseline (~73%). Frozen backbone + head gave bal_acc=0.29 < null.
LoRA adds ~600K adaptable params on attn.qkv of last 4 blocks.
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
import dataset_official as _ds
_root = Path(os.environ.get("CAPSULE_ROOT", ".")).resolve()
_ds.DATA_ROOT = Path(os.environ.get("KVASIR_ROOT", _root / "data/kvasir_capsule/labelled_images"))
if (_ds.DATA_ROOT / "labelled_images").is_dir():
    _ds.DATA_ROOT = _ds.DATA_ROOT / "labelled_images"
_ds.SPLITS_DIR = Path(os.environ.get("KVASIR_SPLITS_DIR", _root / "data/official_splits"))
from dataset_official import (
    KvasirCapsuleOfficial,
    OFFICIAL_CLASSES,
    NUM_CLASSES,
)

DEVICE = "cuda:0"
OUT = _root / "results"
OUT.mkdir(exist_ok=True)


class LoRALinear(nn.Module):
    """LoRA wrapper. Keeps frozen base linear, adds trainable low-rank update."""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
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


class DINOv2LoRAClassifier(nn.Module):
    def __init__(self, n_cls: int = NUM_CLASSES, lora_r: int = 8, lora_blocks: int = 4):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Inject LoRA on last N blocks' attn.qkv
        blocks = self.backbone.blocks[-lora_blocks:]
        self.lora_params = []
        for blk in blocks:
            attn = blk.attn
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                wrapped = LoRALinear(attn.qkv, r=lora_r)
                attn.qkv = wrapped
                self.lora_params.extend([wrapped.lora_a, wrapped.lora_b])
        self.head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_cls),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


def class_balanced_weights(labels, n_cls):
    counts = torch.bincount(labels, minlength=n_cls).float()
    # Effective number of samples (Cui et al. 2019) β=0.999
    beta = 0.999
    eff_n = 1.0 - torch.pow(beta, counts)
    eff_n[counts == 0] = 1.0
    weights = (1.0 - beta) / eff_n
    weights = weights / weights.sum() * n_cls
    weights[counts == 0] = 0
    return weights


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    losses = []
    t0 = time.perf_counter()
    for i, (imgs, labs, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labs = labs.to(device, non_blocking=True)
        logits = model(imgs)
        loss = loss_fn(logits, labs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)), time.perf_counter() - t0


@torch.no_grad()
def evaluate(model, loader, device, n_cls):
    model.eval()
    preds, labs = [], []
    for imgs, lab, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds.extend(logits.argmax(1).cpu().tolist())
        labs.extend(lab.tolist())
    preds = np.array(preds)
    labs = np.array(labs)
    acc = float((preds == labs).mean())
    bal = float(balanced_accuracy_score(labs, preds))
    f1m = float(f1_score(labs, preds, average="macro", zero_division=0))
    majority = int(np.bincount(labs).argmax())
    null_acc = float((np.full_like(labs, majority) == labs).mean())

    per_class = {}
    cm = confusion_matrix(labs, preds, labels=list(range(n_cls)))
    for i, cls in enumerate(OFFICIAL_CLASSES):
        support = int(cm[i].sum())
        per_class[cls] = {
            "acc": float(cm[i, i] / support) if support > 0 else None,
            "n": support,
        }
    return {
        "acc": acc,
        "bal_acc": bal,
        "f1_macro": f1m,
        "null_acc": null_acc,
        "per_class": per_class,
    }


def run_fold(fold: int, epochs: int = 8, batch: int = 48, lr_head: float = 3e-4,
             lr_lora: float = 1e-4, lora_r: int = 8):
    torch.manual_seed(42 + fold)
    np.random.seed(42 + fold)

    train_key = f"split_{fold}"
    test_key = f"split_{1-fold}"
    print(f"\n{'='*60}\n  Fold {fold}: train={train_key}, test={test_key}\n{'='*60}")

    ds_train = KvasirCapsuleOfficial(train_key)
    ds_test = KvasirCapsuleOfficial(test_key)
    print(f"  Train: {len(ds_train)}, Test: {len(ds_test)}")

    train_loader = DataLoader(ds_train, batch_size=batch, shuffle=True,
                              num_workers=6, pin_memory=True, drop_last=True)
    test_loader = DataLoader(ds_test, batch_size=batch*2, shuffle=False,
                             num_workers=6, pin_memory=True)

    model = DINOv2LoRAClassifier(lora_r=lora_r, lora_blocks=4).to(DEVICE)

    # Class-balanced loss
    all_labels = torch.tensor([y for _, y, _ in ds_train.items])
    weights = class_balanced_weights(all_labels, NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    print(f"  Class weights: {weights.cpu().numpy().round(3).tolist()}")
    n_trainable = sum(p.numel() for p in model.trainable_params())
    print(f"  Trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW([
        {"params": model.lora_params, "lr": lr_lora},
        {"params": model.head.parameters(), "lr": lr_head},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_bal = -1
    final_state = None
    history = []
    for ep in range(epochs):
        loss, dt = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        sched.step()
        m = evaluate(model, test_loader, DEVICE, NUM_CLASSES)
        print(f"  Epoch {ep+1}/{epochs}  loss={loss:.4f}  acc={m['acc']:.4f}  "
              f"bal_acc={m['bal_acc']:.4f}  f1={m['f1_macro']:.4f}  null={m['null_acc']:.4f}  "
              f"[{dt:.0f}s]")
        history.append({"epoch": ep + 1, "loss": loss, **{k: v for k, v in m.items() if k != "per_class"}})
        final_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if os.environ.get("SAVE_TEST_BEST", "0") == "1" and m["bal_acc"] > best_bal:
            best_bal = m["bal_acc"]
            torch.save(model.state_dict(), OUT / f"dinov2_lora_official_fold{fold}_best_by_test.pth")
        else:
            best_bal = max(best_bal, m["bal_acc"])

    if final_state is not None:
        torch.save(final_state, OUT / f"dinov2_lora_official_fold{fold}.pth")

    # Final per-class table
    print(f"\n  Per-class (best epoch, fold {fold}):")
    for cls, pm in m["per_class"].items():
        support = pm["n"]
        mark = "  " if support >= 30 else " *"
        a = pm["acc"]
        a_str = f"{a:.4f}" if a is not None else "N/A"
        print(f"    {cls:<25} {mark} acc={a_str} (n={support})")

    return {"fold": fold, "history": history, "final": m, "best_bal": best_bal}


def main():
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print("usage: 06_train_official_lora.py")
        print("Trains final-epoch DINOv2-LoRA checkpoints for TTA: results/dinov2_lora_official_fold{0,1}.pth")
        return
    print("Starting DINOv2-LoRA official 2-fold CV")
    results = {}
    for fold in (0, 1):
        results[f"fold_{fold}"] = run_fold(fold, epochs=8)

    avg = {
        k: float(np.mean([results[f"fold_{i}"]["final"][k] for i in (0, 1)]))
        for k in ("acc", "bal_acc", "f1_macro", "null_acc")
    }
    print(f"\n{'='*60}\n  2-fold CV AVERAGE (LoRA baseline)\n{'='*60}")
    for k, v in avg.items():
        print(f"  {k:<12} = {v:.4f}")
    results["cv_average"] = avg

    with open(OUT / "kvasir_official_lora.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved → {OUT}/kvasir_official_lora.json")


if __name__ == "__main__":
    main()
