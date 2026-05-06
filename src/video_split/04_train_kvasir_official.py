"""
Train DINOv2 classifier on Kvasir-Capsule with domain shift evaluation.
Split: video-based patient split → train (70%) / val (15%) / test (15%)
Since no patient IDs, we use class-stratified split.
"""
import os
import sys
import time
import json
import random
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import KvasirCapsule, TRAIN_TRANSFORM, EVAL_TRANSFORM

DEVICE = "cuda:0"
DATA_ROOT = os.environ.get("CAPSULE_ROOT", ".") + "/data/kvasir_capsule/labelled_images"
OUTPUT_DIR = Path(os.environ.get("CAPSULE_ROOT", ".") + "/results")


class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes, backbone_name="dinov2_vitl14"):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone.embed_dim),
            nn.Linear(self.backbone.embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        with torch.no_grad():
            feat = self.backbone(x)
        return self.head(feat)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, bal_acc, f1


def main():
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print("usage: 04_train_kvasir_official.py")
        print("Legacy Kvasir frame-split trainer. Reads $CAPSULE_ROOT/data/kvasir_capsule/labelled_images.")
        return
    OUTPUT_DIR.mkdir(exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load full dataset
    full_ds = KvasirCapsule(DATA_ROOT, transform=EVAL_TRANSFORM)
    print(f"Full dataset: {len(full_ds)} samples, {len(full_ds.classes)} classes")
    print(f"Classes: {full_ds.classes}")

    # Stratified split: 70/15/15
    labels = [s[1] for s in full_ds.samples]
    idx_train, idx_temp = train_test_split(range(len(full_ds)), test_size=0.3,
                                           stratify=labels, random_state=42)
    labels_temp = [labels[i] for i in idx_temp]
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5,
                                         stratify=labels_temp, random_state=42)

    train_ds = Subset(KvasirCapsule(DATA_ROOT, transform=TRAIN_TRANSFORM), idx_train)
    val_ds = Subset(full_ds, idx_val)
    test_ds = Subset(full_ds, idx_test)

    print(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Build model
    print("Building DINOv2 classifier...")
    model = DINOv2Classifier(num_classes=len(full_ds.classes)).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total_p:,} total ({trainable/total_p*100:.2f}%)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    # Train
    epochs = 15
    best_val_acc = 0
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_acc, val_bal, val_f1 = evaluate(model, val_loader)
        scheduler.step()
        elapsed = time.time() - t0

        record = {"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc,
                  "val_acc": val_acc, "val_bal_acc": val_bal, "val_f1": val_f1, "time_s": elapsed}
        history.append(record)
        print(f"  E{epoch+1:2d}: loss={train_loss:.4f} train={train_acc:.4f} "
              f"val_acc={val_acc:.4f} bal_acc={val_bal:.4f} f1={val_f1:.4f} ({elapsed:.1f}s)")

        if val_bal > best_val_acc:
            best_val_acc = val_bal
            torch.save(model.state_dict(), OUTPUT_DIR / "kvasir_dinov2_best.pth")

    # Final test
    model.load_state_dict(torch.load(OUTPUT_DIR / "kvasir_dinov2_best.pth", weights_only=True))
    test_acc, test_bal, test_f1 = evaluate(model, test_loader)
    print(f"\nTest: acc={test_acc:.4f} bal_acc={test_bal:.4f} f1={test_f1:.4f}")

    # Save
    result = {"history": history, "test_acc": test_acc, "test_bal_acc": test_bal,
              "test_f1": test_f1, "best_val_bal_acc": best_val_acc,
              "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
              "classes": full_ds.classes}
    with open(OUTPUT_DIR / "kvasir_baseline.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {OUTPUT_DIR / 'kvasir_baseline.json'}")


if __name__ == "__main__":
    main()
