"""
TTA Benchmark v4 — Bug-fixed.
Key fix: reload model state_dict before EACH method to prevent state leakage.
"""
import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score, f1_score

DEVICE = "cuda:0"
DATA_ROOT = os.environ.get("CAPSULE_ROOT", ".") + "/data/kvasir_capsule/labelled_images"
OUTPUT_DIR = Path(os.environ.get("CAPSULE_ROOT", ".") + "/results")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import KvasirCapsule, EVAL_TRANSFORM


def get_corruption_transform(severity):
    s = severity / 5.0
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.5*s, contrast=0.6*s, saturation=0.5*s, hue=0.1*s),
        transforms.GaussianBlur(kernel_size=int(5 + 8*s) | 1, sigma=(0.5, 2.0 + 4*s)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        self.head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


def fresh_model(num_classes, head_path):
    """Build a completely fresh model with loaded weights. No state leakage."""
    model = DINOv2Classifier(num_classes).to(DEVICE)
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # Load head
    saved = torch.load(head_path, map_location=DEVICE, weights_only=True)
    clean = {k.replace("net.", ""): v for k, v in saved.items()} if any(k.startswith("net.") for k in saved) else saved
    model.head.load_state_dict(clean)
    model.eval()
    return model


def entropy_loss_filtered(logits, margin):
    p = F.softmax(logits, dim=1)
    ent = -(p * p.clamp(min=1e-8).log()).sum(dim=1)
    mask = ent < margin
    if mask.sum() > 1:
        return ent[mask].mean()
    return ent.mean()


def run_no_adapt(model, loader):
    model.eval()
    preds, labels, times = [], [], []
    with torch.no_grad():
        for imgs, labs, _ in loader:
            imgs = imgs.to(DEVICE)
            t0 = time.perf_counter()
            if True:
                logits = model(imgs)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(labs.tolist())
    return preds, labels, times


def run_ln_adapt(model, loader, steps=1, lr=1e-3):
    """Adapt LayerNorm affine params per batch, restore between batches."""
    margin = 0.4 * np.log(14)
    # Collect LN params and save originals
    ln_params = []
    for m in model.backbone.modules():
        if isinstance(m, nn.LayerNorm):
            ln_params.extend([m.weight, m.bias])
    originals = [p.data.clone() for p in ln_params]
    for p in ln_params:
        p.requires_grad_(True)

    preds, labels, times = [], [], []
    for imgs, labs, _ in loader:
        imgs = imgs.to(DEVICE)
        # Restore before each batch
        for p, o in zip(ln_params, originals):
            p.data.copy_(o)

        t0 = time.perf_counter()
        model.train()
        opt = torch.optim.Adam(ln_params, lr=1e-4)
        for _ in range(steps):
            if True:
                logits = model(imgs)
            loss = entropy_loss_filtered(logits, margin)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(imgs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(labs.tolist())

    # Cleanup
    for p, o in zip(ln_params, originals):
        p.data.copy_(o)
        p.requires_grad_(False)

    return preds, labels, times


def run_sar(model, loader, steps=1, lr=1e-3):
    """SAR: reliable filtering + sharpness-aware on LN params, restore per batch."""
    margin = 0.4 * np.log(14)
    ln_params = []
    for m in model.backbone.modules():
        if isinstance(m, nn.LayerNorm):
            ln_params.extend([m.weight, m.bias])
    originals = [p.data.clone() for p in ln_params]
    for p in ln_params:
        p.requires_grad_(True)

    preds, labels, times = [], [], []
    for imgs, labs, _ in loader:
        imgs = imgs.to(DEVICE)
        for p, o in zip(ln_params, originals):
            p.data.copy_(o)

        t0 = time.perf_counter()
        model.train()
        opt = torch.optim.Adam(ln_params, lr=1e-4)

        for _ in range(steps):
            if True:
                logits = model(imgs)
            loss = entropy_loss_filtered(logits, margin)
            opt.zero_grad()
            loss.backward()
            # SAR perturbation
            with torch.no_grad():
                for p in ln_params:
                    if p.grad is not None:
                        p.add_(0.05 * p.grad.sign())
            if True:
                logits2 = model(imgs)
            loss2 = entropy_loss_filtered(logits2, margin)
            opt.zero_grad()
            loss2.backward()
            with torch.no_grad():
                for p in ln_params:
                    if p.grad is not None:
                        p.sub_(0.05 * p.grad.sign())
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(imgs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(labs.tolist())

    for p, o in zip(ln_params, originals):
        p.data.copy_(o)
        p.requires_grad_(False)
    return preds, labels, times


def run_head_ttt(model, loader, steps=1, lr=1e-4):
    """Head-TTT with filtering, restore per batch."""
    margin = 0.4 * np.log(14)
    head_params = list(model.head.parameters())
    originals = [p.data.clone() for p in head_params]
    for p in head_params:
        p.requires_grad_(True)

    preds, labels, times = [], [], []
    for imgs, labs, _ in loader:
        imgs = imgs.to(DEVICE)
        for p, o in zip(head_params, originals):
            p.data.copy_(o)

        t0 = time.perf_counter()
        model.train()
        opt = torch.optim.Adam(head_params, lr=lr)
        for _ in range(steps):
            if True:
                logits = model(imgs)
            loss = entropy_loss_filtered(logits, margin)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(imgs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(labs.tolist())

    for p, o in zip(head_params, originals):
        p.data.copy_(o)
        p.requires_grad_(False)
    return preds, labels, times


def compute_metrics(preds, labels):
    acc = np.mean(np.array(preds) == np.array(labels))
    bal = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"acc": float(acc), "bal_acc": float(bal), "f1": float(f1)}


def main():
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print("usage: 01_tta_benchmark.py")
        print("Legacy TTA benchmark. Reads $CAPSULE_ROOT/results/kvasir_baseline.json")
        print("and writes $CAPSULE_ROOT/results/tta_benchmark_v5.json.")
        return
    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    with open(OUTPUT_DIR / "kvasir_baseline.json") as f:
        baseline = json.load(f)
    idx_test = baseline["idx_test"]
    classes = baseline["classes"]
    head_path = OUTPUT_DIR / "kvasir_head_best.pth"

    severities = [0, 3, 5]
    batch_size = 128
    all_results = {}

    methods = [
        ("no_adapt", lambda m, l: run_no_adapt(m, l)),
        ("ln_adapt_1", lambda m, l: run_ln_adapt(m, l, steps=1)),
        ("ln_adapt_4", lambda m, l: run_ln_adapt(m, l, steps=4)),
        ("sar_1", lambda m, l: run_sar(m, l, steps=1)),
        ("head_ttt_1", lambda m, l: run_head_ttt(m, l, steps=1)),
        ("head_ttt_4", lambda m, l: run_head_ttt(m, l, steps=4)),
    ]

    for sev in severities:
        print(f"\n{'='*70}")
        print(f"  Severity {sev}")
        print(f"{'='*70}")

        tf = EVAL_TRANSFORM if sev == 0 else get_corruption_transform(sev)
        ds = KvasirCapsule(DATA_ROOT, transform=tf)
        test_subset = Subset(ds, idx_test)
        loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                           num_workers=6, pin_memory=True)

        sev_results = {}
        for method_name, method_fn in methods:
            # FRESH model for each method — no state leakage
            model = fresh_model(len(classes), head_path)
            torch.cuda.reset_peak_memory_stats()

            preds, labels, times = method_fn(model, loader)
            metrics = compute_metrics(preds, labels)
            mem = torch.cuda.max_memory_allocated() / 1e6
            lat = np.mean(times)

            sev_results[method_name] = {
                **metrics, "lat_ms": float(lat), "mem_mb": float(mem),
            }
            print(f"  {method_name:<15} acc={metrics['acc']:.4f} bal={metrics['bal_acc']:.4f} "
                  f"f1={metrics['f1']:.4f} lat={lat:.1f}ms mem={mem:.0f}MB")

            del model
            torch.cuda.empty_cache()

        all_results[f"severity_{sev}"] = sev_results

    # Save
    with open(OUTPUT_DIR / "tta_benchmark_v5.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*90}")
    print(f"{'Sev':>4} {'Method':<15} {'Acc':>7} {'BalAcc':>7} {'F1':>7} {'Lat(ms)':>8} {'Mem(MB)':>8}")
    print(f"{'-'*90}")
    for sev in severities:
        for name, r in all_results[f"severity_{sev}"].items():
            print(f"{sev:>4} {name:<15} {r['acc']:>7.4f} {r['bal_acc']:>7.4f} "
                  f"{r['f1']:>7.4f} {r['lat_ms']:>8.1f} {r['mem_mb']:>8.0f}")
        print()
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
