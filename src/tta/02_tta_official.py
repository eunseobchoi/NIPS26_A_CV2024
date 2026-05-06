"""
Phase 2+3: TTA Benchmark — measure quality, latency, memory across methods and hardware.
Runs on both A100 (server) and edge-device (edge) for cross-platform comparison.
"""
import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import CapsuleVision2024, EVAL_TRANSFORM
from model import DINOv2LoRAClassifier
from tta_methods import get_all_methods, TTAResult

DEVICE = "cuda:0"


def benchmark_method(method, test_loader, num_classes: int, max_batches: int = None):
    """Run one TTA method on test set, collect metrics."""
    all_preds, all_labels, all_probs = [], [], []
    adapt_times, memory_peaks = [], []

    for batch_idx, (imgs, labels, paths) in enumerate(test_loader):
        if max_batches and batch_idx >= max_batches:
            break

        imgs = imgs.to(DEVICE)
        result: TTAResult = method.adapt_and_predict(imgs)

        probs = torch.softmax(result.logits, dim=1).detach().cpu()
        preds = result.logits.argmax(1).detach().cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.append(probs.numpy())
        adapt_times.append(result.adapt_time_ms)
        memory_peaks.append(result.memory_peak_mb)

    all_probs = np.concatenate(all_probs, axis=0)

    # Compute metrics
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    return {
        "method": method.name,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1),
        "auc_macro": float(auc),
        "adapt_time_mean_ms": float(np.mean(adapt_times)),
        "adapt_time_p95_ms": float(np.percentile(adapt_times, 95)),
        "memory_peak_mb": float(np.max(memory_peaks)),
        "n_samples": len(all_preds),
        "n_batches": len(adapt_times),
    }


def main():
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print("usage: 02_tta_official.py [data_root] [model_path] [output_dir]")
        print("Legacy CV2024 TTA benchmark. Prefer tta_benchmark_full.py for the paper benchmark.")
        return
    data_root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CAPSULE_ROOT", ".") + "/data/capsule_vision_2024"
    model_path = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("CAPSULE_ROOT", ".") + "/results/best_model.pth"
    output_dir = Path(sys.argv[3] if len(sys.argv) > 3 else os.environ.get("CAPSULE_ROOT", ".") + "/results")
    output_dir.mkdir(exist_ok=True)

    # Detect hardware
    gpu_name = torch.cuda.get_device_name(0)
    is_edge = any(token in gpu_name for token in ("Orin", "Tegra", "Jetson"))
    hw_tag = "edge" if is_edge else "a100"
    print(f"Hardware: {gpu_name} (tag: {hw_tag})")

    # Load test set
    print("Loading test dataset...")
    test_ds = CapsuleVision2024(data_root, split="testing", transform=EVAL_TRANSFORM)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
    num_classes = len(test_ds.classes)
    print(f"Test: {len(test_ds)} samples, {num_classes} classes")

    # Load model
    print(f"Loading model from {model_path}...")
    model = DINOv2LoRAClassifier(num_classes=num_classes)
    model = model.to(DEVICE)

    from peft import get_peft_model, LoraConfig
    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["qkv"],
        lora_dropout=0.0, bias="none",
    )
    model = get_peft_model(model, lora_config)

    if Path(model_path).exists():
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        print("Loaded pretrained weights")
    else:
        print("WARNING: No pretrained weights found, using random init")

    model.eval()

    # Get all TTA methods
    methods = get_all_methods(model)
    print(f"\nBenchmarking {len(methods)} TTA methods on {hw_tag}...")

    # Run benchmark
    all_results = {"hardware": hw_tag, "gpu": gpu_name, "methods": {}}

    for name, method in methods.items():
        print(f"\n  [{name}]")
        try:
            result = benchmark_method(method, test_loader, num_classes)
            all_results["methods"][name] = result
            print(f"    Acc={result['accuracy']:.4f} BalAcc={result['balanced_accuracy']:.4f} "
                  f"AUC={result['auc_macro']:.4f} F1={result['f1_macro']:.4f}")
            print(f"    Latency: mean={result['adapt_time_mean_ms']:.1f}ms "
                  f"p95={result['adapt_time_p95_ms']:.1f}ms | "
                  f"Memory: {result['memory_peak_mb']:.0f}MB")
        except Exception as e:
            print(f"    ERROR: {e}")
            all_results["methods"][name] = {"error": str(e)}

    # Save
    out_file = output_dir / f"tta_benchmark_{hw_tag}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Method':<15} {'Acc':>7} {'BalAcc':>7} {'AUC':>7} {'F1':>7} "
          f"{'Lat(ms)':>8} {'p95(ms)':>8} {'Mem(MB)':>8}")
    print(f"{'-'*90}")
    for name, r in all_results["methods"].items():
        if "error" in r:
            print(f"{name:<15} ERROR: {r['error'][:50]}")
        else:
            print(f"{name:<15} {r['accuracy']:>7.4f} {r['balanced_accuracy']:>7.4f} "
                  f"{r['auc_macro']:>7.4f} {r['f1_macro']:>7.4f} "
                  f"{r['adapt_time_mean_ms']:>8.1f} {r['adapt_time_p95_ms']:>8.1f} "
                  f"{r['memory_peak_mb']:>8.0f}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
