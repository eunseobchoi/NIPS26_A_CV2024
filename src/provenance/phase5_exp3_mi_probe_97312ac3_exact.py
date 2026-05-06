"""Exp 3 -- Video hold-out membership-inference probe (variant-only).

Purpose (Path B Exp 3):
  Bound instance-level memorization (Feldman 2020 / Arpit 2017 probe).
  Train a DINOv2-LoRA model on CV2024 baseline-fullpool MINUS top-3
  leaked Kvasir videos, then compare per-frame predictive confidence on:
    - held-out 3 videos        (OUT of training, same source distribution)
    - dedup_val                 (OUT of training, different source mix)
    - kvasir_s1 (Smedsrud val)  (OUT of training, independent split)
    - random held-in subset     (IN of training; sampled by seed)

  If held-out-videos confidence ~ dedup_val confidence, memorization is
  not dominant. Compute MI-AUC (held-in vs held-out) via max-softmax as
  the attack score.

Training:
  DINOv2-ViT-L/14 + LoRA r=8 on last 4 qkv blocks, 10 ep, batch=128,
  AdamW (head 3e-4, LoRA 1e-4 from ep 4), label_smoothing 0.05,
  class-balanced weights (cbw beta=0.999).

I/O:
  - Reads `cv2024_training_baseline_minus_top3videos.csv`
  - Reads `cv2024_holdout_top3videos_test.csv` (held-out eval)
  - Reads `cv2024_validation_dedup_le6.csv` (dedup_val)
  - Reads `data/official_splits/split_1.csv` (Smedsrud)
  - Writes `phase5_exp3_mi_variant_n2.json` with per-sample max-softmax
    and argmax for held-out + random held-in subset.
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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(os.environ.get("CAPSULE_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "results"
CV2024_ROOT = ROOT / "data/cv2024/Dataset"

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


class CV2024DS(torch.utils.data.Dataset):
    def __init__(self, csv_path, classes=CV2024_CLASSES, tf=TF):
        df = pd.read_csv(csv_path)
        label_cols = [c for c in df.columns if c in classes]
        df = df[df[label_cols].sum(axis=1) > 0].reset_index(drop=True)
        self.df = df
        self.classes = classes
        self.label_cols = label_cols
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = row["image_path"].replace("\\", "/")
        path = CV2024_ROOT / rel
        img = Image.open(path).convert("RGB")
        label = 0
        for i, c in enumerate(self.label_cols):
            if row[c] == 1:
                label = self.classes.index(c)
                break
        return self.tf(img), label, idx


class KvasirEvalDS(torch.utils.data.Dataset):
    def __init__(self, split="split_1"):
        self.items = []
        with open(_ds.SPLITS_DIR / f"{split}.csv") as f:
            for row in csv.DictReader(f):
                kv_lbl = row["label"]
                cv_lbl = KVASIR_TO_CV.get(kv_lbl)
                if cv_lbl is None or cv_lbl not in CV2024_CLASSES:
                    continue
                if kv_lbl not in LABEL_TO_FOLDER:
                    continue
                folder = LABEL_TO_FOLDER[kv_lbl]
                path = _ds.DATA_ROOT / folder / row["filename"]
                self.items.append((str(path), CV2024_CLASSES.index(cv_lbl)))

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
def ev_with_persample(m, loader, n_cls=NCV, dump_prob=True):
    m.eval()
    P, L, MS, AR = [], [], [], []
    for img, l, _ in loader:
        img = img.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lg = m(img)
        prob = F.softmax(lg.float(), dim=1)
        max_p, arg_p = prob.max(dim=1)
        P.extend(arg_p.cpu().tolist())
        L.extend(l.tolist())
        if dump_prob:
            MS.extend(max_p.cpu().tolist())
            AR.extend(arg_p.cpu().tolist())
    P, L = np.array(P), np.array(L)
    mj = int(np.bincount(L, minlength=n_cls).argmax())
    per_class = {}
    for i, c in enumerate(CV2024_CLASSES):
        mask = L == i
        if mask.sum() > 0:
            per_class[c] = {"recall": float((P[mask] == i).mean()),
                            "n": int(mask.sum())}
    out = {
        "acc": float((P == L).mean()),
        "bal_acc": float(balanced_accuracy_score(L, P)),
        "f1_macro": float(f1_score(L, P, average="macro", zero_division=0)),
        "null_acc": float((np.full_like(L, mj) == L).mean()),
        "n_test": len(L),
        "per_class": per_class,
    }
    if dump_prob:
        out["per_sample"] = {
            "max_softmax": [float(x) for x in MS],
            "argmax": AR,
            "labels": L.tolist(),
        }
    return out


def sample_heldin_subset(full_df, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(full_df), size=min(n, len(full_df)), replace=False)
    return full_df.iloc[idx].reset_index(drop=True)


def train_one_seed(seed, args):
    set_determinism(seed)

    tr = CV2024DS(args.train_csv, tf=TF)

    labels = torch.tensor([tr.classes.index(
        [c for c in tr.label_cols if tr.df.iloc[i][c] == 1][0])
        for i in range(len(tr))])
    w = cbw(labels, NCV).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    g = torch.Generator()
    g.manual_seed(seed)
    ltr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=6,
                     pin_memory=True, persistent_workers=True, drop_last=True,
                     worker_init_fn=worker_init_fn, generator=g)

    dedup_val = CV2024DS(args.dedup_val_csv, tf=TF)
    holdout = CV2024DS(args.holdout_csv, tf=TF)
    kv_s1 = KvasirEvalDS("split_1")

    heldin_df = sample_heldin_subset(tr.df, len(holdout), seed=seed)
    heldin_csv = OUT / f"_tmp_heldin_seed{seed}.csv"
    heldin_df.to_csv(heldin_csv, index=False)
    heldin = CV2024DS(str(heldin_csv), tf=TF)

    l_dv = DataLoader(dedup_val, batch_size=args.batch * 2, shuffle=False, num_workers=6)
    l_ho = DataLoader(holdout, batch_size=args.batch * 2, shuffle=False, num_workers=6)
    l_kv = DataLoader(kv_s1, batch_size=args.batch * 2, shuffle=False, num_workers=6)
    l_hi = DataLoader(heldin, batch_size=args.batch * 2, shuffle=False, num_workers=6)
    print(f"  Train={len(tr)} DedupVal={len(dedup_val)} "
          f"HeldOut={len(holdout)} KvasirS1={len(kv_s1)} HeldIn={len(heldin)}",
          flush=True)

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
                continue
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())

        is_final = (ep == args.epochs - 1)
        m_dv = ev_with_persample(model, l_dv, dump_prob=is_final)
        m_ho = ev_with_persample(model, l_ho, dump_prob=is_final)
        m_kv = ev_with_persample(model, l_kv, dump_prob=is_final)
        m_hi = ev_with_persample(model, l_hi, dump_prob=is_final)

        mi_auc = None
        if is_final:
            score_in = m_hi["per_sample"]["max_softmax"]
            score_out = m_ho["per_sample"]["max_softmax"]
            y = [1] * len(score_in) + [0] * len(score_out)
            s = score_in + score_out
            if len(set(y)) == 2:
                mi_auc = float(roc_auc_score(y, s))

        dt = time.perf_counter() - t0
        print(f"  [seed={seed}] Ep {ep+1} dedup_bal={m_dv['bal_acc']:.4f} "
              f"holdout_bal={m_ho['bal_acc']:.4f} "
              f"kv_s1_bal={m_kv['bal_acc']:.4f} "
              f"heldin_bal={m_hi['bal_acc']:.4f}"
              + (f" | MI_AUC={mi_auc:.4f}" if mi_auc is not None else "")
              + f" [{dt:.0f}s]", flush=True)
        entry = {"epoch": ep + 1,
                 "loss": float(np.mean(losses)) if losses else float("nan"),
                 "dedup_val": m_dv, "holdout": m_ho,
                 "kvasir_s1": m_kv, "heldin": m_hi,
                 "mi_auc": mi_auc, "train_s": float(dt)}
        history.append(entry)

    try:
        heldin_csv.unlink()
    except Exception:
        pass
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv",
                    default="results/cv2024_training_baseline_minus_top3videos.csv")
    ap.add_argument("--holdout_csv",
                    default="results/cv2024_holdout_top3videos_test.csv")
    ap.add_argument("--dedup_val_csv",
                    default="results/cv2024_validation_dedup_le6.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--output", default="phase5_exp3_mi_variant_n2.json")
    args = ap.parse_args()

    for a in ("train_csv", "holdout_csv", "dedup_val_csv"):
        p = getattr(args, a)
        if not os.path.isabs(p):
            setattr(args, a, str(ROOT / p))

    results = {
        "args": vars(args),
        "meta": {
            "task": "exp3_video_holdout_mi_probe_variant_only",
            "script_sha256": file_sha256(__file__),
            "train_csv_md5": file_md5(args.train_csv),
            "holdout_csv_md5": file_md5(args.holdout_csv),
            "dedup_val_csv_md5": file_md5(args.dedup_val_csv),
        },
        "runs": [],
    }
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        history = train_one_seed(seed, args)
        last = history[-1]
        summary = {
            "epoch": last["epoch"],
            "loss": last["loss"],
            "train_s": last["train_s"],
            "dedup_bal": last["dedup_val"]["bal_acc"],
            "holdout_bal": last["holdout"]["bal_acc"],
            "kvasir_s1_bal": last["kvasir_s1"]["bal_acc"],
            "heldin_bal": last["heldin"]["bal_acc"],
            "mi_auc": last["mi_auc"],
        }
        results["runs"].append({"seed": seed, "history": history, "last": summary})
        with open(OUT / args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {OUT / args.output}", flush=True)


if __name__ == "__main__":
    main()
