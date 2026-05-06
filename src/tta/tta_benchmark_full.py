"""Reproducible TTA benchmark for the E&D submission.

Produces the 288-run JSON at
``artifact-root/results/tta/tta_bench_official.json``:
   2 folds x 6 severities x 3 seeds x 8 methods = 288.

This file is the reproducible benchmark entry point for the packaged 288-run
JSON. Earlier lightweight development scripts covered only a subset of these
methods, severities, seeds, and folds; they are retained only as historical
implementation context.

Methods
-------
no_adapt      Frozen-backbone reference. No adaptation.
ln_adapt      LayerNorm-only, entropy minimisation (TENT-style; Wang et al.,
              ICLR 2021, "Tent: Fully Test-Time Adaptation by Entropy
              Minimization").
head_ttt      Classifier-head-only entropy min with reliability filter
              (entropy < margin). Gandelsman et al., NeurIPS 2022, "Test-Time
              Training with Masked Autoencoders" motivates per-sample head
              updates.
lora_ttt      Update only the LoRA wrappers that live on the last 4 attention
              qkv layers. Filtered entropy loss. (author-proposed variant of
              TENT on LoRA parameters; see Hu et al., ICLR 2022, "LoRA:
              Low-Rank Adaptation of Large Language Models").
sar_naive     TENT on LN params plus a naive 1-step FGSM-style sign
              perturbation. Author-proposed lightweight caricature of SAR
              **without** SAM / reliable-sample filtering / reset. Included
              only as an ablation; NOT the official SAR method.
sar_official  Niu et al., ICLR 2023, "Towards Stable Test-Time Adaptation in
              Dynamic Wild World". Ports the reference implementation at
              github.com/mr-eggplant/SAR: reliable-sample entropy filtering
              (< margin), SAM optimiser (rho=0.05) wrapping SGD (lr=2.5e-4,
              momentum=0.9), EMA loss with reset threshold (0.2). LN-only
              parameters, excluding the last 3 transformer blocks (the
              reference's `configure_model` excludes blocks 9-11 of 12; we
              extend this rule to skip the last 3 of 24 blocks in ViT-L).
od_tta        Author-proposed. Entropy-gated on-demand TTA: only adapt when
              per-batch mean entropy exceeds a threshold; otherwise return
              frozen-model logits. Adapts LN params when triggered.
              NO peer-reviewed citation - internal exploration.
hybrid_tta    Author-proposed. If per-batch mean entropy > tau, full LN+head
              adaptation; else head-only lite. NO peer-reviewed citation -
              internal exploration.

Important caveats
-----------------
1. ``get_corruption(severity)`` uses Gaussian blur + ColorJitter. It is NOT
   the ImageNet-C ``motion_blur`` kernel (Hendrycks & Dietterich, ICLR 2019).
   Throughout the paper, references to "motion_blur" or "corruption" at
   severity ``s`` mean this synthetic proxy. Severity 0 applies only the
   ImageNet normalisation - no corruption.

2. SAR-official's exclusion of the "last 3 transformer blocks" mirrors the
   reference repository's ``configure_model``. The reference code was written
   for ViT-B/12; we extrapolate the rule to ViT-L/24 by skipping the last 3
   blocks out of 24. This heuristic is documented in the paper's appendix.

3. AMP is enabled by default (``torch.amp.autocast("cuda", dtype=fp16)``) to
   match the A100 production run that produced the shipped JSON. Running on
   NVIDIA edge-device (sm_110) with AMP enabled destroys DINOv2 LayerNorm stats;
   disable AMP on edge-device.

Reproducibility
---------------
python tta_benchmark_full.py \\
  --data-root /path/to/kvasir_capsule/labelled_images \\
  --splits-dir /path/to/official_splits \\
  --ckpt-dir /path/to/checkpoints \\
  --folds 0,1 \\
  --severities 0,1,2,3,4,5 \\
  --seeds 0,1,2 \\
  --methods no_adapt,ln_adapt,sar_naive,head_ttt,lora_ttt,sar_official,od_tta,hybrid_tta \\
  --output tta_bench_official.json

Dependencies: PyTorch >= 2.1, torchvision, numpy, scikit-learn, PIL. DINOv2
weights are fetched from ``facebookresearch/dinov2`` via ``torch.hub``.
Checkpoint filename convention: ``dinov2_lora_official_fold{fold}.pth``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
DEFAULT_METHODS = [
    "no_adapt",
    "ln_adapt",
    "sar_naive",
    "head_ttt",
    "lora_ttt",
    "sar_official",
    "od_tta",
    "hybrid_tta",
]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip() != ""]


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip() != ""]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Reproducible TTA benchmark (288 runs default).",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("CAPSULE_ROOT", ".") + "/data/kvasir_capsule/labelled_images"),
        help="Root of labelled Kvasir-Capsule images.",
    )
    ap.add_argument(
        "--splits-dir",
        type=Path,
        default=Path(os.environ.get("CAPSULE_ROOT", ".") + "/data/official_splits"),
        help="Directory containing split_0.csv / split_1.csv.",
    )
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path(os.environ.get("CAPSULE_ROOT", ".") + "/results"),
        help="Directory containing dinov2_lora_official_fold{fold}.pth.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("tta_bench_official.json"),
        help="Output JSON path (absolute or relative to --output-dir).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("CAPSULE_ARTIFACT_ROOT", ".")) / "results/tta",
    )
    ap.add_argument("--folds", type=_parse_int_list, default=[0, 1])
    ap.add_argument(
        "--severities", type=_parse_int_list, default=[0, 1, 2, 3, 4, 5]
    )
    ap.add_argument("--seeds", type=_parse_int_list, default=[0, 1, 2])
    ap.add_argument("--methods", type=_parse_str_list, default=DEFAULT_METHODS)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-blocks", type=int, default=4)
    ap.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable AMP (fp16) autocast. Disable on edge-device (sm_110).",
    )
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan and validate CLI only. No data loading.",
    )
    ap.add_argument(
        "--src-dir",
        type=Path,
        default=Path(os.environ.get("CAPSULE_ROOT", ".") + "/src"),
        help="Directory that contains dataset_official.py.",
    )
    return ap


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Corruption pipeline (NOT ImageNet-C motion_blur)
# ---------------------------------------------------------------------------
def get_corruption(severity: int) -> transforms.Compose:
    """Synthetic corruption used as a nuisance proxy.

    We explicitly do NOT use ImageNet-C motion_blur kernels: we apply a
    Gaussian blur plus photometric ColorJitter whose strength scales linearly
    with ``severity / 5``. Severity 0 -> identity photometric + normalize.
    """
    s = severity / 5.0
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(
                brightness=0.5 * s,
                contrast=0.6 * s,
                saturation=0.5 * s,
                hue=0.1 * s,
            ),
            transforms.GaussianBlur(
                kernel_size=int(5 + 8 * s) | 1,
                sigma=(0.5, 2.0 + 4 * s),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Model definition (DINOv2-ViT-L + LoRA on last N qkv + head)
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(
        self, base: nn.Linear, r: int = 8, alpha: float = 16.0
    ) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        d_in, d_out = base.in_features, base.out_features
        self.lora_a = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(d_out, r))
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale
        return self.base(x) + delta


class DINOv2LoRAClassifier(nn.Module):
    def __init__(
        self, n_cls: int, lora_r: int = 8, lora_blocks: int = 4
    ) -> None:
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
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def load_frozen_model(
    fold: int,
    n_cls: int,
    ckpt_dir: Path,
    device: torch.device,
    lora_r: int,
    lora_blocks: int,
) -> DINOv2LoRAClassifier:
    model = DINOv2LoRAClassifier(
        n_cls=n_cls, lora_r=lora_r, lora_blocks=lora_blocks
    ).to(device)
    ckpt_path = ckpt_dir / f"dinov2_lora_official_fold{fold}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Parameter groups
# ---------------------------------------------------------------------------
def get_ln_params(model: DINOv2LoRAClassifier) -> list[torch.Tensor]:
    ps: list[torch.Tensor] = []
    for m in model.backbone.modules():
        if isinstance(m, nn.LayerNorm):
            ps.extend([m.weight, m.bias])
    return ps


def get_ln_params_sar(model: DINOv2LoRAClassifier) -> list[torch.Tensor]:
    """LN params used by SAR-official.

    Excludes the last 3 transformer blocks (see module docstring). Also
    excludes the backbone's trailing ``norm`` module, matching the reference
    SAR implementation at github.com/mr-eggplant/SAR.
    """
    ps: list[torch.Tensor] = []
    n_blocks = len(model.backbone.blocks)
    skip_from = n_blocks - 3
    for idx, blk in enumerate(model.backbone.blocks):
        if idx >= skip_from:
            continue
        for sub in blk.modules():
            if isinstance(sub, nn.LayerNorm):
                ps.extend([sub.weight, sub.bias])
    return ps


def get_head_params(model: DINOv2LoRAClassifier) -> list[torch.Tensor]:
    return list(model.head.parameters())


def get_lora_params(model: DINOv2LoRAClassifier) -> list[torch.Tensor]:
    return [p for w in model.lora_wrappers for p in (w.lora_a, w.lora_b)]


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------
def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=1)
    return -(p * p.clamp(min=1e-8).log()).sum(dim=1)


def entropy_loss(
    logits: torch.Tensor, margin: float | None = None
) -> torch.Tensor:
    ent = entropy_from_logits(logits)
    if margin is None:
        return ent.mean()
    mask = ent < margin
    if mask.sum() > 1:
        return ent[mask].mean()
    return ent.mean() * 0.1


# ---------------------------------------------------------------------------
# SAM optimiser for SAR-official (direct port of mr-eggplant/SAR/sam.py)
# ---------------------------------------------------------------------------
class SAMOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Any,
        base_optimizer: Any,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs: Any,
    ) -> None:
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        dev = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack(
                [
                    (
                        (torch.abs(p) if group["adaptive"] else 1.0) * p.grad
                    ).norm(p=2).to(dev)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )


# ---------------------------------------------------------------------------
# TTA runners
# ---------------------------------------------------------------------------
def _restore(params: list[torch.Tensor], originals: list[torch.Tensor]) -> None:
    for p, o in zip(params, originals):
        p.data.copy_(o)


def run_generic(
    method: str,
    model: DINOv2LoRAClassifier,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    lr: float = 1e-4,
    steps: int = 1,
    use_amp: bool = True,
) -> tuple[list[int], list[int], list[float], float]:
    """Handle no_adapt / ln_adapt / sar_naive / head_ttt / lora_ttt."""
    cfg = {
        "no_adapt": (_no_params(), 0, None, False),
        "ln_adapt": (get_ln_params(model), steps, None, False),
        "sar_naive": (get_ln_params(model), steps, None, True),
        "head_ttt": (get_head_params(model), steps, margin, False),
        "lora_ttt": (get_lora_params(model), steps, margin, False),
    }[method]
    params, n_steps, use_margin, use_sar = cfg

    originals = [p.data.clone() for p in params]
    for p in params:
        p.requires_grad_(True)

    all_preds, all_labs, batch_times = [], [], []
    torch.cuda.reset_peak_memory_stats()

    for imgs, labs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        _restore(params, originals)

        t0 = time.perf_counter()
        if n_steps > 0:
            model.train()
            opt = torch.optim.Adam(params, lr=lr)
            for _ in range(n_steps):
                with torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=torch.float16
                ):
                    logits = model(imgs)
                loss = entropy_loss(logits, margin=use_margin)
                opt.zero_grad()
                loss.backward()
                if use_sar:
                    # NOTE: sar_naive uses sign(grad) perturbation only.
                    # Official SAR uses SAM; see run_sar_official.
                    with torch.no_grad():
                        for p in params:
                            if p.grad is not None:
                                p.add_(0.05 * p.grad.sign())
                    with torch.amp.autocast(
                        "cuda", enabled=use_amp, dtype=torch.float16
                    ):
                        logits2 = model(imgs)
                    loss2 = entropy_loss(logits2, margin=use_margin)
                    opt.zero_grad()
                    loss2.backward()
                    with torch.no_grad():
                        for p in params:
                            if p.grad is not None:
                                p.sub_(0.05 * p.grad.sign())
                opt.step()

        model.eval()
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=use_amp, dtype=torch.float16
        ):
            logits = model(imgs)
        torch.cuda.synchronize()
        batch_times.append((time.perf_counter() - t0) * 1000)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labs.extend(labs.tolist())

    _restore(params, originals)
    for p in params:
        p.requires_grad_(False)

    mem = torch.cuda.max_memory_allocated() / 1e6
    return all_preds, all_labs, batch_times, mem


def _no_params() -> list[torch.Tensor]:
    return []


def run_sar_official(
    model: DINOv2LoRAClassifier,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    lr: float = 2.5e-4,
    reset_thresh: float = 0.2,
    use_amp: bool = True,
) -> tuple[list[int], list[int], list[float], float]:
    params = get_ln_params_sar(model)
    originals = [p.data.clone() for p in params]
    for p in params:
        p.requires_grad_(True)
    opt = SAMOptimizer(
        params, torch.optim.SGD, lr=lr, momentum=0.9, rho=0.05
    )

    ema_loss: float | None = None
    preds, labs_all, times = [], [], []
    torch.cuda.reset_peak_memory_stats()

    for imgs, labs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        _restore(params, originals)  # per-batch reset

        t0 = time.perf_counter()
        model.train()
        with torch.amp.autocast(
            "cuda", enabled=use_amp, dtype=torch.float16
        ):
            out = model(imgs)
        ent = entropy_from_logits(out.float())
        mask1 = ent < margin
        if mask1.sum() > 1:
            loss = ent[mask1].mean()
            opt.zero_grad()
            loss.backward()
            opt.first_step(zero_grad=True)
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=torch.float16
            ):
                out2 = model(imgs)
            ent2 = entropy_from_logits(out2.float())
            mask2 = ent2 < margin
            if mask2.sum() > 1:
                loss2 = ent2[mask2].mean()
                ema_loss = (
                    loss2.item()
                    if ema_loss is None
                    else 0.9 * ema_loss + 0.1 * loss2.item()
                )
                if ema_loss < reset_thresh:
                    _restore(params, originals)
                    ema_loss = None
                else:
                    loss2.backward()
                    opt.second_step(zero_grad=True)

        model.eval()
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=use_amp, dtype=torch.float16
        ):
            logits = model(imgs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        preds.extend(logits.argmax(1).cpu().tolist())
        labs_all.extend(labs.tolist())

    _restore(params, originals)
    for p in params:
        p.requires_grad_(False)
    mem = torch.cuda.max_memory_allocated() / 1e6
    return preds, labs_all, times, mem


def run_od_tta(
    model: DINOv2LoRAClassifier,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    entropy_threshold: float,
    steps: int = 1,
    lr: float = 1e-4,
    use_amp: bool = True,
) -> tuple[list[int], list[int], list[float], float, int]:
    ln_params = get_ln_params(model)
    originals = [p.data.clone() for p in ln_params]
    for p in ln_params:
        p.requires_grad_(True)

    preds, labs_all, times = [], [], []
    triggers = 0
    torch.cuda.reset_peak_memory_stats()

    for imgs, labs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        _restore(ln_params, originals)

        t0 = time.perf_counter()
        model.eval()
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=use_amp, dtype=torch.float16
        ):
            pre_logits = model(imgs)
        pre_ent = entropy_from_logits(pre_logits.float()).mean().item()

        if pre_ent > entropy_threshold:
            triggers += 1
            model.train()
            opt = torch.optim.Adam(ln_params, lr=lr)
            for _ in range(steps):
                with torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=torch.float16
                ):
                    logits = model(imgs)
                loss = entropy_loss(logits, margin=margin)
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=torch.float16
            ):
                post_logits = model(imgs)
        else:
            post_logits = pre_logits

        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        preds.extend(post_logits.argmax(1).cpu().tolist())
        labs_all.extend(labs.tolist())

    _restore(ln_params, originals)
    for p in ln_params:
        p.requires_grad_(False)
    mem = torch.cuda.max_memory_allocated() / 1e6
    return preds, labs_all, times, mem, triggers


def run_hybrid_tta(
    model: DINOv2LoRAClassifier,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    tau: float,
    steps: int = 1,
    use_amp: bool = True,
) -> tuple[list[int], list[int], list[float], float, dict[str, int]]:
    ln_params = get_ln_params(model)
    head_params = get_head_params(model)
    ln_orig = [p.data.clone() for p in ln_params]
    head_orig = [p.data.clone() for p in head_params]
    for p in ln_params + head_params:
        p.requires_grad_(True)

    preds, labs_all, times = [], [], []
    mode_counts: dict[str, int] = {"full": 0, "light": 0}
    torch.cuda.reset_peak_memory_stats()

    for imgs, labs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        _restore(ln_params, ln_orig)
        _restore(head_params, head_orig)

        t0 = time.perf_counter()
        model.eval()
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=use_amp, dtype=torch.float16
        ):
            pre = model(imgs)
        pre_ent = entropy_from_logits(pre.float()).mean().item()

        model.train()
        if pre_ent > tau:
            mode_counts["full"] += 1
            opt = torch.optim.Adam(ln_params + head_params, lr=1e-4)
            for _ in range(steps):
                with torch.amp.autocast(
                    "cuda", enabled=use_amp, dtype=torch.float16
                ):
                    logits = model(imgs)
                loss = entropy_loss(logits, margin=margin)
                opt.zero_grad()
                loss.backward()
                opt.step()
        else:
            mode_counts["light"] += 1
            opt = torch.optim.Adam(head_params, lr=1e-4)
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=torch.float16
            ):
                logits = model(imgs)
            loss = entropy_loss(logits, margin=margin)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=use_amp, dtype=torch.float16
        ):
            post = model(imgs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        preds.extend(post.argmax(1).cpu().tolist())
        labs_all.extend(labs.tolist())

    _restore(ln_params, ln_orig)
    _restore(head_params, head_orig)
    for p in ln_params + head_params:
        p.requires_grad_(False)
    mem = torch.cuda.max_memory_allocated() / 1e6
    return preds, labs_all, times, mem, mode_counts


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------
def compute_metrics(
    preds: list[int],
    labs: list[int],
    official_classes: list[str],
    n_cls: int,
) -> dict[str, Any]:
    a_preds, a_labs = np.asarray(preds), np.asarray(labs)
    acc = float((a_preds == a_labs).mean())
    bal = float(balanced_accuracy_score(a_labs, a_preds))
    f1m = float(f1_score(a_labs, a_preds, average="macro", zero_division=0))
    majority = int(np.bincount(a_labs).argmax())
    null_acc = float((np.full_like(a_labs, majority) == a_labs).mean())
    cm = confusion_matrix(a_labs, a_preds, labels=list(range(n_cls)))
    per_class: dict[str, dict[str, Any]] = {}
    for i, cls in enumerate(official_classes):
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


# ---------------------------------------------------------------------------
# Top-level run planner / dispatcher
# ---------------------------------------------------------------------------
def _format_plan(args: argparse.Namespace) -> str:
    total = (
        len(args.folds)
        * len(args.severities)
        * len(args.seeds)
        * len(args.methods)
    )
    return (
        f"folds={args.folds} severities={args.severities} "
        f"seeds={args.seeds} methods={args.methods} "
        f"total_runs={total}"
    )


def _validate_methods(methods: list[str]) -> None:
    unknown = [m for m in methods if m not in DEFAULT_METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}; valid: {DEFAULT_METHODS}")


def _run_method(
    method: str,
    fold: int,
    sev: int,
    seed: int,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    official_classes: list[str],
    n_cls: int,
    ckpt_dir: Path,
    lora_r: int,
    lora_blocks: int,
    use_amp: bool,
) -> dict[str, Any]:
    """Freshly load the checkpoint, run one method, return one result dict."""
    model = load_frozen_model(
        fold=fold,
        n_cls=n_cls,
        ckpt_dir=ckpt_dir,
        device=device,
        lora_r=lora_r,
        lora_blocks=lora_blocks,
    )
    t0 = time.perf_counter()
    extras: dict[str, Any] = {}

    if method in {"no_adapt", "ln_adapt", "sar_naive", "head_ttt", "lora_ttt"}:
        preds, labs, times, mem = run_generic(
            method=method,
            model=model,
            loader=loader,
            device=device,
            margin=margin,
            lr=1e-4,
            steps=1,
            use_amp=use_amp,
        )
    elif method == "sar_official":
        preds, labs, times, mem = run_sar_official(
            model=model,
            loader=loader,
            device=device,
            margin=margin,
            lr=2.5e-4,
            use_amp=use_amp,
        )
    elif method == "od_tta":
        preds, labs, times, mem, triggers = run_od_tta(
            model=model,
            loader=loader,
            device=device,
            margin=margin,
            entropy_threshold=margin * 0.9,
            use_amp=use_amp,
        )
        extras["triggers"] = int(triggers)
    elif method == "hybrid_tta":
        preds, labs, times, mem, modes = run_hybrid_tta(
            model=model,
            loader=loader,
            device=device,
            margin=margin,
            tau=margin * 0.85,
            use_amp=use_amp,
        )
        extras["modes"] = modes
    else:
        raise AssertionError(f"unreachable method: {method}")

    total_s = time.perf_counter() - t0
    m = compute_metrics(preds, labs, official_classes, n_cls)

    res = {
        "fold": fold,
        "severity": sev,
        "seed": seed,
        "method": method,
        "acc": m["acc"],
        "bal_acc": m["bal_acc"],
        "f1_macro": m["f1_macro"],
        "null_acc": m["null_acc"],
        "lat_ms_mean": float(np.mean(times)),
        "mem_mb": float(mem),
        **extras,
        "per_class": m["per_class"],
    }
    if method in {"no_adapt", "ln_adapt", "sar_naive", "head_ttt", "lora_ttt", "sar_official"}:
        res["total_s"] = float(total_s)

    del model
    torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()
    _validate_methods(args.methods)

    plan = _format_plan(args)
    print(f"[plan] {plan}", flush=True)

    if args.dry_run:
        total = (
            len(args.folds)
            * len(args.severities)
            * len(args.seeds)
            * len(args.methods)
        )
        print(f"[dry-run] validated CLI. Would run {total} total configs.", flush=True)
        print(
            "[dry-run] skipping data load, checkpoint load, and training.",
            flush=True,
        )
        return

    # Lazy import so --dry-run works without the dataset module on path.
    sys.path.insert(0, str(args.src_dir))
    import dataset_official as _ds

    _ds.DATA_ROOT = args.data_root
    _ds.SPLITS_DIR = args.splits_dir
    from dataset_official import (
        KvasirCapsuleOfficial,
        NUM_CLASSES,
        OFFICIAL_CLASSES,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    margin = 0.4 * float(np.log(NUM_CLASSES))
    results: dict[str, Any] = {
        "args": {
            "folds": args.folds,
            "severities": args.severities,
            "seeds": args.seeds,
            "batch": args.batch,
            "lora_r": args.lora_r,
            "output": args.output.name,
        },
        "margin": margin,
        "results": [],
        "paired_corruptions": True,
    }

    out_path = (
        args.output
        if args.output.is_absolute()
        else args.output_dir / args.output
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        test_key = f"split_{1 - fold}"
        for sev in args.severities:
            tf = get_corruption(sev)
            ds = KvasirCapsuleOfficial(test_key, transform=tf)
            for seed in args.seeds:
                set_seed(seed)
                loader = DataLoader(
                    ds,
                    batch_size=args.batch,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                for method in args.methods:
                    # Reset RNG before each method so stochastic corruptions are
                    # paired within each fold/severity/seed comparison.
                    set_seed(seed)
                    res = _run_method(
                        method=method,
                        fold=fold,
                        sev=sev,
                        seed=seed,
                        loader=loader,
                        device=device,
                        margin=margin,
                        official_classes=OFFICIAL_CLASSES,
                        n_cls=NUM_CLASSES,
                        ckpt_dir=args.ckpt_dir,
                        lora_r=args.lora_r,
                        lora_blocks=args.lora_blocks,
                        use_amp=args.amp,
                    )
                    results["results"].append(res)
                    print(
                        f"  F{fold} sev={sev} seed={seed} {method:<13} "
                        f"acc={res['acc']:.4f} bal={res['bal_acc']:.4f} "
                        f"f1={res['f1_macro']:.4f} "
                        f"lat={res['lat_ms_mean']:.0f}ms "
                        f"mem={res['mem_mb']:.0f}MB",
                        flush=True,
                    )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[done] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
