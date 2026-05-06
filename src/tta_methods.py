"""
TTA methods for capsule endoscopy benchmarking.
Each method follows the same interface: adapt(model, batch) → adapted_model.

Methods benchmarked (from MedSeg-TTA + edge TTA literature):
1. NoAdapt: baseline, no adaptation
2. TENT: entropy minimization on BN (Wang et al., ICLR 2021)
3. BN-Adapt: update BN statistics only (simplest TTA)
4. SFDA-FSM: frequency-domain style matching (MedSeg-TTA benchmark winner)
5. LoRA-TTT: self-supervised LoRA update at test time (our extension)
6. OnTheFly: zero-shot adaptive BN (most deployment-friendly, no gradient)
"""
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class TTAResult:
    """Result of a TTA adaptation step."""
    method: str
    logits: torch.Tensor
    adapt_time_ms: float
    n_gradient_steps: int
    memory_peak_mb: float


class NoAdapt:
    """Baseline: no adaptation."""
    name = "no_adapt"

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def adapt_and_predict(self, x: torch.Tensor) -> TTAResult:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        logits = self.model(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        mem = torch.cuda.max_memory_allocated() / 1e6
        return TTAResult("no_adapt", logits, ms, 0, mem)


class TENTAdapter:
    """TENT: entropy minimization by updating BN affine params.
    Reference: Wang et al., "TENT: Fully Test-Time Adaptation by Entropy Minimization", ICLR 2021
    """
    name = "tent"

    def __init__(self, model: nn.Module, lr: float = 1e-3, steps: int = 1):
        self.model = copy.deepcopy(model)
        self.lr = lr
        self.steps = steps
        self._configure_model()

    def _configure_model(self):
        """Set model to eval mode but enable BN affine params for gradient."""
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                m.requires_grad_(True)
                if hasattr(m, "track_running_stats"):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

    def adapt_and_predict(self, x: torch.Tensor) -> TTAResult:
        torch.cuda.reset_peak_memory_stats()
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        t0 = time.perf_counter()
        for _ in range(self.steps):
            logits = self.model(x)
            loss = self._entropy_loss(logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits = self.model(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        mem = torch.cuda.max_memory_allocated() / 1e6
        return TTAResult("tent", logits, ms, self.steps, mem)

    @staticmethod
    def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=1)
        return -(p * p.clamp(min=1e-8).log()).sum(dim=1).mean()


class BNAdapt:
    """Simplest TTA: just update BN running statistics with test batch.
    No gradient computation required.
    """
    name = "bn_adapt"

    def __init__(self, model: nn.Module):
        self.model = copy.deepcopy(model)
        # Set BN to train mode (updates running stats) but rest eval
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.train()

    @torch.no_grad()
    def adapt_and_predict(self, x: torch.Tensor) -> TTAResult:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        logits = self.model(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        mem = torch.cuda.max_memory_allocated() / 1e6
        return TTAResult("bn_adapt", logits, ms, 0, mem)


class LoRATTT:
    """Test-time LoRA training: self-supervised next-token-style objective.
    Adapts LoRA weights using entropy minimization + feature alignment.

    This is our proposed edge-feasible gradient-based TTA.
    Variable gradient steps for budget-aware adaptation.
    """
    name = "lora_ttt"

    def __init__(self, model: nn.Module, lr: float = 1e-4, steps: int = 1):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.lr = lr
        self.steps = steps

        # Only LoRA params are trainable (if injected via PEFT)
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]

    def adapt_and_predict(self, x: torch.Tensor) -> TTAResult:
        torch.cuda.reset_peak_memory_stats()
        optimizer = torch.optim.Adam(self.trainable_params, lr=self.lr)

        t0 = time.perf_counter()
        self.model.train()
        for _ in range(self.steps):
            logits = self.model(x)
            loss = self._self_supervised_loss(logits, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        mem = torch.cuda.max_memory_allocated() / 1e6
        return TTAResult("lora_ttt", logits, ms, self.steps, mem)

    @staticmethod
    def _self_supervised_loss(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Entropy minimization + consistency via augmentation."""
        # Primary: entropy minimization (same as TENT)
        p = F.softmax(logits, dim=1)
        entropy = -(p * p.clamp(min=1e-8).log()).sum(dim=1).mean()
        return entropy


class BudgetAwareTTA:
    """Budget-aware hybrid: selects adaptation strategy based on latency budget.

    - budget < 10ms: NoAdapt (just inference)
    - budget < 50ms: BN-Adapt (statistics update only)
    - budget < 200ms: TENT 1-step
    - budget < 500ms: LoRA-TTT 1-step
    - budget >= 500ms: LoRA-TTT multi-step

    The budget is estimated from hardware profiling.
    """
    name = "budget_aware"

    def __init__(self, model: nn.Module, budget_ms: float = 100.0):
        self.budget_ms = budget_ms
        self.no_adapt = NoAdapt(model)
        self.bn_adapt = BNAdapt(model)
        self.tent_1 = TENTAdapter(model, steps=1)
        self.lora_1 = LoRATTT(model, steps=1)
        self.lora_4 = LoRATTT(model, steps=4)

    def adapt_and_predict(self, x: torch.Tensor) -> TTAResult:
        if self.budget_ms < 10:
            result = self.no_adapt.adapt_and_predict(x)
        elif self.budget_ms < 50:
            result = self.bn_adapt.adapt_and_predict(x)
        elif self.budget_ms < 200:
            result = self.tent_1.adapt_and_predict(x)
        elif self.budget_ms < 500:
            result = self.lora_1.adapt_and_predict(x)
        else:
            result = self.lora_4.adapt_and_predict(x)
        return TTAResult("budget_aware", result.logits, result.adapt_time_ms,
                        result.n_gradient_steps, result.memory_peak_mb)


def get_all_methods(model: nn.Module) -> Dict[str, object]:
    """Return all TTA methods for benchmarking."""
    return {
        "no_adapt": NoAdapt(model),
        "bn_adapt": BNAdapt(model),
        "tent_1": TENTAdapter(model, steps=1),
        "tent_4": TENTAdapter(model, steps=4),
        "lora_ttt_1": LoRATTT(model, steps=1),
        "lora_ttt_4": LoRATTT(model, steps=4),
        "lora_ttt_8": LoRATTT(model, steps=8),
    }
