"""Structured per-step training metrics.

``StepMetrics`` is the single object that captures everything measured during
one optimiser step.  It is built inside the training loop and consumed by
LossGuard, PlateauDetector, PhaseDetector, EarlyStopping, and the logger —
replacing the scattered positional arguments those components used to receive.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(slots=True)
class MoEMetrics:
    """Per-step Mixture-of-Experts diagnostics."""
    aux_loss: float = 0.0
    z_loss: float = 0.0
    load_imbalance: float = 0.0
    expert_utilization: float = 0.0

    @staticmethod
    def from_dict(d: dict) -> Optional[MoEMetrics]:
        """Build from the dict returned by ``trainer.get_moe_stats()``.

        Returns *None* when the dict has ``valid=False``.
        """
        if not d.get("valid", False):
            return None
        return MoEMetrics(
            aux_loss=d.get("aux_loss", 0.0),
            z_loss=d.get("z_loss", 0.0),
            load_imbalance=d.get("load_imbalance", 0.0),
            expert_utilization=d.get("expert_utilization", 0.0),
        )


@dataclass(slots=True)
class StepMetrics:
    """Everything measured during a single training step."""

    step: int = 0
    epoch: float = 0.0
    loss: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    tokens: int = 0
    elapsed_ms: int = 0
    phase: str = ""
    lr_overridden: bool = False
    moe: Optional[MoEMetrics] = None

    # -- derived helpers --

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_ms <= 0:
            return 0.0
        return self.tokens / (self.elapsed_ms / 1000.0)

    def to_dict(self) -> dict:
        """Flat dict suitable for JSON serialisation / wandb logging."""
        d: dict = {
            "step": self.step,
            "epoch": self.epoch,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
            "tokens": self.tokens,
            "elapsed_ms": self.elapsed_ms,
            "tokens_per_second": self.tokens_per_second,
            "phase": self.phase,
            "lr_overridden": self.lr_overridden,
        }
        if self.moe is not None:
            d["moe_aux_loss"] = self.moe.aux_loss
            d["moe_z_loss"] = self.moe.z_loss
            d["moe_load_imbalance"] = self.moe.load_imbalance
            d["moe_expert_utilization"] = self.moe.expert_utilization
        return d
