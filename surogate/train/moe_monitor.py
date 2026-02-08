"""MoE routing health monitor with diagnostic recommendations.

Tracks per-step MoE metrics over a rolling window and emits warnings when
routing degrades (expert collapse, severe imbalance, aux-loss divergence).
``get_routing_diagnostics()`` returns an actionable health report.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from surogate.train.metrics import MoEMetrics


@dataclass(slots=True)
class RoutingDiagnostics:
    """Snapshot of MoE routing health with actionable recommendations."""

    healthy: bool = True
    balance_score: float = 1.0
    utilization_score: float = 1.0
    avg_aux_loss: float = 0.0
    avg_z_loss: float = 0.0
    avg_load_imbalance: float = 1.0
    avg_expert_utilization: float = 1.0
    aux_loss_trend: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class MoEMonitor:
    """Rolling MoE routing health tracker.

    Parameters
    ----------
    logger
        Standard Python logger (``logging.Logger`` or compatible).
    window : int
        Number of recent steps to keep for rolling statistics.
    warmup : int
        Minimum steps before diagnostics become active.
    imbalance_warn : float
        ``load_imbalance`` above this triggers a warning (1.0 = perfect).
    imbalance_severe : float
        ``load_imbalance`` above this triggers a severe warning.
    utilization_warn : float
        ``expert_utilization`` below this triggers a warning (1.0 = all used).
    utilization_critical : float
        ``expert_utilization`` below this indicates expert collapse.
    aux_loss_spike_sigma : float
        Aux-loss jump > mean + sigma*std triggers a warning.
    cooldown : int
        Minimum steps between repeated warnings of the same kind.
    """

    def __init__(
        self,
        logger,
        window: int = 50,
        warmup: int = 10,
        imbalance_warn: float = 3.0,
        imbalance_severe: float = 10.0,
        utilization_warn: float = 0.8,
        utilization_critical: float = 0.5,
        aux_loss_spike_sigma: float = 3.0,
        cooldown: int = 100,
    ):
        self.logger = logger
        self.window = window
        self.warmup = warmup
        self.imbalance_warn = imbalance_warn
        self.imbalance_severe = imbalance_severe
        self.utilization_warn = utilization_warn
        self.utilization_critical = utilization_critical
        self.aux_loss_spike_sigma = aux_loss_spike_sigma
        self.cooldown = cooldown

        # Rolling history
        self._aux_losses: deque[float] = deque(maxlen=window)
        self._z_losses: deque[float] = deque(maxlen=window)
        self._imbalances: deque[float] = deque(maxlen=window)
        self._utilizations: deque[float] = deque(maxlen=window)

        # Cooldown tracking per warning category
        self._last_warn_step: dict[str, int] = {}

    def step(self, moe: Optional[MoEMetrics], step: int) -> None:
        """Feed one step's MoE metrics.  No-op when *moe* is ``None``."""
        if moe is None:
            return

        self._aux_losses.append(moe.aux_loss)
        self._z_losses.append(moe.z_loss)
        self._imbalances.append(moe.load_imbalance)
        self._utilizations.append(moe.expert_utilization)

        if len(self._aux_losses) < self.warmup:
            return

        self._check_imbalance(moe, step)
        self._check_utilization(moe, step)
        self._check_aux_loss_spike(moe, step)

    # ------------------------------------------------------------------
    # Diagnostics report
    # ------------------------------------------------------------------

    def get_routing_diagnostics(self) -> RoutingDiagnostics:
        """Return a health report with recommendations."""
        diag = RoutingDiagnostics()

        if len(self._aux_losses) < self.warmup:
            return diag

        n = len(self._aux_losses)
        diag.avg_aux_loss = sum(self._aux_losses) / n
        diag.avg_z_loss = sum(self._z_losses) / n
        diag.avg_load_imbalance = sum(self._imbalances) / n
        diag.avg_expert_utilization = sum(self._utilizations) / n

        # Balance score: 1.0 = perfect, decays as imbalance grows
        # Normalise so imbalance=1 → score=1, imbalance=10 → score≈0.1
        diag.balance_score = min(1.0, 1.0 / max(diag.avg_load_imbalance, 1.0))

        # Utilization score: direct mapping
        diag.utilization_score = diag.avg_expert_utilization

        # Aux-loss trend (simple: compare second half to first half)
        if n >= 4:
            mid = n // 2
            first_half = list(self._aux_losses)[:mid]
            second_half = list(self._aux_losses)[mid:]
            mean_first = sum(first_half) / len(first_half)
            mean_second = sum(second_half) / len(second_half)
            diag.aux_loss_trend = mean_second - mean_first
        else:
            diag.aux_loss_trend = 0.0

        # --- Recommendations ---
        recs = diag.recommendations

        if diag.balance_score < 0.3:
            recs.append(
                "Severe routing imbalance detected. "
                "Consider increasing router_aux_loss_coef (e.g. 0.01 → 0.05)."
            )
        elif diag.balance_score < 0.7:
            recs.append(
                "Moderate routing imbalance. "
                "Consider increasing router_aux_loss_coef."
            )

        if diag.utilization_score < self.utilization_critical:
            recs.append(
                f"Expert collapse risk: only {diag.utilization_score:.0%} of experts receiving tokens. "
                "Try increasing router_aux_loss_coef or lowering learning rate."
            )
        elif diag.utilization_score < self.utilization_warn:
            recs.append(
                f"Low expert utilization ({diag.utilization_score:.0%}). "
                "Some experts may be underused."
            )

        if diag.aux_loss_trend > 0 and diag.avg_aux_loss > 0:
            relative_trend = diag.aux_loss_trend / max(diag.avg_aux_loss, 1e-8)
            if relative_trend > 0.5:
                recs.append(
                    "Aux-loss is trending upward — routing may be destabilising. "
                    "Consider increasing router_z_loss_coef to regularise router logits."
                )

        diag.healthy = len(recs) == 0
        return diag

    # ------------------------------------------------------------------
    # Internal checks with per-category cooldown
    # ------------------------------------------------------------------

    def _should_warn(self, category: str, step: int) -> bool:
        last = self._last_warn_step.get(category, -self.cooldown)
        return step - last >= self.cooldown

    def _mark_warned(self, category: str, step: int) -> None:
        self._last_warn_step[category] = step

    def _check_imbalance(self, moe: MoEMetrics, step: int) -> None:
        if moe.load_imbalance > self.imbalance_severe:
            if self._should_warn("imbalance_severe", step):
                self.logger.warning(
                    f"[MoE] Severe routing imbalance at step {step}: "
                    f"{moe.load_imbalance:.1f}x "
                    f"(threshold: {self.imbalance_severe:.1f}x). "
                    f"Experts are highly unbalanced — consider increasing "
                    f"router_aux_loss_coef."
                )
                self._mark_warned("imbalance_severe", step)
        elif moe.load_imbalance > self.imbalance_warn:
            if self._should_warn("imbalance_warn", step):
                self.logger.warning(
                    f"[MoE] Routing imbalance at step {step}: "
                    f"{moe.load_imbalance:.1f}x "
                    f"(threshold: {self.imbalance_warn:.1f}x)."
                )
                self._mark_warned("imbalance_warn", step)

    def _check_utilization(self, moe: MoEMetrics, step: int) -> None:
        if moe.expert_utilization < self.utilization_critical:
            if self._should_warn("utilization_critical", step):
                self.logger.warning(
                    f"[MoE] Expert collapse risk at step {step}: "
                    f"utilization={moe.expert_utilization:.0%} "
                    f"(< {self.utilization_critical:.0%}). "
                    f"Most tokens routed to few experts."
                )
                self._mark_warned("utilization_critical", step)
        elif moe.expert_utilization < self.utilization_warn:
            if self._should_warn("utilization_warn", step):
                self.logger.warning(
                    f"[MoE] Low expert utilization at step {step}: "
                    f"{moe.expert_utilization:.0%} "
                    f"(< {self.utilization_warn:.0%})."
                )
                self._mark_warned("utilization_warn", step)

    def _check_aux_loss_spike(self, moe: MoEMetrics, step: int) -> None:
        if len(self._aux_losses) < self.warmup + 1:
            return
        # Compute stats excluding the just-appended value
        history = list(self._aux_losses)[:-1]
        n = len(history)
        mean = sum(history) / n
        variance = sum((x - mean) ** 2 for x in history) / n
        std = variance ** 0.5

        # When std=0 (constant history), fall back to a relative threshold:
        # any value > 2x the mean is a spike.
        if std > 0:
            threshold = mean + self.aux_loss_spike_sigma * std
        else:
            threshold = mean * 2.0 if mean > 0 else 1e-6
        if moe.aux_loss > threshold:
            if self._should_warn("aux_loss_spike", step):
                self.logger.warning(
                    f"[MoE] Aux-loss spike at step {step}: "
                    f"{moe.aux_loss:.4f} vs rolling mean {mean:.4f} "
                    f"(+{self.aux_loss_spike_sigma:.0f}σ threshold: {threshold:.4f}). "
                    f"Router may be destabilising."
                )
                self._mark_warned("aux_loss_spike", step)
