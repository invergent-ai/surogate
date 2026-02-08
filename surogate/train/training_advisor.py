"""Cross-component training intelligence.

The ``TrainingAdvisor`` correlates signals from all monitoring components
(PhaseDetector, GradientTracker, PlateauDetector, LossGuard, MoEMonitor,
LRSchedule) and emits actionable recommendations that no single component
could generate in isolation.

Advisory only — logs warnings/info, never takes automatic actions.
"""

from __future__ import annotations

from typing import Optional

from surogate.train.metrics import StepMetrics
from surogate.train.phase_detector import TrainingPhase


class TrainingAdvisor:
    """Correlate cross-component signals and emit recommendations.

    Parameters
    ----------
    logger
        Standard Python logger.
    phase_detector
        PhaseDetector instance (always present).
    gradient_tracker
        GradientTracker instance (always present).
    plateau_detector
        PlateauDetector instance (always present).
    loss_guard
        LossGuard instance, or *None* if ``auto_lr_reduction`` is disabled.
    moe_monitor
        MoEMonitor instance (always present, but no-ops for non-MoE models).
    lr_schedule
        LRSchedule instance (always present).
    max_steps : int
        Total training steps (used for warmup boundary check).
    warmup_steps : int
        Number of warmup steps in the LR schedule.
    cooldown : int
        Minimum steps between repeated warnings of the same category.
    """

    def __init__(
        self,
        logger,
        phase_detector,
        gradient_tracker,
        plateau_detector,
        loss_guard,
        moe_monitor,
        lr_schedule,
        max_steps: int,
        warmup_steps: int = 0,
        cooldown: int = 200,
    ):
        self.logger = logger
        self.phase_detector = phase_detector
        self.gradient_tracker = gradient_tracker
        self.plateau_detector = plateau_detector
        self.loss_guard = loss_guard          # may be None
        self.moe_monitor = moe_monitor
        self.lr_schedule = lr_schedule
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.cooldown = cooldown

        # Per-category cooldown tracking (same pattern as MoEMonitor)
        self._last_warn_step: dict[str, int] = {}

        # One-shot flags
        self._warmup_checked = False

        # Sustained-condition counters
        self._plateau_grad_flat_count = 0
        self._converging_grad_shrink_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, metrics: StepMetrics, step: int) -> None:
        """Run all cross-signal checks for this training step."""
        phase = self.phase_detector.current_phase

        self._check_plateau_high_lr(metrics, phase, step)
        self._check_diverging_moe_collapse(metrics, phase, step)
        self._check_gradient_vanishing(metrics, phase, step)
        self._check_spike_moe_correlation(metrics, phase, step)
        self._check_lr_reductions_ineffective(metrics, phase, step)
        self._check_warmup_too_short(metrics, phase, step)

    # ------------------------------------------------------------------
    # Cooldown helpers
    # ------------------------------------------------------------------

    def _should_warn(self, category: str, step: int) -> bool:
        last = self._last_warn_step.get(category, -self.cooldown)
        return step - last >= self.cooldown

    def _mark_warned(self, category: str, step: int) -> None:
        self._last_warn_step[category] = step

    # ------------------------------------------------------------------
    # Rule 1: Plateau + flat gradients + high LR
    # ------------------------------------------------------------------

    def _check_plateau_high_lr(
        self, metrics: StepMetrics, phase: TrainingPhase, step: int
    ) -> None:
        if phase != TrainingPhase.PLATEAU:
            self._plateau_grad_flat_count = 0
            return

        # Gradient trend near-zero or negative while on plateau
        if self.gradient_tracker.mean > 0:
            relative_trend = self.gradient_tracker.trend / self.gradient_tracker.mean
        else:
            relative_trend = 0.0

        if relative_trend <= 0.1:  # not growing
            self._plateau_grad_flat_count += 1
        else:
            self._plateau_grad_flat_count = 0

        # Need sustained plateau with flat grads (50 steps)
        if self._plateau_grad_flat_count < 50:
            return

        # Check if LR is still relatively high
        scheduled = self.lr_schedule._scheduled_lr(step)
        if scheduled <= 0 or metrics.lr < scheduled * 0.3:
            return  # LR already reduced significantly

        if self._should_warn("plateau_high_lr", step):
            self.logger.warning(
                f"[Advisor] Plateau with flat gradients at step {step}: "
                f"loss stalled for {self._plateau_grad_flat_count} steps "
                f"while LR={metrics.lr:.2e} is still high. "
                f"Consider reducing learning rate or adjusting schedule."
            )
            self._mark_warned("plateau_high_lr", step)

    # ------------------------------------------------------------------
    # Rule 2: Diverging + gradient trend up + MoE imbalance
    # ------------------------------------------------------------------

    def _check_diverging_moe_collapse(
        self, metrics: StepMetrics, phase: TrainingPhase, step: int
    ) -> None:
        if phase != TrainingPhase.DIVERGING:
            return
        if metrics.moe is None:
            return

        # Gradients trending upward
        if self.gradient_tracker.mean <= 0:
            return
        relative_trend = self.gradient_tracker.trend / self.gradient_tracker.mean
        if relative_trend <= 0.1:
            return

        # MoE routing unhealthy
        diag = self.moe_monitor.get_routing_diagnostics()
        if diag.healthy:
            return

        if self._should_warn("diverging_moe", step):
            self.logger.warning(
                f"[Advisor] MoE routing collapse driving divergence at step {step}: "
                f"loss is diverging with rising gradients (trend={relative_trend:.2%}) "
                f"AND routing is unhealthy (balance={diag.balance_score:.2f}, "
                f"utilization={diag.utilization_score:.0%}). "
                f"Fix routing first — increase router_aux_loss_coef rather than reducing LR."
            )
            self._mark_warned("diverging_moe", step)

    # ------------------------------------------------------------------
    # Rule 3: Converging + gradient shrinking + loss stalling
    # ------------------------------------------------------------------

    def _check_gradient_vanishing(
        self, metrics: StepMetrics, phase: TrainingPhase, step: int
    ) -> None:
        if phase != TrainingPhase.CONVERGING:
            self._converging_grad_shrink_count = 0
            return

        if self.gradient_tracker.mean <= 0:
            return

        relative_trend = self.gradient_tracker.trend / self.gradient_tracker.mean
        # Sustained negative trend = gradients shrinking
        if relative_trend < -0.02:
            self._converging_grad_shrink_count += 1
        else:
            self._converging_grad_shrink_count = 0

        if self._converging_grad_shrink_count < 20:
            return

        # Only warn if loss improvement is also stalling
        if self.plateau_detector.plateau_count < 2:
            return

        if self._should_warn("gradient_vanishing", step):
            self.logger.warning(
                f"[Advisor] Gradient vanishing at step {step}: "
                f"gradients shrinking (trend={relative_trend:.2%}) for "
                f"{self._converging_grad_shrink_count} steps while loss improvement stalls. "
                f"Consider increasing learning rate or reducing weight decay."
            )
            self._mark_warned("gradient_vanishing", step)

    # ------------------------------------------------------------------
    # Rule 4: Loss spike + MoE utilization drop
    # ------------------------------------------------------------------

    def _check_spike_moe_correlation(
        self, metrics: StepMetrics, phase: TrainingPhase, step: int
    ) -> None:
        if self.loss_guard is None or metrics.moe is None:
            return

        # LossGuard triggered recently (within last 10 steps)
        recent_trigger = (step - self.loss_guard.last_trigger_step) <= 10
        if not recent_trigger:
            return

        # MoE utilization dropped
        if metrics.moe.expert_utilization >= self.moe_monitor.utilization_warn:
            return

        if self._should_warn("spike_moe", step):
            self.logger.warning(
                f"[Advisor] Loss spike correlates with MoE routing issues at step {step}: "
                f"LossGuard triggered at step {self.loss_guard.last_trigger_step} "
                f"and expert utilization is low ({metrics.moe.expert_utilization:.0%}). "
                f"Routing instability is likely the root cause — "
                f"increase router_aux_loss_coef before adjusting LR."
            )
            self._mark_warned("spike_moe", step)

    # ------------------------------------------------------------------
    # Rule 5: Multiple LR reductions with no improvement
    # ------------------------------------------------------------------

    def _check_lr_reductions_ineffective(
        self, metrics: StepMetrics, phase: TrainingPhase, step: int
    ) -> None:
        if self.loss_guard is None:
            return

        if self.loss_guard.num_reductions < 2:
            return

        # Despite multiple reductions, still not converging
        if phase == TrainingPhase.CONVERGING:
            return

        if self._should_warn("lr_reductions_ineffective", step):
            self.logger.warning(
                f"[Advisor] LR reductions not helping at step {step}: "
                f"{self.loss_guard.num_reductions} permanent LR reductions applied "
                f"but training is still in {phase.value} phase. "
                f"The problem may not be learning rate — "
                f"check data quality, batch size, or model architecture."
            )
            self._mark_warned("lr_reductions_ineffective", step)

    # ------------------------------------------------------------------
    # Rule 6: Warmup ended but loss still dropping steeply
    # ------------------------------------------------------------------

    def _check_warmup_too_short(
        self, metrics: StepMetrics, phase: TrainingPhase, step: int
    ) -> None:
        if self._warmup_checked:
            return
        if self.warmup_steps <= 0:
            return

        # Check at the step right after warmup ends (within a small window)
        if step < self.warmup_steps or step > self.warmup_steps + 10:
            return

        self._warmup_checked = True

        # Is loss still dropping steeply? Check gradient tracker trend.
        if self.gradient_tracker.mean <= 0:
            return

        # Look at loss trajectory: if recent losses show a strong downward
        # slope, warmup may have been too short.
        if len(self.plateau_detector.history) < 10:
            return

        recent = list(self.plateau_detector.history)[-10:]
        first_half = sum(recent[:5]) / 5
        second_half = sum(recent[5:]) / 5

        if first_half <= 0:
            return

        drop_rate = (first_half - second_half) / first_half

        # A drop > 5% over 10 steps means loss is still falling steeply
        if drop_rate > 0.05:
            self.logger.info(
                f"[Advisor] Warmup may be too short: loss still dropping steeply "
                f"({drop_rate:.1%} over last 10 steps) as warmup ends at step {step}. "
                f"Consider increasing warmup_steps from {self.warmup_steps} "
                f"to ~{int(self.warmup_steps * 1.5)}."
            )
