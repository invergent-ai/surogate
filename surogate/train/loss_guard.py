import math
from collections import deque

import numpy as np


class LossGuard:
    """Detects loss spikes and gradient explosions, reduces LR automatically.

    Monitors a rolling window of loss and grad_norm values.  When an anomaly is
    detected the guard applies a *temporary* LR override that linearly decays
    back to the scheduled value over a grace period.  If anomalies persist the
    guard escalates to a permanent LR reduction.

    Escalation policy (per trigger, respecting cooldown):
      1. First triggers  → temporary override (LR * factor for *grace_period* steps)
      2. After *max_overrides* temporary overrides without recovery → permanent reduction

    Gated by the ``auto_lr_reduction`` config flag — when disabled, this class
    is never instantiated.
    """

    def __init__(
        self,
        lr_schedule,
        logger,
        window: int = 100,
        warmup: int = 50,
        cooldown: int = 50,
        max_reductions: int = 5,
        max_overrides: int = 2,
        grace_period: int = 50,
        loss_std_mult: float = 3.0,
        loss_abs_min: float = 0.5,
        grad_relative: float = 10.0,
        grad_absolute: float = 100.0,
        lr_factor: float = 0.5,
    ):
        self.lr_schedule = lr_schedule
        self.logger = logger

        self.history = deque(maxlen=window)
        self.warmup = warmup
        self.cooldown = cooldown
        self.max_reductions = max_reductions
        self.max_overrides = max_overrides
        self.grace_period = grace_period

        self.loss_std_mult = loss_std_mult
        self.loss_abs_min = loss_abs_min
        self.grad_relative = grad_relative
        self.grad_absolute = grad_absolute
        self.lr_factor = lr_factor

        self.last_trigger_step = -cooldown  # allow immediate first trigger
        self.num_reductions = 0
        self.num_overrides = 0  # temporary overrides since last permanent reduction

    def _apply_reduction(self, step: int, reason: str, loss: float, grad_norm: float) -> None:
        """Decide between temporary override and permanent reduction."""
        scheduled_lr = self.lr_schedule._scheduled_lr(step)
        target_lr = scheduled_lr * self.lr_factor

        if self.num_overrides < self.max_overrides:
            # Temporary override — will blend back to schedule
            self.lr_schedule.override_lr(target_lr, step, self.grace_period)
            self.num_overrides += 1
            self.logger.warning(
                f"Auto LR override: {reason} at step {step} "
                f"(loss={loss:.4f}, grad_norm={grad_norm:.2f}). "
                f"LR: {scheduled_lr:.2e} -> {target_lr:.2e} "
                f"(temporary, {self.grace_period} step grace) "
                f"[override {self.num_overrides}/{self.max_overrides}]"
            )
        else:
            # Escalate to permanent reduction
            old_lr = self.lr_schedule.base_lr
            self.lr_schedule.reduce_lr(self.lr_factor)
            self.num_reductions += 1
            self.num_overrides = 0  # reset override counter
            self.logger.warning(
                f"Auto LR reduction: {reason} at step {step} "
                f"(loss={loss:.4f}, grad_norm={grad_norm:.2f}). "
                f"LR: {old_lr:.2e} -> {self.lr_schedule.base_lr:.2e} (permanent) "
                f"[reduction {self.num_reductions}/{self.max_reductions}]"
            )

        self.last_trigger_step = step

    def step(self, loss: float, grad_norm: float, step: int) -> None:
        # Skip non-finite values — don't pollute the history and trigger
        # an immediate reduction since inf/nan is always abnormal.
        if not math.isfinite(loss) or not math.isfinite(grad_norm):
            if self.num_reductions < self.max_reductions and step - self.last_trigger_step >= self.cooldown:
                self._apply_reduction(step, "non-finite values", loss, grad_norm)
            return

        # Check against history *before* appending so the current value
        # doesn't contaminate its own baseline.
        triggered = False
        if (
            len(self.history) >= self.warmup
            and step - self.last_trigger_step >= self.cooldown
            and self.num_reductions < self.max_reductions
        ):
            losses = np.array([h[0] for h in self.history])
            mean_loss = float(np.mean(losses))
            std_loss = float(np.std(losses))

            grads = np.array([h[1] for h in self.history])
            mean_grad = float(np.mean(grads))

            is_loss_spike = (
                loss > mean_loss + self.loss_std_mult * std_loss
                and loss - mean_loss > self.loss_abs_min
            )
            is_grad_explosion = (
                grad_norm > self.grad_relative * mean_grad
                or grad_norm > self.grad_absolute
            )

            if is_loss_spike or is_grad_explosion:
                reason = "loss spike" if is_loss_spike else "gradient explosion"
                self._apply_reduction(step, reason, loss, grad_norm)
                triggered = True

        # Only add normal values to history — spikes that triggered a
        # reduction are excluded to keep the baseline clean.
        if not triggered:
            self.history.append((loss, grad_norm))
