import math
from collections import deque

import numpy as np


class LossGuard:
    """Detects loss spikes and gradient explosions, reduces LR automatically.

    Monitors a rolling window of loss and grad_norm values. When an anomaly is
    detected the learning-rate schedule is permanently scaled down so the rest
    of training runs at a lower LR.

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

        self.loss_std_mult = loss_std_mult
        self.loss_abs_min = loss_abs_min
        self.grad_relative = grad_relative
        self.grad_absolute = grad_absolute
        self.lr_factor = lr_factor

        self.last_trigger_step = -cooldown  # allow immediate first trigger
        self.num_reductions = 0

    def step(self, loss: float, grad_norm: float, step: int) -> None:
        # Skip non-finite values — don't pollute the history and trigger
        # an immediate reduction since inf/nan is always abnormal.
        if not math.isfinite(loss) or not math.isfinite(grad_norm):
            if self.num_reductions < self.max_reductions and step - self.last_trigger_step >= self.cooldown:
                old_lr = self.lr_schedule.base_lr
                self.lr_schedule.reduce_lr(self.lr_factor)
                self.logger.warning(
                    f"Auto LR reduction: non-finite values at step {step} "
                    f"(loss={loss}, grad_norm={grad_norm}). "
                    f"LR: {old_lr:.2e} -> {self.lr_schedule.base_lr:.2e} "
                    f"[{self.num_reductions + 1}/{self.max_reductions}]"
                )
                self.last_trigger_step = step
                self.num_reductions += 1
            return

        self.history.append((loss, grad_norm))

        if len(self.history) < self.warmup:
            return
        if step - self.last_trigger_step < self.cooldown:
            return
        if self.num_reductions >= self.max_reductions:
            return

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
            old_lr = self.lr_schedule.base_lr
            self.lr_schedule.reduce_lr(self.lr_factor)
            self.logger.warning(
                f"Auto LR reduction: {reason} at step {step} "
                f"(loss={loss:.4f}, grad_norm={grad_norm:.2f}). "
                f"LR: {old_lr:.2e} -> {self.lr_schedule.base_lr:.2e} "
                f"[{self.num_reductions + 1}/{self.max_reductions}]"
            )
            self.last_trigger_step = step
            self.num_reductions += 1
