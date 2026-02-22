import math


class LRSchedule:
    """Learning rate schedule with warmup, main phase, and optional cooldown.

    Supports temporary LR overrides that decay back to the scheduled value
    over a grace period, as well as permanent reductions for severe anomalies.
    """

    def __init__(self, base_lr: float, max_steps: int, warmup_steps: int,
                 cooldown_steps: int, final_lr: float, schedule_type: str,
                 wsd_decay_steps_fraction: float = 0.1):
        self.base_lr = base_lr
        self.max_steps = max_steps
        self.warmup_steps = max(0, warmup_steps)
        self.cooldown_steps = max(0, cooldown_steps)
        self.final_lr = final_lr
        self.schedule_type = schedule_type.lower()

        # WSD decay phase: fraction of total steps
        self.wsd_decay_steps = max(0, int(max_steps * wsd_decay_steps_fraction))

        # Main schedule covers steps between warmup and cooldown
        self.main_steps = max_steps - self.warmup_steps - self.cooldown_steps
        if self.main_steps < 0:
            self.main_steps = 0

        # Temporary override state
        self._override_lr: float | None = None
        self._override_start_step: int = 0
        self._override_grace: int = 0
        self._scheduled_lr_at_override: float = 0.0

    def get_lr(self, step: int) -> float:
        scheduled = self._scheduled_lr(step)

        if self._override_lr is not None:
            elapsed = step - self._override_start_step
            if elapsed >= self._override_grace:
                # Grace period expired — resume normal schedule
                self._override_lr = None
                return scheduled
            # Linearly blend from override back to scheduled LR
            t = elapsed / self._override_grace
            return self._override_lr + (scheduled - self._override_lr) * t

        return scheduled

    def _scheduled_lr(self, step: int) -> float:
        """Compute the base schedule LR (no override)."""
        # Warmup phase: linear ramp from 0 to base_lr
        if step < self.warmup_steps:
            return self.base_lr * step / self.warmup_steps

        # Cooldown phase: 1-sqrt schedule from base_lr to final_lr
        if self.cooldown_steps > 0 and step >= (self.max_steps - self.cooldown_steps):
            cooldown_step = step - (self.max_steps - self.cooldown_steps)
            progress = cooldown_step / self.cooldown_steps
            return self.final_lr + (self.base_lr - self.final_lr) * (1.0 - math.sqrt(progress))

        # Main phase
        if self.main_steps <= 0:
            return self.base_lr

        main_step = step - self.warmup_steps
        progress = min(1.0, main_step / self.main_steps)

        if self.schedule_type == "constant":
            return self.base_lr
        elif self.schedule_type == "cosine":
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.final_lr + (self.base_lr - self.final_lr) * cosine_decay
        elif self.schedule_type == "linear":
            return self.base_lr + (self.final_lr - self.base_lr) * progress
        elif self.schedule_type == "wsd":
            # Warmup-Stable-Decay: stable LR then 1-sqrt decay to final_lr
            decay_start = self.max_steps - self.wsd_decay_steps
            if self.wsd_decay_steps > 0 and step >= decay_start:
                progress = (step - decay_start) / self.wsd_decay_steps
                return self.final_lr + (self.base_lr - self.final_lr) * (1.0 - math.sqrt(progress))
            return self.base_lr
        else:
            # Default to cosine
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.final_lr + (self.base_lr - self.final_lr) * cosine_decay

    def override_lr(self, lr: float, step: int, grace_period: int) -> None:
        """Temporarily override LR, linearly blending back over *grace_period* steps."""
        self._override_lr = lr
        self._override_start_step = step
        self._override_grace = max(1, grace_period)
        self._scheduled_lr_at_override = self._scheduled_lr(step)

    @property
    def has_override(self) -> bool:
        return self._override_lr is not None

    def reduce_lr(self, factor: float) -> None:
        """Permanently scale down base_lr and final_lr by *factor*."""
        self.base_lr *= factor
        self.final_lr *= factor
