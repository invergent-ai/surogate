import math
from collections import deque

import numpy as np


class PlateauDetector:
    """Detects when training loss stops improving and logs a warning.

    Compares the mean loss of the recent half of a rolling window against the
    older half.  When the relative improvement drops below *threshold* for
    *patience* consecutive checks the detector fires a warning.

    Always-on and cheap — no automatic actions, just diagnostics.
    """

    def __init__(
        self,
        logger,
        window: int = 200,
        warmup: int = 100,
        threshold: float = 0.001,
        patience: int = 3,
        cooldown: int = 200,
    ):
        self.logger = logger
        self.history = deque(maxlen=window)
        self.warmup = warmup
        self.threshold = threshold
        self.patience = patience
        self.cooldown = cooldown

        self.plateau_count = 0
        self.last_warn_step = -cooldown

    def step(self, loss: float, step: int) -> None:
        if not math.isfinite(loss):
            return

        self.history.append(loss)

        if len(self.history) < self.warmup:
            return

        half = len(self.history) // 2
        older = list(self.history)[:half]
        recent = list(self.history)[half:]

        older_mean = float(np.mean(older))
        recent_mean = float(np.mean(recent))

        if older_mean <= 0:
            return

        improvement = (older_mean - recent_mean) / older_mean

        if improvement < self.threshold:
            self.plateau_count += 1
        else:
            self.plateau_count = 0

        if (
            self.plateau_count >= self.patience
            and step - self.last_warn_step >= self.cooldown
        ):
            self.logger.warning(
                f"Plateau detected at step {step}: "
                f"loss barely improving over last {len(self.history)} steps "
                f"(older_mean={older_mean:.4f}, recent_mean={recent_mean:.4f}, "
                f"improvement={improvement:.4%}). "
                f"Consider adjusting learning rate or stopping early."
            )
            self.last_warn_step = step
