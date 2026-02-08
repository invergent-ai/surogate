"""Rolling gradient-norm tracker with trend detection.

Always-on and cheap — tracks a short window of recent gradient norms and
exposes summary statistics.  Logs a warning when gradients are accelerating
(sustained upward trend) which can predict an imminent explosion before any
single step exceeds a hard threshold.
"""

import math
from collections import deque

import numpy as np


class GradientTracker:
    """Track gradient norm history and detect acceleration trends.

    The tracker maintains a short rolling window and computes:
      - ``mean``: rolling mean of recent grad norms
      - ``max``: rolling max
      - ``trend``: slope of a linear fit over the window (positive = growing)

    It warns when the trend exceeds *trend_threshold* for *patience*
    consecutive checks, signalling that gradients are accelerating toward
    an explosion even if no single step has triggered LossGuard yet.
    """

    def __init__(
        self,
        logger,
        window: int = 20,
        warmup: int = 10,
        trend_threshold: float = 0.5,
        patience: int = 5,
        cooldown: int = 100,
    ):
        self.logger = logger
        self.history: deque[float] = deque(maxlen=window)
        self.warmup = warmup
        self.trend_threshold = trend_threshold
        self.patience = patience
        self.cooldown = cooldown

        self.trend_count = 0
        self.last_warn_step = -cooldown

        # Expose latest summary stats (updated every step after warmup)
        self.mean: float = 0.0
        self.max: float = 0.0
        self.trend: float = 0.0

    def step(self, grad_norm: float, step: int) -> None:
        if not math.isfinite(grad_norm):
            return

        self.history.append(grad_norm)

        if len(self.history) < self.warmup:
            return

        norms = np.array(self.history)
        self.mean = float(np.mean(norms))
        self.max = float(np.max(norms))

        # Linear regression slope over the window: positive = growing
        x = np.arange(len(norms), dtype=np.float64)
        x_mean = x.mean()
        y_mean = norms.mean()
        denom = float(((x - x_mean) ** 2).sum())
        if denom > 0:
            self.trend = float(((x - x_mean) * (norms - y_mean)).sum() / denom)
        else:
            self.trend = 0.0

        # Normalise trend relative to mean so threshold is scale-independent
        relative_trend = self.trend / max(self.mean, 1e-8)

        if relative_trend > self.trend_threshold:
            self.trend_count += 1
        else:
            self.trend_count = 0

        if (
            self.trend_count >= self.patience
            and step - self.last_warn_step >= self.cooldown
        ):
            self.logger.warning(
                f"Gradient acceleration at step {step}: "
                f"grad norms trending upward over last {len(self.history)} steps "
                f"(mean={self.mean:.4f}, max={self.max:.4f}, "
                f"trend={self.trend:.4f}, relative={relative_trend:.2%}). "
                f"Consider reducing learning rate."
            )
            self.last_warn_step = step
