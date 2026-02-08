import math
from collections import deque
from enum import Enum

import numpy as np


class TrainingPhase(Enum):
    WARMUP = "warmup"
    CONVERGING = "converging"
    PLATEAU = "plateau"
    UNSTABLE = "unstable"
    DIVERGING = "diverging"


class PhaseDetector:
    """Classifies the current training phase and logs transitions.

    Phases
    ------
    WARMUP      First *warmup* steps (loss dropping fast, statistics unreliable).
    CONVERGING  Loss is steadily decreasing.
    PLATEAU     Loss improvement is negligible.
    UNSTABLE    Loss variance is high relative to the mean.
    DIVERGING   Loss is trending upward.

    Only logs when the phase *changes*, so it produces very little output.
    """

    def __init__(
        self,
        logger,
        window: int = 100,
        warmup: int = 50,
        plateau_threshold: float = 0.001,
        diverge_threshold: float = 0.01,
        instability_cv: float = 0.15,
    ):
        self.logger = logger
        self.history = deque(maxlen=window)
        self.warmup = warmup
        self.plateau_threshold = plateau_threshold
        self.diverge_threshold = diverge_threshold
        self.instability_cv = instability_cv

        self.current_phase = TrainingPhase.WARMUP
        self.phase_start_step = 0
        self.steps_seen = 0

    # ------------------------------------------------------------------
    def step(self, loss: float, step: int) -> TrainingPhase:
        if not math.isfinite(loss):
            return self.current_phase

        self.history.append(loss)
        self.steps_seen += 1

        if self.steps_seen <= self.warmup:
            return self.current_phase

        new_phase = self._classify()
        if new_phase != self.current_phase:
            duration = step - self.phase_start_step
            self.logger.info(
                f"Training phase: {self.current_phase.value} -> {new_phase.value} "
                f"at step {step} (previous phase lasted {duration} steps)"
            )
            self.current_phase = new_phase
            self.phase_start_step = step

        return self.current_phase

    # ------------------------------------------------------------------
    def _classify(self) -> TrainingPhase:
        losses = np.array(self.history)
        half = len(losses) // 2
        if half == 0:
            return TrainingPhase.WARMUP

        older = losses[:half]
        recent = losses[half:]

        older_mean = float(np.mean(older))
        recent_mean = float(np.mean(recent))
        recent_std = float(np.std(recent))

        if older_mean <= 0:
            return TrainingPhase.CONVERGING

        # Coefficient of variation of recent losses — high means unstable.
        cv = recent_std / max(recent_mean, 1e-8)
        if cv > self.instability_cv:
            return TrainingPhase.UNSTABLE

        improvement = (older_mean - recent_mean) / older_mean

        if improvement > self.diverge_threshold * -1 and improvement < self.plateau_threshold:
            # Near-zero or very small improvement
            return TrainingPhase.PLATEAU

        if improvement < 0:
            # Loss increased
            return TrainingPhase.DIVERGING

        return TrainingPhase.CONVERGING
