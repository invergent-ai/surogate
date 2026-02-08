import math
from collections import deque

import numpy as np

from surogate.train.phase_detector import TrainingPhase


class EarlyStopping:
    """Multi-criteria early stopping for training runs.

    Monitors four independent signals and stops training when ANY fires:

    1. **Convergence score > threshold** — eval loss has stabilised and is no
       longer improving meaningfully (checked every eval cycle).
    2. **Compute efficiency collapsed** — loss reduction per FLOP has dropped
       below 50 % of the peak observed during training.  Uses the standard
       ``6N`` approximation (FLOPs/token ≈ 6 × model_params).
    3. **Training diverged** — PhaseDetector reports DIVERGING for too many
       consecutive steps.
    4. **Plateau persisted** — PhaseDetector reports PLATEAU for too many
       consecutive steps.

    Gated by the ``early_stop`` config flag — when disabled this class is
    never instantiated.
    """

    def __init__(
        self,
        logger,
        num_params: int,
        tokens_per_step: int,
        patience: int = 5,
        convergence_threshold: float = 0.85,
        efficiency_drop: float = 0.50,
        diverge_steps: int = 200,
        plateau_steps: int = 500,
        efficiency_window: int = 50,
        warmup: int = 100,
    ):
        self.logger = logger

        # FLOPs bookkeeping
        self.flops_per_step = 6 * num_params * tokens_per_step
        self.warmup = warmup

        # Criterion 1 — convergence (eval-based)
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.eval_losses: list[float] = []
        self.convergence_count = 0

        # Criterion 2 — compute efficiency (step-based)
        self.efficiency_drop = efficiency_drop
        self.efficiency_history: deque[float] = deque(maxlen=efficiency_window)
        self.peak_efficiency = 0.0
        self.prev_loss: float | None = None

        # Criteria 3 & 4 — phase persistence (step-based)
        self.diverge_steps = diverge_steps
        self.plateau_steps = plateau_steps
        self.diverge_count = 0
        self.plateau_count = 0

        self.reason: str | None = None
        self.steps_seen = 0

    # ------------------------------------------------------------------
    # Eval-based check (called every eval_steps)
    # ------------------------------------------------------------------
    def check_eval(self, eval_loss: float, step: int) -> bool:
        """Return *True* if training should stop based on eval convergence."""
        if not math.isfinite(eval_loss) or self.reason is not None:
            return self.reason is not None

        self.eval_losses.append(eval_loss)

        if len(self.eval_losses) < 3:
            return False

        # --- convergence score ---
        # stability: 1 - coefficient of variation of last 5 evals
        window = self.eval_losses[-min(5, len(self.eval_losses)):]
        mean = float(np.mean(window))
        std = float(np.std(window))
        cv = std / max(mean, 1e-8)
        stability = max(0.0, min(1.0, 1.0 - cv))

        # improvement rate: relative change from previous eval
        prev = self.eval_losses[-2]
        improvement = (prev - eval_loss) / max(abs(prev), 1e-8)
        # Normalise: 0 improvement → 1.0, large improvement → 0.0
        # Clamp negative improvement (loss got worse) to 0.0 contribution
        norm_improvement = max(0.0, min(1.0, 1.0 - improvement * 20.0))

        score = 0.6 * stability + 0.4 * norm_improvement

        if score > self.convergence_threshold:
            self.convergence_count += 1
        else:
            self.convergence_count = 0

        if self.convergence_count >= self.patience:
            self.reason = (
                f"convergence (score {score:.3f} > {self.convergence_threshold} "
                f"for {self.patience} consecutive evals)"
            )
            self.logger.warning(
                f"Early stopping at step {step}: {self.reason}"
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Step-based check (called every training step)
    # ------------------------------------------------------------------
    def check_step(self, loss: float, phase: TrainingPhase, step: int) -> bool:
        """Return *True* if training should stop based on per-step criteria."""
        if not math.isfinite(loss) or self.reason is not None:
            return self.reason is not None

        self.steps_seen += 1

        # ---- Criterion 2: compute efficiency ----
        if self.prev_loss is not None and self.flops_per_step > 0:
            delta = abs(self.prev_loss - loss)
            efficiency = delta / self.flops_per_step
            self.efficiency_history.append(efficiency)

            if len(self.efficiency_history) == self.efficiency_history.maxlen:
                rolling_mean = float(np.mean(self.efficiency_history))
                self.peak_efficiency = max(self.peak_efficiency, rolling_mean)

                if (
                    self.steps_seen > self.warmup
                    and self.peak_efficiency > 0
                    and rolling_mean < self.peak_efficiency * self.efficiency_drop
                ):
                    self.reason = (
                        f"compute efficiency collapsed "
                        f"(current {rolling_mean:.2e} < "
                        f"{self.efficiency_drop:.0%} of peak {self.peak_efficiency:.2e})"
                    )
                    self.logger.warning(
                        f"Early stopping at step {step}: {self.reason}"
                    )
                    self.prev_loss = loss
                    return True

        self.prev_loss = loss

        # Skip phase checks during warmup
        if self.steps_seen <= self.warmup:
            return False

        # ---- Criterion 3: sustained divergence ----
        if phase == TrainingPhase.DIVERGING:
            self.diverge_count += 1
        else:
            self.diverge_count = 0

        if self.diverge_count >= self.diverge_steps:
            self.reason = (
                f"sustained divergence "
                f"(DIVERGING for {self.diverge_steps} consecutive steps)"
            )
            self.logger.warning(
                f"Early stopping at step {step}: {self.reason}"
            )
            return True

        # ---- Criterion 4: sustained plateau ----
        if phase == TrainingPhase.PLATEAU:
            self.plateau_count += 1
        else:
            self.plateau_count = 0

        if self.plateau_count >= self.plateau_steps:
            self.reason = (
                f"sustained plateau "
                f"(PLATEAU for {self.plateau_steps} consecutive steps)"
            )
            self.logger.warning(
                f"Early stopping at step {step}: {self.reason}"
            )
            return True

        return False