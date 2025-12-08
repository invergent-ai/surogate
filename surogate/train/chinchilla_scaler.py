import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from datasets import Dataset
from transformers import PreTrainedModel

from surogate.core.config.sft_config import SFTConfig
from surogate.utils.logger import get_logger

logger = get_logger()

@dataclass
class TrainingStepMetrics:
    """Holds the raw and derived metrics for a specific training step."""
    step: int
    epoch: float
    loss: float
    grad_norm: float
    learning_rate: float
    tokens_seen: int
    timestamp: float
    loss_reduction_rate: float = 0.0
    gradient_variance: float = 0.0
    compute_efficiency_score: float = 0.0
    convergence_confidence: float = 0.0

class TrainingStabilityMonitor:
    """Monitors loss and gradient history to detect plateaus or divergence."""

    def __init__(self, observation_window_size: int = 100):
        self.window_size = observation_window_size
        self.loss_history = deque(maxlen=observation_window_size)
        self.grad_norm_history = deque(maxlen=observation_window_size)

    def record_step(self, current_loss: float, current_grad_norm: float):
        self.loss_history.append(current_loss)
        self.grad_norm_history.append(current_grad_norm)

    def check_for_loss_plateau(self, improvement_threshold: float = 0.001) -> Tuple[bool, float]:
        if len(self.loss_history) < self.window_size:
            return False, 1.0

        # Split history into two halves to compare trends
        recent_window = list(self.loss_history)[-self.window_size//2:]
        historical_window = list(self.loss_history)[:self.window_size//2]

        recent_avg_loss = np.mean(recent_window)
        historical_avg_loss = np.mean(historical_window)

        # Calculate relative improvement
        improvement_rate = (historical_avg_loss - recent_avg_loss) / max(historical_avg_loss, 1e-8)
        is_stagnant = abs(improvement_rate) < improvement_threshold

        return is_stagnant, improvement_rate

    def check_for_divergence(self, divergence_threshold: float = 0.1) -> Tuple[bool, float]:
        if len(self.loss_history) < 10:
            return False, 0.0

        current_window = list(self.loss_history)[-10:]
        # Use a slightly older window for comparison
        baseline_window = list(self.loss_history)[-20:-10] if len(self.loss_history) >= 20 else current_window

        current_avg = np.mean(current_window)
        baseline_avg = np.mean(baseline_window)

        divergence_rate = (current_avg - baseline_avg) / max(baseline_avg, 1e-8)
        is_diverging = divergence_rate > divergence_threshold

        return is_diverging, divergence_rate

    def calculate_gradient_variance(self) -> float:
        if len(self.grad_norm_history) < 10:
            return 1.0

        return float(np.std(list(self.grad_norm_history)[-20:]))

    def calculate_convergence_score(self) -> float:
        """Calculates a composite score (0.0 to 1.0) indicating training health."""
        if len(self.loss_history) < self.window_size:
            return 0.0

        # 1. Analyze Loss Stability (Standard Deviation)
        recent_losses = list(self.loss_history)[-20:]
        loss_std_dev = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        loss_stability_score = 1.0 - min(loss_std_dev / max(loss_mean, 1e-8), 1.0)

        # 2. Analyze Improvement Rate
        _, improvement_rate = self.check_for_loss_plateau()
        improvement_score = 1.0 - min(abs(improvement_rate) * 100, 1.0)

        # 3. Analyze Gradient Stability
        grad_variance = self.calculate_gradient_variance()
        grad_stability_score = 1.0 / (1.0 + grad_variance)

        # Weighted composite score
        convergence_score = (
                0.4 * loss_stability_score +
                0.4 * improvement_score +
                0.2 * grad_stability_score
        )

        return convergence_score


class ComputeEfficiencyAnalyzer:
    """Tracks how efficiently the model turns FLOPs into loss reduction."""

    def __init__(self, theoretical_flops_per_token: Optional[float] = None):
        self.flops_per_token = theoretical_flops_per_token
        self.metrics_log = []
        self.efficiency_history = deque(maxlen=100)

    def calculate_theoretical_flops(self, num_params: int, sequence_length: int) -> float:
        # Standard approximation: 6 FLOPs per parameter per token
        return 6 * num_params * sequence_length

    def record_efficiency_step(self, tokens_processed: int, loss_reduction: float):
        if self.flops_per_token is None:
            return

        total_flops_used = tokens_processed * self.flops_per_token

        # Efficiency = (Gain in Loss) / (Cost in FLOPs)
        efficiency_ratio = loss_reduction / max(total_flops_used, 1e-20)

        self.efficiency_history.append(efficiency_ratio)
        self.metrics_log.append({
            'tokens': tokens_processed,
            'loss_reduction': loss_reduction,
            'efficiency_ratio': efficiency_ratio
        })

    def get_current_efficiency_score(self) -> float:
        if not self.efficiency_history:
            return 1.0
        return float(np.mean(list(self.efficiency_history)[-20:]))

    def check_efficiency_degradation(self, degradation_threshold: float = 0.5) -> Tuple[bool, float]:
        if len(self.efficiency_history) < 50:
            return False, 0.0

        current_window = list(self.efficiency_history)[-25:]
        baseline_window = list(self.efficiency_history)[-50:-25]

        current_avg_efficiency = np.mean(current_window)
        baseline_avg_efficiency = np.mean(baseline_window)

        decline_ratio = (baseline_avg_efficiency - current_avg_efficiency) / max(baseline_avg_efficiency, 1e-8)
        is_declining = decline_ratio > degradation_threshold

        return is_declining, decline_ratio

class CurriculumPacingController:
    """Manages training difficulty based on the learning velocity."""

    def __init__(self):
        self.difficulty_schedule = []
        self.loss_reduction_velocity = deque(maxlen=50)

    def record_learning_velocity(self, loss_reduction: float):
        self.loss_reduction_velocity.append(loss_reduction)

    def suggest_curriculum_difficulty(self) -> float:
        if len(self.loss_reduction_velocity) < 10:
            return 0.3

        avg_recent_velocity = np.mean(list(self.loss_reduction_velocity)[-10:])

        if avg_recent_velocity > 0.01:
            # Learning is fast; increase difficulty
            return min(0.9, 0.5 + avg_recent_velocity * 20)
        else:
            # Learning has slowed; decrease difficulty
            return max(0.2, 0.5 - abs(avg_recent_velocity) * 10)

class ComputeOptimalEpochScheduler:
    """
    Adjusts training epochs based on Chinchilla scaling laws and real-time
    training stability/efficiency metrics.
    """
    def __init__(self, config: SFTConfig, model: PreTrainedModel, dataset: Dataset):
        self.config = config
        self.model = model
        self.dataset = dataset

        # Configuration Parameters
        self.tokens_per_param_target = getattr(config, 'chinchilla_multiplier', 20)
        self.min_epoch_bound = getattr(config, 'min_auto_epochs', 1)
        self.max_epoch_bound = getattr(config, 'max_auto_epochs', 50)

        # Feature Flags
        self.use_loss_landscape = getattr(config, 'enable_loss_landscape', True)
        self.use_compute_efficiency = getattr(config, 'enable_compute_efficiency', True)
        self.use_adaptive_curriculum = getattr(config, 'enable_adaptive_curriculum', True)
        self.use_early_stopping = getattr(config, 'enable_early_stopping', True)

        # Thresholds
        self.plateau_patience_steps = getattr(config, 'plateau_patience', 5)
        self.efficiency_decline_threshold = getattr(config, 'efficiency_decline_threshold', 0.3)
        self.convergence_threshold = getattr(config, 'convergence_threshold', 0.85)

        # Sub-components
        self.stability_monitor = TrainingStabilityMonitor(observation_window_size=50)
        self.efficiency_analyzer = ComputeEfficiencyAnalyzer()
        self.pacing_controller = CurriculumPacingController()

        # State tracking
        self.metrics_history: List[TrainingStepMetrics] = []
        self.initial_loss_value = None
        self.best_loss_value = float('inf')
        self.plateau_counter = 0
        self.total_tokens_processed = 0

        # Model and Data stats
        self.total_model_params = sum(p.numel() for p in model.parameters())
        self.params_in_billions = self.total_model_params / 1e9

        self.total_dataset_tokens = self._calculate_total_dataset_tokens()
        self.dataset_tokens_in_billions = self.total_dataset_tokens / 1e9

        # Initialize Efficiency Tracker
        sequence_length = getattr(config, 'seq_length', 2048)
        self.efficiency_analyzer.flops_per_token = self.efficiency_analyzer.calculate_theoretical_flops(
            self.total_model_params, sequence_length
        )

        # Calculate Epochs
        self.theoretical_optimal_epochs = self._calculate_chinchilla_optimal_epochs()
        self.current_adjusted_epochs = self.theoretical_optimal_epochs

        self._log_initialization_summary()

    def _calculate_total_dataset_tokens(self) -> int:
        try:
            if hasattr(self.dataset, '__len__'):
                seq_length = self.config.sequence_len
                return len(self.dataset) * seq_length
            else:
                return 1_000_000_000
        except Exception as e:
            print(f"Warning: Could not estimate dataset tokens: {e}")
            return 1_000_000_000

    def _calculate_chinchilla_optimal_epochs(self) -> int:
        """
        Calculates epochs needed to satisfy the Chinchilla ratio (e.g., 20 tokens per param).
        $$ Epochs = \frac{Multiplier \times Parameters}{DatasetTokens} $$
        """
        target_total_tokens = self.tokens_per_param_target * self.total_model_params
        tokens_per_epoch = self.total_dataset_tokens

        calculated_epochs = math.ceil(target_total_tokens / tokens_per_epoch)

        # Clamp between min and max bounds
        return max(self.min_epoch_bound, min(calculated_epochs, self.max_epoch_bound))

    def _log_initialization_summary(self):
        logger.header("Compute-Optimal Epoch Scheduler")
        logger.metric("Model: parameters", f"{self.total_model_params:,}")
        logger.metric("Model: FLOPs/token", f"{self.efficiency_analyzer.flops_per_token:.2e}")
        logger.metric("Dataset: total tokens", f"{self.total_dataset_tokens:,}")
        logger.metric("Dataset: samples", f"{len(self.dataset):,}")
        logger.metric("Scaling: target ratio", f"x{self.tokens_per_param_target} (tokens/param)")
        logger.metric("Scaling: optimal tokens", f"{self.tokens_per_param_target * self.total_model_params:,}")
        logger.metric("Scaling: base optimal epochs", f"{self.theoretical_optimal_epochs}")
        logger.metric("Scaling: epoch constraints", f"[{self.min_epoch_bound}, {self.max_epoch_bound}]")

        optimal_tokens_billions = (self.tokens_per_param_target * self.total_model_params) / 1e9
        actual_budget_billions = self.dataset_tokens_in_billions * self.theoretical_optimal_epochs
        coverage_percentage = (actual_budget_billions / optimal_tokens_billions) * 100

        logger.metric("Token Budget: target", f"{optimal_tokens_billions:.3f}", "B tokens")
        logger.metric("Token Budget: actual training", f"{actual_budget_billions:.3f}", "B tokens")
        logger.metric("Token Budget: coverage", f"{coverage_percentage:.2f}%", "of target")

        if coverage_percentage < 50:
            logger.warning(f"Significantly under Chinchilla recommendation")
        elif coverage_percentage > 150:
            logger.warning(f"Exceeding Chinchilla recommendation")
        else:
            logger.success(f"Within reasonable range")