import datetime
import json
import math
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, Any

import numpy as np
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, trainer

from surogate.config.sft_config import SFTConfig
from surogate.train.noop_callbacks import NoopTrainerCallback, NoopPrinterCallback
from surogate.utils.logger import get_logger
from swift import Seq2SeqTrainer

logger = get_logger()

trainer.DEFAULT_PROGRESS_CALLBACK = NoopTrainerCallback
trainer.PrinterCallback = NoopPrinterCallback


class SuppressStderr:
    """Context manager to suppress stderr output (for LAPACK warnings)."""

    def __enter__(self):
        self.old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self.old_stderr


@dataclass
class TrainingMetrics:
    epoch: int
    step: int
    loss: float
    grad_norm: float
    learning_rate: float
    memory_usage: Dict[str, float]
    timestamp: datetime.datetime

    def to_dict(self):
        """Convert to dictionary with proper serialization."""
        result = {
            'epoch': self.epoch,
            'step': self.step,
            'loss': self.loss,
            'grad_norm': self.grad_norm,
            'learning_rate': self.learning_rate,
            'memory_usage': self.memory_usage,
            'timestamp': self.timestamp.isoformat()
        }
        return result


@dataclass
class AdaptiveDecision:
    """Represents an adaptive decision made by the intelligence system."""
    decision_type: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str
    expected_improvement: float
    timestamp: datetime.datetime


class AdaptiveHyperparameterOptimizer:
    def __init__(self):
        self.optimization_history = []
        self.current_search_space = {}
        self.performance_buffer = deque(maxlen=50)
        self.last_adjustment_step = 0

    def should_adjust_learning_rate(self, current_metrics):
        """Decide whether to adjust learning rate."""

        if len(self.performance_buffer) > 0:
            steps_since_last = current_metrics.step - self.last_adjustment_step
            if steps_since_last < 50:  # Don't adjust too often
                self.performance_buffer.append(current_metrics)
                return None

        self.performance_buffer.append(current_metrics)
        recent_losses = [m.loss for m in list(self.performance_buffer)[-20:]]
        very_recent = [m.loss for m in list(self.performance_buffer)[-5:]]

        # 1. PLATEAU - If loss barely changing
        if np.std(very_recent) < 0.01 and np.mean(very_recent) > 0.5:
            self.last_adjustment_step = current_metrics.step
            return {
                'action': 'increase',
                'factor': 1.5,
                'reasoning': f'Loss plateau: std={np.std(very_recent):.4f}',
                'emergency': False,
            }

        # 2. DIVERGENCE - If loss increasing
        recent_mean = np.mean(very_recent)
        older_mean = np.mean(recent_losses[-15:-10]) if len(recent_losses) >= 15 else recent_mean
        if recent_mean > older_mean + 0.3:
            self.last_adjustment_step = current_metrics.step
            return {
                'action': 'decrease',
                'factor': 0.5,
                'reasoning': f'Loss increasing: {older_mean:.3f} → {recent_mean:.3f}',
                'emergency': False
            }

        # 3. GOOD PROGRESS - If steadily decreasing
        if recent_mean < older_mean - 0.1 and np.std(very_recent) < 0.05:
            self.last_adjustment_step = current_metrics.step
            return {
                'action': 'increase',
                'factor': 1.2,
                'reasoning': 'Steady improvement, accelerating'
            }
        # Check for instability
        grad_norms = [m.grad_norm for m in list(self.performance_buffer)[-5:]]
        if np.mean(grad_norms) > 10.0:
            return {
                'action': 'decrease',
                'factor': 0.7,
                'reasoning': 'High gradient norms detected, reducing LR for stability',
                'emergency': False
            }

        return None

    def optimize_batch_size(self, current_metrics, memory_usage):
        """Dynamically optimize batch size based on performance and memory."""
        current_memory_usage = memory_usage.get('gpu_memory_percent', 0)

        # If memory usage is low and performance is good, increase batch size
        if current_memory_usage < 70 and current_metrics.loss < 2.0:
            return {
                'action': 'increase',
                'new_size': int(current_metrics.step * 1.25),
                'reasoning': 'Low memory usage and good performance, increasing batch size'
            }

        # If memory usage is high, decrease batch size
        if current_memory_usage > 90:
            return {
                'action': 'decrease',
                'new_size': max(1, int(current_metrics.step * 0.8)),
                'reasoning': 'High memory usage, reducing batch size'
            }

        return None


class RealTimeAnalytics:
    def __init__(self):
        self.metrics_buffer = deque(maxlen=1000)
        self.anomaly_detector = None
        self.trend_analyzer = None

        # Configurable thresholds
        self.anomaly_thresholds = {
            'loss_spike_std_multiplier': 2.0,
            'loss_spike_min_increase': 0.1,
            'gradient_explosion_threshold': 100.0,
            'gradient_explosion_relative': 10.0,
            'min_buffer_size': 50,
            'recent_window': 10,
        }

    def detect_training_anomalies(self, current_metrics):
        """Detect unusual patterns in training using adaptive thresholds."""
        if len(self.metrics_buffer) < self.anomaly_thresholds['min_buffer_size']:
            self.metrics_buffer.append(current_metrics)
            return None

        self.metrics_buffer.append(current_metrics)

        # Configurable windows
        recent_window = self.anomaly_thresholds['recent_window']
        recent_losses = [m.loss for m in list(self.metrics_buffer)[-recent_window:]]
        historical_losses = [m.loss for m in list(self.metrics_buffer)[-50:-recent_window]]

        if not historical_losses:
            return None

        recent_mean = np.mean(recent_losses)
        historical_mean = np.mean(historical_losses)
        historical_std = np.std(historical_losses)

        anomalies = []

        # Adaptive loss spike detection
        std_multiplier = self.anomaly_thresholds['loss_spike_std_multiplier']
        min_increase = self.anomaly_thresholds['loss_spike_min_increase']

        threshold = historical_mean + std_multiplier * historical_std
        absolute_increase = recent_mean - historical_mean

        if recent_mean > threshold and absolute_increase > min_increase:
            severity = 'critical' if absolute_increase > 1.0 else 'high'
            anomalies.append({
                'type': 'loss_spike',
                'severity': severity,
                'description': f'Loss increased significantly: {recent_mean:.3f} vs {historical_mean:.3f} (+{absolute_increase:.3f})',
                'relative_increase': absolute_increase / historical_mean
            })

        # Adaptive gradient explosion detection
        abs_threshold = self.anomaly_thresholds['gradient_explosion_threshold']
        relative_threshold = self.anomaly_thresholds['gradient_explosion_relative']

        # Calculate historical gradient norm mean
        historical_grad_norms = [m.grad_norm for m in list(self.metrics_buffer)[-50:-recent_window] if m.grad_norm > 0]

        is_explosion = current_metrics.grad_norm > abs_threshold
        if historical_grad_norms:
            hist_grad_mean = np.mean(historical_grad_norms)
            is_explosion = is_explosion or (current_metrics.grad_norm > hist_grad_mean * relative_threshold)

        if is_explosion:
            anomalies.append({
                'type': 'gradient_explosion',
                'severity': 'critical',
                'description': f'Gradient norm extremely high: {current_metrics.grad_norm:.2f}',
                'threshold_used': abs_threshold
            })

        return anomalies if anomalies else None

    def analyze_loss_dynamics(self, recent_metrics):
        """Analyze loss curve dynamics for insights with robust error handling."""
        if len(recent_metrics) < 10:
            return None

        try:
            losses = [m.loss for m in recent_metrics]
            steps = [m.step for m in recent_metrics]

            if any(math.isnan(l) or math.isinf(l) for l in losses):
                logger.warning("Invalid loss values detected in dynamics analysis")
                return None

            losses_array = np.array(losses, dtype=np.float64)
            steps_array = np.array(steps, dtype=np.float64)

            loss_mean = np.mean(losses_array)
            loss_std = np.std(losses_array) + 1e-8  # Avoid division by zero
            normalized_losses = (losses_array - loss_mean) / loss_std

            step_mean = np.mean(steps_array)
            step_std = np.std(steps_array) + 1e-8
            normalized_steps = (steps_array - step_mean) / step_std

            try:
                # Try quadratic (degree 2)
                with SuppressStderr():
                    coeffs = np.polyfit(normalized_steps, normalized_losses, 2, full=False)
            except np.linalg.LinAlgError:
                logger.debug("Degree-2 polyfit failed, falling back to linear")
                try:
                    # Fall back to linear (degree 1)
                    with SuppressStderr():
                        coeffs = np.polyfit(normalized_steps, normalized_losses, 1, full=False)
                    coeffs = np.array([0.0, coeffs[0], coeffs[1]])
                except np.linalg.LinAlgError:
                    logger.warning("All polyfit attempts failed, using simple trend")
                    # Fallback: simple slope calculation
                    trend = (normalized_losses[-1] - normalized_losses[0]) / (
                                normalized_steps[-1] - normalized_steps[0])
                    coeffs = np.array([0.0, trend, normalized_losses[0]])

            # Analyze curvature and trend
            curvature = coeffs[0]
            trend = coeffs[1]

            insights = {
                'trend_direction': 'decreasing' if trend < 0 else 'increasing',
                'trend_strength': abs(trend),
                'curvature': 'concave_up' if curvature > 0 else 'concave_down',
                'predicted_convergence': self._predict_convergence(coeffs, steps[-1])
            }

            return insights

        except Exception as e:
            logger.debug(f"Error in loss dynamics analysis: {e}")
            return None


class MetaLearningEngine:
    """Learns how to train more effectively over time."""

    def __init__(self, orchestrator=None):
        self.training_history = []
        self.successful_strategies = []
        self.meta_model = None
        self.adaptation_buffer = deque(maxlen=1000)
        self.orchestrator = orchestrator  # Store reference to get model params

    def _synthesize_suggestions(self, successful_patterns, current_metrics):
        """Synthesize hyperparameter suggestions from successful patterns."""
        if not successful_patterns:
            return {}

        # Average successful hyperparameters
        avg_lr = np.mean([p['config'].get('learning_rate', self.orchestrator.config.learning_rate if self.orchestrator else 0.001)
                          for p in successful_patterns])

        suggestions = {
            'learning_rate': {
                'value': avg_lr,
                'confidence': min(len(successful_patterns) / 10.0, 0.9)
            }
        }

        # Add batch size suggestions if available
        batch_sizes = [p['config'].get('batch_size') for p in successful_patterns if 'batch_size' in p['config']]
        if batch_sizes:
            avg_batch_size = int(np.mean(batch_sizes))
            suggestions['batch_size'] = {
                'value': avg_batch_size,
                'confidence': min(len(batch_sizes) / 10.0, 0.8)
            }

        return suggestions

    def record_training_outcome(self, config, metrics, final_performance):
        """Record the outcome of a training run for meta-learning."""

        # Validate metrics have to_dict method
        metrics_dicts = []
        for m in metrics:
            if hasattr(m, 'to_dict'):
                metrics_dicts.append(m.to_dict())
            elif isinstance(m, dict):
                metrics_dicts.append(m)
            else:
                logger.warning(f"Metric object {type(m)} has no to_dict() method, converting to dict")
                metrics_dicts.append(asdict(m) if hasattr(m, '__dataclass_fields__') else {})

        outcome = {
            'config': self._serialize_config(config),
            'metrics_progression': metrics_dicts,
            'final_performance': final_performance,
            'training_duration': len(metrics),
            'success_score': self._calculate_success_score(metrics, final_performance)
        }
        self.training_history.append(outcome)
        self._update_meta_model()

    def _update_meta_model(self):
        """Update the meta-learning model based on training history."""
        # This is a placeholder for future meta-learning implementation
        # For now, just track successful strategies
        if len(self.training_history) > 0:
            recent_run = self.training_history[-1]
            if recent_run['success_score'] > 0.7:
                # Extract successful hyperparameters
                strategy = {
                    'learning_rate': recent_run['config'].get('learning_rate'),
                    'batch_size': recent_run['config'].get('batch_size'),
                    'success_score': recent_run['success_score'],
                    'timestamp': time.time()
                }
                self.successful_strategies.append(strategy)

                # Keep only top 20 strategies
                self.successful_strategies.sort(key=lambda x: x['success_score'], reverse=True)
                self.successful_strategies = self.successful_strategies[:20]

    def suggest_hyperparameters(self, current_metrics, config):
        """Suggest hyperparameter adjustments based on meta-learning."""
        if len(self.training_history) < 3:
            return self._conservative_suggestions(current_metrics)

        # Get model params from orchestrator
        current_params = 0
        current_device = 'cpu'
        if self.orchestrator and self.orchestrator.model:
            current_params = sum(p.numel() for p in self.orchestrator.model.parameters())
            current_device = str(self.orchestrator.device.type)

        # Find similar training scenarios
        similar_runs = self._find_similar_runs(current_metrics, config, current_params, current_device)

        # Extract successful patterns
        successful_patterns = [run for run in similar_runs if run['success_score'] > 0.7]

        if not successful_patterns:
            return self._exploratory_suggestions(current_metrics)

        # Generate suggestions based on successful patterns
        suggestions = self._synthesize_suggestions(successful_patterns, current_metrics)

        return suggestions

    def _conservative_suggestions(self, current_metrics):
        """Conservative hyperparameter suggestions for cold start."""
        return {
            'learning_rate': {'value': current_metrics.learning_rate * 0.9, 'confidence': 0.5},
            'batch_size': {'value': None, 'confidence': 0.3}  # Don't change batch size conservatively
        }

    def _exploratory_suggestions(self, current_metrics):
        """Exploratory suggestions when no similar runs found."""
        return {
            'learning_rate': {'value': current_metrics.learning_rate * 1.1, 'confidence': 0.4},
            'warmup_steps': {'value': 500, 'confidence': 0.5}
        }

    def _find_similar_runs(self, current_metrics, config, current_model_params, current_device):
        """Find training runs with similar characteristics using multi-dimensional similarity."""
        similar = []

        for run in self.training_history:
            if len(run['metrics_progression']) == 0:
                continue

            similarity_score = self._calculate_run_similarity(
                current_metrics,
                run,
                current_model_params,
                current_device,
                config  # FIX: Pass config explicitly
            )

            # Use threshold of 0.6 for similarity
            if similarity_score > 0.6:
                similar.append((run, similarity_score))

        # Return sorted by similarity (most similar first)
        similar.sort(key=lambda x: x[1], reverse=True)
        return [run for run, score in similar]

    def _calculate_run_similarity(self, current_metrics, historical_run, current_params, current_device, config):
        """Calculate multi-dimensional similarity score between current and historical runs."""
        score = 0.0

        # Loss similarity (weight: 0.4)
        initial_loss = historical_run['metrics_progression'][0].get('loss', float('inf'))
        if initial_loss < float('inf'):
            loss_diff = abs(current_metrics.loss - initial_loss)
            loss_similarity = max(0, 1.0 - loss_diff / 5.0)  # Normalize by max expected diff
            score += 0.4 * loss_similarity

        # Model size similarity (weight: 0.3)
        if 'model_params' in historical_run and current_params > 0:
            hist_params = historical_run['model_params']
            size_ratio = min(current_params, hist_params) / max(current_params, hist_params)
            score += 0.3 * size_ratio

        # Hardware similarity (weight: 0.2)
        if historical_run.get('device_type') == current_device:
            score += 0.2

        # Architecture similarity (weight: 0.1) - FIX: Use passed config parameter
        if historical_run['config'].get('use_moe') == getattr(config, 'use_moe', False):
            score += 0.05
        if historical_run['config'].get('use_mod') == getattr(config, 'use_mod', False):
            score += 0.05

        return score

    def predict_training_trajectory(self, current_metrics, config):
        """Predict how training will progress."""
        if len(self.adaptation_buffer) < 10:
            return None

        recent_metrics = list(self.adaptation_buffer)[-10:]
        loss_trend = np.polyfit(range(len(recent_metrics)),
                                [m.loss for m in recent_metrics], 1)[0]

        # Predict plateau, convergence, or divergence
        if abs(loss_trend) < 1e-4:
            return {
                'prediction': 'plateau',
                'confidence': 0.8,
                'suggested_action': 'increase_lr_or_change_architecture',
                'expected_improvement': 0.1
            }
        elif loss_trend < -1e-3:
            return {
                'prediction': 'healthy_convergence',
                'confidence': 0.9,
                'suggested_action': 'continue',
                'expected_improvement': abs(loss_trend) * 100
            }
        else:
            return {
                'prediction': 'potential_divergence',
                'confidence': 0.7,
                'suggested_action': 'reduce_lr_or_add_regularization',
                'expected_improvement': 0.05
            }

    def _serialize_config(self, config):
        """Convert config to serializable format."""
        return {
            attr: getattr(config, attr) for attr in dir(config)
            if not attr.startswith('_') and not callable(getattr(config, attr))
        }

    def _calculate_success_score(self, metrics, final_performance):
        """Calculate how successful a training run was."""
        if not metrics:
            return 0.0

        # Factors: convergence speed, final performance, stability
        convergence_speed = 1.0 / len(metrics) if len(metrics) > 0 else 0
        stability = 1.0 - np.std([m.loss for m in metrics[-10:]])

        return 0.4 * final_performance + 0.3 * convergence_speed + 0.3 * stability


class AdaptiveTrainerCallback(TrainerCallback):
    def __init__(self, trainer, monitoring_queue: queue.Queue):
        self.trainer = trainer
        self.monitoring_queue = monitoring_queue

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info(f"Starting training epoch {state.epoch + 1}")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info(f"Completed training epoch {state.epoch}")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        lr = state.log_history[-1].get('learning_rate', 0.0) if state.log_history else args.learning_rate
        self.trainer.current_lr = lr
        metrics = TrainingMetrics(
            epoch=math.floor(state.epoch),
            step=state.global_step,
            learning_rate=lr,
            loss=state.log_history[-1].get('loss', 0.0) if state.log_history else 0.0,
            grad_norm=state.log_history[-1].get('grad_norm', 0.0) if state.log_history else 0.0,
            memory_usage=self._get_memory_usage(),
            timestamp=datetime.datetime.now()
        )
        self.monitoring_queue.put(metrics)
        logger.info(f"Step {state.global_step}: {metrics}")

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_stats = {}

        try:
            if torch.cuda.is_available():
                memory_stats['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
                memory_stats['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
                memory_stats['gpu_memory_percent'] = (
                        torch.cuda.memory_allocated() /
                        torch.cuda.get_device_properties(0).total_memory * 100
                )
        except Exception as e:
            logger.debug(f"Could not get memory usage: {e}")

        return memory_stats


class AdaptiveSFTTrainer(Seq2SeqTrainer):
    def __init__(self, config: SFTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sg_config = config
        self.global_step = 0
        self.current_epoch = 0
        self.current_lr = config.learning_rate
        self.max_consecutive_errors = 5
        self.min_override_threshold = 0.1

        self.meta_learner = MetaLearningEngine(orchestrator=self)
        self.hyperparameter_optimizer = AdaptiveHyperparameterOptimizer()
        self.analytics = RealTimeAnalytics()

        # Training state
        self.training_metrics_history = []
        self.adaptive_decisions = []
        self.current_metrics = None
        self.is_training = False
        self.should_stop = False

        self.monitoring_thread = None
        self.monitoring_queue = queue.Queue(maxsize=1000)

        self._setup_signal_handlers()
        self.add_callback(AdaptiveTrainerCallback(self, self.monitoring_queue))
        self._load_meta_learning_state()

    def train(self, *args, **kwargs):
        self.is_training = True
        start_time = datetime.datetime.now()

        try:
            self.start_real_time_monitoring()
            result = super().train(*args, **kwargs)
            end_time = datetime.datetime.now()
            training_duration = (end_time - start_time).total_seconds()
            final_performance = self._calculate_final_performance()
            self.meta_learner.record_training_outcome(
                self.sg_config, self.training_metrics_history, final_performance
            )
            self._generate_adaptive_insights_report(training_duration, final_performance)
            self._save_meta_learning_state()
            return result
        except Exception as e:
            import traceback as tb
            error_trace = tb.format_exc()
            logger.error(f"Adaptive training failed: {e}")
            logger.error(error_trace)
            raise
        finally:
            self.is_training = False
            self.cleanup()

    def start_real_time_monitoring(self):
        def monitoring_loop():
            consecutive_errors = 0

            while self.is_training and not self.should_stop:
                try:
                    # Get latest metrics with timeout to allow checking should_stop
                    try:
                        metrics = self.monitoring_queue.get(timeout=1.0)
                        self._process_real_time_metrics(metrics)
                    except queue.Empty:
                        # No metrics available, continue loop to check should_stop
                        continue

                except (KeyboardInterrupt, SystemExit):
                    # Allow graceful shutdown
                    break
                except Exception as e:
                    consecutive_errors += 1

                    # Only log first error and every 10th to avoid spam
                    if consecutive_errors == 1 or consecutive_errors % 10 == 0:
                        logger.error(f"Error in monitoring loop (count: {consecutive_errors}): {e}")
                        if consecutive_errors <= 2:
                            import traceback
                            logger.error(traceback.format_exc())

                    # Stop if too many errors
                    if consecutive_errors >= self.max_consecutive_errors:
                        logger.error(f"Too many errors ({consecutive_errors}), stopping monitoring")
                        break

                    time.sleep(0.5)

        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def _setup_signal_handlers(self):
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, saving adaptive learning state...")
            self.should_stop = True
            self.should_stop = True
            self._save_meta_learning_state()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _save_meta_learning_state(self):
        pass

    def _load_meta_learning_state(self):
        pass

    def _calculate_final_performance(self):
        """Calculate final performance metrics."""
        if not self.training_metrics_history:
            return 0.0

        recent_metrics = self.training_metrics_history[-10:]
        avg_loss = np.mean([m.loss for m in recent_metrics])

        # Normalize performance (lower loss = higher performance)
        performance = max(0, 1.0 - min(avg_loss / 10.0, 1.0))
        return performance

    def _calculate_convergence_rate(self, losses):
        """Calculate how quickly the model converged."""
        if len(losses) < 10:
            return 0.0

        # Fit exponential decay to estimate convergence
        steps = np.arange(len(losses))
        try:
            # Simple linear fit to log losses (exponential decay)
            log_losses = np.log(np.array(losses) + 1e-8)
            coeffs = np.polyfit(steps, log_losses, 1)
            return abs(coeffs[0])
        except:
            return 0.0

    def _generate_adaptive_insights_report(self, training_duration, final_performance):
        report = {
            'experiment_name': self.sg_config.run_name,
            'training_duration_seconds': training_duration,
            'final_performance': final_performance,
            'total_adaptive_decisions': len(self.adaptive_decisions),
            'metrics_collected': len(self.training_metrics_history),
            'timestamp': datetime.datetime.now().isoformat()
        }

        decision_types = {}
        for decision in self.adaptive_decisions:
            decision_type = decision.decision_type
            if decision_type not in decision_types:
                decision_types[decision_type] = []
            decision_types[decision_type].append(decision.confidence)

        report['decision_breakdown'] = {}
        for decision_type, confidences in decision_types.items():
            report['decision_breakdown'][decision_type] = {
                'count': len(confidences),
                'avg_confidence': np.mean(confidences),
                'success_rate': len([c for c in confidences if c > 0.7]) / len(confidences)
            }

        if len(self.training_metrics_history) > 10:
            losses = [m.loss for m in self.training_metrics_history]
            report['performance_trends'] = {
                'initial_loss': losses[0],
                'final_loss': losses[-1],
                'best_loss': min(losses),
                'convergence_rate': self._calculate_convergence_rate(losses),
                'stability_score': 1.0 - np.std(losses[-20:]) if len(losses) > 20 else 0.5
            }

        if len(self.meta_learner.training_history) > 1:
            report['meta_learning'] = {
                'historical_runs': len(self.meta_learner.training_history),
                'improvement_over_baseline': final_performance - 0.5,
                'learned_strategies': len(self.meta_learner.successful_strategies)
            }

        logger.info(json.dumps(report, indent=4))

    def _process_real_time_metrics(self, metrics):
        self.current_metrics = metrics
        self.training_metrics_history.append(metrics)
        self.analytics.metrics_buffer.append(metrics)

        anomalies = self.analytics.detect_training_anomalies(metrics)
        if anomalies:
            for anomaly in anomalies:
                logger.warning(f"⚠️ Training anomaly detected: {anomaly}")
                self._handle_training_anomaly(anomaly)

    def _handle_training_anomaly(self, anomaly):
        if anomaly['type'] == 'gradient_explosion':
            # ✅ Mark as emergency
            adjustment = {
                'factor': 0.1,
                'reasoning': 'EMERGENCY: Gradient explosion detected',
                'emergency': True
            }
            self._apply_learning_rate_adjustment(adjustment)
        elif anomaly['type'] == 'loss_spike':
            # ✅ Mark as emergency if severe
            severity = anomaly.get('severity', 'medium')
            adjustment = {
                'factor': 0.5 if severity == 'critical' else 0.8,
                'reasoning': f'Loss spike detected (severity: {severity})',
                'emergency': severity == 'critical'
            }
            self._apply_learning_rate_adjustment(adjustment)

    def _apply_learning_rate_adjustment(self, adjustment):
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = current_lr * adjustment['factor']
        is_emergency = adjustment.get('emergency', False)

        if is_emergency:
            # Emergency changes apply immediately
            logger.warning(
                f"🚨 Emergency LR adjustment: {current_lr:.6f} -> {new_lr:.6f} due to {adjustment['reasoning']}")
        else:
            # Check if change is significant enough
            min_threshold = self.min_override_threshold
            change_ratio = abs(new_lr - current_lr) / current_lr
            if change_ratio < min_threshold:
                return

            logger.warning(f"⚠️ LR adjustment: {current_lr:.6f} -> {new_lr:.6f} due to {adjustment['reasoning']}")

        decision = AdaptiveDecision(
            decision_type='learning_rate_adjustment',
            parameters={
                'old_lr': current_lr,
                'new_lr': new_lr,
                'factor': adjustment['factor'],
                'emergency': is_emergency
            },
            confidence=0.9 if is_emergency else 0.7,
            reasoning=adjustment['reasoning'],
            expected_improvement=0.1,
            timestamp=datetime.datetime.now()
        )
        self._execute_adaptive_decision(decision)

    def _execute_adaptive_decision(self, decision):
        self.adaptive_decisions.append(decision)

        try:
            if decision.decision_type == 'learning_rate_adjustment':
                self.adjust_learning_rate(decision['new_lr'], grace_period=20 if decision['emergency'] else 10)
            elif decision.decision_type == 'corrective_lr_reduction':
                factor = decision.parameters.get('factor', 0.8)
                current_lr = self.current_lr or self.sg_config.learning_rate
                new_lr = current_lr * factor
                self.adjust_learning_rate(new_lr, grace_period=10, emergency=False)
            elif decision.decision_type == 'optimization_lr_increase':
                factor = decision.parameters.get('factor', 1.1)
                current_lr = self.current_lr or self.sg_config.learning_rate
                new_lr = current_lr * factor
                self.adjust_learning_rate(new_lr, grace_period=10, emergency=False)
            elif decision.decision_type == 'emergency_lr_reduction':
                self.emergency_lr_reduction(decision.parameters['factor'])
            elif decision.decision_type == 'plateau_intervention':
                action = decision.parameters.get('action', 'increase_lr')
                if 'increase_lr' in action:
                    current_lr = self.current_lr or self.sg_config.learning_rate
                    new_lr = current_lr * 1.5
                    self.adjust_learning_rate(new_lr, grace_period=15, emergency=False)
            elif decision.decision_type == 'divergence_prevention':
                factor = decision.parameters.get('factor', 0.5)
                self.emergency_lr_reduction(factor)
            else:
                logger.warning(f"Unknown decision type: {decision.decision_type}")

        except Exception as e:
            logger.error(f"Failed to execute adaptive decision {decision.decision_type}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def adjust_learning_rate(self, new_lr: float, grace_period: int = 10, emergency: bool = False):
        old_lr = self.optimizer.param_groups[0]['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.current_lr = new_lr
        pass

    def emergency_lr_reduction(self, reduction_factor: float = 10.0):
        old_lr = self.sg_config.learning_rate
        new_lr = old_lr / reduction_factor
        self.sg_config.learning_rate = new_lr

        if self.deepspeed:
            for param_group in self.deepspeed.optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            # Update standard optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

        # Reset scheduler if it exists
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'base_lrs'):
            self.lr_scheduler.base_lrs = [new_lr for _ in self.lr_scheduler.base_lrs]

    def cleanup(self):
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.is_training = False
            self.monitoring_thread.join(timeout=10)

        try:
            while not self.monitoring_queue.empty():
                try:
                    self.monitoring_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception:
            pass

        self._save_meta_learning_state()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                pass
