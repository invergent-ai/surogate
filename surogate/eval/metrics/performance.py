# surogate/eval/metrics/performance.py
from typing import Dict, Any, Optional, Union, List
import time
import statistics

from surogate.utils.logger import get_logger
from .base import MetricResult, MetricType, BaseMetric
from .registry import register_metric
from surogate.eval.datasets.test_case import TestCase, MultiTurnTestCase
from surogate.eval.targets.base import TargetResponse

logger = get_logger()


@register_metric(MetricType.LATENCY)
class LatencyMetric(BaseMetric):
    """Measure response latency (time to complete request)."""

    def _validate_config(self):
        """Validate configuration."""
        if 'threshold_ms' not in self.config:
            self.config['threshold_ms'] = 5000  # 5 seconds default

    def evaluate(
            self,
            test_case: Union[TestCase, MultiTurnTestCase],
            actual_output: str,
            target_response: Optional[TargetResponse] = None
    ) -> MetricResult:
        """
        Evaluate latency from target response timing.

        Args:
            test_case: Test case
            actual_output: Actual output from target
            target_response: Full response with timing data

        Returns:
            Metric result with latency measurements
        """
        if not target_response or not target_response.timing:
            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=0.0,
                success=False,
                reason="No timing data available"
            )

        try:
            # Get timing data
            total_time = target_response.timing.get('total_time', 0)
            total_time_ms = total_time * 1000  # Convert to milliseconds

            # Get time to first token if available
            time_to_first_token_ms = None
            if 'time_to_first_token' in target_response.timing:
                time_to_first_token_ms = target_response.timing['time_to_first_token'] * 1000

            # Check against threshold
            threshold_ms = self.config['threshold_ms']
            meets_threshold = total_time_ms <= threshold_ms

            # Calculate score (inverse of latency, normalized)
            # Score is 1.0 if latency is 0, approaches 0 as latency increases
            # Using threshold as the normalization point
            score = max(0.0, min(1.0, 1.0 - (total_time_ms / (threshold_ms * 2))))

            metadata = {
                'total_time_ms': round(total_time_ms, 2),
                'total_time_s': round(total_time, 3),
                'threshold_ms': threshold_ms,
                'meets_threshold': meets_threshold
            }

            if time_to_first_token_ms is not None:
                metadata['time_to_first_token_ms'] = round(time_to_first_token_ms, 2)

            # Add percentile info if available
            if 'percentile' in target_response.timing:
                metadata['percentile'] = target_response.timing['percentile']

            reason = f"Latency: {total_time_ms:.0f}ms (threshold: {threshold_ms}ms)"

            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=score,
                success=meets_threshold,
                reason=reason,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Latency measurement failed: {e}")
            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=0.0,
                success=False,
                reason=f"Measurement error: {str(e)}"
            )


@register_metric(MetricType.THROUGHPUT)
class ThroughputMetric(BaseMetric):
    """Measure throughput (requests per second)."""

    def _validate_config(self):
        """Validate configuration."""
        if 'min_rps' not in self.config:
            self.config['min_rps'] = 1.0  # Minimum 1 request/second

    def evaluate(
            self,
            test_case: Union[TestCase, MultiTurnTestCase],
            actual_output: str,
            target_response: Optional[TargetResponse] = None
    ) -> MetricResult:
        """
        Evaluate throughput (single request, used in batch for true throughput).

        Args:
            test_case: Test case
            actual_output: Actual output from target
            target_response: Full response with timing data

        Returns:
            Metric result with throughput measurements
        """
        if not target_response or not target_response.timing:
            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=0.0,
                success=False,
                reason="No timing data available"
            )

        try:
            total_time = target_response.timing.get('total_time', 0)

            if total_time == 0:
                return MetricResult(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    score=0.0,
                    success=False,
                    reason="Zero latency detected (invalid)"
                )

            # Calculate requests per second for this single request
            rps = 1.0 / total_time

            # Check against minimum threshold
            min_rps = self.config['min_rps']
            meets_threshold = rps >= min_rps

            # Score based on RPS relative to minimum
            score = min(1.0, rps / (min_rps * 2))  # Scale so 2x minimum = score of 1.0

            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=score,
                success=meets_threshold,
                reason=f"Throughput: {rps:.2f} req/s (min: {min_rps} req/s)",
                metadata={
                    'requests_per_second': round(rps, 3),
                    'latency_ms': round(total_time * 1000, 2),
                    'min_rps': min_rps,
                    'meets_threshold': meets_threshold
                }
            )

        except Exception as e:
            logger.error(f"Throughput measurement failed: {e}")
            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=0.0,
                success=False,
                reason=f"Measurement error: {str(e)}"
            )

    def evaluate_batch(
            self,
            test_cases: List[Union[TestCase, MultiTurnTestCase]],
            actual_outputs: List[str],
            target_responses: Optional[List[TargetResponse]] = None
    ) -> 'BatchMetricResult':
        """
        Override batch evaluation to calculate true throughput across all requests.

        Args:
            test_cases: List of test cases
            actual_outputs: List of outputs
            target_responses: List of responses with timing

        Returns:
            Batch metric result with aggregate throughput
        """
        from .base import BatchMetricResult

        # Call parent to get individual results
        batch_result = super().evaluate_batch(test_cases, actual_outputs, target_responses)

        if not target_responses:
            return batch_result

        # Calculate aggregate throughput
        try:
            # Get time span of entire batch
            start_times = [r.timing.get('start_time', 0) for r in target_responses if r and r.timing]
            end_times = [r.timing.get('end_time', 0) for r in target_responses if r and r.timing]

            if start_times and end_times:
                total_duration = max(end_times) - min(start_times)
                num_requests = len(test_cases)

                if total_duration > 0:
                    aggregate_rps = num_requests / total_duration
                    logger.metric("Aggregate Throughput", f"{aggregate_rps:.2f} req/s")

                    # Update batch result metadata
                    if not hasattr(batch_result, 'aggregate_metadata'):
                        batch_result.aggregate_metadata = {}

                    batch_result.aggregate_metadata = {
                        'aggregate_requests_per_second': round(aggregate_rps, 3),
                        'total_requests': num_requests,
                        'total_duration_s': round(total_duration, 3),
                        'avg_latency_ms': round((total_duration / num_requests) * 1000, 2)
                    }

        except Exception as e:
            logger.warning(f"Failed to calculate aggregate throughput: {e}")

        return batch_result


@register_metric(MetricType.TOKEN_GENERATION_SPEED)
class TokenGenerationSpeedMetric(BaseMetric):
    """Measure token generation speed (tokens per second)."""

    def _validate_config(self):
        """Validate configuration."""
        if 'min_tokens_per_sec' not in self.config:
            self.config['min_tokens_per_sec'] = 10.0  # Minimum 10 tokens/sec

    def evaluate(
            self,
            test_case: Union[TestCase, MultiTurnTestCase],
            actual_output: str,
            target_response: Optional[TargetResponse] = None
    ) -> MetricResult:
        """
        Evaluate token generation speed.

        Args:
            test_case: Test case
            actual_output: Actual output from target
            target_response: Full response with timing and token data

        Returns:
            Metric result with token generation speed
        """
        if not target_response or not target_response.timing:
            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=0.0,
                success=False,
                reason="No timing data available"
            )

        try:
            total_time = target_response.timing.get('total_time', 0)

            if total_time == 0:
                return MetricResult(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    score=0.0,
                    success=False,
                    reason="Zero latency detected (invalid)"
                )

            # Try to get token count from response metadata
            tokens_generated = None

            # Check usage data (OpenAI format)
            if target_response.metadata and 'usage' in target_response.metadata:
                usage = target_response.metadata['usage']
                if isinstance(usage, dict):
                    tokens_generated = usage.get('completion_tokens')

            # Fallback: estimate from output text
            if tokens_generated is None:
                # Simple whitespace-based token estimation
                # Real tokenizers vary, but ~1.3 tokens per word is a rough average
                words = len(actual_output.split())
                tokens_generated = int(words * 1.3)
                estimated = True
            else:
                estimated = False

            if tokens_generated == 0:
                return MetricResult(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    score=0.5,
                    success=True,
                    reason="No tokens generated (empty response)",
                    metadata={'tokens': 0, 'estimated': estimated}
                )

            # Calculate tokens per second
            tokens_per_sec = tokens_generated / total_time

            # Check against minimum threshold
            min_tokens_per_sec = self.config['min_tokens_per_sec']
            meets_threshold = tokens_per_sec >= min_tokens_per_sec

            # Score based on tokens/sec relative to minimum
            score = min(1.0, tokens_per_sec / (min_tokens_per_sec * 2))

            # Calculate time per token (useful metric)
            time_per_token_ms = (total_time / tokens_generated) * 1000

            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=score,
                success=meets_threshold,
                reason=f"Speed: {tokens_per_sec:.1f} tokens/s (min: {min_tokens_per_sec} tokens/s)",
                metadata={
                    'tokens_per_second': round(tokens_per_sec, 2),
                    'tokens_generated': tokens_generated,
                    'time_per_token_ms': round(time_per_token_ms, 3),
                    'total_time_s': round(total_time, 3),
                    'min_tokens_per_sec': min_tokens_per_sec,
                    'meets_threshold': meets_threshold,
                    'estimated_tokens': estimated
                }
            )

        except Exception as e:
            logger.error(f"Token generation speed measurement failed: {e}")
            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=0.0,
                success=False,
                reason=f"Measurement error: {str(e)}"
            )


class PerformanceAggregator:
    """Helper class to aggregate performance metrics across multiple runs."""

    @staticmethod
    def calculate_percentiles(values: List[float], percentiles: List[int] = None) -> Dict[str, float]:
        """
        Calculate percentiles from a list of values.

        Args:
            values: List of numeric values
            percentiles: List of percentiles to calculate (default: [50, 90, 95, 99])

        Returns:
            Dictionary of percentile values
        """
        if not values:
            return {}

        if percentiles is None:
            percentiles = [50, 90, 95, 99]

        sorted_values = sorted(values)
        n = len(sorted_values)

        result = {}
        for p in percentiles:
            idx = int((p / 100.0) * n)
            idx = min(idx, n - 1)
            result[f'p{p}'] = sorted_values[idx]

        return result

    @staticmethod
    def aggregate_latencies(responses: List[TargetResponse]) -> Dict[str, Any]:
        """
        Aggregate latency statistics from multiple responses.

        Args:
            responses: List of target responses

        Returns:
            Dictionary of latency statistics
        """
        latencies = [
            r.timing.get('total_time', 0) * 1000
            for r in responses
            if r and r.timing
        ]

        if not latencies:
            return {}

        return {
            'min_ms': round(min(latencies), 2),
            'max_ms': round(max(latencies), 2),
            'mean_ms': round(statistics.mean(latencies), 2),
            'median_ms': round(statistics.median(latencies), 2),
            'stdev_ms': round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0.0,
            'percentiles': PerformanceAggregator.calculate_percentiles(latencies)
        }

    @staticmethod
    def aggregate_token_speeds(responses: List[TargetResponse], outputs: List[str]) -> Dict[str, Any]:
        """
        Aggregate token generation speed statistics.

        Args:
            responses: List of target responses
            outputs: List of output strings

        Returns:
            Dictionary of token speed statistics
        """
        speeds = []

        for response, output in zip(responses, outputs):
            if not response or not response.timing:
                continue

            total_time = response.timing.get('total_time', 0)
            if total_time == 0:
                continue

            # Get tokens
            tokens = None
            if response.metadata and 'usage' in response.metadata:
                usage = response.metadata['usage']
                if isinstance(usage, dict):
                    tokens = usage.get('completion_tokens')

            if tokens is None:
                words = len(output.split())
                tokens = int(words * 1.3)

            if tokens > 0:
                tokens_per_sec = tokens / total_time
                speeds.append(tokens_per_sec)

        if not speeds:
            return {}

        return {
            'min_tokens_per_sec': round(min(speeds), 2),
            'max_tokens_per_sec': round(max(speeds), 2),
            'mean_tokens_per_sec': round(statistics.mean(speeds), 2),
            'median_tokens_per_sec': round(statistics.median(speeds), 2),
            'stdev_tokens_per_sec': round(statistics.stdev(speeds), 2) if len(speeds) > 1 else 0.0,
        }