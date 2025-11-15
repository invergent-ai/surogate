# surogate/eval/stress/stress_tester.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import time
import asyncio
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from surogate.utils.logger import get_logger
from surogate.eval.targets.base import BaseTarget, TargetRequest, TargetResponse
from surogate.eval.datasets.test_case import TestCase
from .resource_monitor import ResourceMonitor

logger = get_logger()


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""

    # Load configuration
    num_concurrent: int = 10  # Number of concurrent requests
    duration_seconds: Optional[int] = None  # Test duration (None = use num_requests)
    num_requests: Optional[int] = 100  # Total requests (None = unlimited for duration)

    # Progressive load testing
    progressive: bool = False  # Enable progressive load increase
    start_concurrent: int = 1  # Starting concurrency level
    step_concurrent: int = 2  # Increase per step
    step_duration_seconds: int = 30  # Duration of each step

    # Resource monitoring
    monitor_resources: bool = True  # Enable resource monitoring
    monitoring_interval: float = 0.5  # Seconds between resource snapshots

    # Failure handling
    max_failures: int = 10  # Stop test after this many failures
    retry_on_failure: bool = False  # Retry failed requests

    # Warmup
    warmup_requests: int = 5  # Number of warmup requests before test


@dataclass
class StressTestResult:
    """Results from stress testing."""

    config: StressTestConfig

    # Timing statistics
    total_duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int

    # Throughput
    requests_per_second: float
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float

    # Token statistics (if available)
    total_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = None
    avg_tokens_per_request: Optional[float] = None

    # Resource usage
    resource_summary: Dict[str, Any] = field(default_factory=dict)

    # Errors
    errors: List[str] = field(default_factory=list)

    # Breaking point (for progressive tests)
    breaking_point_concurrent: Optional[int] = None
    breaking_point_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config': {
                'num_concurrent': self.config.num_concurrent,
                'duration_seconds': self.config.duration_seconds,
                'num_requests': self.config.num_requests,
                'progressive': self.config.progressive,
            },
            'total_duration': self.total_duration,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'requests_per_second': self.requests_per_second,
            'latency': {
                'avg_ms': self.avg_latency_ms,
                'median_ms': self.median_latency_ms,
                'p95_ms': self.p95_latency_ms,
                'p99_ms': self.p99_latency_ms,
                'min_ms': self.min_latency_ms,
                'max_ms': self.max_latency_ms,
            },
            'tokens': {
                'total': self.total_tokens,
                'per_second': self.tokens_per_second,
                'avg_per_request': self.avg_tokens_per_request,
            },
            'resources': self.resource_summary,
            'errors': self.errors,
            'breaking_point': {
                'concurrent': self.breaking_point_concurrent,
                'reason': self.breaking_point_reason,
            }
        }


class StressTester:
    """Stress test vLLM and local model targets."""

    def __init__(self, target: BaseTarget, test_cases: List[TestCase]):
        """
        Initialize stress tester.

        Args:
            target: Target to stress test
            test_cases: Test cases to use for requests
        """
        self.target = target
        self.test_cases = test_cases

        if not test_cases:
            raise ValueError("At least one test case required for stress testing")

    def run(self, config: StressTestConfig) -> StressTestResult:
        """
        Run stress test.

        Args:
            config: Stress test configuration

        Returns:
            Stress test results
        """
        logger.separator(char="═")
        logger.header("STRESS TEST")
        logger.separator(char="═")
        logger.info(f"Target: {self.target.name}")
        logger.info(f"Test cases: {len(self.test_cases)}")
        logger.info(f"Concurrent requests: {config.num_concurrent}")

        if config.progressive:
            return self._run_progressive(config)
        else:
            return self._run_fixed_load(config)

    def _run_fixed_load(self, config: StressTestConfig) -> StressTestResult:
        """Run stress test with fixed concurrency level."""
        # Start resource monitoring
        monitor = None
        if config.monitor_resources:
            monitor = ResourceMonitor(interval=config.monitoring_interval)
            monitor.start()

        try:
            # Warmup
            if config.warmup_requests > 0:
                logger.info(f"Running {config.warmup_requests} warmup requests...")
                self._run_warmup(config.warmup_requests)
                logger.success("Warmup complete")

            # Run main test
            logger.info(f"Starting stress test with {config.num_concurrent} concurrent requests")
            start_time = time.time()

            results = self._execute_concurrent_requests(
                num_concurrent=config.num_concurrent,
                num_requests=config.num_requests,
                duration_seconds=config.duration_seconds,
                max_failures=config.max_failures
            )

            end_time = time.time()
            total_duration = end_time - start_time

            # Stop monitoring
            if monitor:
                monitor.stop()

            # Process results
            return self._create_result(config, results, total_duration, monitor)

        finally:
            if monitor:
                monitor.stop()

    def _run_progressive(self, config: StressTestConfig) -> StressTestResult:
        """Run progressive load test with increasing concurrency."""
        logger.info("Starting progressive load test")
        logger.info(f"Start: {config.start_concurrent} → Step: +{config.step_concurrent}")

        # Start resource monitoring
        monitor = None
        if config.monitor_resources:
            monitor = ResourceMonitor(interval=config.monitoring_interval)
            monitor.start()

        try:
            current_concurrent = config.start_concurrent
            all_results = []
            total_start_time = time.time()
            breaking_point = None
            breaking_reason = None

            while True:
                logger.separator(char="─")
                logger.info(f"Testing with {current_concurrent} concurrent requests")

                step_results = self._execute_concurrent_requests(
                    num_concurrent=current_concurrent,
                    num_requests=None,  # Use duration instead
                    duration_seconds=config.step_duration_seconds,
                    max_failures=config.max_failures
                )

                all_results.extend(step_results)

                # Check for breaking point
                failed_ratio = len([r for r in step_results if r['error']]) / len(step_results)
                if failed_ratio > 0.2:  # More than 20% failures
                    breaking_point = current_concurrent
                    breaking_reason = f"High failure rate: {failed_ratio:.1%}"
                    logger.warning(f"Breaking point reached at {current_concurrent} concurrent: {breaking_reason}")
                    break

                # Calculate step throughput
                step_duration = step_results[-1]['timing']['end_time'] - step_results[0]['timing']['start_time']
                step_rps = len(step_results) / step_duration
                logger.metric("Step RPS", f"{step_rps:.2f}")

                # Increase concurrency
                current_concurrent += config.step_concurrent

                # Stop if we've reached max concurrent
                if config.num_concurrent and current_concurrent > config.num_concurrent:
                    logger.info(f"Reached max concurrency: {config.num_concurrent}")
                    break

            total_duration = time.time() - total_start_time

            # Stop monitoring
            if monitor:
                monitor.stop()

            # Create result with breaking point info
            result = self._create_result(config, all_results, total_duration, monitor)
            result.breaking_point_concurrent = breaking_point
            result.breaking_point_reason = breaking_reason

            return result

        finally:
            if monitor:
                monitor.stop()

    def _run_warmup(self, num_requests: int):
        """Run warmup requests sequentially."""
        for i in range(num_requests):
            test_case = self.test_cases[i % len(self.test_cases)]
            request = TargetRequest(prompt=test_case.input)
            try:
                self.target.send_request(request)
            except Exception as e:
                logger.debug(f"Warmup request {i + 1} failed: {e}")

    def _execute_concurrent_requests(
            self,
            num_concurrent: int,
            num_requests: Optional[int],
            duration_seconds: Optional[int],
            max_failures: int
    ) -> List[Dict[str, Any]]:
        """Execute requests concurrently."""
        results = []
        failures = 0
        request_count = 0
        start_time = time.time()

        # Use ThreadPoolExecutor for concurrent execution
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = []

            while True:
                # Check stop conditions
                if num_requests and request_count >= num_requests:
                    break
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    break
                if failures >= max_failures:
                    logger.warning(f"Stopping test: reached max failures ({max_failures})")
                    break

                # Submit new requests up to concurrency limit
                while len(futures) < num_concurrent:
                    if num_requests and request_count >= num_requests:
                        break

                    test_case = self.test_cases[request_count % len(self.test_cases)]
                    future = executor.submit(self._send_single_request, test_case, request_count)
                    futures.append(future)
                    request_count += 1

                # Wait for at least one to complete (FIXED: longer timeout)
                # Wait for at least one to complete (FIXED: longer timeout)
                if futures:
                    done_futures = []
                    try:
                        # Use 30 second timeout instead of 1 second
                        for future in as_completed(futures, timeout=30):
                            result = future.result()
                            results.append(result)

                            if result['error']:
                                failures += 1

                            done_futures.append(future)
                    except TimeoutError:
                        # If timeout occurs, log warning but continue
                        logger.warning(f"Timeout waiting for {len(futures)} futures, continuing...")
                        # Try to get results from any completed futures
                        done_futures = []  # Reset to avoid duplicates
                        for future in list(futures):  # Create a copy to iterate
                            if future.done():
                                try:
                                    result = future.result(timeout=0)
                                    results.append(result)
                                    if result['error']:
                                        failures += 1
                                    done_futures.append(future)
                                except Exception as e:
                                    logger.debug(f"Error getting future result: {e}")

                    # Remove completed futures (using set difference to avoid duplicates)
                    futures = [f for f in futures if f not in done_futures]

                    # Check duration for infinite requests mode
                    if not num_requests and duration_seconds:
                        if (time.time() - start_time) >= duration_seconds:
                            break

                    # Small sleep to prevent tight loop
                    if not done_futures:
                        time.sleep(0.1)

            # Wait for remaining futures with longer timeout
            logger.info(f"Waiting for {len(futures)} remaining requests...")
            for future in futures:
                try:
                    result = future.result(timeout=60)  # 60 second timeout for final futures
                    results.append(result)
                    if result['error']:
                        failures += 1
                except Exception as e:
                    logger.error(f"Future failed: {e}")
                    # Add error result
                    results.append({
                        'request_id': -1,
                        'success': False,
                        'error': str(e),
                        'timing': {'start_time': time.time(), 'end_time': time.time(), 'duration': 0},
                        'tokens': None,
                        'output_length': 0
                    })
                    failures += 1

        return results

    def _send_single_request(self, test_case: TestCase, request_id: int) -> Dict[str, Any]:
        """Send a single request and capture metrics."""
        start_time = time.time()

        try:
            request = TargetRequest(prompt=test_case.input)
            response = self.target.send_request(request)

            end_time = time.time()

            # Extract token count if available
            tokens = None
            if response.metadata and 'usage' in response.metadata:
                usage = response.metadata['usage']
                if isinstance(usage, dict):
                    tokens = usage.get('completion_tokens')

            # Estimate tokens from output if not provided
            if tokens is None and response.content:
                words = len(response.content.split())
                tokens = int(words * 1.3)

            return {
                'request_id': request_id,
                'success': response.error is None,
                'error': response.error,
                'timing': {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                },
                'tokens': tokens,
                'output_length': len(response.content) if response.content else 0,
            }

        except Exception as e:
            end_time = time.time()
            logger.debug(f"Request {request_id} failed: {e}")
            return {
                'request_id': request_id,
                'success': False,
                'error': str(e),
                'timing': {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                },
                'tokens': None,
                'output_length': 0,
            }

    def _create_result(
            self,
            config: StressTestConfig,
            results: List[Dict[str, Any]],
            total_duration: float,
            monitor: Optional[ResourceMonitor]
    ) -> StressTestResult:
        """Create stress test result from raw results."""
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        # Calculate latency statistics
        latencies = [r['timing']['duration'] * 1000 for r in successful]  # Convert to ms

        if latencies:
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)

            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            p95_latency = latencies_sorted[int(0.95 * n)]
            p99_latency = latencies_sorted[int(0.99 * n)]
            min_latency = min(latencies)
            max_latency = max(latencies)
        else:
            avg_latency = median_latency = p95_latency = p99_latency = min_latency = max_latency = 0.0

        # Calculate throughput
        requests_per_second = len(results) / total_duration

        # Calculate token statistics
        token_results = [r for r in successful if r['tokens'] is not None]
        if token_results:
            total_tokens = sum(r['tokens'] for r in token_results)
            tokens_per_second = total_tokens / total_duration
            avg_tokens_per_request = total_tokens / len(token_results)
        else:
            total_tokens = tokens_per_second = avg_tokens_per_request = None

        # Get resource summary
        resource_summary = {}
        if monitor:
            resource_summary = monitor.get_summary()

        # Collect unique errors
        errors = list(set(r['error'] for r in failed if r['error']))

        # Log summary
        logger.separator(char="─")
        logger.success(f"Stress test complete: {len(results)} requests in {total_duration:.1f}s")
        logger.metric("Throughput", f"{requests_per_second:.2f} req/s")
        logger.metric("Avg Latency", f"{avg_latency:.1f}ms")
        logger.metric("P95 Latency", f"{p95_latency:.1f}ms")
        logger.metric("Success Rate", f"{len(successful) / len(results):.1%}")

        if tokens_per_second:
            logger.metric("Token Throughput", f"{tokens_per_second:.1f} tokens/s")

        if resource_summary:
            if 'gpu' in resource_summary:
                logger.metric("GPU Memory", f"{resource_summary['gpu'].get('memory_avg_mb', 0):.1f}MB avg")
                logger.metric("GPU Utilization", f"{resource_summary['gpu'].get('utilization_avg', 0):.1f}% avg")

        return StressTestResult(
            config=config,
            total_duration=total_duration,
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            requests_per_second=requests_per_second,
            avg_latency_ms=avg_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second,
            avg_tokens_per_request=avg_tokens_per_request,
            resource_summary=resource_summary,
            errors=errors,
        )