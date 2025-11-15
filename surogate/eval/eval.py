# surogate/eval/eval.py
from pathlib import Path
from typing import Dict, Any, List

from surogate.eval.backend import LocalBackend
from surogate.eval.benchmarks import BenchmarkRegistry, BenchmarkConfig
from surogate.eval.config import ConfigParser, ConfigValidator
from surogate.eval.datasets import DatasetLoader, DatasetValidator
from surogate.eval.targets import BaseTarget, TargetFactory
from surogate.utils.logger import get_logger

logger = get_logger()


class SurogateEval:
    """Main evaluation orchestrator."""

    def __init__(self, config: str, **kwargs):
        """
        Initialize Surogate Eval.

        Args:
            config: Path to configuration file
            **kwargs: Additional arguments
        """
        self.config_path = config
        self.kwargs = kwargs
        self.config = None
        self.targets: List[BaseTarget] = []

        # Consolidated results structure
        self.consolidated_results = {
            "project": {},
            "timestamp": None,
            "summary": {
                "total_targets": 0,
                "total_evaluations": 0,
                "total_test_cases": 0
            },
            "targets": []
        }

    def run(self):
        """Run the evaluation pipeline."""
        from datetime import datetime

        logger.banner("SUROGATE EVAL")

        # Set timestamp at start
        self.consolidated_results["timestamp"] = datetime.now().isoformat()

        # 1. Parse configuration
        parser = ConfigParser(self.config_path)
        self.config = parser.parse()

        # Store project info
        self.consolidated_results["project"] = self.config.get('project', {})

        # 2. Validate configuration
        validator = ConfigValidator(self.config)
        is_valid, errors = validator.validate()

        if not is_valid:
            logger.error("Configuration validation failed")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError("Invalid configuration")

        # 3. Process each target
        try:
            self._process_targets()
        finally:
            # 4. Cleanup
            self._cleanup()

        # 5. Save consolidated results
        self._save_consolidated_results()

        logger.success("Surogate Eval completed")

    def _process_targets(self):
        """Process all targets from config."""
        target_configs = self.config.get('targets', [])

        if not target_configs:
            logger.warning("No targets specified in configuration")
            return

        logger.info(f"Processing {len(target_configs)} target(s)")

        self.consolidated_results["summary"]["total_targets"] = len(target_configs)

        for target_config in target_configs:
            target_name = target_config.get('name', 'unnamed')
            logger.separator(char="═")
            logger.header(f"Target: {target_name}")
            logger.separator(char="═")

            try:
                target_results = self._process_single_target(target_config)
                if target_results:
                    self.consolidated_results["targets"].append(target_results)
            except Exception as e:
                logger.error(f"Failed to process target '{target_name}': {e}")
                import traceback
                traceback.print_exc()

                # Add failed target to results
                self.consolidated_results["targets"].append({
                    "name": target_name,
                    "status": "failed",
                    "error": str(e),
                    "evaluations": []
                })
                continue

    def _process_single_target(self, target_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single target with its evaluations.

        Args:
            target_config: Target configuration dictionary

        Returns:
            Target results dictionary
        """
        target_name = target_config.get('name')

        # 1. Create target
        logger.info(f"Creating target: {target_name}")
        target = TargetFactory.create_target(target_config)
        self.targets.append(target)

        # 2. Health check
        if not target.health_check():
            logger.error(f"Target '{target_name}' health check failed - skipping")
            return {
                "name": target_name,
                "type": target.target_type.value,
                "model": target.config.get('model', 'unknown'),
                "status": "unhealthy",
                "evaluations": []
            }

        logger.success(f"Target '{target_name}' is healthy")

        # Initialize target result structure
        target_result = {
            "name": target_name,
            "type": target.target_type.value,
            "model": target.config.get('model', 'unknown'),
            "provider": target.config.get('provider', 'unknown'),
            "status": "success",
            "evaluations": []
        }

        # 3. Setup backend (if infrastructure specified)
        backend = self._setup_target_backend(target_config)

        # 4. Run evaluations
        evaluations = target_config.get('evaluations', [])
        if evaluations:
            logger.info(f"Running {len(evaluations)} evaluation(s) for target '{target_name}'")
            self.consolidated_results["summary"]["total_evaluations"] += len(evaluations)

            for eval_config in evaluations:
                eval_result = self._run_evaluation(target, eval_config, backend)
                if eval_result:
                    target_result["evaluations"].append(eval_result)
                    # Add test case count to summary
                    self.consolidated_results["summary"]["total_test_cases"] += eval_result.get("num_test_cases", 0)
        else:
            logger.warning(f"No evaluations specified for target '{target_name}'")

        # 4b. Run benchmarks (if specified in evaluations)
        for eval_config in evaluations:
            benchmarks_config = eval_config.get('benchmarks', [])

            if benchmarks_config:
                logger.info(f"Running {len(benchmarks_config)} benchmark(s)")

                benchmark_results = self._run_benchmarks(target, benchmarks_config)

                # Add to target results
                if benchmark_results:
                    target_result['benchmarks'] = benchmark_results

        # 5. Run stress testing (if enabled)
        stress_testing = target_config.get('stress_testing', {})
        if stress_testing.get('enabled'):
            logger.info(f"Running stress testing for target '{target_name}'")
            stress_result = self._run_stress_testing(target, stress_testing)
            if stress_result:
                target_result["stress_testing"] = stress_result

        # 6. Run red teaming (if enabled)
        red_teaming = target_config.get('red_teaming', {})
        if red_teaming.get('enabled'):
            logger.info(f"Running red teaming for target '{target_name}'")
            red_team_result = self._run_red_teaming(target, red_teaming)
            if red_team_result:
                target_result["red_teaming"] = red_team_result

        # 7. Test guardrails (if enabled)
        guardrails = target_config.get('guardrails', {})
        if guardrails.get('enabled'):
            logger.info(f"Testing guardrails for target '{target_name}'")
            guardrails_result = self._run_guardrails_testing(target, guardrails)
            if guardrails_result:
                target_result["guardrails"] = guardrails_result

        # 8. Shutdown backend
        if backend:
            backend.shutdown()

        return target_result

    def _setup_target_backend(self, target_config: Dict[str, Any]) -> Any:
        """
        Setup execution backend for a target.

        Args:
            target_config: Target configuration

        Returns:
            Backend instance or None
        """
        infra_config = target_config.get('infrastructure', {})

        if not infra_config:
            logger.debug("No infrastructure config - using default")
            return None

        backend_type = infra_config.get('backend', 'local')

        if backend_type == 'local':
            backend = LocalBackend(infra_config)
            logger.success(f"Local backend initialized with {infra_config.get('workers', 1)} workers")
            return backend
        else:
            raise NotImplementedError(f"Backend '{backend_type}' not implemented yet")

    def _run_evaluation(
            self,
            target: BaseTarget,
            eval_config: Dict[str, Any],
            backend: Any = None
    ) -> Dict[str, Any]:
        """
        Run a single evaluation on a target.

        Args:
            target: Target to evaluate
            eval_config: Evaluation configuration
            backend: Execution backend (optional)

        Returns:
            Evaluation result dictionary
        """
        from datetime import datetime

        eval_name = eval_config.get('name', 'unnamed')
        dataset_path = eval_config.get('dataset')

        logger.separator(char="─")
        logger.header(f"Evaluation: {eval_name}")
        logger.info(f"Dataset: {dataset_path}")
        logger.separator(char="─")

        if not dataset_path:
            logger.warning(f"No dataset specified for evaluation '{eval_name}'")
            return None

        try:
            # Load dataset
            loader = DatasetLoader()
            dataset_type = loader.detect_dataset_type(dataset_path)
            logger.info(f"Dataset type: {dataset_type}")

            test_cases = loader.load_test_cases(dataset_path)
            logger.info(f"Loaded {len(test_cases)} test cases")

            # Validate dataset
            validator = DatasetValidator()
            df = loader.load(dataset_path)
            is_valid, errors = validator.validate(df)

            if not is_valid:
                logger.error("Dataset validation failed:")
                for error in errors:
                    logger.error(f"  - {error}")
                return {
                    "name": eval_name,
                    "dataset": dataset_path,
                    "dataset_type": dataset_type,
                    "status": "validation_failed",
                    "errors": errors
                }

            # Load metrics
            from surogate.eval.metrics import MetricRegistry

            metric_configs = eval_config.get('metrics', [])
            if not metric_configs:
                logger.warning(f"No metrics specified for evaluation '{eval_name}'")
                return None

            # Filter metrics by dataset type
            filtered_metric_configs = self._filter_metrics_by_dataset_type(
                metric_configs,
                dataset_type
            )

            if not filtered_metric_configs:
                logger.error(f"No compatible metrics for dataset type: {dataset_type}")
                return None

            logger.info(f"Using {len(filtered_metric_configs)} metric(s)")

            metrics = MetricRegistry.create_metrics(filtered_metric_configs)

            # Run inference
            logger.info(f"Running inference on {len(test_cases)} test cases...")
            target_outputs = []
            target_responses = []

            for idx, test_case in enumerate(test_cases):
                try:
                    from surogate.eval.targets.base import TargetRequest
                    from surogate.eval.datasets.test_case import TestCase, MultiTurnTestCase

                    if isinstance(test_case, TestCase):
                        request = TargetRequest(prompt=test_case.input)
                    elif isinstance(test_case, MultiTurnTestCase):
                        request = TargetRequest(messages=test_case.get_context())
                    else:
                        logger.error(f"Unknown test case type: {type(test_case)}")
                        continue

                    response = target.send_request(request)
                    target_outputs.append(response.content)
                    target_responses.append(response)

                    if (idx + 1) % 10 == 0:
                        logger.step(idx + 1, len(test_cases), f"Progress: {idx + 1}/{len(test_cases)} test cases")

                except Exception as e:
                    logger.error(f"Failed to get output for test case {idx}: {e}")
                    target_outputs.append("")
                    target_responses.append(None)

            logger.success(f"Completed inference on {len(test_cases)} test cases")

            # Run metrics
            metric_results = {}
            detailed_results = []

            for metric in metrics:
                logger.info(f"Running metric: {metric.name}")

                try:
                    # Set judge target if needed
                    from surogate.eval.metrics import LLMJudgeMetric
                    if isinstance(metric, LLMJudgeMetric):
                        judge_config = metric.config.get('judge_model', {})
                        judge_target_name = judge_config.get('target')

                        if judge_target_name:
                            judge_target = self._find_target_by_name(judge_target_name)
                            if judge_target:
                                metric.set_judge_target(judge_target)
                                logger.debug(f"Set judge target '{judge_target_name}'")
                            else:
                                logger.warning(f"Judge target '{judge_target_name}' not found")

                    # Evaluate batch
                    batch_result = metric.evaluate_batch(
                        test_cases,
                        target_outputs,
                        target_responses
                    )

                    # Store aggregated results
                    metric_results[metric.name] = batch_result.to_dict()

                    # Store detailed per-test-case results
                    for i, individual_result in enumerate(batch_result.results):
                        if i >= len(detailed_results):
                            from surogate.eval.datasets.test_case import TestCase
                            if isinstance(test_cases[i], TestCase):
                                input_preview = test_cases[i].input[:100]
                            else:
                                input_preview = f"multi-turn ({len(test_cases[i].turns)} turns)"

                            detailed_results.append({
                                'test_case_index': i,
                                'input': input_preview,
                                'output': target_outputs[i][:200] if target_outputs[i] else "",
                                'metrics': {}
                            })

                        detailed_results[i]['metrics'][metric.name] = {
                            'score': individual_result.score,
                            'success': individual_result.success,
                            'reason': individual_result.reason,
                            'metadata': individual_result.metadata
                        }

                    # Use metric() method for displaying results
                    logger.metric(f"{metric.name} - Avg Score", f"{batch_result.avg_score:.3f}")
                    logger.metric(f"{metric.name} - Success Rate", f"{batch_result.success_rate:.3f}")

                except Exception as e:
                    logger.error(f"Metric {metric.name} failed: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    metric_results[metric.name] = {
                        'error': str(e),
                        'status': 'failed'
                    }

            # Create evaluation result
            return {
                "name": eval_name,
                "dataset": dataset_path,
                "dataset_type": dataset_type,
                "num_test_cases": len(test_cases),
                "num_metrics": len(metrics),
                "status": "completed",
                "metrics_summary": metric_results,
                "detailed_results": detailed_results
            }

        except Exception as e:
            logger.error(f"Failed to run evaluation '{eval_name}': {e}")
            import traceback
            traceback.print_exc()

            return {
                "name": eval_name,
                "dataset": dataset_path,
                "status": "failed",
                "error": str(e)
            }

    # In eval.py - simplify _run_benchmarks

    def _run_benchmarks(
            self,
            target: BaseTarget,
            benchmark_configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run benchmarks on target.

        Args:
            target: Target to evaluate
            benchmark_configs: List of benchmark configurations

        Returns:
            List of benchmark results
        """
        if not benchmark_configs:
            return []

        logger.separator(char="─")
        logger.header(f"Running {len(benchmark_configs)} Benchmark(s)")
        logger.separator(char="─")

        benchmark_results = []

        for bench_config in benchmark_configs:
            bench_result = self._run_single_benchmark(target, bench_config)
            if bench_result:
                benchmark_results.append(bench_result)

        return benchmark_results

    # surogate/eval/eval.py - Update _run_single_benchmark method

    def _run_single_benchmark(
            self,
            target: BaseTarget,
            bench_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single benchmark on target."""
        benchmark_name = bench_config.get('name')
        logger.info(f"Running benchmark: {benchmark_name}")

        try:
            from surogate.eval.benchmarks import BenchmarkConfig

            config = BenchmarkConfig(
                name=benchmark_name,
                path=bench_config.get('path'),
                num_fewshot=bench_config.get('num_fewshot'),
                limit=bench_config.get('limit'),
                tasks=bench_config.get('tasks'),
                subset=bench_config.get('subset'),
                use_cache=bench_config.get('use_cache', True),
                cache_dir=bench_config.get('cache_dir'),
                backend_params=bench_config.get('backend_params', {}),
            )

            # Create benchmark instance
            benchmark = BenchmarkRegistry.create_benchmark(config)

            # Get judge target if specified
            judge_model_config = bench_config.get('judge_model')
            if judge_model_config:
                judge_target_name = judge_model_config.get('target')
                judge_target = self._find_target_by_name(judge_target_name)
                if judge_target:
                    # Pass judge to backend via backend_params
                    benchmark.config.backend_params['judge_target'] = judge_target
                    logger.info(f"Using judge '{judge_target_name}' for benchmark '{benchmark_name}'")
                else:
                    logger.warning(f"Judge target '{judge_target_name}' not found")

            # Validate target compatibility
            if not benchmark.validate_target(target):
                target_type = target.target_type.value
                required = benchmark.REQUIRED_TARGET_TYPES
                logger.error(
                    f"Target '{target.name}' (type: {target_type}) not compatible with "
                    f"benchmark '{benchmark_name}' (requires: {required})"
                )
                return {
                    'benchmark': benchmark_name,
                    'status': 'incompatible',
                    'error': f'Benchmark requires {required} target, got {target_type}'
                }

            # Run benchmark
            result = benchmark.evaluate(target)

            # Convert to dict
            result_dict = result.to_dict()
            result_dict['status'] = 'completed'

            logger.success(f"Benchmark '{benchmark_name}' completed")
            logger.metric(f"{benchmark_name} - Overall Score", f"{result.overall_score:.4f}")

            return result_dict

        except Exception as e:
            logger.error(f"Benchmark '{benchmark_name}' failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

            return {
                'benchmark': benchmark_name,
                'status': 'failed',
                'error': str(e)
            }

    def _run_stress_testing(self, target: BaseTarget, stress_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run stress testing on target.

        Args:
            target: Target to stress test
            stress_config: Stress testing configuration

        Returns:
            Stress test results
        """
        from surogate.eval.metrics.stress import StressTester, StressTestConfig
        from surogate.eval.datasets import DatasetLoader

        logger.info(f"Running stress test for target '{target.name}'")

        try:
            # Load test dataset
            dataset_path = stress_config.get('dataset')
            if not dataset_path:
                logger.error("No dataset specified for stress testing")
                return {"status": "error", "reason": "No dataset specified"}

            loader = DatasetLoader()
            test_cases = loader.load_test_cases(dataset_path)

            logger.info(f"Loaded {len(test_cases)} test cases for stress testing")

            # Create stress test config
            config = StressTestConfig(
                num_concurrent=stress_config.get('num_concurrent', 10),
                duration_seconds=stress_config.get('duration_seconds'),
                num_requests=stress_config.get('num_requests', 100),
                progressive=stress_config.get('progressive', False),
                start_concurrent=stress_config.get('start_concurrent', 1),
                step_concurrent=stress_config.get('step_concurrent', 2),
                step_duration_seconds=stress_config.get('step_duration_seconds', 30),
                monitor_resources=stress_config.get('monitor_resources', True),
                warmup_requests=stress_config.get('warmup_requests', 5),
            )

            # Run stress test
            tester = StressTester(target, test_cases)
            result = tester.run(config)

            return result.to_dict()

        except Exception as e:
            logger.error(f"Stress testing failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "reason": str(e)}

    def _find_target_by_name(self, name: str) -> BaseTarget:
        """Find a target by name from created targets."""
        for target in self.targets:
            if target.name == name:
                return target
        return None

    def _filter_metrics_by_dataset_type(
            self,
            metric_configs: List[Dict[str, Any]],
            dataset_type: str
    ) -> List[Dict[str, Any]]:
        """
        Filter metrics based on dataset type compatibility.

        Args:
            metric_configs: List of metric configurations
            dataset_type: 'single_turn' or 'multi_turn'

        Returns:
            Filtered list of compatible metric configurations
        """
        single_turn_metrics = {
            'g_eval',
            'dag',
            'multimodal_g_eval',  # ADD THIS LINE
            'toxicity',
            'bias',
            'harm',
            'embedding_similarity',
            'classification',
            'latency',
            'throughput',
            'token_generation_speed',
        }

        multi_turn_metrics = {
            'conversational_g_eval',
            'conversation_coherence',
            'context_retention',
            'turn_analysis',
            'conversational_dag',
            'multimodal_g_eval',  # ADD THIS LINE TOO (works for both)
            'toxicity',
            'bias',
            'harm',
            'latency',
            'throughput',
            'token_generation_speed',
        }

        filtered_configs = []
        skipped_metrics = []

        for config in metric_configs:
            metric_type = config.get('type')
            metric_name = config.get('name', metric_type)

            is_compatible = False

            if dataset_type == 'single_turn':
                is_compatible = metric_type in single_turn_metrics
            elif dataset_type == 'multi_turn':
                is_compatible = metric_type in multi_turn_metrics

            if is_compatible:
                filtered_configs.append(config)
                logger.debug(f"Metric '{metric_name}' is compatible with {dataset_type}")
            else:
                skipped_metrics.append(metric_name)
                logger.warning(f"Skipping metric '{metric_name}' - incompatible with {dataset_type}")

        if skipped_metrics:
            logger.info(f"Skipped {len(skipped_metrics)} incompatible metrics: {', '.join(skipped_metrics)}")

        return filtered_configs

    def _run_red_teaming(self, target: BaseTarget, red_team_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run red teaming tests on target.

        Args:
            target: Target to test
            red_team_config: Red teaming configuration

        Returns:
            Red teaming results
        """
        logger.info(f"Red teaming not yet implemented for target '{target.name}'")
        logger.debug(f"Red teaming config: {red_team_config}")

        return {
            "status": "not_implemented",
            "config": red_team_config
        }

    def _run_guardrails_testing(self, target: BaseTarget, guardrails_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test guardrails on target.

        Args:
            target: Target to test
            guardrails_config: Guardrails configuration

        Returns:
            Guardrails test results
        """
        logger.info(f"Guardrails testing not yet implemented for target '{target.name}'")
        logger.debug(f"Guardrails config: {guardrails_config}")

        return {
            "status": "not_implemented",
            "config": guardrails_config
        }

    def _save_consolidated_results(self):
        """Save consolidated results to a single file."""
        try:
            from pathlib import Path
            import json
            from datetime import datetime

            # Create results directory
            results_dir = Path("eval_results")
            results_dir.mkdir(exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_{timestamp}.json"
            filepath = results_dir / filename

            # Save JSON results with default handler for non-serializable types
            with open(filepath, 'w') as f:
                json.dump(self.consolidated_results, f, indent=2, default=str)  # ← ADD default=str

            logger.separator(char="═")
            logger.success(f"Consolidated results saved to: {filepath}")
            logger.separator(char="═")

            # Create summary report
            self._create_summary_report(self.consolidated_results, results_dir, timestamp)

        except Exception as e:
            logger.error(f"Failed to save consolidated results: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _create_summary_report(
            self,
            results: Dict[str, Any],
            results_dir: Path,
            timestamp: str
    ):
        """
        Create a human-readable summary report.

        Args:
            results: Consolidated results dictionary
            results_dir: Directory to save report
            timestamp: Timestamp for filename
        """
        try:
            report_file = results_dir / f"report_{timestamp}.md"

            with open(report_file, 'w') as f:
                # Header
                f.write(f"# Evaluation Report\n\n")
                project_info = results.get('project', {})
                f.write(f"**Project:** {project_info.get('name', 'N/A')}\n")
                f.write(f"**Version:** {project_info.get('version', 'N/A')}\n")
                f.write(f"**Generated:** {results.get('timestamp', 'N/A')}\n\n")

                # Summary
                summary = results.get('summary', {})
                f.write(f"## Summary\n\n")
                f.write(f"- **Total Targets:** {summary.get('total_targets', 0)}\n")
                f.write(f"- **Total Evaluations:** {summary.get('total_evaluations', 0)}\n")
                f.write(f"- **Total Test Cases:** {summary.get('total_test_cases', 0)}\n\n")

                # Per-target results
                for target in results.get('targets', []):
                    f.write(f"## Target: {target.get('name', 'Unknown')}\n\n")
                    f.write(f"- **Type:** {target.get('type', 'N/A')}\n")
                    f.write(f"- **Model:** {target.get('model', 'N/A')}\n")
                    f.write(f"- **Provider:** {target.get('provider', 'N/A')}\n")
                    f.write(f"- **Status:** {target.get('status', 'N/A')}\n\n")

                    # Evaluations for this target
                    evaluations = target.get('evaluations', [])
                    if evaluations:
                        f.write(f"### Evaluations ({len(evaluations)})\n\n")

                        for eval_result in evaluations:
                            eval_name = eval_result.get('name', 'Unknown')
                            f.write(f"#### {eval_name}\n\n")
                            f.write(f"- **Dataset:** {eval_result.get('dataset', 'N/A')}\n")
                            f.write(f"- **Dataset Type:** {eval_result.get('dataset_type', 'N/A')}\n")
                            f.write(f"- **Test Cases:** {eval_result.get('num_test_cases', 0)}\n")
                            f.write(f"- **Status:** {eval_result.get('status', 'N/A')}\n\n")

                            # Metrics table
                            if 'metrics_summary' in eval_result:
                                f.write(f"##### Metrics Performance\n\n")
                                f.write(f"| Metric | Avg Score | Success Rate | Status |\n")
                                f.write(f"|--------|-----------|--------------|--------|\n")

                                metrics_summary = eval_result.get('metrics_summary', {})
                                for metric_name, metric_data in metrics_summary.items():
                                    if 'error' in metric_data:
                                        f.write(f"| {metric_name} | N/A | N/A | ❌ Failed |\n")
                                    else:
                                        avg_score = metric_data.get('avg_score', 0)
                                        success_rate = metric_data.get('success_rate', 0)

                                        if success_rate >= 0.8:
                                            status = "✅ Excellent"
                                        elif success_rate >= 0.6:
                                            status = "⚠️  Good"
                                        else:
                                            status = "❌ Needs Work"

                                        f.write(
                                            f"| {metric_name} | {avg_score:.3f} | {success_rate:.3f} | {status} |\n")

                                f.write(f"\n")

                    # Benchmark results - FIX: benchmarks is a LIST, not a dict
                    benchmarks = target.get('benchmarks', [])
                    if benchmarks and isinstance(benchmarks, list):  # FIXED: Check if it's a list
                        f.write(f"\n### Benchmarks ({len(benchmarks)})\n\n")
                        f.write(f"| Benchmark | Overall Score | Backend | Status |\n")
                        f.write(f"|-----------|---------------|---------|--------|\n")

                        for bench_result in benchmarks:
                            bench_name = bench_result.get('benchmark_name',
                                                          bench_result.get('benchmark', 'Unknown'))
                            overall_score = bench_result.get('overall_score', 0.0)
                            backend = bench_result.get('backend', 'unknown')
                            status = bench_result.get('status', 'unknown')

                            status_emoji = "✅" if status == "completed" else "❌"
                            f.write(f"| {bench_name} | {overall_score:.4f} | {backend} | {status_emoji} |\n")

                        f.write(f"\n")

                    # Stress testing results
                    if 'stress_testing' in target:
                        f.write(f"\n### Stress Testing\n\n")
                        stress = target['stress_testing']
                        f.write(f"Status: {stress.get('status', 'N/A')}\n")
                        if 'metrics' in stress:
                            metrics = stress['metrics']
                            f.write(f"- **Avg Latency:** {metrics.get('avg_latency_ms', 0):.2f} ms\n")
                            f.write(f"- **Throughput:** {metrics.get('throughput_rps', 0):.2f} RPS\n")
                            f.write(f"- **Error Rate:** {metrics.get('error_rate', 0):.2%}\n")
                        f.write(f"\n")

                    # Red teaming results
                    if 'red_teaming' in target:
                        f.write(f"\n### Red Teaming\n\n")
                        f.write(f"Status: {target['red_teaming'].get('status', 'N/A')}\n\n")

                    # Guardrails results
                    if 'guardrails' in target:
                        f.write(f"\n### Guardrails\n\n")
                        f.write(f"Status: {target['guardrails'].get('status', 'N/A')}\n\n")

                f.write(f"\n---\n\n")
                f.write(f"**Full Results:** `eval_results/eval_{timestamp}.json`\n")

            logger.success(f"Summary report saved to: {report_file}")

        except Exception as e:
            logger.error(f"Failed to create summary report: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _cleanup(self):
        """Cleanup all resources."""
        logger.info("Cleaning up resources")

        for target in self.targets:
            try:
                target.cleanup()
                logger.debug(f"Cleaned up target: {target.name}")
            except Exception as e:
                logger.error(f"Error cleaning up target {target.name}: {e}")

        logger.success("Cleanup complete")

    def get_results(self) -> Dict[str, Any]:
        """Get consolidated evaluation results."""
        return self.consolidated_results

    def get_targets(self) -> List[BaseTarget]:
        """Get configured targets."""
        return self.targets