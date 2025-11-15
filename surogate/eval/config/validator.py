# surogate/eval/config/validator.py
from typing import Dict, Any, List, Tuple, Set
import jsonschema
from pathlib import Path
from .schema import CONFIG_SCHEMA
from surogate.utils.logger import get_logger

logger = get_logger()


class ConfigValidator:
    """Validate configuration against schema."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator with configuration.

        Args:
            config: Configuration dictionary to validate
        """
        self.config = config
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration against schema.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        logger.info("Validating configuration")

        # Schema validation
        try:
            jsonschema.validate(instance=self.config, schema=CONFIG_SCHEMA)
            logger.info("✓ Schema validation passed")
        except jsonschema.ValidationError as e:
            self.errors.append(f"Schema validation error: {e.message}")
            logger.error(f"✗ Schema validation failed: {e.message}")
            return False, self.errors

        # Custom validation rules
        self._validate_targets()
        self._validate_target_references()
        self._validate_file_paths()

        # Log warnings
        for warning in self.warnings:
            logger.warning(warning)

        if self.errors:
            for error in self.errors:
                logger.error(error)
            return False, self.errors

        logger.info("✓ Configuration validation passed")
        return True, []

    def _validate_targets(self) -> None:
        """Validate targets configuration."""
        targets = self.config.get('targets', [])

        if not targets:
            self.errors.append("At least one target must be specified")
            return

        # Check for duplicate target names
        target_names = [t.get('name') for t in targets]
        duplicates = [name for name in target_names if target_names.count(name) > 1]
        if duplicates:
            self.errors.append(f"Duplicate target names found: {set(duplicates)}")

        # Validate each target
        for idx, target in enumerate(targets):
            target_name = target.get('name', f'target-{idx}')
            self._validate_single_target(target, target_name)

    def _validate_single_target(self, target: Dict[str, Any], target_name: str) -> None:
        """Validate a single target configuration."""
        target_type = target.get('type')
        provider = target.get('provider')

        # Validate model configuration based on provider
        if provider == 'local' and target_type in ['llm', 'multimodal']:
            if not target.get('model_path') and not target.get('model'):
                self.errors.append(
                    f"Target '{target_name}': model_path or model required for local LLM"
                )

        # Validate infrastructure
        infrastructure = target.get('infrastructure', {})
        if infrastructure:
            self._validate_infrastructure(infrastructure, target_name)

        # Validate evaluations
        evaluations = target.get('evaluations', [])
        if evaluations:
            self._validate_evaluations(evaluations, target_name)

        # Validate red teaming
        red_teaming = target.get('red_teaming', {})
        if red_teaming.get('enabled') and not red_teaming.get('attacks'):
            self.warnings.append(
                f"Target '{target_name}': red_teaming enabled but no attacks specified"
            )

        # Validate guardrails
        guardrails = target.get('guardrails', {})
        if guardrails.get('enabled'):
            if not guardrails.get('input_guards') and not guardrails.get('output_guards'):
                self.warnings.append(
                    f"Target '{target_name}': guardrails enabled but no guards specified"
                )

    def _validate_infrastructure(self, infra: Dict[str, Any], target_name: str) -> None:
        """Validate infrastructure configuration for a target."""
        workers = infra.get('workers', 1)
        if workers <= 0:
            self.errors.append(f"Target '{target_name}': workers must be positive")
        elif workers > 32:
            self.warnings.append(
                f"Target '{target_name}': workers is very high ({workers}), may cause resource issues"
            )

        # Check parallel execution
        parallel = infra.get('parallel_execution', {})
        if parallel.get('enabled'):
            max_workers = parallel.get('max_workers', 0)
            if max_workers <= 0:
                self.errors.append(
                    f"Target '{target_name}': max_workers must be positive when parallel_execution is enabled"
                )

    # surogate/eval/config/validator.py

    def _validate_evaluations(self, evaluations: List[Dict[str, Any]], target_name: str) -> None:
        """Validate evaluations for a target."""
        if not evaluations:
            self.warnings.append(f"Target '{target_name}': no evaluations specified")
            return

        # Check for duplicate evaluation names
        eval_names = [e.get('name', f'eval-{i}') for i, e in enumerate(evaluations)]
        duplicates = [name for name in eval_names if eval_names.count(name) > 1]
        if duplicates:
            self.warnings.append(
                f"Target '{target_name}': duplicate evaluation names found: {set(duplicates)}"
            )

        for idx, evaluation in enumerate(evaluations):
            eval_name = evaluation.get('name', f'evaluation-{idx}')

            # Check that evaluation has EITHER dataset OR benchmarks (or both)
            has_dataset = bool(evaluation.get('dataset'))
            has_benchmarks = bool(evaluation.get('benchmarks'))
            has_metrics = bool(evaluation.get('metrics'))

            if not has_dataset and not has_benchmarks:
                self.errors.append(
                    f"Target '{target_name}', Evaluation '{eval_name}': "
                    f"must have either 'dataset' (for custom metrics) or 'benchmarks' (for standard benchmarks)"
                )

            # If has dataset, must have metrics
            if has_dataset and not has_metrics:
                self.warnings.append(
                    f"Target '{target_name}', Evaluation '{eval_name}': "
                    f"dataset specified but no metrics defined"
                )

            # If has benchmarks, validate benchmark configs
            if has_benchmarks:
                benchmarks = evaluation.get('benchmarks', [])
                if not benchmarks:
                    self.warnings.append(
                        f"Target '{target_name}', Evaluation '{eval_name}': "
                        f"benchmarks key present but empty"
                    )
                else:
                    # Validate each benchmark has a name
                    for bench_idx, benchmark in enumerate(benchmarks):
                        if not benchmark.get('name'):
                            self.errors.append(
                                f"Target '{target_name}', Evaluation '{eval_name}': "
                                f"benchmark at index {bench_idx} missing 'name' field"
                            )

    def _validate_target_references(self) -> None:
        """Validate that judge model references point to existing targets."""
        # Get all target names
        target_names: Set[str] = {t.get('name') for t in self.config.get('targets', [])}

        # Check all judge model references
        for target in self.config.get('targets', []):
            target_name = target.get('name')
            evaluations = target.get('evaluations', [])

            for eval_config in evaluations:
                eval_name = eval_config.get('name', 'unnamed')
                metrics = eval_config.get('metrics', [])

                for metric in metrics:
                    judge_model = metric.get('judge_model', {})
                    judge_target = judge_model.get('target')

                    if judge_target and judge_target not in target_names:
                        self.errors.append(
                            f"Target '{target_name}', Evaluation '{eval_name}': "
                            f"judge target '{judge_target}' not found"
                        )

    def _validate_file_paths(self) -> None:
        """Validate that specified file paths exist."""
        project_root = Path.cwd()

        for target in self.config.get('targets', []):
            target_name = target.get('name')
            evaluations = target.get('evaluations', [])

            for evaluation in evaluations:
                eval_name = evaluation.get('name', 'unnamed')

                # Check dataset path
                dataset = evaluation.get('dataset')
                if dataset:
                    dataset_path = Path(dataset)
                    if not dataset_path.is_absolute():
                        dataset_path = project_root / dataset
                    if not dataset_path.exists():
                        self.warnings.append(
                            f"Target '{target_name}', Evaluation '{eval_name}': "
                            f"dataset path does not exist: {dataset}"
                        )

                # NEW: Check benchmark custom paths
                benchmarks = evaluation.get('benchmarks', [])
                for bench in benchmarks:
                    bench_path = bench.get('path')
                    if bench_path:
                        path = Path(bench_path)
                        if not path.is_absolute():
                            path = project_root / path
                        if not path.exists():
                            self.warnings.append(
                                f"Target '{target_name}', Benchmark '{bench.get('name')}': "
                                f"custom path does not exist: {bench_path}"
                            )