# surogate/eval/config/validator.py
from typing import Dict, Any, List, Tuple
import jsonschema
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
            logger.info("Schema validation passed")
        except jsonschema.ValidationError as e:
            self.errors.append(f"Schema validation error: {e.message}")
            logger.error(f"Schema validation failed: {e.message}")
            return False, self.errors

        # Custom validation rules
        self._validate_targets()
        self._validate_evaluation()
        self._validate_infrastructure()
        self._validate_file_paths()

        # Log warnings
        for warning in self.warnings:
            logger.warning(warning)

        if self.errors:
            for error in self.errors:
                logger.error(error)
            return False, self.errors

        logger.info("Configuration validation passed")
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
            self.warnings.append(f"Duplicate target names found: {set(duplicates)}")

        # Validate each target
        for idx, target in enumerate(targets):
            target_type = target.get('type')
            provider = target.get('provider')

            # RAG/Agent/Chatbot/MCP need endpoints
            if target_type in ['rag', 'agent', 'chatbot', 'mcp']:
                if not target.get('endpoint'):
                    self.errors.append(
                        f"Target '{target.get('name', f'target-{idx}')}': "
                        f"endpoint required for type '{target_type}'"
                    )

            # Local models need model_path
            if provider == 'local' and target_type == 'llm':
                if not target.get('model_path'):
                    self.errors.append(
                        f"Target '{target.get('name', f'target-{idx}')}': "
                        f"model_path required for local LLM"
                    )

    def _validate_evaluation(self) -> None:
        """Validate evaluation configuration."""
        evaluation = self.config.get('evaluation', {})

        if not evaluation.get('enabled'):
            return

        # Check datasets
        datasets = evaluation.get('datasets', [])
        if not datasets:
            self.warnings.append("Evaluation enabled but no datasets specified")

        # Check metrics
        metrics = evaluation.get('metrics', {})
        if not metrics.get('functional') and not metrics.get('performance'):
            self.warnings.append("Evaluation enabled but no metrics specified")

        # Check benchmarks
        benchmarks = evaluation.get('benchmarks', {})
        if benchmarks.get('standard') or benchmarks.get('third_party'):
            if not datasets:
                self.errors.append("Benchmarks specified but no datasets provided")

    def _validate_infrastructure(self) -> None:
        """Validate infrastructure configuration."""
        infra = self.config.get('infrastructure', {})

        # Check workers
        workers = infra.get('workers', 1)
        if workers <= 0:
            self.errors.append("infrastructure.workers must be positive")
        elif workers > 32:
            self.warnings.append(
                f"infrastructure.workers is very high ({workers}). "
                "This may cause resource issues."
            )

        # Check parallel execution
        parallel = infra.get('parallel_execution', {})
        if parallel.get('enabled'):
            max_workers = parallel.get('max_workers', 0)
            if max_workers <= 0:
                self.errors.append("max_workers must be positive when parallel_execution is enabled")

        # Check sandbox
        sandbox = infra.get('sandbox', {})
        if sandbox.get('enabled'):
            timeout = sandbox.get('timeout_seconds', 0)
            if timeout <= 0:
                self.warnings.append("sandbox timeout_seconds should be positive")

    def _validate_file_paths(self) -> None:
        """Validate that specified file paths are accessible."""
        from pathlib import Path

        # Check dataset paths
        datasets = self.config.get('evaluation', {}).get('datasets', [])
        for dataset in datasets:
            path = dataset.get('path')
            if path and not Path(path).exists():
                self.warnings.append(f"Dataset path does not exist: {path}")

        # Check custom guard paths
        guards = self.config.get('guardrails', {}).get('custom_guards', [])
        for guard in guards:
            path = guard.get('path')
            if path and not Path(path).exists():
                self.warnings.append(f"Custom guard path does not exist: {path}")