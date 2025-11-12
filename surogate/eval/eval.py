# surogate/eval/eval.py
from typing import Dict, Any, List

from surogate.eval.backend import LocalBackend
from surogate.eval.backend.aggregator import ResultAggregator
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
        self.backend = None
        self.targets: List[BaseTarget] = []
        self.results = []

    def run(self):
        """Run the evaluation pipeline."""
        logger.info("Starting Surogate Eval")

        # 1. Parse configuration
        parser = ConfigParser(self.config_path)
        self.config = parser.parse()

        # 2. Validate configuration
        validator = ConfigValidator(self.config)
        is_valid, errors = validator.validate()

        if not is_valid:
            logger.error("Configuration validation failed")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError("Invalid configuration")

        # 3. Initialize backend
        infra_config = self.config.get('infrastructure', {})
        backend_type = infra_config.get('backend', 'local')

        if backend_type == 'local':
            self.backend = LocalBackend(infra_config)
        else:
            raise NotImplementedError(f"Backend '{backend_type}' not implemented yet")

        # 4. Setup targets
        self._setup_targets()

        # 5. Run evaluation based on config
        try:
            self._run_evaluation()
        finally:
            # 6. Cleanup
            self._cleanup()

        logger.info("Surogate Eval completed")

    def _setup_targets(self):
        """Setup evaluation targets from config."""
        target_configs = self.config.get('targets', [])

        if not target_configs:
            logger.warning("No targets specified in configuration")
            return

        logger.info(f"Setting up {len(target_configs)} target(s)")

        # Create targets using factory
        self.targets = TargetFactory.create_multiple_targets(target_configs)

        if not self.targets:
            raise ValueError("Failed to create any targets from configuration")

        logger.info(f"Successfully initialized {len(self.targets)} target(s)")

        # Health check all targets
        self._health_check_targets()

    def _health_check_targets(self):
        """Perform health checks on all targets."""
        logger.info("Performing health checks on targets...")

        healthy_count = 0
        for target in self.targets:
            try:
                if target.health_check():
                    logger.info(f"✓ Target '{target.name}' ({target.target_type.value}) is healthy")
                    healthy_count += 1
                else:
                    logger.warning(f"✗ Target '{target.name}' ({target.target_type.value}) health check failed")
            except Exception as e:
                logger.error(f"✗ Target '{target.name}' health check error: {e}")

        if healthy_count == 0:
            raise RuntimeError("No healthy targets available")

        logger.info(f"Health check complete: {healthy_count}/{len(self.targets)} targets healthy")

    def _run_evaluation(self):
        """Execute the evaluation based on configuration."""
        if not self.targets:
            logger.error("No targets available for evaluation")
            return

        # Check what's enabled
        eval_config = self.config.get('evaluation', {})
        red_team_config = self.config.get('red_teaming', {})
        guardrails_config = self.config.get('guardrails', {})

        # Track what was run
        ran_something = False

        if eval_config.get('enabled'):
            logger.info("Running functional evaluation")
            self._run_functional_evaluation(eval_config)
            ran_something = True

        if red_team_config.get('enabled'):
            logger.info("Running red teaming")
            self._run_red_teaming(red_team_config)
            ran_something = True

        if guardrails_config.get('enabled'):
            logger.info("Testing guardrails")
            self._run_guardrails_testing(guardrails_config)
            ran_something = True

        if not ran_something:
            logger.warning("No evaluation tasks were enabled in configuration")
            return

        # Aggregate results if multiple runs
        if len(self.results) > 1:
            logger.info("Aggregating results from multiple runs")
            aggregated = ResultAggregator.aggregate(self.results)
            logger.info(f"Aggregated results: {aggregated.get('summary', {})}")
            self.results.append(aggregated)

    def _run_functional_evaluation(self, eval_config: Dict[str, Any]):
        """
        Run functional evaluation on targets.

        Args:
            eval_config: Evaluation configuration
        """
        dataset_path = eval_config.get('dataset')

        if not dataset_path:
            logger.warning("No dataset specified for evaluation")
            return

        # Load dataset
        try:
            loader = DatasetLoader()
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
                return

            # TODO: Run metrics on test cases
            logger.info("Running evaluation metrics (not yet implemented)")

            result = {
                "type": "functional_evaluation",
                "status": "dataset_loaded",
                "num_test_cases": len(test_cases),
                "targets": [t.name for t in self.targets]
            }
            self.results.append(result)

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            result = {
                "type": "functional_evaluation",
                "status": "error",
                "error": str(e)
            }
            self.results.append(result)

    def _run_red_teaming(self, red_team_config: Dict[str, Any]):
        """
        Run red teaming tests on targets.

        Args:
            red_team_config: Red teaming configuration
        """
        # TODO: Section 8 - Implement red teaming logic
        # This will include:
        # - Adversarial attacks
        # - Vulnerability scanning
        # - Risk assessment

        logger.info("Red teaming not yet implemented")
        logger.debug(f"Red teaming config: {red_team_config}")

        # Placeholder for now
        result = {
            "type": "red_teaming",
            "status": "not_implemented",
            "targets": [t.name for t in self.targets]
        }
        self.results.append(result)

    def _run_guardrails_testing(self, guardrails_config: Dict[str, Any]):
        """
        Test guardrails on targets.

        Args:
            guardrails_config: Guardrails configuration
        """
        # TODO: Section 9 - Implement guardrails testing
        # This will include:
        # - Input/output validation
        # - Content filtering tests
        # - Runtime guardrails

        logger.info("Guardrails testing not yet implemented")
        logger.debug(f"Guardrails config: {guardrails_config}")

        # Placeholder for now
        result = {
            "type": "guardrails_testing",
            "status": "not_implemented",
            "targets": [t.name for t in self.targets]
        }
        self.results.append(result)

    def _cleanup(self):
        """Cleanup all resources."""
        logger.info("Cleaning up resources")

        # Cleanup targets
        for target in self.targets:
            try:
                target.cleanup()
                logger.debug(f"Cleaned up target: {target.name}")
            except Exception as e:
                logger.error(f"Error cleaning up target {target.name}: {e}")

        # Cleanup backend
        if self.backend:
            try:
                self.backend.shutdown()
                logger.debug("Backend shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down backend: {e}")

        logger.info("Cleanup complete")

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get evaluation results.

        Returns:
            List of result dictionaries
        """
        return self.results

    def get_targets(self) -> List[BaseTarget]:
        """
        Get configured targets.

        Returns:
            List of target instances
        """
        return self.targets