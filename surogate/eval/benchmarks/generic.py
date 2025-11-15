# surogate/eval/benchmarks/generic.py

from typing import Dict, Any, List
from .base import BaseBenchmark, BenchmarkResult, BenchmarkConfig
from .registry import BenchmarkRegistry
from surogate.eval.targets.base import BaseTarget
from surogate.utils.logger import get_logger

logger = get_logger()


class GenericBenchmark(BaseBenchmark):
    """Generic benchmark that delegates to EvalScope."""

    # Define which benchmarks need specific target types
    VISION_BENCHMARKS = {
        'mmmu', 'mathvista', 'chartqa', 'docvqa', 'infovqa',
        'ocrvqa', 'ai2d', 'scienceqa', 'seedbench'
    }

    EMBEDDING_BENCHMARKS = {
        'mteb', 'cmteb', 'beir'
    }

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark and set requirements."""
        super().__init__(config)

        # Set required target types based on benchmark name
        if self.name.lower() in self.VISION_BENCHMARKS:
            self.REQUIRED_TARGET_TYPES = ['multimodal']
            logger.debug(f"Benchmark '{self.name}' requires multimodal target")
        elif self.name.lower() in self.EMBEDDING_BENCHMARKS:
            self.REQUIRED_TARGET_TYPES = ['embedding']
            logger.debug(f"Benchmark '{self.name}' requires embedding target")
        else:
            self.REQUIRED_TARGET_TYPES = []  # Accept any

    def evaluate(self, target: BaseTarget) -> BenchmarkResult:
        """Evaluate target on this benchmark."""
        logger.info(f"Evaluating {target.name} on {self.name}")

        # Prepare config for EvalScope
        backend_config = {
            'tasks': self.config.tasks or [self.name],
            'num_fewshot': self.config.num_fewshot,
            'limit': self.config.limit,
            'subset': self.config.subset,
            'path': self.config.path,  # Custom dataset path
            'batch_size': self.config.backend_params.get('batch_size', 1),
            'backend_params': self.config.backend_params,
        }

        # Run evaluation via EvalScope
        backend_results = self.backend.evaluate(
            target=target,
            benchmark_name=self.name,
            config=backend_config
        )

        # Parse results
        result = self._parse_results(backend_results)

        logger.success(f"{self.name} completed. Score: {result.overall_score:.4f}")
        return result

    def _parse_results(self, backend_results: Dict[str, Any]) -> BenchmarkResult:
        """Parse EvalScope results."""
        overall_score = backend_results.get('overall_score', 0.0)
        task_results = backend_results.get('task_results', {})

        # Count samples
        num_samples = sum(
            task.get('n_samples', 0)
            for task in task_results.values()
            if isinstance(task, dict)
        )

        return BenchmarkResult(
            benchmark_name=self.name,
            overall_score=overall_score,
            num_samples=num_samples,
            backend="evalscope",
            task_results=task_results,
            metadata=backend_results.get('metadata', {})
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            'name': self.name,
            'description': f'Benchmark: {self.name}',
            'backend': 'evalscope',
            'required_target_types': self.REQUIRED_TARGET_TYPES,
        }