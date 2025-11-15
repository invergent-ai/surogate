# surogate/eval/benchmarks/base.py
"""Base classes for benchmark evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from surogate.eval.targets.base import BaseTarget
from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""
    name: str

    # Dataset configuration
    path: Optional[str] = None  # NEW: Custom dataset path
    num_fewshot: Optional[int] = None
    limit: Optional[Union[int, float]] = None

    # Task-specific configuration
    tasks: Optional[List[str]] = None
    subset: Optional[Union[str, List[str]]] = None  # NEW: Subsets

    # Additional parameters
    backend_params: Dict[str, Any] = field(default_factory=dict)

    # Cache configuration
    use_cache: bool = True
    cache_dir: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Results from benchmark evaluation."""
    # Required fields first
    benchmark_name: str
    overall_score: float
    num_samples: int

    # Fields with defaults last
    backend: str = "evalscope"
    task_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'benchmark_name': self.benchmark_name,
            'backend': self.backend,
            'overall_score': self.overall_score,
            'num_samples': self.num_samples,
            'task_results': self.task_results,
            'metadata': self.metadata,
        }

    def get_summary(self) -> str:
        """Get a summary string of results."""
        lines = [
            f"Benchmark: {self.benchmark_name}",
            f"Overall Score: {self.overall_score:.4f}",
            f"Samples: {self.num_samples}",
        ]

        if self.task_results:
            lines.append("\nTask Results:")
            for task_name, task_data in self.task_results.items():
                score = task_data.get('score', task_data.get('accuracy', 'N/A'))
                lines.append(f"  {task_name}: {score}")

        return "\n".join(lines)



class BaseBenchmark(ABC):
    """Base class for all benchmarks."""

    # Add class-level attribute to specify required target types
    REQUIRED_TARGET_TYPES: List[str] = []  # Empty = any target type

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark."""
        self.config = config
        self.name = config.name

        from .backends.evalscope_backend import EvalScopeBackend
        self.backend = EvalScopeBackend()

        logger.info(f"Initialized benchmark: {self.name} (EvalScope)")

    def validate_target(self, target: BaseTarget) -> bool:
        """
        Validate that target is compatible with this benchmark.

        Args:
            target: Target to validate

        Returns:
            True if compatible
        """
        # If no specific requirements, accept all targets
        if not self.REQUIRED_TARGET_TYPES:
            return True

        # Check if target type matches requirements
        from surogate.eval.targets.base import TargetType
        target_type_str = target.target_type.value if hasattr(target.target_type, 'value') else str(target.target_type)

        return target_type_str in self.REQUIRED_TARGET_TYPES

    @abstractmethod
    def evaluate(self, target: BaseTarget) -> BenchmarkResult:
        """Evaluate target on this benchmark."""
        pass

    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the benchmark dataset."""
        pass