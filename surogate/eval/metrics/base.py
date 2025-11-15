# surogate/eval/metrics/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from surogate.eval.datasets.test_case import TestCase, MultiTurnTestCase
from surogate.eval.targets.base import TargetResponse


class MetricType(Enum):
    """Types of metrics."""
    G_EVAL = "g_eval"
    CONVERSATIONAL_G_EVAL = "conversational_g_eval"
    MULTIMODAL_G_EVAL = "multimodal_g_eval"
    ARENA_G_EVAL = "arena_g_eval"
    DAG = "dag"
    CONVERSATIONAL_DAG = "conversational_dag"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    TOKEN_GENERATION_SPEED = "token_generation_speed"

    # Multi-turn
    CONVERSATION_COHERENCE = "conversation_coherence"
    CONTEXT_RETENTION = "context_retention"
    TURN_ANALYSIS = "turn_analysis"

    # Safety
    TOXICITY = "toxicity"
    BIAS = "bias"
    HARM = "harm"

    # Non-LLM
    EMBEDDING_SIMILARITY = "embedding_similarity"
    CLASSIFICATION = "classification"


@dataclass
class MetricResult:
    """Result from a metric evaluation."""

    metric_name: str
    metric_type: MetricType
    score: float  # 0.0 to 1.0
    success: bool
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric_name': self.metric_name,
            'metric_type': self.metric_type.value,
            'score': self.score,
            'success': self.success,
            'reason': self.reason,
            'metadata': self.metadata
        }


@dataclass
class BatchMetricResult:
    """Results from batch evaluation."""

    metric_name: str
    metric_type: MetricType
    results: List[MetricResult]

    @property
    def avg_score(self) -> float:
        """Calculate average score."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric_name': self.metric_name,
            'metric_type': self.metric_type.value,
            'num_evaluations': len(self.results),
            'avg_score': self.avg_score,
            'success_rate': self.success_rate,
            'results': [r.to_dict() for r in self.results]
        }


class BaseMetric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metric.

        Args:
            config: Metric configuration
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.metric_type = MetricType(config.get('type'))
        self._validate_config()

    @abstractmethod
    def _validate_config(self):
        """Validate metric-specific configuration."""
        pass

    @abstractmethod
    def evaluate(
            self,
            test_case: Union[TestCase, MultiTurnTestCase],
            actual_output: str,
            target_response: Optional[TargetResponse] = None
    ) -> MetricResult:
        """
        Evaluate a single test case.

        Args:
            test_case: Test case to evaluate
            actual_output: Actual output from target
            target_response: Full response from target (optional)

        Returns:
            Metric result
        """
        pass

    def evaluate_batch(
            self,
            test_cases: List[Union[TestCase, MultiTurnTestCase]],
            actual_outputs: List[str],
            target_responses: Optional[List[TargetResponse]] = None
    ) -> BatchMetricResult:
        """
        Evaluate multiple test cases.

        Args:
            test_cases: List of test cases
            actual_outputs: List of actual outputs
            target_responses: List of full responses (optional)

        Returns:
            Batch metric result
        """
        if len(test_cases) != len(actual_outputs):
            raise ValueError("Number of test cases must match number of outputs")

        if target_responses and len(target_responses) != len(test_cases):
            raise ValueError("Number of target responses must match test cases")

        results = []
        for i, (test_case, actual_output) in enumerate(zip(test_cases, actual_outputs)):
            target_response = target_responses[i] if target_responses else None
            result = self.evaluate(test_case, actual_output, target_response)
            results.append(result)

        return BatchMetricResult(
            metric_name=self.name,
            metric_type=self.metric_type,
            results=results
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, type={self.metric_type.value})"


class LLMJudgeMetric(BaseMetric):
    """Base class for metrics that use LLM as judge."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM judge metric.

        Args:
            config: Metric configuration including judge_model
        """
        super().__init__(config)
        self.judge_model_config = config.get('judge_model', {})
        self.judge_target = None  # Will be set by runner

    def set_judge_target(self, judge_target):
        """Set the judge target for evaluation."""
        self.judge_target = judge_target