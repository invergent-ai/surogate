# surogate/eval/metrics/embeddings.py
from typing import Dict, Any, Optional, Union, List
import numpy as np

from surogate.utils.logger import get_logger
from .base import MetricResult, MetricType, BaseMetric
from .registry import register_metric
from surogate.eval.datasets.test_case import TestCase, MultiTurnTestCase
from surogate.eval.targets.base import TargetResponse

logger = get_logger()


@register_metric(MetricType.EMBEDDING_SIMILARITY)
class EmbeddingSimilarityMetric(BaseMetric):
    """Measure similarity between embeddings."""

    def _validate_config(self):
        """Validate configuration."""
        if 'similarity_function' not in self.config:
            self.config['similarity_function'] = 'cosine'  # or 'euclidean', 'dot_product'

        if 'threshold' not in self.config:
            self.config['threshold'] = 0.8

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product > 0 else 0.0

    def _euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate euclidean distance (inverted to similarity)."""
        distance = np.linalg.norm(vec1 - vec2)
        # Convert to similarity score (0-1 range)
        return 1.0 / (1.0 + distance)

    def evaluate(
            self,
            test_case: Union[TestCase, MultiTurnTestCase],
            actual_output: str,
            target_response: Optional[TargetResponse] = None
    ) -> MetricResult:
        """
        Evaluate embedding similarity.

        Args:
            test_case: Test case (should contain expected embedding in metadata)
            actual_output: Not used for embeddings
            target_response: Should contain embedding in metadata

        Returns:
            Metric result
        """
        try:
            similarity_func = self.config['similarity_function']
            threshold = self.config['threshold']

            # Extract embeddings
            if not target_response or 'embedding' not in target_response.metadata:
                return MetricResult(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    score=0.0,
                    success=False,
                    reason="No embedding found in target response"
                )

            actual_embedding = np.array(target_response.metadata['embedding'])

            # Get expected embedding from test case metadata
            if 'expected_embedding' not in test_case.metadata:
                return MetricResult(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    score=0.0,
                    success=False,
                    reason="No expected embedding in test case"
                )

            expected_embedding = np.array(test_case.metadata['expected_embedding'])

            # Calculate similarity
            if similarity_func == 'cosine':
                similarity = self._cosine_similarity(actual_embedding, expected_embedding)
            elif similarity_func == 'euclidean':
                similarity = self._euclidean_distance(actual_embedding, expected_embedding)
            elif similarity_func == 'dot_product':
                similarity = np.dot(actual_embedding, expected_embedding)
            else:
                raise ValueError(f"Unknown similarity function: {similarity_func}")

            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=similarity,
                success=similarity >= threshold,
                reason=f"Embedding similarity ({similarity_func}): {similarity:.4f}",
                metadata={
                    'similarity_function': similarity_func,
                    'threshold': threshold,
                    'embedding_dim': len(actual_embedding)
                }
            )

        except Exception as e:
            logger.error(f"Embedding similarity evaluation failed: {e}")
            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=0.0,
                success=False,
                reason=f"Evaluation error: {str(e)}"
            )


@register_metric(MetricType.CLASSIFICATION)
class ClassificationMetric(BaseMetric):
    """Classification metrics (accuracy, precision, recall, F1)."""

    def _validate_config(self):
        """Validate configuration."""
        if 'metric_type' not in self.config:
            self.config['metric_type'] = 'accuracy'  # or 'precision', 'recall', 'f1'

    def evaluate(
            self,
            test_case: Union[TestCase, MultiTurnTestCase],
            actual_output: str,
            target_response: Optional[TargetResponse] = None
    ) -> MetricResult:
        """
        Evaluate classification.

        Args:
            test_case: Should have expected_output as the correct class
            actual_output: Predicted class
            target_response: Full response

        Returns:
            Metric result
        """
        try:
            metric_type = self.config['metric_type']

            expected = test_case.expected_output if hasattr(test_case, 'expected_output') else None
            if not expected:
                return MetricResult(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    score=0.0,
                    success=False,
                    reason="No expected output for classification"
                )

            # Simple exact match for single example
            # For batch evaluation, this will accumulate true/false positives
            is_correct = actual_output.strip().lower() == expected.strip().lower()
            score = 1.0 if is_correct else 0.0

            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=score,
                success=is_correct,
                reason=f"Classification: {'correct' if is_correct else 'incorrect'}",
                metadata={
                    'metric_type': metric_type,
                    'expected': expected,
                    'actual': actual_output
                }
            )

        except Exception as e:
            logger.error(f"Classification evaluation failed: {e}")
            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=0.0,
                success=False,
                reason=f"Evaluation error: {str(e)}"
            )