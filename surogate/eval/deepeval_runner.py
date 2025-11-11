import json
import os
from typing import Optional

from deepeval import evaluate, TestCase, Metric
from deepeval.metrics import FaithfulnessMetric, RelevanceMetric, CorrectnessMetric
from deepeval.models import OpenAIModel

from swift.utils import get_logger

logger = get_logger()


class SurogateDeepEval:
    """
    Surogate wrapper for DeepEval – allows running DeepEval test suites
    via CLI (`python -m surogate deepeval ...`).

    Example usage:
    python -m surogate deepeval --config configs/mistral.yaml \
                                --metric factuality \
                                --dataset evals/qa.json
    """

    def __init__(self,
                 config: str,
                 metric: Optional[str] = None,
                 dataset: Optional[str] = None,
                 model: Optional[str] = None,
                 output: Optional[str] = "results/deepeval.json",
                 **kwargs):

        self.config = config
        self.metric_name = metric or "faithfulness"
        self.dataset_path = dataset
        self.model_name = model or "gpt-4o-mini"
        self.output_path = output

        self.metric = self._get_metric(self.metric_name)
        self.model = OpenAIModel(model=self.model_name)

    def _get_metric(self, name: str) -> Metric:
        """Map metric names to DeepEval metrics."""
        name = name.lower()
        if name in ["faithfulness", "factuality"]:
            return FaithfulnessMetric(model=self.model)
        elif name in ["relevance", "context_relevance"]:
            return RelevanceMetric(model=self.model)
        elif name in ["correctness", "accuracy"]:
            return CorrectnessMetric(model=self.model)
        else:
            raise ValueError(f"Unsupported metric: {name}")

    def _load_dataset(self):
        """Load test cases from JSON dataset file."""
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Expected dataset structure:
        # [
        #   {"input": "...", "expected_output": "..."},
        #   {"input": "...", "expected_output": "..."},
        #   ...
        # ]
        test_cases = [
            TestCase(
                input=item.get("input"),
                actual_output=None,
                expected_output=item.get("expected_output")
            )
            for item in data
        ]
        return test_cases

    def run(self):
        logger.info(f"Running DeepEval on {self.dataset_path} using {self.metric_name}")

        test_cases = self._load_dataset()
        if not test_cases:
            raise ValueError("No test cases found in dataset.")

        results = evaluate(
            test_cases=test_cases,
            metrics=[self.metric],
        )

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results.to_dict(), f, indent=2)

        logger.info(f"DeepEval complete. Results saved to {self.output_path}")
