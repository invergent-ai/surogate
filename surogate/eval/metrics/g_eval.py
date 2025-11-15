# surogate/eval/metrics/g_eval.py
from typing import Dict, Any, Optional, Union

from surogate.utils.logger import get_logger
from .base import MetricResult, MetricType, LLMJudgeMetric
from .registry import register_metric
from .adapters.deepeval_adapter import DeepEvalAdapter
from surogate.eval.datasets.test_case import TestCase, MultiTurnTestCase
from surogate.eval.targets.base import TargetResponse

logger = get_logger()


@register_metric(MetricType.G_EVAL)
class GEvalMetric(DeepEvalAdapter):
    """G-Eval metric using LLM as judge."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize G-Eval metric."""
        config['deepeval_metric_type'] = 'g_eval'
        config['type'] = MetricType.G_EVAL.value

        if 'parameters' not in config:
            evaluation_params = config.get('evaluation_params', [])

            try:
                from deepeval.test_case import LLMTestCaseParams

                param_mapping = {
                    'input': LLMTestCaseParams.INPUT,
                    'actual_output': LLMTestCaseParams.ACTUAL_OUTPUT,
                    'expected_output': LLMTestCaseParams.EXPECTED_OUTPUT,
                    'context': LLMTestCaseParams.CONTEXT,
                    'retrieval_context': LLMTestCaseParams.RETRIEVAL_CONTEXT,
                }

                enum_params = []
                for param in evaluation_params:
                    if isinstance(param, str):
                        param_lower = param.lower().replace(' ', '_')
                        if param_lower in param_mapping:
                            enum_params.append(param_mapping[param_lower])
                        else:
                            logger.warning(f"Unknown evaluation param: {param}, skipping")
                    else:
                        enum_params.append(param)

                if not enum_params:
                    enum_params = [LLMTestCaseParams.ACTUAL_OUTPUT]

                config['parameters'] = {
                    'name': config.get('name', 'G-Eval'),
                    'criteria': config.get('criteria', 'Correctness'),
                    'evaluation_params': enum_params,
                }
            except ImportError:
                logger.warning("Could not import LLMTestCaseParams from deepeval")
                config['parameters'] = {
                    'name': config.get('name', 'G-Eval'),
                    'criteria': config.get('criteria', 'Correctness'),
                    'evaluation_params': evaluation_params,
                }

        super().__init__(config)


@register_metric(MetricType.CONVERSATIONAL_G_EVAL)
class ConversationalGEvalMetric(DeepEvalAdapter):
    """Conversational G-Eval for multi-turn conversations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Conversational G-Eval metric."""
        config['deepeval_metric_type'] = 'conversational_g_eval'
        config['type'] = MetricType.CONVERSATIONAL_G_EVAL.value

        if 'parameters' not in config:
            config['parameters'] = {
                'name': config.get('name', 'Conversational G-Eval'),
                'criteria': config.get('criteria', 'Coherence'),
            }

        super().__init__(config)


@register_metric(MetricType.MULTIMODAL_G_EVAL)
class MultimodalGEvalMetric(DeepEvalAdapter):
    """Multimodal G-Eval for vision+language models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Multimodal G-Eval metric."""
        config['deepeval_metric_type'] = 'multimodal_g_eval'
        config['type'] = MetricType.MULTIMODAL_G_EVAL.value

        if 'parameters' not in config:
            evaluation_params = config.get('evaluation_params', [])

            try:
                from deepeval.test_case import MLLMTestCaseParams

                param_mapping = {
                    'input': MLLMTestCaseParams.INPUT,
                    'actual_output': MLLMTestCaseParams.ACTUAL_OUTPUT,
                    'expected_output': MLLMTestCaseParams.EXPECTED_OUTPUT,
                    'context': MLLMTestCaseParams.CONTEXT,
                }

                enum_params = []
                for param in evaluation_params:
                    if isinstance(param, str):
                        param_lower = param.lower().replace(' ', '_')
                        if param_lower in param_mapping:
                            enum_params.append(param_mapping[param_lower])
                        else:
                            logger.warning(f"Unknown evaluation param: {param}, skipping")
                    else:
                        enum_params.append(param)

                if not enum_params:
                    enum_params = [MLLMTestCaseParams.ACTUAL_OUTPUT]

                config['parameters'] = {
                    'name': config.get('name', 'Multimodal G-Eval'),
                    'criteria': config.get('criteria', 'Image-Text Alignment'),
                    'evaluation_params': enum_params,
                }

                if 'evaluation_steps' in config:
                    config['parameters']['evaluation_steps'] = config['evaluation_steps']

            except ImportError:
                logger.warning("Could not import MLLMTestCaseParams from deepeval")
                config['parameters'] = {
                    'name': config.get('name', 'Multimodal G-Eval'),
                    'criteria': config.get('criteria', 'Image-Text Alignment'),
                    'evaluation_params': evaluation_params,
                }

        if 'rubric' in config:
            rubric = config['rubric']

            if 'parameters' not in config:
                config['parameters'] = {}

            try:
                from deepeval.metrics.g_eval import Rubric

                if isinstance(rubric, dict):
                    logger.warning("Converting rubric from dict to Rubric objects")
                    rubric_list = []
                    for key, value in rubric.items():
                        import re
                        match = re.search(r'\((\d+)-(\d+)\)', key)
                        if match:
                            min_score = int(match.group(1))
                            max_score = int(match.group(2))
                            rubric_list.append(
                                Rubric(
                                    score_range=(min_score, max_score),  # tuple, not list
                                    expected_outcome=value
                                )
                            )
                        else:
                            logger.error(f"Could not parse score range from rubric key: {key}")
                    config['parameters']['rubric'] = rubric_list

                elif isinstance(rubric, list):
                    # Convert list of dicts to Rubric objects
                    rubric_objects = []
                    for item in rubric:
                        if isinstance(item, dict):
                            if 'score_range' not in item or 'expected_outcome' not in item:
                                logger.error(
                                    f"Invalid rubric item format: {item}. Must have 'score_range' and 'expected_outcome' keys"
                                )
                                continue

                            score_range = item['score_range']
                            # Convert list to tuple if needed
                            if isinstance(score_range, list):
                                score_range = tuple(score_range)

                            rubric_objects.append(
                                Rubric(
                                    score_range=score_range,
                                    expected_outcome=item['expected_outcome']
                                )
                            )
                        else:
                            logger.error(f"Invalid rubric item type: {type(item)}. Must be dict")
                    config['parameters']['rubric'] = rubric_objects
                else:
                    logger.error(f"Invalid rubric format: {type(rubric)}. Must be dict or list")

            except ImportError:
                logger.error("Could not import Rubric from deepeval.metrics.g_eval")
                # Fallback: try to pass as-is and let DeepEval handle it
                config['parameters']['rubric'] = rubric

        super().__init__(config)


@register_metric(MetricType.ARENA_G_EVAL)
class ArenaGEvalMetric(LLMJudgeMetric):
    """Arena G-Eval for comparing multiple model outputs."""

    def _validate_config(self):
        """Validate configuration."""
        if 'comparison_mode' not in self.config:
            self.config['comparison_mode'] = 'pairwise'

    def evaluate(
            self,
            test_case: Union[TestCase, MultiTurnTestCase],
            actual_output: str,
            target_response: Optional[TargetResponse] = None
    ) -> MetricResult:
        """Evaluate in arena mode."""
        logger.warning("Arena G-Eval not fully implemented yet")

        return MetricResult(
            metric_name=self.name,
            metric_type=self.metric_type,
            score=0.0,
            success=False,
            reason="Arena evaluation not implemented"
        )