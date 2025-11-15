# surogate/eval/metrics/adapters/deepeval_adapter.py
from typing import Dict, Any, Optional, Union, List

try:
    from deepeval.metrics import (
        GEval,
        ConversationalGEval,
        MultimodalGEval,
    )
    from deepeval.test_case import (
        LLMTestCase,
        ConversationalTestCase,
        Turn as DeepEvalTurn,
        MLLMTestCase,
        MLLMImage,
    )
    # DAG components
    from deepeval.metrics.dag import (
        DeepAcyclicGraph,
        TaskNode,
        BinaryJudgementNode,
        NonBinaryJudgementNode,
        VerdictNode,
    )

    DEEPEVAL_AVAILABLE = True
except ImportError as e:
    DEEPEVAL_AVAILABLE = False
    DEEPEVAL_IMPORT_ERROR = str(e)

from surogate.utils.logger import get_logger
from ..base import BaseMetric, MetricResult, MetricType, LLMJudgeMetric
from surogate.eval.datasets.test_case import TestCase, MultiTurnTestCase
from surogate.eval.targets.base import TargetResponse

logger = get_logger()


class DeepEvalAdapter(LLMJudgeMetric):
    """Adapter for DeepEval metrics."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DeepEval adapter.

        Args:
            config: Metric configuration
        """
        if not DEEPEVAL_AVAILABLE:
            raise ImportError(
                f"DeepEval is not installed. Install it with: pip install deepeval\n"
                f"Import error: {DEEPEVAL_IMPORT_ERROR if 'DEEPEVAL_IMPORT_ERROR' in globals() else 'Unknown'}"
            )

        # IMPORTANT: Set flags BEFORE calling super().__init__()
        deepeval_type = config.get('deepeval_metric_type')
        self.is_conversational = (deepeval_type in ['conversational_g_eval', 'conversational_dag'])
        self.is_multimodal = (deepeval_type == 'multimodal_g_eval')

        super().__init__(config)
        self.deepeval_metric = None
        self._initialize_deepeval_metric()

    def _validate_config(self):
        """Validate configuration."""
        required = ['type', 'deepeval_metric_type']
        for field in required:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

    def _initialize_deepeval_metric(self):
        """Initialize the underlying DeepEval metric."""
        deepeval_type = self.config.get('deepeval_metric_type')
        parameters = self.config.get('parameters', {})

        # Map our metric types to DeepEval metrics
        if deepeval_type == 'g_eval':
            self.deepeval_metric = GEval(**parameters)
        elif deepeval_type == 'conversational_g_eval':
            self.deepeval_metric = ConversationalGEval(**parameters)
        elif deepeval_type == 'multimodal_g_eval':
            self.deepeval_metric = MultimodalGEval(**parameters)
        elif deepeval_type == 'dag':
            dag = parameters.get('dag')
            if not dag:
                raise ValueError("DAG metric requires 'dag' parameter")

            from deepeval.metrics import DAGMetric as DeepEvalDAG
            self.deepeval_metric = DeepEvalDAG(
                name=parameters.get('name', 'DAG Metric'),
                dag=dag,
                threshold=parameters.get('threshold', 0.5),
                model=parameters.get('model'),
                strict_mode=parameters.get('strict_mode', False),
                async_mode=parameters.get('async_mode', True),
                verbose_mode=parameters.get('verbose_mode', False),
            )
        elif deepeval_type == 'conversational_dag':
            dag = parameters.get('dag')
            if not dag:
                raise ValueError("Conversational DAG metric requires 'dag' parameter")

            from deepeval.metrics import ConversationalDAGMetric as DeepEvalConvDAG
            self.deepeval_metric = DeepEvalConvDAG(
                name=parameters.get('name', 'Conversational DAG Metric'),
                dag=dag,
                threshold=parameters.get('threshold', 0.5),
                model=parameters.get('model'),
                strict_mode=parameters.get('strict_mode', False),
                async_mode=parameters.get('async_mode', True),
                verbose_mode=parameters.get('verbose_mode', False),
            )
        else:
            raise ValueError(f"Unknown DeepEval metric type: {deepeval_type}")

        logger.debug(f"Initialized DeepEval metric: {deepeval_type}")

    def evaluate(
            self,
            test_case: Union[TestCase, MultiTurnTestCase],
            actual_output: str,
            target_response: Optional[TargetResponse] = None
    ) -> MetricResult:
        """
        Evaluate using DeepEval metric.

        Args:
            test_case: Test case to evaluate
            actual_output: Actual output from target
            target_response: Full response from target (optional)

        Returns:
            Metric result
        """
        try:
            # Check if we have actual output
            if not actual_output:
                return MetricResult(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    score=0.0,
                    success=False,
                    reason="No actual output to evaluate"
                )

            # Check for metric-dataset mismatch
            if isinstance(test_case, TestCase) and self.is_conversational:
                return MetricResult(
                    metric_name=self.name,
                    metric_type=self.metric_type,
                    score=0.0,
                    success=False,
                    reason=f"Conversational metric '{self.name}' requires multi-turn test cases"
                )

            if isinstance(test_case, MultiTurnTestCase) and not self.is_conversational:
                logger.info(f"Converting multi-turn test case to single-turn for metric '{self.name}'")

            # Convert our test case to DeepEval format
            if isinstance(test_case, TestCase):
                # Check if this is a multimodal metric
                if self.is_multimodal:
                    # Convert to MLLMTestCase for multimodal evaluation
                    images = test_case.metadata.get('images', [])
                    if isinstance(images, str):
                        images = [images]

                    # Convert image URLs to MLLMImage objects
                    mllm_images = []
                    for img in images:
                        if isinstance(img, str):
                            # MLLMImage will auto-detect if it's local or remote
                            mllm_images.append(MLLMImage(url=img))
                        elif isinstance(img, MLLMImage):
                            mllm_images.append(img)

                    # Build input list: text + images
                    input_list = []
                    if test_case.input:
                        input_list.append(test_case.input)
                    input_list.extend(mllm_images)

                    # Build actual_output list
                    actual_output_list = []
                    if actual_output:
                        actual_output_list.append(actual_output)

                    # Build expected_output list if provided
                    expected_output_list = None
                    if test_case.expected_output:
                        expected_output_list = [test_case.expected_output]

                    # Build context list if provided
                    context_list = None
                    context_data = test_case.metadata.get('context')
                    if context_data:
                        if isinstance(context_data, list):
                            context_list = context_data
                        else:
                            context_list = [context_data]

                    deepeval_test_case = MLLMTestCase(
                        input=input_list,
                        actual_output=actual_output_list,
                        expected_output=expected_output_list,
                        context=context_list,
                    )
                else:
                    # Standard single-turn test case
                    deepeval_test_case = LLMTestCase(
                        input=test_case.input,
                        actual_output=actual_output,
                        expected_output=test_case.expected_output,
                        context=test_case.metadata.get('context'),
                    )

            elif isinstance(test_case, MultiTurnTestCase):
                # Multi-turn test case
                if self.is_conversational:
                    # Use conversational format
                    deepeval_turns = []
                    for turn in test_case.turns:
                        deepeval_turn = DeepEvalTurn(
                            role=turn.role,
                            content=turn.content
                        )
                        deepeval_turns.append(deepeval_turn)

                    # Add the actual output as the final assistant turn
                    if actual_output:
                        final_turn = DeepEvalTurn(
                            role="assistant",
                            content=actual_output
                        )
                        deepeval_turns.append(final_turn)

                    deepeval_test_case = ConversationalTestCase(
                        turns=deepeval_turns,
                        expected_outcome=test_case.expected_final_output
                    )
                else:
                    # Convert multi-turn to single-turn
                    conversation_context = []
                    for turn in test_case.turns:
                        conversation_context.append(f"{turn.role}: {turn.content}")

                    last_user_turn = None
                    for turn in reversed(test_case.turns):
                        if turn.role == "user":
                            last_user_turn = turn.content
                            break

                    input_text = last_user_turn or "\n".join(conversation_context)

                    deepeval_test_case = LLMTestCase(
                        input=input_text,
                        actual_output=actual_output,
                        expected_output=test_case.expected_final_output,
                        context="\n".join(conversation_context),
                    )

                    logger.debug(f"Converted multi-turn to single-turn for non-conversational metric")
            else:
                raise ValueError(f"Unknown test case type: {type(test_case)}")

            logger.debug(f"Evaluating with DeepEval: {self.config.get('deepeval_metric_type')}")

            # Measure the metric
            self.deepeval_metric.measure(deepeval_test_case, _show_indicator=False)

            # Extract results
            score = self.deepeval_metric.score
            success = self.deepeval_metric.is_successful()
            reason = self.deepeval_metric.reason if hasattr(self.deepeval_metric, 'reason') else None

            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=score,
                success=success,
                reason=reason,
                metadata={
                    'deepeval_type': self.config.get('deepeval_metric_type'),
                    'evaluation_model': self.deepeval_metric.evaluation_model if hasattr(self.deepeval_metric,
                                                                                         'evaluation_model') else None,
                    'is_conversational': self.is_conversational,
                    'is_multimodal': self.is_multimodal,
                    'test_case_type': type(test_case).__name__,
                }
            )

        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return MetricResult(
                metric_name=self.name,
                metric_type=self.metric_type,
                score=0.0,
                success=False,
                reason=f"Evaluation error: {str(e)}"
            )