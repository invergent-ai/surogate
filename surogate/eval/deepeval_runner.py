# surogate/eval/deepeval_runner.py
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import json

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    SummarizationMetric,
    BiasMetric,
    ToxicityMetric,
    GEval,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from surogate.utils.logger import get_logger

from surogate.utils.config import load_config

logger = get_logger()



class CustomLLM(DeepEvalBaseLLM):
    """Custom LLM wrapper for DeepEval to use vLLM or any OpenAI-compatible endpoint"""

    def __init__(self, model: str, api_url: str, api_key: str = "", **kwargs):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.generation_config = kwargs.get('generation_config', {})
        self.timeout = kwargs.get('timeout', 120.0)  # Default 2 minutes

        # Initialize OpenAI client with custom base URL and timeout
        from openai import OpenAI
        self.client = OpenAI(
            base_url=api_url,
            api_key=api_key or "dummy-key",
            timeout=self.timeout
        )

    def load_model(self):
        """No-op: model is already served"""
        return self.model

    def generate(self, prompt: str, schema: Any = None) -> Any:
        """Generate response using the custom API endpoint"""
        try:
            # If schema is provided but not supported, return raw text
            # DeepEval will fall back to JSON parsing
            if schema is not None:
                logger.warning(f"Schema-guided generation not supported for {self.model}, returning raw text")
                # Still generate, but DeepEval will handle parsing via trimAndLoadJson

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.generation_config.get('temperature', 0.7),
                max_tokens=self.generation_config.get('max_tokens', 500),
                top_p=self.generation_config.get('top_p', 0.9),
                stop=self.generation_config.get('stop'),
                timeout=self.timeout
            )

            content = response.choices[0].message.content

            # Return raw string - DeepEval will parse it via trimAndLoadJson
            return content

        except Exception as e:
            logger.error(f"Error generating response from {self.api_url}: {e}")
            raise TypeError(f"Generation failed: {e}")  # Raise TypeError for DeepEval's exception handling

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        """Async generate - falls back to sync"""
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        """Return model name"""
        return self.model


class SurogateDeepEval:
    """DeepEval integration for custom metrics evaluation"""

    METRIC_MAP = {
        'answer_relevancy': AnswerRelevancyMetric,
        'faithfulness': FaithfulnessMetric,
        'contextual_recall': ContextualRecallMetric,
        'contextual_precision': ContextualPrecisionMetric,
        'contextual_relevancy': ContextualRelevancyMetric,
        'hallucination': HallucinationMetric,
        'summarization': SummarizationMetric,
        'bias': BiasMetric,
        'toxicity': ToxicityMetric,
        'g_eval': GEval,
    }

    def __init__(self, **kwargs):
        self.args = kwargs
        self.config = load_config(self.args['config'])
        self._resolve_env_vars()
        self.judge_llm = None

    def _resolve_env_vars(self):
        """Resolve environment variables in config"""

        def resolve(obj):
            if isinstance(obj, str):
                import re
                def replace_env(match):
                    var_name = match.group(1) or match.group(2)
                    return os.getenv(var_name, match.group(0))

                return re.sub(r'\$\{([^}]+)\}|\$([A-Z_][A-Z0-9_]*)', replace_env, obj)
            elif isinstance(obj, dict):
                return {k: resolve(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve(item) for item in obj]
            return obj

        self.config = resolve(self.config)

    def _build_judge_model(self) -> Optional[Any]:
        """
        Build judge model - supports OpenAI, vLLM, or any OpenAI-compatible endpoint.
        """
        # Check for explicit judge model first
        judge_config = self.config.get('judge_model')

        # If no explicit judge, use target as judge
        if not judge_config:
            logger.info("No explicit judge_model found, using target model as judge")
            judge_config = self.config.get('target', {})
        else:
            logger.info("Using explicit judge_model configuration")

        if not judge_config:
            logger.warning("No judge model or target configuration found")
            return None

        model_name = judge_config.get('model')
        api_url = judge_config.get('api_url', '')
        api_key = judge_config.get('api_key', '')
        timeout = judge_config.get('timeout', 180.0)  # Increased default to 3 minutes
        max_retries = judge_config.get('max_retries', 3)

        if not model_name:
            logger.warning("Judge model requires 'model' field")
            return None

        # Detect if it's OpenAI or custom endpoint
        is_openai = (
                'openai.com' in api_url.lower() or
                model_name.startswith('gpt-') or
                model_name.startswith('o1-')
        )

        if is_openai:
            # Use OpenAI directly (DeepEval's default behavior)
            logger.info(f"Using OpenAI judge model: {model_name} (timeout: {timeout}s, max_retries: {max_retries})")

            # Verify API key is set
            if not api_key and not os.getenv('OPENAI_API_KEY'):
                raise ValueError(
                    "OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide api_key in config.")

            # Set API key in environment if provided
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key

            # Set timeout and retry configuration
            os.environ['OPENAI_TIMEOUT'] = str(timeout)
            os.environ['OPENAI_MAX_RETRIES'] = str(max_retries)

            # Return model name string - DeepEval will use its default OpenAI handler
            return model_name
        else:
            # Use CustomLLM wrapper for vLLM or other endpoints
            if not api_url:
                logger.warning("Custom judge model requires 'api_url' field")
                return None

            logger.info(f"Using custom judge model: {model_name} at {api_url} (timeout: {timeout}s)")

            return CustomLLM(
                model=model_name,
                api_url=api_url,
                api_key=api_key,
                generation_config=judge_config.get('generation_config', {}),
                timeout=timeout
            )

    def _build_metric(self, metric_config: Dict[str, Any]) -> Any:
        """Build a DeepEval metric from config"""
        metric_type = metric_config.get('type')

        if metric_type not in self.METRIC_MAP:
            logger.warning(f"Unknown metric type: {metric_type}")
            return None

        MetricClass = self.METRIC_MAP[metric_type]

        # Build metric kwargs
        kwargs = {}

        # Add threshold if specified
        if 'threshold' in metric_config:
            kwargs['threshold'] = metric_config['threshold']

        # Add judge model for LLM-based metrics
        if self.judge_llm and metric_type in [
            'answer_relevancy', 'faithfulness', 'contextual_recall',
            'contextual_precision', 'contextual_relevancy', 'hallucination',
            'summarization', 'bias', 'toxicity', 'g_eval'
        ]:
            kwargs['model'] = self.judge_llm

        # Special handling for G-Eval
        if metric_type == 'g_eval':
            kwargs['name'] = metric_config.get('name', 'G-Eval')
            kwargs['criteria'] = metric_config.get('criteria', 'Determine if output is good')
            kwargs['evaluation_params'] = metric_config.get('evaluation_params', [
                'Fluency', 'Coherence', 'Relevance'
            ])

        # Pass through any other custom parameters
        for key, value in metric_config.items():
            if key not in ['type', 'threshold'] and key not in kwargs:
                kwargs[key] = value

        try:
            return MetricClass(**kwargs)
        except Exception as e:
            logger.error(f"Error building metric {metric_type}: {e}")
            return None

    # Update _build_target_model to pass timeout

    def _build_target_model(self) -> Optional[CustomLLM]:
        """Build target model to generate actual outputs"""
        target_config = self.config.get('target')

        if not target_config:
            logger.info("No target model specified, using pre-defined actual_outputs from test cases")
            return None

        model_name = target_config.get('model')
        api_url = target_config.get('api_url')
        timeout = target_config.get('timeout', 120.0)  # Default 2 minutes

        if not model_name or not api_url:
            logger.warning("Target model requires 'model' and 'api_url'")
            return None

        logger.info(f"Using target model: {model_name} at {api_url} (timeout: {timeout}s)")

        return CustomLLM(
            model=model_name,
            api_url=api_url,
            api_key=target_config.get('api_key', ''),
            generation_config=target_config.get('generation_config', {}),
            timeout=timeout
        )

    # Add this helper method to SurogateDeepEval class

    def _clean_output(self, text: str) -> str:
        """Clean output by removing thinking tags if configured"""
        strip_thinking = self.config.get('strip_thinking_tags', True)  # Default to True

        if not strip_thinking:
            return text

        # Remove <think>...</think> tags and their content
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.strip()

        return text

    def _generate_actual_outputs(self, test_cases: List[LLMTestCase], target_model: CustomLLM) -> List[LLMTestCase]:
        """Generate actual outputs from target model for test cases"""
        logger.info(f"Generating actual outputs from target model for {len(test_cases)} test cases...")

        for i, test_case in enumerate(test_cases):
            if test_case.actual_output:
                logger.debug(f"Test case {i} already has actual_output, skipping")
                continue

            try:
                # Generate response from target model
                actual_output = target_model.generate(test_case.input)

                # Clean output (remove thinking tags)
                actual_output = self._clean_output(actual_output)

                test_case.actual_output = actual_output
                logger.debug(f"Generated output for test case {i}: {actual_output[:100]}...")

            except Exception as e:
                logger.error(f"Error generating output for test case {i}: {e}")
                test_case.actual_output = ""

        return test_cases

    def _load_test_cases(self) -> List[LLMTestCase]:
        """Load test cases from dataset file or inline config"""

        # Check for inline test cases first
        if 'test_cases' in self.config:
            logger.info("Using inline test cases from config")
            test_cases = []
            for item in self.config['test_cases']:
                test_case = LLMTestCase(
                    input=item.get('input') or item.get('question'),
                    actual_output=item.get('actual_output'),  # Can be None, will be generated
                    expected_output=item.get('expected_output'),
                    context=item.get('context'),
                    retrieval_context=item.get('retrieval_context'),
                )
                test_cases.append(test_case)
            logger.info(f"Loaded {len(test_cases)} inline test cases")
            return test_cases

        # Otherwise load from file
        dataset_path = self.args.get('dataset') or self.config.get('dataset')

        if not dataset_path:
            raise ValueError(
                "No dataset or test_cases specified. Use 'dataset' file, 'test_cases' inline, or --dataset CLI arg.")

        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        # Load dataset based on format
        if dataset_path.suffix == '.json':
            with open(dataset_path) as f:
                data = json.load(f)
        elif dataset_path.suffix == '.jsonl':
            data = []
            with open(dataset_path) as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

        # Convert to LLMTestCase objects
        test_cases = []
        for item in data:
            test_case = LLMTestCase(
                input=item.get('input') or item.get('question'),
                actual_output=item.get('actual_output'),  # Can be None
                expected_output=item.get('expected_output'),
                context=item.get('context'),
                retrieval_context=item.get('retrieval_context'),
            )
            test_cases.append(test_case)

        logger.info(f"Loaded {len(test_cases)} test cases from {dataset_path}")
        return test_cases

    def _build_metrics(self) -> List[Any]:
        """Build all metrics from config"""
        metrics_config = self.config.get('metrics', [])

        if not metrics_config:
            raise ValueError("No metrics specified in config")

        metrics = []
        for metric_config in metrics_config:
            metric = self._build_metric(metric_config)
            if metric:
                metrics.append(metric)
                logger.info(f"Added metric: {metric_config.get('type')}")

        return metrics

    def run(self):
        """Run DeepEval evaluation"""
        try:
            # Build judge model first (needed for metrics)
            self.judge_llm = self._build_judge_model()

            if not self.judge_llm:
                raise ValueError("Failed to build judge model")

            # Build target model (optional - for generating actual outputs)
            target_model = self._build_target_model()

            # Load test cases
            test_cases = self._load_test_cases()

            # Generate actual outputs if target model is provided
            if target_model:
                test_cases = self._generate_actual_outputs(test_cases, target_model)
            else:
                # Verify all test cases have actual_output
                missing_outputs = [i for i, tc in enumerate(test_cases) if not tc.actual_output]
                if missing_outputs:
                    raise ValueError(
                        f"Test cases at indices {missing_outputs} are missing 'actual_output'. "
                        "Either provide 'actual_output' in test cases or specify 'target' model in config."
                    )

            # Build metrics
            metrics = self._build_metrics()

            # Configure Confident AI settings
            confident_ai_config = self.config.get('confident_ai', {})
            show_confident_prompts = confident_ai_config.get('show_prompts', False)

            # Disable telemetry and prompts by default
            if not show_confident_prompts:
                os.environ['DEEPEVAL_TELEMETRY_OPT_OUT'] = 'YES'

            # Set Confident AI API key if provided
            if confident_ai_config.get('api_key'):
                os.environ['CONFIDENT_API_KEY'] = confident_ai_config['api_key']

            # Run evaluation
            logger.info(f"Starting evaluation with {len(metrics)} metrics on {len(test_cases)} test cases")

            # Control verbosity
            verbose = self.config.get('verbose', True)
            import contextlib

            if not verbose:
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        results = evaluate(test_cases=test_cases, metrics=metrics)
            else:
                results = evaluate(test_cases=test_cases, metrics=metrics)

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

        # Save results
        output_path = Path(self.args.get('output', 'results/deepeval.json'))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to dict for saving
        judge_name = self.judge_llm if isinstance(self.judge_llm, str) else self.judge_llm.get_model_name()
        target_name = target_model.get_model_name() if target_model else "pre-defined"

        results_dict = {
            'summary': {
                'total_cases': len(test_cases),
                'metrics': [m.__class__.__name__ for m in metrics],
                'judge_model': judge_name,
                'target_model': target_name,
            },
            'results': []
        }

        # Extract detailed results
        try:
            if hasattr(results, 'test_results'):
                for idx, test_result in enumerate(results.test_results):
                    result_item = {
                        'test_case_index': idx,
                        'success': test_result.success if hasattr(test_result, 'success') else None,
                        'metrics_data': []
                    }

                    # Get test case details
                    if hasattr(test_result, 'input'):
                        result_item['input'] = str(test_result.input)[:200]
                    if hasattr(test_result, 'actual_output'):
                        result_item['actual_output'] = str(test_result.actual_output)[:500]
                    if hasattr(test_result, 'expected_output'):
                        result_item['expected_output'] = str(test_result.expected_output)[:200]
                    if hasattr(test_result, 'retrieval_context'):
                        result_item['retrieval_context'] = test_result.retrieval_context

                    # Extract metrics data
                    if hasattr(test_result, 'metrics_data'):
                        for metric_data in test_result.metrics_data:
                            result_item['metrics_data'].append({
                                'name': metric_data.name if hasattr(metric_data, 'name') else 'unknown',
                                'score': metric_data.score if hasattr(metric_data, 'score') else None,
                                'threshold': metric_data.threshold if hasattr(metric_data, 'threshold') else None,
                                'success': metric_data.success if hasattr(metric_data, 'success') else None,
                                'reason': metric_data.reason if hasattr(metric_data, 'reason') else None,
                                'strict_mode': metric_data.strict_mode if hasattr(metric_data, 'strict_mode') else None,
                                'evaluation_model': metric_data.evaluation_model if hasattr(metric_data,
                                                                                            'evaluation_model') else None,
                                'evaluation_cost': metric_data.evaluation_cost if hasattr(metric_data,
                                                                                          'evaluation_cost') else None,
                            })

                    results_dict['results'].append(result_item)

            # Calculate aggregate statistics
            if hasattr(results, 'test_results') and results.test_results:
                all_metrics = {}
                total_cost = 0.0

                for test_result in results.test_results:
                    if hasattr(test_result, 'metrics_data'):
                        for metric_data in test_result.metrics_data:
                            metric_name = metric_data.name if hasattr(metric_data, 'name') else 'unknown'

                            if metric_name not in all_metrics:
                                all_metrics[metric_name] = {
                                    'scores': [],
                                    'successes': [],
                                    'total_cost': 0.0
                                }

                            if hasattr(metric_data, 'score') and metric_data.score is not None:
                                all_metrics[metric_name]['scores'].append(metric_data.score)
                            if hasattr(metric_data, 'success') and metric_data.success is not None:
                                all_metrics[metric_name]['successes'].append(metric_data.success)
                            if hasattr(metric_data, 'evaluation_cost') and metric_data.evaluation_cost is not None:
                                all_metrics[metric_name]['total_cost'] += metric_data.evaluation_cost
                                total_cost += metric_data.evaluation_cost

                # Add aggregate stats to summary
                results_dict['summary']['aggregate_metrics'] = {}
                for metric_name, data in all_metrics.items():
                    results_dict['summary']['aggregate_metrics'][metric_name] = {
                        'average_score': sum(data['scores']) / len(data['scores']) if data['scores'] else None,
                        'pass_rate': sum(data['successes']) / len(data['successes']) * 100 if data[
                            'successes'] else None,
                        'total_cost': data['total_cost']
                    }

                results_dict['summary']['total_evaluation_cost'] = total_cost

        except Exception as e:
            logger.warning(f"Error parsing detailed results: {e}")
            results_dict['results'] = [
                {
                    'test_case_index': i,
                    'input': str(tc.input)[:100],
                    'note': f'Detailed metrics unavailable: {str(e)}'
                }
                for i, tc in enumerate(test_cases)
            ]

        with open(output_path, 'w') as f:
            json.dump(results_dict, indent=2, fp=f)

        logger.info(f"Results saved to {output_path}")

        # Calculate pass rate
        successful_tests = len([r for r in results_dict['results'] if r.get('success')])
        pass_rate = (successful_tests / len(test_cases) * 100) if test_cases else 0
        logger.info(f"Evaluation completed! Pass rate: {pass_rate:.1f}%")

        return results

    def debug_config(self):
        """Debug: print the configuration"""
        import json
        print("\n" + "=" * 60)
        print("DeepEval Configuration:")
        print("=" * 60)
        print(json.dumps(self.config, indent=2, default=str))
        print("=" * 60 + "\n")