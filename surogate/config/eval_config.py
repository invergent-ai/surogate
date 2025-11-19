from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path

from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class ProjectConfig:
    """
    Project metadata configuration.

    Args:
        name (str): Project name
        version (Optional[str]): Project version. Default is None.
        description (Optional[str]): Project description. Default is None.
    """
    name: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None

    def __init__(self, cfg: DictDefault):
        self.name = cfg['name']
        self.version = cfg['version']
        self.description = cfg['description']


@dataclass
class TargetConfig:
    """
    Target configuration for evaluation.

    Supports multiple target types:
    - API targets (OpenAI, Anthropic, Cohere, Azure, etc.)
    - Local models (vLLM, Ollama, Transformers)
    - Embedding models
    - Vision-language models
    - Custom targets

    Args:
        # Basic target identification
        name (str): Target name identifier
        type (str): Target type ('llm', 'multimodal', 'embedding', 'reranker', 'clip', 'custom')

        # Model configuration
        provider (str): Model provider ('openai', 'anthropic', 'azure', 'cohere', 'huggingface', 'vllm', 'ollama', 'local', 'custom')
        model (str): Model identifier or name
        model_path (Optional[str]): Path to local model files. Default is None.
        base_url (Optional[str]): Base URL for API endpoints. Default is None.
        api_key (Optional[str]): API key for authentication. Default is None.
        endpoint (Optional[str]): Custom endpoint URL. Default is None.

        # Model loading configuration
        backend (Optional[str]): Backend engine ('transformers', 'vllm'). Default is None.
        device (Optional[str]): Device to run on ('cuda', 'cpu', etc.). Default is None.
        timeout (Optional[float]): Request timeout in seconds. Default is None.
        headers (Optional[Dict[str, str]]): Custom HTTP headers. Default is None.
        health_endpoint (Optional[str]): Health check endpoint. Default is None.
        load_in_8bit (Optional[bool]): Load model in 8-bit precision. Default is None.
        load_in_4bit (Optional[bool]): Load model in 4-bit precision. Default is None.
        tensor_parallel_size (Optional[int]): Tensor parallelism size. Default is None.

        # Serving configuration (for local/vLLM targets)
        infer_backend (Optional[Literal['vllm', 'pytorch', 'sglang']]): Backend to use for inference. Default is None.
        host (Optional[str]): The host address to bind the server to. Default is '0.0.0.0'.
        port (Optional[int]): The port number to bind the server to. Default is 8000.
        served_name (Optional[str]): The name of the model being served. Default is None.
        max_logprobs (Optional[int]): Max number of logprobs to return. Default is 20.
        seed (Optional[int]): Random seed for reproducibility. Default is 1234.
        deterministic (Optional[bool]): Whether to use deterministic inference. Default is False.
        max_context (Optional[int]): Maximum context length for the model. Default is None.
        tensor_parallel (Optional[int]): Tensor parallelism size. Default is 1.
        max_memory (Optional[float]): Maximum GPU memory utilization. Default is 0.9 (90%).
        use_chat_template (Optional[bool]): Whether to use model's chat template. Default is True.

        # Evaluation configuration
        infrastructure (Optional[Dict[str, Any]]): Infrastructure settings for parallel execution. Default is None.
        evaluations (Optional[List[Dict[str, Any]]]): List of evaluation configurations. Default is None.
        stress_testing (Optional[Dict[str, Any]]): Stress testing configuration. Default is None.
        red_teaming (Optional[Dict[str, Any]]): Red teaming (security testing) configuration. Default is None.
        guardrails (Optional[Dict[str, Any]]): Guardrails (refusal behavior) testing configuration. Default is None.
        comment (Optional[str]): Additional comments about this target. Default is None.
    """
    # Basic target identification
    name: Optional[str] = None
    type: Optional[str] = None

    # Model configuration
    provider: Optional[str] = None
    model: Optional[str] = None
    model_path: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None

    # Model loading
    backend: Optional[str] = None
    device: Optional[str] = None
    timeout: Optional[float] = None
    headers: Optional[Dict[str, str]] = None
    health_endpoint: Optional[str] = None
    load_in_8bit: Optional[bool] = None
    load_in_4bit: Optional[bool] = None
    tensor_parallel_size: Optional[int] = None

    # Serving configuration
    infer_backend: Optional[Literal['vllm', 'pytorch', 'sglang']] = None
    host: Optional[str] = None
    port: Optional[int] = None
    served_name: Optional[str] = None
    max_logprobs: Optional[int] = None
    seed: Optional[int] = None
    deterministic: Optional[bool] = None
    max_context: Optional[int] = None
    tensor_parallel: Optional[int] = None
    max_memory: Optional[float] = None
    use_chat_template: Optional[bool] = None

    # Evaluation configuration
    infrastructure: Optional[Dict[str, Any]] = None
    evaluations: Optional[List[Dict[str, Any]]] = None
    stress_testing: Optional[Dict[str, Any]] = None
    red_teaming: Optional[Dict[str, Any]] = None
    guardrails: Optional[Dict[str, Any]] = None
    comment: Optional[str] = None

    def __init__(self, cfg: DictDefault):
        # Basic target config
        self.name = cfg['name']
        self.type = cfg['type']
        self.provider = cfg['provider']
        self.model = cfg['model']
        self.model_path = cfg['model_path']
        self.base_url = cfg['base_url']
        self.api_key = cfg['api_key']
        self.endpoint = cfg['endpoint']

        # Model loading
        self.backend = cfg['backend']
        self.device = cfg['device']
        self.timeout = cfg['timeout']
        self.headers = cfg['headers']
        self.health_endpoint = cfg['health_endpoint']
        self.load_in_8bit = cfg['load_in_8bit']
        self.load_in_4bit = cfg['load_in_4bit']
        self.tensor_parallel_size = cfg['tensor_parallel_size']

        # Serving configuration (with defaults matching ServeConfig)
        self.infer_backend = cfg['infer_backend']
        self.host = cfg['host'] or '0.0.0.0'
        self.port = cfg['port'] or 8000
        self.served_name = cfg['served_name']
        self.max_logprobs = cfg['max_logprobs'] or 20
        self.seed = cfg['seed'] or 1234
        self.deterministic = cfg['deterministic'] or False
        self.max_context = cfg['max_context']
        self.tensor_parallel = cfg['tensor_parallel'] or 1
        self.max_memory = cfg['max_memory'] or 0.9
        self.use_chat_template = cfg['use_chat_template'] if cfg['use_chat_template'] is not None else True

        # Evaluation configuration
        self.infrastructure = cfg['infrastructure']
        self.evaluations = cfg['evaluations']
        self.stress_testing = cfg['stress_testing']
        self.red_teaming = cfg['red_teaming']
        self.guardrails = cfg['guardrails']
        self.comment = cfg['comment']

    def to_dict(self) -> Dict[str, Any]:
        """Convert TargetConfig to dictionary for compatibility with TargetFactory."""
        result = {}

        # Only include non-None values
        if self.name is not None:
            result['name'] = self.name
        if self.type is not None:
            result['type'] = self.type
        if self.provider is not None:
            result['provider'] = self.provider
        if self.model is not None:
            result['model'] = self.model
        if self.model_path is not None:
            result['model_path'] = self.model_path
        if self.base_url is not None:
            result['base_url'] = self.base_url
        if self.api_key is not None:
            result['api_key'] = self.api_key
        if self.endpoint is not None:
            result['endpoint'] = self.endpoint
        if self.backend is not None:
            result['backend'] = self.backend
        if self.device is not None:
            result['device'] = self.device
        if self.timeout is not None:
            result['timeout'] = self.timeout
        if self.headers is not None:
            result['headers'] = self.headers
        if self.health_endpoint is not None:
            result['health_endpoint'] = self.health_endpoint
        if self.load_in_8bit is not None:
            result['load_in_8bit'] = self.load_in_8bit
        if self.load_in_4bit is not None:
            result['load_in_4bit'] = self.load_in_4bit
        if self.tensor_parallel_size is not None:
            result['tensor_parallel_size'] = self.tensor_parallel_size

        return result


@dataclass
class EvalConfig:
    """
    Main evaluation configuration.

    Validates:
    - Project metadata existence
    - At least one target
    - Target names are unique
    - Target types are valid
    - Required fields per provider
    - Judge target references exist
    - File paths exist (warnings only)

    Args:
        project (ProjectConfig): Project metadata
        targets (List[TargetConfig]): List of evaluation targets
    """
    project: Optional[ProjectConfig] = None
    targets: Optional[List[TargetConfig]] = None

    def __init__(self, cfg: DictDefault):
        self.project = ProjectConfig(cfg['project']) if cfg['project'] else None
        self.targets = [TargetConfig(t) for t in cfg['targets']] if cfg['targets'] else []
        self.__post_init__()

    def __post_init__(self):
        """Validate configuration after initialization."""
        errors = []
        warnings = []

        # Validate project
        if not self.project:
            errors.append("Project configuration is required")
        elif not self.project.name:
            errors.append("Project name is required")

        # Validate targets exist
        if not self.targets:
            errors.append("At least one target must be specified")
        else:
            # Check for duplicate target names
            target_names = [t.name for t in self.targets]
            duplicates = [name for name in target_names if target_names.count(name) > 1]
            if duplicates:
                errors.append(f"Duplicate target names found: {set(duplicates)}")

            # Validate each target
            for target in self.targets:
                target_errors, target_warnings = self._validate_target(target)
                errors.extend(target_errors)
                warnings.extend(target_warnings)

            # Validate target references (judge models)
            ref_errors = self._validate_target_references()
            errors.extend(ref_errors)

            # Validate file paths
            path_warnings = self._validate_file_paths()
            warnings.extend(path_warnings)

        # Log warnings
        for warning in warnings:
            logger.warning(warning)

        # Raise if errors
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

        logger.info("✓ Configuration validation passed")

    def _validate_target(self, target: TargetConfig) -> tuple[list[str], list[str]]:
        """Validate a single target configuration. Returns (errors, warnings)."""
        errors = []
        warnings = []

        if not target.name:
            errors.append("Target name is required")
            return errors, warnings  # Can't continue without name

        # Validate type
        if not target.type:
            errors.append(f"Target '{target.name}': type is required")
        else:
            valid_types = ['llm', 'multimodal', 'embedding', 'reranker', 'clip', 'custom']
            if target.type not in valid_types:
                errors.append(
                    f"Target '{target.name}': invalid type '{target.type}'. "
                    f"Valid types: {valid_types}"
                )

        # Validate provider and model for llm/multimodal
        if target.type in ['llm', 'multimodal']:
            if not target.provider:
                errors.append(f"Target '{target.name}': provider is required for type '{target.type}'")
            if not target.model:
                errors.append(f"Target '{target.name}': model is required for type '{target.type}'")

        # Validate local provider
        if target.provider == 'local' and not target.model and not target.model_path:
            errors.append(f"Target '{target.name}': model or model_path required for local provider")

        # Validate API key (only for non-local endpoints)
        providers_needing_key = ['anthropic', 'cohere', 'azure']
        if target.provider in providers_needing_key and not target.api_key:
            errors.append(f"Target '{target.name}': api_key is required for provider '{target.provider}'")

        # OpenAI needs API key unless it's a local endpoint (localhost, 127.0.0.1)
        if target.provider == 'openai' and not target.api_key:
            if not target.base_url or not any(x in target.base_url for x in ['localhost', '127.0.0.1', '0.0.0.0']):
                errors.append(
                    f"Target '{target.name}': api_key is required for OpenAI provider (except local endpoints)")

        # Validate evaluations
        if target.evaluations:
            eval_errors, eval_warnings = self._validate_evaluations(target.evaluations, target.name)
            errors.extend(eval_errors)
            warnings.extend(eval_warnings)

        # Validate infrastructure
        if target.infrastructure:
            infra_warnings = self._validate_infrastructure(target.infrastructure, target.name)
            warnings.extend(infra_warnings)

        # Validate red teaming
        if target.red_teaming and target.red_teaming.get('enabled'):
            if not target.red_teaming.get('vulnerabilities') and not target.red_teaming.get('attacks'):
                warnings.append(
                    f"Target '{target.name}': red_teaming enabled but no vulnerabilities or attacks specified")

        # Validate stress testing
        if target.stress_testing and target.stress_testing.get('enabled'):
            if not target.stress_testing.get('dataset'):
                errors.append(f"Target '{target.name}': stress_testing enabled but no dataset specified")

        return errors, warnings

    def _validate_evaluations(self, evaluations: List[Dict[str, Any]], target_name: str) -> tuple[list[str], list[str]]:
        """Validate evaluations for a target. Returns (errors, warnings)."""
        errors = []
        warnings = []

        if not evaluations:
            warnings.append(f"Target '{target_name}': no evaluations specified")
            return errors, warnings

        # Check for duplicate evaluation names
        eval_names = [e.get('name', f'eval-{i}') for i, e in enumerate(evaluations)]
        duplicates = [name for name in eval_names if eval_names.count(name) > 1]
        if duplicates:
            warnings.append(f"Target '{target_name}': duplicate evaluation names found: {set(duplicates)}")

        for idx, evaluation in enumerate(evaluations):
            eval_name = evaluation.get('name', f'evaluation-{idx}')

            # Check that evaluation has EITHER dataset OR benchmarks (or both)
            has_dataset = bool(evaluation.get('dataset'))
            has_benchmarks = bool(evaluation.get('benchmarks'))
            has_metrics = bool(evaluation.get('metrics'))

            if not has_dataset and not has_benchmarks:
                errors.append(
                    f"Target '{target_name}', Evaluation '{eval_name}': "
                    f"must have either 'dataset' (for custom metrics) or 'benchmarks' (for standard benchmarks)"
                )

            # If has dataset, should have metrics
            if has_dataset and not has_metrics:
                warnings.append(
                    f"Target '{target_name}', Evaluation '{eval_name}': "
                    f"dataset specified but no metrics defined"
                )

            # Validate benchmarks
            if has_benchmarks:
                benchmarks = evaluation.get('benchmarks', [])
                if not benchmarks:
                    warnings.append(
                        f"Target '{target_name}', Evaluation '{eval_name}': "
                        f"benchmarks key present but empty"
                    )
                else:
                    for bench_idx, benchmark in enumerate(benchmarks):
                        if not benchmark.get('name'):
                            errors.append(
                                f"Target '{target_name}', Evaluation '{eval_name}': "
                                f"benchmark at index {bench_idx} missing 'name' field"
                            )

        return errors, warnings

    def _validate_infrastructure(self, infra: Dict[str, Any], target_name: str) -> list[str]:
        """Validate infrastructure configuration. Returns warnings."""
        warnings = []

        workers = infra.get('workers', 1)
        if workers > 32:
            warnings.append(
                f"Target '{target_name}': workers is very high ({workers}), may cause resource issues"
            )

        parallel = infra.get('parallel_execution', {})
        if parallel.get('enabled'):
            max_workers = parallel.get('max_workers', 0)
            if max_workers <= 0:
                warnings.append(
                    f"Target '{target_name}': max_workers should be positive when parallel_execution is enabled"
                )

        return warnings

    def _validate_target_references(self) -> list[str]:
        """Validate that judge model references point to existing targets."""
        errors = []
        target_names = {t.name for t in self.targets}

        for target in self.targets:
            if not target.evaluations:
                continue

            for eval_config in target.evaluations:
                eval_name = eval_config.get('name', 'unnamed')
                metrics = eval_config.get('metrics', [])

                # Check metric judge models
                for metric in metrics:
                    judge_model = metric.get('judge_model', {})
                    judge_target = judge_model.get('target')

                    if judge_target and judge_target not in target_names:
                        errors.append(
                            f"Target '{target.name}', Evaluation '{eval_name}': "
                            f"judge target '{judge_target}' not found in configured targets"
                        )

                # Check benchmark judge models
                benchmarks = eval_config.get('benchmarks', [])
                for benchmark in benchmarks:
                    judge_model = benchmark.get('judge_model', {})
                    judge_target = judge_model.get('target')

                    if judge_target and judge_target not in target_names:
                        errors.append(
                            f"Target '{target.name}', Benchmark '{benchmark.get('name')}': "
                            f"judge target '{judge_target}' not found in configured targets"
                        )

            # Check guardrails judge model
            if target.guardrails and target.guardrails.get('enabled'):
                refusal_judge = target.guardrails.get('refusal_judge_model', {})
                judge_target = refusal_judge.get('target')

                if judge_target and judge_target not in target_names:
                    errors.append(
                        f"Target '{target.name}': "
                        f"guardrails refusal_judge_model target '{judge_target}' not found in configured targets"
                    )

        return errors

    def _validate_file_paths(self) -> list[str]:
        """Validate that specified file paths exist. Returns warnings."""
        warnings = []
        project_root = Path.cwd()

        for target in self.targets:
            # Check stress testing dataset
            if target.stress_testing and target.stress_testing.get('enabled'):
                dataset = target.stress_testing.get('dataset')
                if dataset:
                    dataset_path = Path(dataset)
                    if not dataset_path.is_absolute():
                        dataset_path = project_root / dataset
                    if not dataset_path.exists():
                        warnings.append(
                            f"Target '{target.name}': "
                            f"stress_testing dataset path does not exist: {dataset}"
                        )

            # Check guardrails safe prompts dataset
            if target.guardrails and target.guardrails.get('enabled'):
                safe_dataset = target.guardrails.get('safe_prompts_dataset')
                if safe_dataset:
                    dataset_path = Path(safe_dataset)
                    if not dataset_path.is_absolute():
                        dataset_path = project_root / dataset_path
                    if not dataset_path.exists():
                        warnings.append(
                            f"Target '{target.name}': "
                            f"guardrails safe_prompts_dataset does not exist: {safe_dataset}"
                        )

            if not target.evaluations:
                continue

            for evaluation in target.evaluations:
                eval_name = evaluation.get('name', 'unnamed')

                # Check dataset path
                dataset = evaluation.get('dataset')
                if dataset:
                    dataset_path = Path(dataset)
                    if not dataset_path.is_absolute():
                        dataset_path = project_root / dataset
                    if not dataset_path.exists():
                        warnings.append(
                            f"Target '{target.name}', Evaluation '{eval_name}': "
                            f"dataset path does not exist: {dataset}"
                        )

                # Check benchmark custom paths
                benchmarks = evaluation.get('benchmarks', [])
                for bench in benchmarks:
                    bench_path = bench.get('path')
                    if bench_path:
                        path = Path(bench_path)
                        if not path.is_absolute():
                            path = project_root / path
                        if not path.exists():
                            warnings.append(
                                f"Target '{target.name}', Benchmark '{bench.get('name')}': "
                                f"custom path does not exist: {bench_path}"
                            )

        return warnings


    def get_target_by_name(self, name: str) -> Optional[TargetConfig]:
        """Find target configuration by name."""
        for target in self.targets:
            if target.name == name:
                return target
        return None