from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

from surogate.utils.dict import DictDefault


@dataclass
class GRPOModelConfig:
    """
    Configuration for a model.

    Args:
        name: Name or path of the HF model to use.
        lora_adapter: Name of the LoRA adapter to register with vLLM. When set, weight broadcasts
            are routed through vLLM's `/load_lora_adapter` endpoint and the served model name
            switches to this adapter after the first update. Required when training with LoRA.
    """

    name: str | None = None
    lora_adapter: str | None = None

    def __init__(self, cfg: DictDefault):
        self.name = cfg.get("name", self.name)
        self.lora_adapter = cfg.get("lora_adapter", self.lora_adapter)


@dataclass
class GRPOClientConfig:
    """
    Configures the GRPO orchestrator client for RL training.

    Args:
        timeout: Timeout in seconds. Defaults to 1200 seconds.
        base_url: Base URLs to use for the OpenAI API. By default, it is set to a single server on localhost at port 8000 which matches the default local vLLM server configuration. If you specify more than one URL, the client will round-robin (chat) completion requests across all servers.
        api_key_var: Name of environment variable containing the API key to use for the inference API. Can be set to an arbitrary string if the inference server is not protected by an API key. If multiple URLs are specified, the same API key will be used for all servers.
        headers: Headers to use for the OpenAI API. By default, it is set to an empty dictionary.
        skip_model_check: Whether to skip checking if the model is available in the inference pool. Useful for external APIs or API Keys that don't support the /models endpoint.
    """

    timeout: int | None = 1200
    base_url: list[str] | None = field(default_factory=lambda: ["http://localhost:8000/v1"])
    api_key_var: str | None = "VLLM_API_KEY"
    headers: dict[str, str] | None = field(default_factory=dict)
    skip_model_check: bool | None = False

    def __init__(self, cfg: DictDefault):
        self.timeout = cfg.get("timeout", 1200)
        self.base_url = cfg.get("base_url", ["http://localhost:8000/v1"])
        self.api_key_var = cfg.get("api_key_var", "VLLM_API_KEY")
        self.headers = cfg.get("headers", {})
        self.skip_model_check = cfg.get("skip_model_check", False)


@dataclass
class TeacherModelConfig(GRPOModelConfig, GRPOClientConfig):
    """
    Configures the teacher model for computing teacher logprobs (e.g. for distillation).

    Args:
        model: The model configuration for the teacher model.
        client: The OAI client configuration for the teacher model.
    """

    model: GRPOModelConfig | None = None
    client: GRPOClientConfig | None = None

    def __init__(self, cfg: DictDefault):
        self.model = GRPOModelConfig(cfg.get("model", {}))
        self.client = GRPOClientConfig(cfg.get("client", {}))


@dataclass
class GRPOTemperatureSchedulerConfig:
    """
    Configures temperature scheduling over training steps. Use this OR sampling.temperature, not both.

    Args:
        type: Schedule shape. Linear interpolates linearly; cosine uses smooth, monotonic curve.
        start_temperature: Temperature at step 0.
        end_temperature: Temperature at the final step.
        total_steps: Number of steps to reach end_temperature. Defaults to orchestrator max_steps if None.
    """

    type: Literal["linear", "cosine"] | None = "linear"
    start_temperature: float | None = None
    end_temperature: float | None = None
    total_steps: int | None = None

    def __init__(self, cfg: DictDefault):
        self.type = cfg.get("type", self.type)
        self.start_temperature = cfg.get("start_temperature", self.start_temperature)
        self.end_temperature = cfg.get("end_temperature", self.end_temperature)
        self.total_steps = cfg.get("total_steps", self.total_steps)


@dataclass
class GRPOSamplingConfig:
    """
    Configures how tokens are sampled from the model for training. Largely follows the vLLM sampling parameters.

    Args:
        temperature: Constant temperature for sampling. Defaults to 1.0 if neither this nor temp_scheduler is set. Cannot be set together with temp_scheduler.
        temp_scheduler: Temperature schedule over training steps. Set this OR temperature, not both.
        top_p: Cumulative probability of the top tokens to consider. If 1, all tokens are considered.
        repetition_penalty: Penalty for repeating tokens. Values > 1.0 discourage repetition, values < 1.0 encourage repetition, and 1.0 means no penalty.
        max_tokens: Maximum number of output tokens to generate per turn. If None, will generate until maximum context length or EOS token is hit.
        min_tokens: Minimum number of output tokens to generate per sequence.
        seed: Random seed for sampling. If None, a random seed will be used.
        extra_body: Extra body to pass with each request to the inference server. By default, it is set to an empty dictionary.
    """

    temperature: float | None = None
    temp_scheduler: GRPOTemperatureSchedulerConfig | None = None
    top_p: float | None = 1.0
    repetition_penalty: float | None = 1.0
    max_tokens: int | None = None
    min_tokens: int | None = 0
    seed: int | None = None
    extra_body: dict[str, Any] | None = field(default_factory=dict)

    def __init__(self, cfg: DictDefault):
        self.temperature = cfg.get("temperature", self.temperature)
        if cfg.get("temp_scheduler") is not None:
            self.temp_scheduler = GRPOTemperatureSchedulerConfig(cfg.get("temp_scheduler"))
        self.top_p = cfg.get("top_p", self.top_p)
        self.repetition_penalty = cfg.get("repetition_penalty", self.repetition_penalty)
        self.max_tokens = cfg.get("max_tokens", self.max_tokens)
        self.min_tokens = cfg.get("min_tokens", self.min_tokens)
        self.seed = cfg.get("seed", self.seed)
        self.extra_body = cfg.get("extra_body", {})


@dataclass
class GRPOEnvConfig:
    """
    Configures an environment for training.

    Args:
        id: ID of the environment to use.
        args: Arguments to pass to the environment.
        name: Name of the environment to use.
        address: Address of the environment server. If None, will spawn an environment server in a subprocess automatically.If given, will try to connect an environment client to the environment server at this address.
        extra_env_kwargs: Extra kwargs passed to an env (e.g. seq_len, score_rollouts). This field is auto-populated with the seq_len, and score_rollouts for training envs on the orchestrator. It is generally NOT recommended for this field to be overriden by the user. It's main use case is to match the extra_env_kwargs when running an env in an isolated environment server.
        max_retries: Maximum number of times the environment will retry a failed rollout.
        path: Optional directory prepended to sys.path before importing the env module. Useful when the env is checked out locally (e.g. .../my-envs/<id>) and not pip-installed. The directory must contain a Python package named after `id` (with dashes converted to underscores) that exposes a top-level `load_environment` function.
    """

    id: str | None = None
    args: dict[str, Any] | None = field(default_factory=dict)
    name: str | None = None
    address: str | None = None
    extra_env_kwargs: dict[str, Any] | None = field(default_factory=dict)
    max_retries: int | None = 0
    path: str | None = None

    @property
    def resolved_name(self) -> str:
        return self.name or self.id.split("@")[0]

    def __init__(self, cfg: DictDefault):
        self.id = cfg.get("id", self.id)
        self.args = cfg.get("args", {})
        self.name = cfg.get("name", self.name)
        self.address = cfg.get("address", self.address)
        self.extra_env_kwargs = cfg.get("extra_env_kwargs", {})
        self.max_retries = cfg.get("max_retries", self.max_retries)
        self.path = cfg.get("path", self.path)

        if self.resolved_name == "all":
            raise ValueError(
                'Environment name "all" is reserved for global metric aggregation. Use a different name or id.'
            )


@dataclass
class GRPOEvalEnvConfig(GRPOEnvConfig):
    """
    Configures an environment for evaluation.

    Inherits base environment fields (`id`, `args`, `name`, `address`,
    `extra_env_kwargs`, `max_retries`) and adds eval-specific overrides.

    Args:
        num_examples: Number of examples to evaluate for this environment.
            If None, falls back to eval.num_examples.
        rollouts_per_example: Number of rollouts per example for this
            environment. If None, falls back to eval.rollouts_per_example.
        skip_first: Number of initial examples to skip. Defaults to 0.
    """

    num_examples: int | None = None
    rollouts_per_example: int | None = None
    skip_first: int | None = 0

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.num_examples = cfg.get("num_examples", self.num_examples)
        self.rollouts_per_example = cfg.get("rollouts_per_example", self.rollouts_per_example)
        self.skip_first = cfg.get("skip_first", self.skip_first)


@dataclass
class GRPOEvalSamplingConfig:
    """
    Configures how tokens are sampled from the model for evaluation. Largely follows the vLLM sampling parameters.

    Args:
        temperature: Scales the output probability distribution. Lower values => more deterministic, higher values => more random. If 0, will sample greedily. Defaults to None, which means we fall back to the inference server's default value.
        repetition_penalty: Penalty for repeating tokens. Values > 1.0 discourage repetition, values < 1.0 encourage repetition, and 1.0 means no penalty. Defaults to None, which means we fall back to the inference server's default value.
        top_p: Cumulative probability of the top tokens to consider. If 1, all tokens are considered. Defaults to None, which means we fall back to the inference server's default value.
        top_k: Number of top tokens to consider. If 0, all tokens are considered. Defaults to None, which means we fall back to the inference server's default value.
        min_p: Minimum probability of tokens to consider. Defaults to None, which means we fall back to the inference server's default value.
        max_tokens: Maximum number of output tokens to generate per turn. If None, will generate until maximum context length or EOS token is hit. Defaults to None, which means we fall back to the inference server's default value.
        min_tokens: Minimum number of output tokens to generate per sequence. Defaults to None, which means we fall back to the inference server's default value.
        reasoning_effort: Constrains effort on reasoning for reasoning models. Currently supported values are minimal, low, medium, and high. Defaults to None, which means we fall back to the inference server's default value.
        seed: Random seed for sampling. If None, a random seed will be used. Defaults to None, which means we fall back to the inference server's default value.
        extra_body: Extra body to use for the OpenAI API. By default, it is set to an empty dictionary.
    """

    temperature: float | None = None
    repetition_penalty: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    min_tokens: int | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    seed: int | None = None
    extra_body: dict[str, Any] | None = None

    def __init__(self, cfg: DictDefault):
        self.temperature = cfg.get("temperature", self.temperature)
        self.repetition_penalty = cfg.get("repetition_penalty", self.repetition_penalty)
        self.top_p = cfg.get("top_p", self.top_p)
        self.top_k = cfg.get("top_k", self.top_k)
        self.min_p = cfg.get("min_p", self.min_p)
        self.max_tokens = cfg.get("max_tokens", self.max_tokens)
        self.min_tokens = cfg.get("min_tokens", self.min_tokens)
        self.reasoning_effort = cfg.get("reasoning_effort", self.reasoning_effort)
        self.seed = cfg.get("seed", self.seed)
        self.extra_body = cfg.get("extra_body", self.extra_body)


@dataclass
class GRPOEvalConfig:
    """
    Configures evaluation using verifiers environments.

    Args:
        env: List of environment configurations for evaluation.
        sampling: Shared sampling configuration for evals; can differ from training sampling.
        num_examples: Number of examples to evaluate per environment.
        rollouts_per_example: Number of samples to generate per example for each environment.
        interval: Interval at which to evaluate the model.
        eval_base_model: Whether to evaluate the base model we are training on.
        skip_eval_on_resume: If True and resuming the orchestrator from a checkpoint, skip the (potentially redundant) online eval that would otherwise run immediately at the resumed checkpoint step.
        cancel_inflight_rollouts_on_eval: Whether to cancel in-flight training rollouts before starting online evals. This is useful to avoid congestion (e.g. do not have training + eval rollouts happening at the same time) but leads to slower training steps as rollouts get cancelled and the pipeline has to fill up after each eval.
    """

    env: list[GRPOEvalEnvConfig] | None = None
    sampling: GRPOEvalSamplingConfig | None = None
    num_examples: int | None = -1
    rollouts_per_example: int | None = 1
    interval: int | None = 100
    eval_base_model: bool | None = True
    skip_eval_on_resume: bool | None = True
    cancel_inflight_rollouts_on_eval: bool | None = False

    def __init__(self, cfg: DictDefault):
        env_cfgs = cfg.get("env", [])
        self.env = [GRPOEvalEnvConfig(env_cfg) for env_cfg in env_cfgs]
        self.sampling = GRPOEvalSamplingConfig(cfg.get("sampling", {}))
        self.num_examples = cfg.get("num_examples", self.num_examples)
        self.rollouts_per_example = cfg.get("rollouts_per_example", self.rollouts_per_example)
        self.interval = cfg.get("interval", self.interval)
        self.eval_base_model = cfg.get("eval_base_model", self.eval_base_model)
        self.skip_eval_on_resume = cfg.get("skip_eval_on_resume", self.skip_eval_on_resume)
        self.cancel_inflight_rollouts_on_eval = cfg.get(
            "cancel_inflight_rollouts_on_eval", self.cancel_inflight_rollouts_on_eval
        )


@dataclass
class GRPOBufferConfig:
    """
    Configures the buffer for the orchestrator.

    Args:
        seed: Random seed to use for the buffer. If set, the sampling from the buffer will be deterministic.
        env_ratios: Ratios for sampling from each environment. If None, samples uniformly across all available problems (not environments).
        easy_threshold: Threshold for easy difficulty classification. If average reward >= this threshold, mark as easy.
        hard_threshold: Threshold for hard difficulty classification. If average reward <= this threshold, mark as hard.
        easy_fraction: Fraction of easy problems to convert to normal when resuming or starting training. Only problems with difficulty 'normal' are sampled.
        hard_fraction: Fraction of hard problems to convert to normal when resuming or starting training. Only problems with difficulty 'normal' are sampled.
        online_difficulty_filtering: Whether to filter rollouts based on difficulty. If True, rollouts with average reward 0.0 or 1.0 are not added to the buffer.
        hash_keys: Keys to use for computing example hashes. Will be used to match examples from buffer checkpoints and determine buffer resume behavior.
    """

    seed: int | None = None
    env_ratios: list[float] | None = None
    easy_threshold: float | None = None
    hard_threshold: float | None = None
    easy_fraction: float | None = 0.0
    hard_fraction: float | None = 0.0
    online_difficulty_filtering: bool | None = False
    hash_keys: list[str] | None = field(default_factory=lambda: ["task", "prompt"])

    def __init__(self, cfg: DictDefault):
        self.seed = cfg.get("seed", self.seed)
        self.env_ratios = cfg.get("env_ratios", self.env_ratios)
        self.easy_threshold = cfg.get("easy_threshold", self.easy_threshold)
        self.hard_threshold = cfg.get("hard_threshold", self.hard_threshold)
        self.easy_fraction = cfg.get("easy_fraction", self.easy_fraction)
        self.hard_fraction = cfg.get("hard_fraction", self.hard_fraction)
        self.online_difficulty_filtering = cfg.get("online_difficulty_filtering", self.online_difficulty_filtering)
        self.hash_keys = cfg.get("hash_keys", ["task", "prompt"])
        self.__post_init__()

    def __post_init__(self):
        if self.easy_threshold is not None and self.hard_threshold is not None:
            assert self.easy_threshold > self.hard_threshold, "easy_threshold must be greater than hard_threshold."

        if self.env_ratios is not None:
            assert all(ratio > 0 for ratio in self.env_ratios), "All env_ratios must be positive."


@dataclass
class GRPOVerificationConfig:
    """Configures rollout verification and rubric scoring."""

    enabled: bool | None = True

    def __init__(self, cfg: DictDefault):
        self.enabled = cfg.get("enabled", self.enabled)


@dataclass
class GRPOAdvantageConfig:
    """Config for the default advantage."""

    type: Literal["default"] | None = "default"
    length_weighted_mean: bool | None = False

    def __init__(self, cfg: DictDefault):
        self.type = cfg.get("type", self.type)
        self.length_weighted_mean = cfg.get("length_weighted_mean", self.length_weighted_mean)


@dataclass
class GRPOCustomAdvantageConfig:
    """
    Config for a custom external advantage function.

    Args:
        type: Must be "custom" for this advantage type.
        import_path: Import path to the advantage function (e.g., 'my_module.my_advantage').
        kwargs: Kwargs to pass to the advantage function.
    """

    type: Literal["custom"] | None = "custom"
    import_path: str | None = None
    kwargs: dict[str, Any] | None = None

    def __init__(self, cfg: DictDefault):
        self.type = cfg.get("type", self.type)
        self.import_path = cfg.get("import_path", self.import_path)
        self.kwargs = cfg.get("kwargs", self.kwargs)


AdvantageConfigType: TypeAlias = GRPOAdvantageConfig | GRPOCustomAdvantageConfig


@dataclass
class GRPOGibberishFilterConfig:
    """
    Flags rare tokens generated at high entropy (Section 5.2, https://arxiv.org/abs/2510.02387).

    Args:
        type: Must be "gibberish" for this filter type.
        enforce: If True, mask detected rollouts so they don't contribute to training. If False, only track detection metrics.
        token_id_threshold: Token IDs above this are candidates for gibberish. BPE tokens are sorted by merge order.
        logprob_offset: Offset from uniform distribution logprob. Threshold = -log(vocab_size) - logprob_offset.
    """

    type: Literal["gibberish"] | None = "gibberish"
    enforce: bool | None = False
    token_id_threshold: int | None = 100_000
    logprob_offset: float | None = 2.0

    def __init__(self, cfg: DictDefault):
        self.type = cfg.get("type", self.type)
        self.enforce = cfg.get("enforce", self.enforce)
        self.token_id_threshold = cfg.get("token_id_threshold", self.token_id_threshold)
        self.logprob_offset = cfg.get("logprob_offset", self.logprob_offset)


@dataclass
class GRPORepetitionFilterConfig:
    """
    Flags rollouts where the model gets stuck in a repetition loop, emitting high-confidence tokens
    for an extended stretch. A rollout is flagged when `window` consecutive tokens are each sampled
    with probability above `prob_threshold`. (Section 3.2, https://arxiv.org/abs/2506.13585)

    Args:
        type: Must be "repetition" for this filter type.
        enforce: If True, mask detected rollouts so they don't contribute to training. If False, only track detection metrics.
        window: Number of consecutive high-probability steps before flagging.
        prob_threshold: Probability threshold for high-confidence tokens. Tokens sampled with probability above this are considered repetitive. Consecutive such tokens count toward the window.
    """

    type: Literal["repetition"] | None = "repetition"
    enforce: bool | None = False
    window: int | None = 3_000
    prob_threshold: float | None = 0.99

    def __init__(self, cfg: DictDefault):
        self.type = cfg.get("type", self.type)
        self.enforce = cfg.get("enforce", self.enforce)
        self.window = cfg.get("window", self.window)
        self.prob_threshold = cfg.get("prob_threshold", self.prob_threshold)


FilterConfigType: TypeAlias = GRPOGibberishFilterConfig | GRPORepetitionFilterConfig


@dataclass
class GRPOLogConfig:
    """
    Configures the GRPO logger.

    Args:
        level: Logging level for the process. Will determine the logging verbosity and format.
        vf_level: Logging level for the verifiers package. Will determine the logging verbosity and format.
        file: Whether to log to a file. If True, will log to a file in the output directory.
        env_worker_logs: Whether env workers log to files. If True, workers write to logs/env_workers/{env_name}.log.
        log_data: Whether to log the first data sample to the logger.
        json_logging: Emit JSON logs (newline-delimited) for log aggregation (Loki, Grafana, etc.).
    """

    level: str | None = "info"
    vf_level: str | None = "info"
    file: bool | None = True
    env_worker_logs: bool | None = False
    log_data: bool | None = False
    json_logging: bool | None = False

    def __init__(self, cfg: DictDefault):
        self.level = cfg.get("level", self.level)
        self.vf_level = cfg.get("vf_level", self.vf_level)
        self.file = cfg.get("file", self.file)
        self.env_worker_logs = cfg.get("env_worker_logs", self.env_worker_logs)
        self.log_data = cfg.get("log_data", self.log_data)
        self.json_logging = cfg.get("json_logging", self.json_logging)


@dataclass
class GRPOReportingConfig:
    """
    Configures logging to W&B and other reporting platforms.

    Args:
        project: The W&B project to log to.
        offline: Whether to run W&B in offline mode.
        run_name: The run name to log to. If None, a random ID will be generated. If you want to resume a run, you can set the ID to the run ID you want to resume.
        samples: Whether to log prompt/response samples.
        distributions: Whether to log distributions (like rewards, advantages, etc.).
        interval: Step interval at which to log samples and distributions.
    """

    project: str | None = "Surogate"
    name: str | None = None
    offline: bool | None = False
    run_name: str | None = None
    samples: bool | None = None
    distributions: bool | None = None
    interval: int | None = 10

    def __init__(self, cfg: DictDefault):
        self.project = cfg.get("project", self.project)
        self.name = cfg.get("name", self.name)
        self.offline = cfg.get("offline", self.offline)
        self.run_name = cfg.get("run_name", self.run_name)
        self.samples = cfg.get("samples", self.samples)
        self.distributions = cfg.get("distributions", self.distributions)
        self.interval = cfg.get("interval", self.interval)


@dataclass
class GRPOCheckpointConfig:
    """
    Configures checkpointing for the orchestrator.

    Args:
        interval: Interval at which to save the checkpoint.
        resume_step: Step to resume orchestrator from. If None, will start from scratch. If -1, will restart from latest checkpoint available.
        wait_for_weights_timeout: When resuming, wait up to this many seconds for the weight directory to appear. Useful when the orchestrator restarts while the trainer is still saving weights. If None (default), fail immediately if weights are not found.
        keep_last: Keep at most this many recent step checkpoints on disk. If None, never clean old checkpoints based on recency.
        keep_interval: Keep checkpoints at every N steps permanently (e.g., keep_interval=100 keeps step 100, 200, ...). If None, no interval-based keeping.
        skip_progress: Whether to skip loading the progress from checkpoint.
        skip_buffer: Whether to skip loading the buffer from checkpoint.
    """

    interval: int | None = None
    resume_step: int | None = None
    wait_for_weights_timeout: int | None = None
    keep_last: int | None = None
    keep_interval: int | None = None
    skip_progress: bool | None = False
    skip_buffer: bool | None = False

    def __init__(self, cfg: DictDefault):
        self.interval = cfg.get("interval", self.interval)
        self.resume_step = cfg.get("resume_step", self.resume_step)
        self.wait_for_weights_timeout = cfg.get("wait_for_weights_timeout", self.wait_for_weights_timeout)
        self.keep_last = cfg.get("keep_last", self.keep_last)
        self.keep_interval = cfg.get("keep_interval", self.keep_interval)
        self.skip_progress = cfg.get("skip_progress", self.skip_progress)
        self.skip_buffer = cfg.get("skip_buffer", self.skip_buffer)


@dataclass
class GRPOValConfig:
    """
    Configures the validation of the model.

    Args:
        num_examples: Number of examples to use for validation. If -1, will use all examples.
        rollouts_per_example: Number of samples to generate per example for validation.
        interval: Interval at which to validate the model.
    """

    num_examples: int | None = 16
    rollouts_per_example: int | None = 1
    interval: int | None = 10

    def __init__(self, cfg: DictDefault):
        self.num_examples = cfg.get("num_examples", self.num_examples)
        self.rollouts_per_example = cfg.get("rollouts_per_example", self.rollouts_per_example)
        self.interval = cfg.get("interval", self.interval)


@dataclass
class FileSystemWeightBroadcastConfig:
    """Configures the filesystem weight broadcast."""

    type: Literal["filesystem"] | None = "filesystem"

    def __init__(self, cfg: DictDefault):
        self.type = cfg.get("type", self.type)


@dataclass
class NCCLWeightBroadcastConfig:
    """Configures the NCCL weight broadcast."""

    type: Literal["nccl"] | None = "nccl"
    host: str | None = "localhost"
    port: int | None = 29501
    timeout: int | None = 1200

    def __init__(self, cfg: DictDefault):
        self.type = cfg.get("type", self.type)
        self.host = cfg.get("host", self.host)
        self.port = cfg.get("port", self.port)
        self.timeout = cfg.get("timeout", self.timeout)


@dataclass
class ColocateWeightBroadcastConfig:
    """Configures colocate weight broadcast (zero-copy shared GPU memory)."""

    type: Literal["colocate"] | None = "colocate"

    def __init__(self, cfg: DictDefault):
        self.type = cfg.get("type", self.type)


@dataclass
class FileSystemTransportConfig:
    """Configures filesystem-based transport for training examples."""

    type: Literal["filesystem"] | None = "filesystem"

    def __init__(self, cfg: DictDefault):
        self.type = cfg.get("type", self.type)


@dataclass
class ZMQTransportConfig:
    """
    Configures ZMQ-based transport for training examples.

    Args:
        host: Hostname for ZMQ transport. Default is localhost.
        port: Port for ZMQ transport. Default is 5555.
        hwm: High-water mark (max messages in queue) for ZMQ transport. Default is 10.
    """

    type: Literal["zmq"] | None = "zmq"
    host: str | None = "localhost"
    port: int | None = 5555
    hwm: int | None = 10

    def __init__(self, cfg: DictDefault):
        self.type = cfg.get("type", self.type)
        self.host = cfg.get("host", self.host)
        self.port = cfg.get("port", self.port)
        self.hwm = cfg.get("hwm", self.hwm)


TransportConfigType: TypeAlias = FileSystemTransportConfig | ZMQTransportConfig


@dataclass
class GRPOOrchestratorConfig:
    """
    GRPO orchestrator configuration.

    Args:
        client: The OAI client configuration
        teacher_model: The teacher model configuration for computing teacher logprobs (e.g. for distillation). If provided, teacher logprobs will be computed using the specified model. If None, no teacher model will be used.
        learning_rate: Per-run learning rate for multi-run training.
        output_dir: Directory to write outputs to. Will be populated with checkpoints, weights, rollouts and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details.
        max_concurrent: Maximum number of concurrent rollouts to generate and score per-environment. If None, will not limit concurrency.
        tasks_per_minute: Rate limit for tasks per environment worker, in tasks per minute. Recommended for sandbox-backed environments to prevent sandbox-not-ready errors during autoscaling. When set to None, no rate limiting is applied. Note: with multiple workers, the effective total rate equals workers × this value.
        batch_size: Number of samples to train on per step (rollout-based batching). Set this OR token_batch_size.
        oversampling_factor: Factor by which to oversample the batch. Will lead to more in-flight group rollout requests at the same time. Default is 1.0 (no oversampling).
        rollouts_per_example: Number of output sequences to return per example during training.
        sequence_len: Sequence length to use for training. If a sample is shorter than this, it will be padded. If a sequence is longer than this, it will be truncated.
        num_train_workers: Number of training workers to use for training.
        max_steps: Maximum number of training steps to run. If None, will run indefinitely.
        max_off_policy_steps: Maximum number of policies that are allowed to generate a single rollout. Rollouts that are generated from more than `max_off_policy_steps` steps ahead of training will be discarded. Higher values yield better throughput, but lead to more off-policyness in training.
        max_async_level: Maximum number of steps the inference can be ahead of training. If 0, will degenerate to synchronous on-policy RL. If >=1, training and inference will be overlapped.
        strict_async_level: Whether to strictly enforce the max async level. If True, will always ensure that the policy used for generating rollouts is exactly `max_async_level` steps ahead of training. If False, any policy that is at most `max_async_level` steps ahead of training is allowed, i.e. we always use the latest available policy.
        bench: Whether to run in benchmark mode. It will automatically set the maximum number of steps to run to 5, max async level to ~infinity and disable reporting.
        seed: Random seed for the orchestrator.
        use_token_client: Whether to use the token-in-token-out (TITO) client for training across all environments. WARNING: Only use this if your environment has a linear history and the chat template has the extension property (i.e. no tokens are ever removed or inserted by the chat template)
        token_batch_size: Number of tokens to train on per step (token-based batching). Set this OR batch_size.
        max_inflight_rollouts: Maximum number of rollouts to keep in-flight. Required for token-based batching. If batch_size is set and this is unset, defaults to batch_size * oversampling_factor (or batch_size when oversampling_factor is unset).
        verification: Rollout verification configuration
        dump_metrics: Whether to dump metrics to '/tmp/grpo_metrics.jsonl' at each logging step.
    """

    client: GRPOClientConfig | None = None
    model: GRPOModelConfig | None = None
    teacher_model: TeacherModelConfig | None = None
    learning_rate: float | None = 1e-4
    sampling: GRPOSamplingConfig | None = None
    env: list[GRPOEnvConfig] | None = None
    eval: GRPOEvalConfig | None = None
    buffer: GRPOBufferConfig | None = None
    verification: GRPOVerificationConfig | None = None
    advantage: GRPOAdvantageConfig | GRPOCustomAdvantageConfig | None = None
    filters: list[GRPOGibberishFilterConfig | GRPORepetitionFilterConfig] | None = None
    log: GRPOLogConfig | None = None
    report_to: GRPOReportingConfig | None = None
    ckpt: GRPOCheckpointConfig | None = None
    val: GRPOValConfig | None = None
    weight_broadcast: (
        FileSystemWeightBroadcastConfig | NCCLWeightBroadcastConfig | ColocateWeightBroadcastConfig | None
    ) = None
    rollout_transport: FileSystemTransportConfig | ZMQTransportConfig | None = None
    output_dir: str | None = "outputs/run_default"
    max_concurrent: int | None = None
    tasks_per_minute: float | None = None
    batch_size: int | None = 128
    oversampling_factor: float | None = None
    rollouts_per_example: int | None = 1
    sequence_len: int | None = 2048
    num_train_workers: int | None = 1
    max_steps: int | None = None
    max_off_policy_steps: int | None = 8
    max_async_level: int | None = 1
    strict_async_level: bool | None = False
    bench: bool | None = False
    seed: int | None = 42
    use_token_client: bool | None = True
    token_batch_size: int | None = None
    max_inflight_rollouts: int | None = None
    dump_metrics: bool | None = False

    def __init__(self, cfg: DictDefault):
        self.client = GRPOClientConfig(cfg.get("client", {}))
        self.model = GRPOModelConfig(cfg.get("model", {}))

        if cfg.get("teacher_model") is not None:
            self.teacher_model = TeacherModelConfig(cfg.get("teacher_model"))

        self.learning_rate = cfg.get("learning_rate", self.learning_rate)
        self.sampling = GRPOSamplingConfig(cfg.get("sampling", {}))

        env_cfgs = cfg.get("env", [])
        self.env = [GRPOEnvConfig(env_cfg) for env_cfg in env_cfgs]

        if cfg.get("eval") is not None:
            self.eval = GRPOEvalConfig(cfg.get("eval"))

        self.buffer = GRPOBufferConfig(cfg.get("buffer", {}))

        if cfg.get("verification") is not None:
            self.verification = GRPOVerificationConfig(cfg.get("verification", {}))
        else:
            self.verification = GRPOVerificationConfig({})

        if cfg.get("advantage") is not None:
            if cfg.get("advantage").get("type") == "custom":
                self.advantage = GRPOCustomAdvantageConfig(cfg.get("advantage", {}))
            else:
                self.advantage = GRPOAdvantageConfig(cfg.get("advantage", {}))
        else:
            self.advantage = GRPOAdvantageConfig({})

        if cfg.get("filters") is not None:
            self.filters = []
            for filter_cfg in cfg.get("filters"):
                if filter_cfg.get("type") not in ["gibberish", "repetition"]:
                    raise ValueError(
                        f"Unsupported filter type: {filter_cfg.get('type')}. Supported types are 'gibberish' and 'repetition'."
                    )

                if filter_cfg.get("type") == "gibberish":
                    filter_config = GRPOGibberishFilterConfig(filter_cfg)
                else:
                    filter_config = GRPORepetitionFilterConfig(filter_cfg)

                self.filters.append(filter_config)
        else:
            self.filters = [GRPOGibberishFilterConfig({}), GRPORepetitionFilterConfig({})]

        self.log = GRPOLogConfig(cfg.get("log", {}))

        if cfg.get("report_to") is not None:
            self.report_to = GRPOReportingConfig(cfg.get("report_to", {}))

        if cfg.get("ckpt") is not None:
            self.ckpt = GRPOCheckpointConfig(cfg.get("ckpt"))

        if cfg.get("val") is not None:
            self.val = GRPOValConfig(cfg.get("val"))

        if cfg.get("weight_broadcast") is not None:
            wb_type = cfg.get("weight_broadcast").get("type")
            if wb_type == "nccl":
                self.weight_broadcast = NCCLWeightBroadcastConfig(cfg.get("weight_broadcast", {}))
            elif wb_type == "colocate":
                self.weight_broadcast = ColocateWeightBroadcastConfig(cfg.get("weight_broadcast", {}))
            else:
                self.weight_broadcast = FileSystemWeightBroadcastConfig(cfg.get("weight_broadcast", {}))
        else:
            self.weight_broadcast = FileSystemWeightBroadcastConfig({})

        if cfg.get("rollout_transport") is not None:
            if cfg.get("rollout_transport").get("type") == "zmq":
                self.rollout_transport = ZMQTransportConfig(cfg.get("rollout_transport", {}))
            else:
                self.rollout_transport = FileSystemTransportConfig(cfg.get("rollout_transport", {}))
        else:
            self.rollout_transport = FileSystemTransportConfig({})

        self.output_dir = Path(cfg.get("output_dir", self.output_dir))
        self.max_concurrent = cfg.get("max_concurrent", self.max_concurrent)
        self.tasks_per_minute = cfg.get("tasks_per_minute", self.tasks_per_minute)
        self.batch_size = cfg.get("batch_size", self.batch_size)
        self.oversampling_factor = cfg.get("oversampling_factor", self.oversampling_factor)
        self.rollouts_per_example = cfg.get("rollouts_per_example", self.rollouts_per_example)
        self.sequence_len = cfg.get("sequence_len", self.sequence_len)
        configured_num_train_workers = cfg.get("num_train_workers")

        if configured_num_train_workers is not None:
            self.num_train_workers = configured_num_train_workers
        elif self.rollout_transport.type == "zmq":
            rollout_transport_cfg = cfg.get("rollout_transport", {})
            zmq_connections = rollout_transport_cfg.get("connections")

            if isinstance(zmq_connections, int):
                self.num_train_workers = max(1, zmq_connections)
            elif isinstance(zmq_connections, (list, tuple, set)):
                self.num_train_workers = max(1, len(zmq_connections))
            elif isinstance(self.client.base_url, (list, tuple, set)):
                self.num_train_workers = max(1, len(self.client.base_url))
            else:
                self.num_train_workers = 1
        else:
            self.num_train_workers = self.num_train_workers

        self.max_steps = cfg.get("max_steps", self.max_steps)
        self.max_off_policy_steps = cfg.get("max_off_policy_steps", self.max_off_policy_steps)
        self.max_async_level = cfg.get("max_async_level", self.max_async_level)
        self.strict_async_level = cfg.get("strict_async_level", self.strict_async_level)
        self.bench = cfg.get("bench", self.bench)
        self.seed = cfg.get("seed", self.seed)
        self.use_token_client = cfg.get("use_token_client", self.use_token_client)
        self.token_batch_size = cfg.get("token_batch_size", self.token_batch_size)
        self.max_inflight_rollouts = cfg.get("max_inflight_rollouts", self.max_inflight_rollouts)

        self.dump_metrics = cfg.get("dump_metrics", self.dump_metrics)

        self.__post_init__()

    def __post_init__(self):
        types = [f.type for f in self.filters]
        if len(types) != len(set(types)):
            raise ValueError(f"Duplicate filter types: {types}. Each filter type may only appear once.")

        if self.max_concurrent is not None and self.max_concurrent < self.rollouts_per_example:
            raise ValueError("max_concurrent must be at least the number of rollouts per example")

        if self.weight_broadcast.type == "nccl":
            if not self.max_async_level == 1:
                raise ValueError("max_async_level must be 1 for NCCL broadcast")

        has_rollout_batch = self.batch_size is not None
        has_token_batch = self.token_batch_size is not None
        if has_rollout_batch and has_token_batch:
            raise ValueError("Set exactly one of batch_size or token_batch_size")

        if not has_rollout_batch and not has_token_batch:
            self.batch_size = 128

        if has_token_batch:
            if self.oversampling_factor is not None:
                raise ValueError("oversampling_factor can only be set when batch_size is set")

            if self.max_inflight_rollouts is None:
                raise ValueError("max_inflight_rollouts must be set when token_batch_size is set")
        else:
            assert self.batch_size is not None
            if self.batch_size % self.rollouts_per_example != 0:
                raise ValueError("Batch size must be divisible by the number of samples per problem")
            if self.max_inflight_rollouts is not None and self.oversampling_factor is not None:
                expected_max_inflight_rollouts = int(self.batch_size * self.oversampling_factor)
                if self.max_inflight_rollouts != expected_max_inflight_rollouts:
                    raise ValueError("max_inflight_rollouts conflicts with oversampling_factor * batch_size")
            if self.max_inflight_rollouts is None:
                oversampling_factor = self.oversampling_factor if self.oversampling_factor is not None else 1.0
                self.max_inflight_rollouts = int(self.batch_size * oversampling_factor)

        if self.max_inflight_rollouts is not None and self.max_inflight_rollouts < self.rollouts_per_example:
            raise ValueError("max_inflight_rollouts must be at least the number of rollouts per example")

        if self.buffer.env_ratios is not None:
            assert len(self.buffer.env_ratios) == len(self.env), "env_ratios length must match number of environments"

        if self.bench:
            self.max_steps = 4  # Run for 1 warmup step + 3 evaluation steps
            self.max_async_level = int(1e9)  # Never wait for RL weight checkpoints

            # Disable evaluation
            self.eval = None
            if self.report_to:
                self.report_to = None

        train_extra_env_kwargs = dict(
            max_seq_len=self.sequence_len,
            score_rollouts=not self.verification.enabled,
        )
        for env in self.env:
            # extra_env_kwargs is not meant to be used by the user, we shamelessly override here
            env.extra_env_kwargs.update(train_extra_env_kwargs)

        has_temp = self.sampling.temperature is not None
        has_scheduler = self.sampling.temp_scheduler is not None

        if has_temp and has_scheduler:
            raise ValueError("Set either sampling.temperature OR sampling.temp_scheduler, not both")

        # Default to temperature=1.0 if neither is set
        if not has_temp and not has_scheduler:
            self.sampling.temperature = 1.0

        if has_scheduler:
            scheduler = self.sampling.temp_scheduler
            if scheduler.total_steps is None and self.max_steps is None:
                raise ValueError("temp_scheduler.total_steps must be set when max_steps is None")

        if not self.verification and self.buffer.online_difficulty_filtering:
            raise ValueError(
                "verification.enabled cannot be False when buffer.online_difficulty_filtering is True. "
                "These features depend on rewards which are disabled when verification.enabled=False."
            )

        if not self.verification and self.buffer.easy_threshold is not None:
            raise ValueError(
                "verification.enabled cannot be False when buffer.easy_threshold is set. "
                "Easy threshold depends on rewards which are disabled when verification.enabled=False."
            )

        if not self.verification and self.buffer.hard_threshold is not None:
            raise ValueError(
                "verification.enabled cannot be False when buffer.hard_threshold is set. "
                "Hard threshold depends on rewards which are disabled when verification.enabled=False."
            )
