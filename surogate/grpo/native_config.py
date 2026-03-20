"""Configuration for native single-process GRPO training.

NativeGRPOConfig extends SFTConfig with generation, reward, evaluation,
and filtering settings. Replaces the 3-component system (vLLM inference +
orchestrator + trainer) with a single-process loop using the C++ engine's
built-in generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Any

from surogate.core.config.grpo_orch_config import (
    AdvantageConfigType,
    FilterConfigType,
    GRPOAdvantageConfig,
    GRPOCustomAdvantageConfig,
    GRPOEnvConfig,
    GRPOEvalEnvConfig,
    GRPOGibberishFilterConfig,
    GRPORepetitionFilterConfig,
    GRPOReportingConfig,
    GRPOSamplingConfig,
    GRPOTemperatureSchedulerConfig,
)
from surogate.core.config.sft_config import SFTConfig
from surogate.grpo.config import GRPOLossConfig
from surogate.utils.dict import DictDefault


@dataclass
class NativeGRPOGenerationConfig:
    """Generation parameters for native GRPO.

    Args:
        num_completions: Number of completions to generate per prompt.
        max_gen_len: Maximum number of tokens to generate per completion.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling. 0 means disabled.
        min_p: Minimum probability threshold for sampling.
    """

    num_completions: int = 4
    max_gen_len: int = 512
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0

    def __init__(self, cfg: DictDefault):
        self.num_completions = cfg.get("num_completions", self.num_completions)
        self.max_gen_len = cfg.get("max_gen_len", self.max_gen_len)
        self.top_p = cfg.get("top_p", self.top_p)
        self.top_k = cfg.get("top_k", self.top_k)
        self.min_p = cfg.get("min_p", self.min_p)


@dataclass
class NativeGRPORewardConfig:
    """Reward configuration — verifiers environments or a custom callback.

    Args:
        mode: "verifiers" to use vf.Environment scoring, or "callback" for a
            user-provided reward function.
        env: List of verifiers environment configs (mode="verifiers").
        reward_fn_import_path: Dotted import path to a reward function
            (mode="callback"). Signature: fn(prompts, completions) -> rewards.
        dataset_path: Path to a JSONL dataset of prompts (mode="callback").
            Each line must have a "prompt" field.
    """

    mode: Literal["verifiers", "callback"] = "verifiers"
    env: Optional[List[GRPOEnvConfig]] = None
    reward_fn_import_path: Optional[str] = None
    dataset_path: Optional[str] = None
    prompts: Optional[List[str]] = None
    multiturn: bool = False

    def __init__(self, cfg: DictDefault):
        self.mode = cfg.get("mode", self.mode)
        env_cfgs = cfg.get("env", [])
        self.env = [GRPOEnvConfig(DictDefault(e)) for e in env_cfgs] if env_cfgs else None
        self.reward_fn_import_path = cfg.get("reward_fn_import_path", self.reward_fn_import_path)
        self.dataset_path = cfg.get("dataset_path", self.dataset_path)
        self.prompts = cfg.get("prompts", self.prompts)
        self.multiturn = cfg.get("multiturn", self.multiturn)


@dataclass
class NativeGRPOEvalConfig:
    """Evaluation configuration for native GRPO.

    Args:
        env: List of verifiers environment configs for evaluation.
        num_examples: Number of examples to evaluate per environment.
        rollouts_per_example: Number of completions per example.
        interval: Evaluate every N training steps.
        temperature: Sampling temperature for evaluation (0.0 = greedy).
        max_gen_len: Max generation length for eval. If None, uses training config.
    """

    env: Optional[List[GRPOEvalEnvConfig]] = None
    num_examples: int = 100
    rollouts_per_example: int = 1
    interval: int = 10
    temperature: float = 0.0
    max_gen_len: Optional[int] = None

    def __init__(self, cfg: DictDefault):
        env_cfgs = cfg.get("env", [])
        self.env = [GRPOEvalEnvConfig(DictDefault(e)) for e in env_cfgs] if env_cfgs else None
        self.num_examples = cfg.get("num_examples", self.num_examples)
        self.rollouts_per_example = cfg.get("rollouts_per_example", self.rollouts_per_example)
        self.interval = cfg.get("interval", self.interval)
        self.temperature = cfg.get("temperature", self.temperature)
        self.max_gen_len = cfg.get("max_gen_len", self.max_gen_len)


def _parse_advantage_config(cfg: DictDefault) -> AdvantageConfigType | None:
    """Parse advantage config from YAML dict."""
    if not cfg:
        return GRPOAdvantageConfig(DictDefault({}))
    adv_type = cfg.get("type", "default")
    if adv_type == "custom":
        return GRPOCustomAdvantageConfig(cfg)
    return GRPOAdvantageConfig(cfg)


def _parse_filter_configs(cfg_list: list) -> list[FilterConfigType]:
    """Parse filter configs from YAML list."""
    filters = []
    for fc in cfg_list:
        fc = DictDefault(fc) if isinstance(fc, dict) else fc
        ftype = fc.get("type", "")
        if ftype == "gibberish":
            filters.append(GRPOGibberishFilterConfig(fc))
        elif ftype == "repetition":
            filters.append(GRPORepetitionFilterConfig(fc))
    return filters


@dataclass
class NativeGRPOConfig(SFTConfig):
    """Single-process GRPO trainer configuration.

    Extends SFTConfig with GRPO-specific settings for generation, reward,
    loss, advantage, evaluation, monitoring, and filtering.
    """

    # GRPO loss
    loss: Optional[GRPOLossConfig] = None

    # Generation
    generation: Optional[NativeGRPOGenerationConfig] = None

    # Sampling (temperature, temp_scheduler — reuse orchestrator's config)
    sampling: Optional[GRPOSamplingConfig] = None

    # Reward
    reward: Optional[NativeGRPORewardConfig] = None

    # Advantage
    advantage: Optional[AdvantageConfigType] = None

    # Teacher model for KL distillation (path to a HuggingFace model)
    teacher_model: Optional[str] = None

    # Monitoring
    report_to: Optional[GRPOReportingConfig] = None

    # Evaluation
    eval: Optional[NativeGRPOEvalConfig] = None

    # Filters
    filters: Optional[List[FilterConfigType]] = None

    # Batch
    problems_per_step: int = 8

    # Checkpointing
    save_steps: int = 0
    checkpoint_dir: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None

    def __init__(self, cfg: DictDefault):
        # GRPO-specific defaults (same as GRPOTrainConfig)
        if "master_dtype" not in cfg:
            cfg["master_dtype"] = "fp32"
        if "gradient_dtype" not in cfg:
            cfg["gradient_dtype"] = "fp32"
        cfg["sample_packing"] = "false"
        cfg["datasets"] = []

        super().__init__(cfg)

        # Parse nested configs
        loss_dict = cfg.get("loss", {})
        if isinstance(loss_dict, dict) and loss_dict:
            self.loss = GRPOLossConfig(**loss_dict)
        elif isinstance(loss_dict, GRPOLossConfig):
            self.loss = loss_dict
        else:
            self.loss = GRPOLossConfig()

        gen_dict = cfg.get("generation", {})
        self.generation = NativeGRPOGenerationConfig(DictDefault(gen_dict))

        sampling_dict = cfg.get("sampling", {})
        if sampling_dict:
            self.sampling = GRPOSamplingConfig(DictDefault(sampling_dict))
        else:
            # Default: constant temperature 1.0
            self.sampling = GRPOSamplingConfig(DictDefault({"temperature": 1.0}))

        reward_dict = cfg.get("reward", {})
        if reward_dict:
            self.reward = NativeGRPORewardConfig(DictDefault(reward_dict))
        else:
            self.reward = None

        self.teacher_model = cfg.get("teacher_model", self.teacher_model)

        adv_dict = cfg.get("advantage", {})
        self.advantage = _parse_advantage_config(DictDefault(adv_dict))

        report_dict = cfg.get("report_to", None)
        if report_dict is not None:
            self.report_to = GRPOReportingConfig(DictDefault(report_dict))
        else:
            self.report_to = None

        eval_dict = cfg.get("eval", None)
        if eval_dict is not None:
            self.eval = NativeGRPOEvalConfig(DictDefault(eval_dict))
        else:
            self.eval = None

        filter_list = cfg.get("filters", [])
        self.filters = _parse_filter_configs(filter_list) if filter_list else []

        self.problems_per_step = cfg.get("problems_per_step", self.problems_per_step)
        self.save_steps = cfg.get("save_steps", self.save_steps)
        self.checkpoint_dir = cfg.get("checkpoint_dir", self.checkpoint_dir)
        self.resume_from_checkpoint = cfg.get("resume_from_checkpoint", self.resume_from_checkpoint)

        # Initialize inherited config
        self.__post_init__()

        # Disable CUDA graphs (same as GRPOTrainConfig)
        if self.use_cuda_graphs:
            self.use_cuda_graphs = False
            self.runtime_config.use_cuda_graphs = False
