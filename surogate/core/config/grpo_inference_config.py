from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Literal, Optional

from surogate.core.config.ct_config import DictDefault
from surogate.grpo.utils.utils import rgetattr, rsetattr
from surogate.utils.logger import get_logger

# TODO: Set thinking/ solution budget

logger = get_logger()


# Valid vLLM max_lora_rank values (from vllm/config/lora.py)
# TODO: on newer vLLM, can import via `get_args(vllm.config.lora.MaxLoRARanks)`
VALID_VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)

@dataclass
class GRPOInferenceConfig:
    """
    Configures GRPO inference.

    Args:
        host: The host to bind the inference server to. Passed to vLLM as `--host`. Default is None, which means binding to all interfaces.
        port: The port to bind the inference server to. Passed to vLLM as `--port`. Default is 8000.
        model: Name or path of the HF model to use.
        dtype: Data type for model weights and activations. Passed to vLLM as `--dtype`.
        max_model_len: Maximum model context length. Passed to vLLM as `--max-model-len`.
        enforce_eager: Whether to enforce eager mode. Passed to vLLM as `--enforce-eager`.
        trust_remote_code: Whether to trust remote code. Passed to vLLM engine init.
        enable_auto_tool_choice: Whether to enable auto tool choice. Passed to vLLM as `--enable-auto-tool-choice`.
        tool_call_parser: The tool call parser to use. Passed to vLLM as `--tool-call-parser`.
        reasoning_parser: Parser for extracting reasoning content. Passed to vLLM as `--reasoning-parser`.
        rope_scaling: RoPE scaling configuration. Passed to vLLM as `--rope-scaling`.
        tp: The tensor parallel size. Passed to vLLM as `--tensor-parallel-size`. Default is 1.
        dp: The data parallel size. Passed to vLLM as `--data-parallel-size`. Default is 1.
        enable_lora: Whether to enable LoRA. Passed to vLLM as `--enable-lora`. Default is True.
        max_loras: The maximum number of LoRAs to use. Passed to vLLM as `--max-loras`. Default is 8.
        max_cpu_loras: The maximum number of LoRAs to use on CPU. Passed to vLLM as `--max-cpu-loras`. Default is 100.
        max_lora_rank: The maximum LoRA rank to use. Passed to vLLM as `--max-lora-rank`. Default is None, which means no limit.
        enable_prefix_caching: Whether to enable prefix caching. Passed to vLLM as `--enable-prefix-caching`. Default is None, which means vLLM's default.
        gpu_memory_utilization: The GPU memory utilization to use. Passed to vLLM as `--gpu-memory-utilization`. Default is 0.9.
        api_server_count: The number of API servers to use. Passed to vLLM as `--api-server-count`. Default is 1, but will be automatically increased to match data parallel size if necessary.
        seed: The seed to use for inference. Passed to vLLM as `--seed`. Default is 0.
        weight_broadcast_type: The type of weight broadcast to use. Default is "filesystem".
    """

    # VLLM server configuration
    host: Optional[str] = None
    port: Optional[int] = 8000

    # Model configuration (formerly ModelConfig)
    model: Optional[str] = None
    dtype: Optional[Literal["auto", "float16", "bfloat16", "float32"]] = "auto"
    max_model_len: Optional[int] = None
    enforce_eager: Optional[bool] = False
    trust_remote_code: Optional[bool] = False
    enable_auto_tool_choice: Optional[bool] = False
    tool_call_parser: Optional[str] = "hermes"
    reasoning_parser: Optional[str] = None
    rope_scaling: Optional[dict[str, Any] | str] = None

    # Parallel configuration (formerly ParallelConfig)
    tp: Optional[int] = 1
    dp: Optional[int] = 1

    enable_lora: Optional[bool] = True
    max_loras: Optional[int] = 8
    # TODO: The default value is very high because our areal impl for lora isn't ideal
    # We add a lora with the same name instead of changing weights inplace
    # Because we dont cancel requests that are past max_async, these requests could be using a LoRA that gets unloaded which will crash the inference server
    max_cpu_loras: Optional[int] = 100
    max_lora_rank: Optional[int] = None
    enable_prefix_caching: Optional[bool] = None
    gpu_memory_utilization: Optional[float] = 0.9
    api_server_count: Optional[int] = 1
    seed: Optional[int] = 0
    weight_broadcast_type: Literal["nccl", "filesystem"] = "filesystem"

    def __init__(self, cfg: DictDefault):
        self.host = cfg.get("host", self.host)
        self.port = cfg.get("port", self.port)
        self.model = cfg.get("model", self.model)
        self.dtype = cfg.get("dtype", self.dtype)
        self.max_model_len = cfg.get("max_model_len", self.max_model_len)
        self.enforce_eager = cfg.get("enforce_eager", self.enforce_eager)
        self.trust_remote_code = cfg.get("trust_remote_code", self.trust_remote_code)
        self.enable_auto_tool_choice = cfg.get("enable_auto_tool_choice", self.enable_auto_tool_choice)
        self.tool_call_parser = cfg.get("tool_call_parser", self.tool_call_parser)
        self.reasoning_parser = cfg.get("reasoning_parser", self.reasoning_parser)
        self.rope_scaling = cfg.get("rope_scaling", self.rope_scaling)
        self.tp = cfg.get("tp", self.tp)
        self.dp = cfg.get("dp", self.dp)
        self.enable_lora = cfg.get("enable_lora", self.enable_lora)
        self.max_loras = cfg.get("max_loras", self.max_loras)
        self.max_cpu_loras = cfg.get("max_cpu_loras", self.max_cpu_loras)
        self.max_lora_rank = cfg.get("max_lora_rank", self.max_lora_rank)
        self.enable_prefix_caching = cfg.get("enable_prefix_caching", self.enable_prefix_caching)
        self.gpu_memory_utilization = cfg.get("gpu_memory_utilization", self.gpu_memory_utilization)
        self.api_server_count = cfg.get("api_server_count", self.api_server_count)
        self.seed = cfg.get("seed", self.seed)
        self.weight_broadcast_type = cfg.get("weight_broadcast_type", self.weight_broadcast_type)
        self.__post_init__()

    def __post_init__(self):
        if self.max_lora_rank is not None:
            original_rank = self.max_lora_rank
            for valid_rank in VALID_VLLM_LORA_RANKS:
                if valid_rank >= self.max_lora_rank:
                    self.max_lora_rank = valid_rank
                    break
            else:
                raise ValueError(f"max_lora_rank={original_rank} exceeds vLLM maximum of {VALID_VLLM_LORA_RANKS[-1]}")

        if self.api_server_count < self.dp:
            self.api_server_count = self.dp

        if self.enable_lora:
            self.api_server_count = 1  # LoRA requires only one API server

    def to_vllm(self) -> Namespace:
        """Convert InferenceConfig to vLLM-compatible Namespace."""
        namespace = Namespace()
        to_vllm = {
            "host": "host",
            "port": "port",
            "model": "model",
            "dtype": "dtype",
            "max_model_len": "max_model_len",
            "enforce_eager": "enforce_eager",
            "trust_remote_code": "trust_remote_code",
            "enable_auto_tool_choice": "enable_auto_tool_choice",
            "tool_call_parser": "tool_call_parser",
            "reasoning_parser": "reasoning_parser",
            "rope_scaling": "rope_scaling",
            "tp": "tensor_parallel_size",
            "dp": "data_parallel_size",
            "enable_lora": "enable_lora",
            "enable_prefix_caching": "enable_prefix_caching",
            "max_loras": "max_loras",
            "max_cpu_loras": "max_cpu_loras",
            "max_lora_rank": "max_lora_rank",
            "gpu_memory_utilization": "gpu_memory_utilization",
            "api_server_count": "api_server_count",
        }

        for key, vllm_key in to_vllm.items():
            value = getattr(self, key, None)
            rsetattr(namespace, vllm_key, value)

        # Set `logprobs_mode` to `processed_logprobs` by default
        rsetattr(namespace, "logprobs_mode", "processed_logprobs")

        # Remove reasoning_parser if not set (vLLM doesn't accept None)
        if namespace.reasoning_parser is None:
            delattr(namespace, "reasoning_parser")

        # Remove rope_scaling if not set (vLLM doesn't accept None)
        if hasattr(namespace, "rope_scaling"):
            if namespace.rope_scaling is None:
                delattr(namespace, "rope_scaling")
        
        rsetattr(namespace, "disable_uvicorn_access_log", "true")
        
        return namespace
