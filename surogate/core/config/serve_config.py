from dataclasses import dataclass
from typing import Literal, Optional

from surogate.utils.dict import DictDefault


@dataclass
class ServeConfig:
    """Configuration for native OpenAI-compatible inference serving.

    Parameters are grouped by purpose (server, runtime, generation defaults).
    Values can come from YAML config and/or CLI overrides.
    """

    # Server
    # Bind address for the HTTP server.
    host: str = "0.0.0.0"
    # TCP port used by the HTTP server.
    port: int = 8000
    # Uvicorn/server logging verbosity (for example: "debug", "info", "warning").
    log_level: str = "info"
    # Optional bearer token required by the API when set.
    api_key: Optional[str] = None

    # Model / runtime
    # Model identifier or local model path (required).
    model: Optional[str] = None
    # Optional public model id returned in API responses; defaults to `model` when unset.
    model_id: Optional[str] = None
    # Runtime compute dtype.
    dtype: Literal["bf16", "fp32"] = "bf16"
    # Number of GPUs to initialize for the runtime.
    gpus: int = 1
    # Maximum runtime batch capacity used by serving scheduler/generation.
    batch_size: int = 8
    # Runtime sequence length cap used to size model buffers.
    sequence_len: int = 4096
    # Whether Hugging Face remote code is trusted during model/tokenizer load.
    trust_remote_code: bool = True
    # Optional explicit floor for runtime stack arena size in MiB.
    min_stack_mb: Optional[int] = None
    # Target fraction of GPU memory budgeted for serving allocations.
    gpu_memory_utilization: float = 0.9
    # Enable MoE expert offloading (CPU/GPU streaming) when model/runtime supports it.
    offload_experts: bool = False

    # Generation defaults
    # Default maximum number of generated tokens per request (unless overridden by request).
    max_gen_len: int = 512
    # Default sampling temperature.
    temperature: float = 1.0
    # Default top-k sampling cutoff (0 means disabled).
    top_k: int = 0
    # Default nucleus sampling probability (top-p).
    top_p: float = 1.0
    # Default minimum probability sampling floor.
    min_p: float = 0.0
    # Default repetition penalty (> 0).
    repetition_penalty: float = 1.0
    # Enable CUDA graph execution in inference path.
    use_cuda_graphs: bool = True
    # Prefill chunk size for long-prompt chunked prefill (0 disables chunking).
    prefill_chunk_size: int = 256

    def __init__(self, cfg: DictDefault):
        self.host = cfg.get("host", self.host)
        self.port = int(cfg.get("port", self.port))
        self.log_level = cfg.get("log_level", self.log_level)
        self.api_key = cfg.get("api_key", self.api_key)

        self.model = cfg.get("model", self.model)
        self.model_id = cfg.get("model_id", self.model_id)
        self.dtype = cfg.get("dtype", self.dtype)
        self.gpus = int(cfg.get("gpus", self.gpus))
        self.batch_size = int(cfg.get("batch_size", self.batch_size))
        self.sequence_len = int(cfg.get("sequence_len", self.sequence_len))
        self.trust_remote_code = bool(
            cfg.get("trust_remote_code", self.trust_remote_code)
        )
        min_stack_raw = cfg.get("min_stack_mb", self.min_stack_mb)
        self.min_stack_mb = None if min_stack_raw is None else int(min_stack_raw)
        self.gpu_memory_utilization = float(
            cfg.get("gpu_memory_utilization", self.gpu_memory_utilization)
        )
        self.offload_experts = bool(cfg.get("offload_experts", self.offload_experts))

        self.max_gen_len = int(cfg.get("max_gen_len", self.max_gen_len))
        self.temperature = float(cfg.get("temperature", self.temperature))
        self.top_k = int(cfg.get("top_k", self.top_k))
        self.top_p = float(cfg.get("top_p", self.top_p))
        self.min_p = float(cfg.get("min_p", self.min_p))
        self.repetition_penalty = float(
            cfg.get("repetition_penalty", self.repetition_penalty)
        )
        self.use_cuda_graphs = bool(cfg.get("use_cuda_graphs", self.use_cuda_graphs))
        self.prefill_chunk_size = int(
            cfg.get("prefill_chunk_size", self.prefill_chunk_size)
        )

        self.__post_init__()

    def __post_init__(self):
        if not self.model:
            raise ValueError("ServeConfig: `model` is required")
        if self.dtype not in ("bf16", "fp32"):
            raise ValueError("ServeConfig: `dtype` must be one of: bf16, fp32")
        if self.port <= 0:
            raise ValueError("ServeConfig: `port` must be > 0")
        if self.sequence_len <= 0:
            raise ValueError("ServeConfig: `sequence_len` must be > 0")
        if self.max_gen_len <= 0:
            raise ValueError("ServeConfig: `max_gen_len` must be > 0")
        if self.gpu_memory_utilization <= 0.0 or self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "ServeConfig: `gpu_memory_utilization` must be in (0, 1]"
            )
        if self.min_stack_mb is not None and self.min_stack_mb <= 0:
            self.min_stack_mb = None
        if self.prefill_chunk_size < 0:
            self.prefill_chunk_size = 0
        if self.top_k < 0:
            self.top_k = 0
        if self.top_p <= 0.0:
            self.top_p = 1.0
        if self.min_p < 0.0:
            self.min_p = 0.0
        if self.repetition_penalty <= 0.0:
            self.repetition_penalty = 1.0
