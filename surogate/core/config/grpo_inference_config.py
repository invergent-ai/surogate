from dataclasses import dataclass

from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class GRPOInferenceConfig:
    """
    Configures the GRPO rollout server: this repository's own serving engine, started by
    `surogate.grpo.inference.surogate_engine`, which turns these fields into engine flags.

    Args:
        host: Bind address. None binds every interface.
        port: Bind port.
        model: HuggingFace directory, hub id or GGUF the engine builds its artifact from.
        max_model_len: Maximum context length (`--max-model-len`).
        max_num_seqs: Concurrency cap (`--max-num-seqs`).
        decode_cache_bytes: Persistent cache byte budget on the shared training path;
            0 selects 25% of free VRAM after trainer allocation.
        kv_cache_dtype: KV cache dtype, e.g. `fp8` (`--kv-dtype`).
        tp: GPUs per replica. With `dp`, the number of GPUs a split run hands the server.
        dp: Replicas.
        enable_lora: Enable LoRA hot-loading, which is how the trainer's adapter reaches the
            server (`--enable-lora`).
        max_loras: Adapters held at once (`--max-loras`).
        max_lora_rank: Largest adapter rank the server accepts (`--max-lora-rank`).
        seed: Sampling seed (`--seed`).
    """

    host: str | None = None
    port: int | None = 8000
    model: str | None = None
    max_model_len: int | None = None
    # Concurrency cap. Sizes the activation buffers, so it is the knob that decides
    # whether a big model + LoRA fits. Lowering the KV budget does NOT help -- it
    # shrinks the very pool those buffers draw from.
    max_num_seqs: int | None = None
    # Native shared-model cache storage, including recurrent state and tables.
    # Zero selects 25% of free VRAM after allocating the resident trainer.
    decode_cache_bytes: int = 0
    # fp8 KV halves cache bytes/token, ~doubling concurrency on a KV-bound server. It
    # also perturbs sampled logprobs, which feed GRPO's importance ratio -- measure
    # mismatch_kl before adopting.
    kv_cache_dtype: str | None = None
    tp: int | None = 1
    dp: int | None = 1
    enable_lora: bool | None = True
    max_loras: int | None = 8
    max_lora_rank: int | None = None
    seed: int | None = 0

    def __init__(self, cfg: DictDefault):
        self.host = cfg.get("host", self.host)
        self.port = cfg.get("port", self.port)
        self.model = cfg.get("model", self.model)
        self.max_model_len = cfg.get("max_model_len", self.max_model_len)
        self.max_num_seqs = cfg.get("max_num_seqs", self.max_num_seqs)
        self.decode_cache_bytes = int(cfg.get("decode_cache_bytes", self.decode_cache_bytes))
        if self.decode_cache_bytes < 0:
            raise ValueError("decode_cache_bytes must be nonnegative")
        self.kv_cache_dtype = cfg.get("kv_cache_dtype", self.kv_cache_dtype)
        self.tp = cfg.get("tp", self.tp)
        self.dp = cfg.get("dp", self.dp)
        self.enable_lora = cfg.get("enable_lora", self.enable_lora)
        self.max_loras = cfg.get("max_loras", self.max_loras)
        self.max_lora_rank = cfg.get("max_lora_rank", self.max_lora_rank)
        self.seed = cfg.get("seed", self.seed)
