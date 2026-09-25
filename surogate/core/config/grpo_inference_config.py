from dataclasses import dataclass
from typing import Literal

from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()


SERVING_OFFLOAD_FIELDS = (
    "gpu_layers",
    "host_moe_layers",
    "expert_slots",
    "host_expert_bank",
    "cpu_moe_share",
    "cpu_moe_prefill_share",
    "cpu_moe_min_tokens",
)


def _offload_count(name, value, choices=()):
    if value is None or (isinstance(value, str) and value in choices):
        return value
    if isinstance(value, str) and value.isdecimal():
        value = int(value)
    # Match the serving CLI's signed 32-bit count range without truncating floats
    # or treating YAML booleans as layer counts.
    if type(value) is int and 0 <= value <= 2**31 - 1:
        return value
    alternatives = "".join(f" or '{choice}'" for choice in choices)
    raise ValueError(f"{name} must be an integer between 0 and {2**31 - 1}{alternatives}")


def _offload_share(name, value, *, allow_auto=False):
    if value is None or (allow_auto and isinstance(value, str) and value == "auto"):
        return value
    if type(value) in (int, float, str):
        try:
            share = float(value)
            if 0 <= share <= 1:
                return share
        except (ValueError, OverflowError):
            pass
    alternative = " or 'auto'" if allow_auto else ""
    raise ValueError(f"{name} must be a finite number between 0 and 1{alternative}")


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
        enable_auto_tool_choice: Permit ``tool_choice: "auto"``. Required for a
            tool-using environment, whose loop reads structured ``tool_calls``
            back off the assistant message. Off by default.
        tool_call_parser: How the model marks a tool call. Takes vLLM's names
            (``hermes`` and ``qwen3_xml`` are the same format). Unset leaves the
            engine's own default.
        max_num_seqs: Concurrency cap (`--max-num-seqs`).
        max_pending_requests: How many requests past `max_num_seqs` the engine
            will hold rather than refuse (`--max-pending-requests`). The two
            together are its admission bound, and a request past them gets a
            429. Rollout clients give up after ten of those, so a run whose
            in-flight rollouts exceed the bound loses whole rollouts; holding
            them instead costs a worker thread each and a wait. `grpo/utils/capacity.py`
            derives a default from the orchestrator's in-flight count, called from
            `split.py`, the only place both numbers are known. Unset leaves the engine's own
            default of 16.
        pending_timeout_ms: How long a held request may wait before the engine
            gives up on it (`--pending-timeout-ms`). Sized just BELOW the rollout
            client's own timeout: past that deadline nobody is waiting for the
            answer, while the held request still occupies its admission slot and
            worker, so the caller's retry queues behind a request it has already
            abandoned. Unset leaves the engine's default of 30000.
        adapter_update_timeout_ms: How long a weight update may wait for the
            adapter to stop being read (`--adapter-update-timeout-ms`). A request
            claims the adapter when admitted rather than when it starts
            generating, so this bounds draining the admitted backlog, not one
            generation. It must exceed `pending_timeout_ms`, and the admin
            client's own timeout must exceed it, so the engine's 503 naming the
            adapter wins the race against an opaque client timeout. Unset leaves
            the engine inheriting `pending_timeout_ms`, which is what it did for
            both jobs before this existed.
        gpu_layers: Decoder layers kept on GPU; 0 offloads all, 'all' keeps all resident.
        host_moe_layers: MoE layers with experts in CPU RAM; a count, 'auto', or 'all'.
        expert_slots: GPU expert-cache slots for offloaded MoE models.
        host_expert_bank: CPU expert storage format: 'auto', 'w8', or 'q4'.
        cpu_moe_share: Fraction of decode expert cache misses computed on CPU, or 'auto'.
        cpu_moe_prefill_share: Fraction of prefill expert work computed on CPU.
        cpu_moe_min_tokens: Minimum token count for CPU expert computation.
        decode_cache_bytes: Persistent cache byte budget on the shared training path;
            0 selects 25% of free VRAM after trainer allocation.
        decode_memory_bytes: Combined decode cache/workspace budget on the shared
            training path; 0 selects 80% of free VRAM after trainer allocation.
        decode_prefill_chunk: Maximum prefill tokens per scheduler round on the
            shared training path; reduced automatically during decode and memory pressure.
        decode_prefix_entries: Maximum reusable prompt snapshots on that path; 0 disables caching.
        kv_cache_dtype: KV cache dtype, e.g. `fp8` (`--kv-dtype`).
        tp: GPUs per replica. With `dp`, the number of GPUs a split run hands the server.
        dp: Replicas.
        enable_lora: Enable LoRA hot-loading, which is how the trainer's adapter reaches the
            server (`--enable-lora`).
        max_loras: Adapters held at once (`--max-loras`).
        max_lora_rank: Largest adapter rank the server accepts (`--max-lora-rank`).
        seed: Sampling seed (`--seed`). **Leave unset for GRPO.** The engine
            applies it as a process-level override on every request, so a set
            seed makes each rollout in a group sample identically. Identical
            rollouts score identically, and both advantage functions return
            exactly zero for a group with no reward spread, so training receives
            no gradient at all. Unset, the engine draws a fresh seed per
            request, which is what rollouts need and what colocate already does.
            Set it only to make a server reproducible, never to train.
    """

    host: str | None = None
    port: int | None = 8000
    model: str | None = None
    max_model_len: int | None = None
    # Concurrency cap. Sizes the activation buffers, so it is the knob that decides
    # whether a big model + LoRA fits. Lowering the KV budget does NOT help -- it
    # shrinks the very pool those buffers draw from.
    max_num_seqs: int | None = None
    max_pending_requests: int | None = None
    pending_timeout_ms: int | None = None
    adapter_update_timeout_ms: int | None = None
    # These controls apply to the separate serving engine. None preserves its defaults.
    gpu_layers: int | Literal["all"] | None = None
    host_moe_layers: int | Literal["auto", "all"] | None = None
    expert_slots: int | None = None
    host_expert_bank: Literal["auto", "w8", "q4"] | None = None
    cpu_moe_share: float | Literal["auto"] | None = None
    cpu_moe_prefill_share: float | None = None
    cpu_moe_min_tokens: int | None = None
    # Native shared-model cache storage, including recurrent state and tables.
    # Zero selects 25% of free VRAM after allocating the resident trainer.
    decode_cache_bytes: int = 0
    # Incremental decode cache + workspace budget; 0 chooses 80% of free VRAM.
    decode_memory_bytes: int = 0
    decode_prefill_chunk: int = 256
    decode_prefix_entries: int = 32
    # fp8 KV halves cache bytes/token, ~doubling concurrency on a KV-bound server. It
    # also perturbs sampled logprobs, which feed GRPO's importance ratio -- measure
    # mismatch_kl before adopting.
    kv_cache_dtype: str | None = None
    # Tool calling, the two flags the engine takes for it.
    #
    # A verifiers ToolEnv reads structured ``tool_calls`` back off the assistant
    # message to decide whether its loop is done, and the engine only fills that
    # field for a ``tool_choice: auto`` request when the gate is open -- without
    # it the request is refused and every rollout of such an environment fails.
    #
    # Off by default, and only the environments that need it ask: with the gate
    # open the engine parses a tool call out of every completion, which a run
    # that never calls one has no reason to carry. ``tool_call_parser`` is left
    # None so the engine's own default (``qwen3_xml``) stands unless a caller
    # names one; it takes vLLM's names, so ``hermes`` is the same format.
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = None
    tp: int | None = 1
    dp: int | None = 1
    enable_lora: bool | None = True
    max_loras: int | None = 8
    max_lora_rank: int | None = None
    seed: int | None = None

    def __init__(self, cfg: DictDefault):
        self.host = cfg.get("host", self.host)
        self.port = cfg.get("port", self.port)
        self.model = cfg.get("model", self.model)
        self.max_model_len = cfg.get("max_model_len", self.max_model_len)
        self.max_num_seqs = cfg.get("max_num_seqs", self.max_num_seqs)
        self.max_pending_requests = cfg.get("max_pending_requests", self.max_pending_requests)
        self.pending_timeout_ms = cfg.get("pending_timeout_ms", self.pending_timeout_ms)
        self.adapter_update_timeout_ms = cfg.get("adapter_update_timeout_ms", self.adapter_update_timeout_ms)
        self.enable_auto_tool_choice = cfg.get("enable_auto_tool_choice", self.enable_auto_tool_choice)
        self.tool_call_parser = cfg.get("tool_call_parser", self.tool_call_parser)
        # The engine refuses this pair at startup, and it does so after execv,
        # so the run would show a server that never turns healthy rather than
        # the engine's own message. Only the two "off" spellings are checked:
        # the parser registry grows between releases, so a copy of it here
        # would go stale and reject a name the engine accepts.
        if self.enable_auto_tool_choice and self.tool_call_parser in ("none", "off"):
            raise ValueError(
                f"enable_auto_tool_choice needs a tool_call_parser; {self.tool_call_parser!r} turns parsing off"
            )
        self.gpu_layers = _offload_count("gpu_layers", cfg.get("gpu_layers"), ("all",))
        self.host_moe_layers = _offload_count("host_moe_layers", cfg.get("host_moe_layers"), ("auto", "all"))
        self.expert_slots = _offload_count("expert_slots", cfg.get("expert_slots"))
        self.cpu_moe_min_tokens = _offload_count("cpu_moe_min_tokens", cfg.get("cpu_moe_min_tokens"))
        self.host_expert_bank = cfg.get("host_expert_bank")
        if self.host_expert_bank is not None and self.host_expert_bank not in ("auto", "w8", "q4"):
            raise ValueError("host_expert_bank must be 'auto', 'w8', or 'q4'")
        self.cpu_moe_share = _offload_share("cpu_moe_share", cfg.get("cpu_moe_share"), allow_auto=True)
        self.cpu_moe_prefill_share = _offload_share("cpu_moe_prefill_share", cfg.get("cpu_moe_prefill_share"))
        self.decode_cache_bytes = int(cfg.get("decode_cache_bytes", self.decode_cache_bytes))
        if self.decode_cache_bytes < 0:
            raise ValueError("decode_cache_bytes must be nonnegative")
        self.decode_memory_bytes = int(cfg.get("decode_memory_bytes", self.decode_memory_bytes))
        if self.decode_memory_bytes < 0:
            raise ValueError("decode_memory_bytes must be nonnegative")
        self.decode_prefill_chunk = int(cfg.get("decode_prefill_chunk", self.decode_prefill_chunk))
        self.decode_prefix_entries = int(cfg.get("decode_prefix_entries", self.decode_prefix_entries))
        if self.decode_prefill_chunk <= 0 or self.decode_prefix_entries < 0:
            raise ValueError("decode_prefill_chunk must be positive and decode_prefix_entries nonnegative")
        # The engine refuses 0 for both, and it does so after execv, so the run
        # shows a server that never turns healthy instead of the reason.
        if self.max_pending_requests is not None and self.max_pending_requests <= 0:
            raise ValueError("max_pending_requests must be positive")
        if self.pending_timeout_ms is not None and self.pending_timeout_ms <= 0:
            raise ValueError("pending_timeout_ms must be positive")
        if self.adapter_update_timeout_ms is not None and self.adapter_update_timeout_ms <= 0:
            raise ValueError("adapter_update_timeout_ms must be positive")
        if (
            self.adapter_update_timeout_ms is not None
            and self.pending_timeout_ms is not None
            and self.adapter_update_timeout_ms <= self.pending_timeout_ms
        ):
            # A request may sit pending for the whole hold, so an update that
            # gives up sooner aborts runs that were only busy.
            raise ValueError(
                f"adapter_update_timeout_ms ({self.adapter_update_timeout_ms}) must exceed "
                f"pending_timeout_ms ({self.pending_timeout_ms}): a pending request holds the "
                f"adapter claim a weight update waits on"
            )
        self.kv_cache_dtype = cfg.get("kv_cache_dtype", self.kv_cache_dtype)
        self.tp = cfg.get("tp", self.tp)
        self.dp = cfg.get("dp", self.dp)
        self.enable_lora = cfg.get("enable_lora", self.enable_lora)
        self.max_loras = cfg.get("max_loras", self.max_loras)
        self.max_lora_rank = cfg.get("max_lora_rank", self.max_lora_rank)
        self.seed = cfg.get("seed", self.seed)
