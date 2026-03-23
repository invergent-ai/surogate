from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer

from surogate import _surogate
from surogate.core.config.serve_config import ServeConfig
from surogate.core.model.registry import ModelInfo
from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.kernels.jit_compile import compile_jit_kernels
from surogate.utils.hf import get_model_weights_path
from surogate.utils.logger import get_logger

logger = get_logger()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer", "tool"]
    content: Any


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    stream: bool = False
    stop: Optional[str | list[str]] = None
    stream_options: Optional[StreamOptions] = None

    # Non-standard but useful for parity with existing internal generation knobs.
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None


class CompletionsRequest(BaseModel):
    model: Optional[str] = None
    prompt: str | list[str]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    stream: bool = False
    stop: Optional[str | list[str]] = None
    stream_options: Optional[StreamOptions] = None

    # Non-standard sampling knobs.
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None


@dataclass
class GeneratedChoice:
    index: int
    text: str
    token_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class GenerationParams:
    n: int
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    repetition_penalty: float


@dataclass
class PendingGeneration:
    prompt_ids: list[int]
    stop_strings: list[str]
    params: GenerationParams
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[list[GeneratedChoice]] = None
    error: Optional[Exception] = None


@dataclass
class ActiveGeneration:
    pending: PendingGeneration
    generated_ids: list[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: str = "stop"
    done_signaled: bool = False
    is_padding: bool = False


class ContextLengthExceededError(ValueError):
    def __init__(self, message: str, *, param: str):
        super().__init__(message)
        self.param = param


def _normalize_stop(stop: Optional[str | list[str]]) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [s for s in stop if isinstance(s, str) and s]


def _apply_stop(text: str, stop_strings: Sequence[str]) -> tuple[str, bool]:
    if not stop_strings:
        return text, False
    min_pos = -1
    for s in stop_strings:
        p = text.find(s)
        if p >= 0 and (min_pos < 0 or p < min_pos):
            min_pos = p
    if min_pos >= 0:
        return text[:min_pos], True
    return text, False


def _sse(data: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def _openai_context_length_error(message: str, param: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": "context_length_exceeded",
            }
        },
    )


def _token_text_chunks(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    chunks: list[str] = []
    for tok in token_ids:
        piece = tokenizer.decode([int(tok)], skip_special_tokens=True)
        if piece:
            chunks.append(piece)
    return chunks


class NativeServingRuntime:
    def __init__(self, config: ServeConfig):
        self.config = config
        self.model_id = config.model_id or config.model or "native"

        model_dir = self._resolve_model_dir(config.model or "")
        self._apply_model_generation_config_defaults(model_dir, config)
        qlora_config, prequant_method, is_moe_model, model_max_len = (
            self._resolve_prequant_qlora_config(model_dir)
        )
        self._is_moe_model = bool(is_moe_model)

        logger.info(f"Loading tokenizer for {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=config.trust_remote_code
        )
        self.eos_id = self.tokenizer.eos_token_id or 151643

        ir_json = build_dsl_ir_for_model(model_dir)
        jit_manifests = compile_jit_kernels(ir_json)
        requested_batch_size = max(1, int(config.max_num_seqs))
        requested_seq_len = self._resolve_effective_max_model_len(
            config,
            model_max_len=model_max_len,
            tokenizer=self.tokenizer,
        )
        trainer_seq_len = self._resolve_runtime_seq_len_budget(
            config,
            requested_seq_len=requested_seq_len,
            requested_batch_size=requested_batch_size,
        )
        self._max_context_len = int(requested_seq_len)

        # Prequant NVFP4 on Blackwell is most stable/fast with cuDNN FP4 backend.
        # Allow override via env for A/B testing.
        serve_fp4_backend: Optional[str] = None
        if prequant_method == "prequant_nvfp4":
            fp4_backend = os.getenv("SUROGATE_SERVE_FP4_BACKEND", "cudnn").strip().lower()
            if fp4_backend not in {"cudnn", "cutlass"}:
                fp4_backend = "cudnn"
            serve_fp4_backend = fp4_backend
            options = _surogate.RuntimeOptions(
                fp4_backend=fp4_backend,
                fp4_enable_four_over_six=False,
            )
        else:
            options = _surogate.RuntimeOptions()
        options.dsl_ir_json = ir_json
        if jit_manifests:
            options.jit_kernel_manifests = jit_manifests
        prequant_recipe_env = os.getenv("SUROGATE_SERVE_PREQUANT_RECIPE", "").strip()
        if prequant_recipe_env:
            prequant_recipe_enabled = prequant_recipe_env != "0"
            prequant_recipe_auto = False
        else:
            prequant_recipe_enabled = True
            prequant_recipe_auto = True
            # Default to BF16 runtime for prequant serving.
            # Enable quant runtime recipe only via explicit override:
            #   SUROGATE_SERVE_PREQUANT_RECIPE=1
            if prequant_method in {"prequant_nvfp4", "prequant_fp8"}:
                prequant_recipe_enabled = False
        if prequant_recipe_enabled:
            if prequant_method == "prequant_nvfp4":
                options.set_recipe("nvfp4")
                logger.info(
                    "Serve runtime recipe set to nvfp4 for prequant NVFP4 model (fp4_backend=%s, four_over_six=false).",
                    serve_fp4_backend or "cudnn",
                )
            elif prequant_method == "prequant_fp8":
                options.set_recipe("fp8-hybrid")
                logger.info("Serve runtime recipe set to fp8-hybrid for prequant FP8 model.")
        elif prequant_method is not None:
            if prequant_recipe_auto:
                logger.info(
                    "Serve runtime recipe disabled for prequant model (prequant_method=%s). "
                    "Set SUROGATE_SERVE_PREQUANT_RECIPE=1 to force quant recipe.",
                    prequant_method,
                )
            else:
                logger.info(
                    "Prequant recipe override disabled by SUROGATE_SERVE_PREQUANT_RECIPE=0; using bf16 recipe."
                )
        # Auto-enable FP8 KV-cache for prequant FP8 models (halves KV memory).
        if (
            prequant_method == "prequant_fp8"
            and not os.getenv("SUROGATE_ENABLE_FP8_KV_CACHE")
        ):
            os.environ["SUROGATE_ENABLE_FP8_KV_CACHE"] = "1"
            logger.info(
                "Auto-enabled FP8 KV-cache for prequant FP8 model. "
                "Set SUROGATE_ENABLE_FP8_KV_CACHE=0 to disable."
            )

        options.use_cuda_graphs = bool(config.use_cuda_graphs)
        if (
            options.use_cuda_graphs
            and prequant_method in {"prequant_nvfp4", "prequant_fp8", "prequant_mxfp4"}
            and os.getenv("SUROGATE_SERVE_FULL_CUDA_GRAPH_MODE") is None
        ):
            os.environ["SUROGATE_SERVE_FULL_CUDA_GRAPH_MODE"] = "0"
            logger.info(
                "Defaulting SUROGATE_SERVE_FULL_CUDA_GRAPH_MODE=0 for prequant serving "
                "with CUDA graphs (decode graphs stay enabled). Set "
                "SUROGATE_SERVE_FULL_CUDA_GRAPH_MODE=1 to force full-step capture."
            )
        options.doc_masking = True
        options.recompute = "true"
        if hasattr(options, "selective_expert_dequant"):
            options.selective_expert_dequant = True
        self._offload_experts = bool(config.offload_experts)
        if hasattr(options, "offload_experts"):
            options.offload_experts = self._offload_experts
        elif self._offload_experts:
            logger.warning(
                "offload_experts requested, but runtime binding does not expose this option."
            )
        # Serve uses generate-only paths. Keep training-specific buffers lightweight.
        options.offload_grads = True
        options.lmhead_chunks = self._resolve_lmhead_chunks(
            config,
            effective_seq_len=requested_seq_len,
            max_num_seqs=requested_batch_size,
        )
        options.min_stack_mb = self._resolve_min_stack_mb(config)
        logger.info(
            "Serving runtime stack floor=%d MiB lmhead_chunks=%d offload_grads=%s (gpu_memory_utilization=%.2f)",
            int(options.min_stack_mb),
            int(options.lmhead_chunks),
            bool(options.offload_grads),
            float(config.gpu_memory_utilization),
        )

        model_weights_path = get_model_weights_path(model_dir)
        (
            self._runtime_batch_size,
            self._trainer_seq_len,
            self.trainer,
        ) = self._build_trainer(
            model_dir=model_dir,
            config=config,
            options=options,
            qlora_config=qlora_config,
            model_weights_path=model_weights_path,
            batch_size=requested_batch_size,
            seq_len=trainer_seq_len,
        )
        self._runtime_seq_len = int(self._max_context_len)
        self._log_fp8_kv_cache_default(config.gpus)
        logger.info(
            "Serve startup capacity: max_num_seqs=%d max_model_len=%d runtime_seq_len=%d",
            self._runtime_batch_size,
            self._runtime_seq_len,
            self._trainer_seq_len,
        )
        self._pending_lock = threading.Lock()
        self._pending_cv = threading.Condition(self._pending_lock)
        self._pending: list[PendingGeneration] = []
        self._shutdown = False
        self._continuous_step_tokens = max(
            1,
            int(os.getenv("SUROGATE_SERVE_CONTINUOUS_STEP_TOKENS", "32")),
        )
        self._continuous_idle_step_tokens = max(
            self._continuous_step_tokens,
            int(os.getenv("SUROGATE_SERVE_CONTINUOUS_IDLE_STEP_TOKENS", "256")),
        )
        if self._is_moe_model:
            default_moe_cap = "64" if self._offload_experts else "32"
            moe_step_cap = max(
                1,
                int(os.getenv("SUROGATE_SERVE_MOE_MAX_STEP_TOKENS", default_moe_cap)),
            )
            self._continuous_step_tokens = min(self._continuous_step_tokens, moe_step_cap)
            self._continuous_idle_step_tokens = min(
                self._continuous_idle_step_tokens, moe_step_cap
            )
            logger.info(
                "Applied MoE decode-step cap=%d tokens (offload_experts=%s).",
                moe_step_cap,
                bool(self._offload_experts),
            )
        configured_cap = max(1, int(self._runtime_batch_size))
        env_cap = int(os.getenv("SUROGATE_SERVE_MAX_BATCH_SEQUENCES", str(configured_cap)))
        self._max_batch_sequences = max(1, min(configured_cap, env_cap))
        if self._max_batch_sequences <= 1:
            logger.warning(
                "Serve runtime running with max_num_seqs=1; concurrent requests will be serialized. "
                "Set --max_num_seqs>1 to improve throughput."
            )
        self._batch_worker = threading.Thread(
            target=self._continuous_engine_loop,
            name="surogate-continuous-engine",
            daemon=True,
        )
        self._batch_worker.start()
        logger.info(
            "Native inference runtime ready (step_tokens=%d idle_step_tokens=%d)",
            int(self._continuous_step_tokens),
            int(self._continuous_idle_step_tokens),
        )

    @staticmethod
    def _is_cuda_oom_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return (
            "cuda oom" in msg
            or "out of memory" in msg
            or "cuda_error_out_of_memory" in msg
        )

    @staticmethod
    def _log_fp8_kv_cache_default(ngpu: int) -> None:
        try:
            enable_env = os.getenv("SUROGATE_ENABLE_FP8_KV_CACHE", "").strip()
            if (not enable_env) or enable_env.lower() in {"0", "false"}:
                return
            infos = _surogate.SystemInfo.get_gpu_info()
            if not infos:
                return
            use_count = max(1, int(ngpu))
            selected = infos[:use_count]
            if not selected:
                return
            sms = [
                int(info.compute_capability_major) * 10
                + int(info.compute_capability_minor)
                for info in selected
            ]
            if all(sm >= 89 for sm in sms):
                sm_list = ", ".join(f"SM{sm}" for sm in sms)
                logger.info("FP8 KV-cache enabled (%s).", sm_list)
        except Exception:
            # Best-effort log only.
            pass

    def _build_trainer(
        self,
        *,
        model_dir: str,
        config: ServeConfig,
        options: Any,
        qlora_config: Any,
        model_weights_path: str,
        batch_size: int,
        seq_len: int,
    ) -> tuple[int, int, Any]:
        batch_size = max(1, int(batch_size))
        seq_len = max(1, int(seq_len))
        options.lmhead_chunks = self._resolve_lmhead_chunks(
            config,
            effective_seq_len=seq_len,
            max_num_seqs=batch_size,
        )
        logger.info(
            "Serve runtime init with batch_size=%d seq_len=%d lmhead_chunks=%d",
            batch_size,
            seq_len,
            int(options.lmhead_chunks),
        )
        trainer = None
        try:
            trainer = _surogate.SurogateTrainer(
                ngpu=config.gpus,
                config=_surogate.PretrainedConfig.from_pretrained(model_dir, config.dtype),
                options=options,
                batch_size=batch_size,
                seq_len=seq_len,
                grad_accum=1,
                memcpy_all_gather=True,
                memcpy_send_recv=True,
                qlora_config=qlora_config,
            )
            logger.info(f"Importing weights from {model_weights_path}")
            trainer.import_weights(model_weights_path)
            return batch_size, seq_len, trainer
        except RuntimeError as exc:
            if trainer is not None:
                del trainer
            if self._is_cuda_oom_error(exc):
                raise RuntimeError(
                    "Serve startup failed due to CUDA OOM "
                    f"(batch_size={batch_size}, seq_len={seq_len}). "
                ) from exc
            raise

    @staticmethod
    def _resolve_min_stack_mb(config: ServeConfig) -> int:
        explicit_min = config.min_stack_mb
        if explicit_min is not None:
            return max(64, int(explicit_min))

        # For inference we keep the stack floor small by default.
        # gpu_memory_utilization controls overall serving budget; stack floor should
        # be a small slice of reserved headroom, not multi-GB.
        auto_min_mb = 512
        try:
            import torch

            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info(0)
                mb = 1024 * 1024
                free_mb = int(free_b // mb)
                total_mb = int(total_b // mb)
                util = float(config.gpu_memory_utilization)
                reserve_mb = max(0.0, (1.0 - util) * float(total_mb))
                # Keep stack floor in a tight inference range [256, 1024] MiB.
                by_reserve = int(reserve_mb * 0.05)  # 5% of reserved headroom
                by_free = int(max(256, free_mb * 0.10))  # never exceed 10% of currently free memory
                auto_min_mb = max(256, min(1024, by_reserve, by_free))
                logger.info(
                    "Auto min_stack_mb=%d MiB from free=%d MiB total=%d MiB reserve=%d MiB gpu_memory_utilization=%.2f",
                    auto_min_mb,
                    free_mb,
                    total_mb,
                    int(reserve_mb),
                    float(config.gpu_memory_utilization),
                )
        except Exception as exc:
            logger.warning(
                "Falling back to default min_stack_mb=%d MiB: %s",
                auto_min_mb,
                exc,
            )
        return int(auto_min_mb)

    @staticmethod
    def _normalize_context_len(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            normalized = int(value)
        except (TypeError, ValueError):
            return None
        if normalized <= 0:
            return None
        # Ignore sentinel "infinite/unknown" tokenizer limits.
        if normalized >= int(1e9):
            return None
        return normalized

    @staticmethod
    def _resolve_prequant_qlora_config(model_dir: str):
        """Auto-detect pre-quantized models and return QLoRA + model metadata."""
        try:
            model_info = ModelInfo.create(model_dir)
            quant_info = model_info.quant_info or {}
            quant_method = (model_info.quant_method or "").lower()
            modules_to_not_convert = quant_info.get("modules_to_not_convert") or []
            model_max_len = NativeServingRuntime._normalize_context_len(model_info.max_model_len)

            if quant_method == "prequant_fp8":
                qlora = _surogate.QLoRAConfig.prequant_fp8()
            elif quant_method == "prequant_nvfp4":
                qlora = _surogate.QLoRAConfig.prequant_nvfp4()
            elif quant_method == "prequant_mxfp4":
                qlora = _surogate.QLoRAConfig.prequant_mxfp4()
            else:
                return None, None, bool(model_info.is_moe_model), model_max_len

            if modules_to_not_convert:
                qlora.modules_to_not_convert = list(modules_to_not_convert)
            logger.info(
                "Detected pre-quantized model (%s); enabling prequant serve loading path.",
                quant_method,
            )
            return qlora, quant_method, bool(model_info.is_moe_model), model_max_len
        except Exception as exc:
            logger.warning(
                "Prequant auto-detection failed for %s (%s). Proceeding without prequant config.",
                model_dir,
                exc,
            )
            return None, None, False, None

    @staticmethod
    def _resolve_lmhead_chunks(
        config: ServeConfig,
        *,
        effective_seq_len: int,
        max_num_seqs: Optional[int] = None,
    ) -> int:
        # Training runtime persists an output buffer shaped [B*T/lmhead_chunks, V].
        # Keep per-chunk token count small for inference startup memory stability.
        tokens_per_chunk = max(
            128,
            int(os.getenv("SUROGATE_SERVE_LMHEAD_TOKENS_PER_CHUNK", "1024")),
        )
        bsz = max(
            1,
            int(config.max_num_seqs if max_num_seqs is None else max_num_seqs),
        )
        total_tokens = max(1, bsz * int(effective_seq_len))
        chunks = max(1, math.ceil(total_tokens / tokens_per_chunk))
        return int(chunks)

    @staticmethod
    def _resolve_effective_max_model_len(
        config: ServeConfig,
        *,
        model_max_len: Optional[int],
        tokenizer: Any,
    ) -> int:
        if config.max_model_len is not None:
            return max(1, int(config.max_model_len))

        inferred_model_max_len = NativeServingRuntime._normalize_context_len(model_max_len)
        if inferred_model_max_len is None:
            inferred_model_max_len = NativeServingRuntime._normalize_context_len(
                getattr(tokenizer, "model_max_length", None)
            )
        if inferred_model_max_len is None:
            raise ValueError(
                "ServeConfig: `max_model_len` is not set and model max context length "
                "could not be inferred. Set --max_model_len explicitly."
            )
        logger.info(
            "Serve max_model_len not set; defaulting to model max context length=%d.",
            inferred_model_max_len,
        )
        return inferred_model_max_len

    @staticmethod
    def _resolve_runtime_seq_len_budget(
        config: ServeConfig,
        *,
        requested_seq_len: int,
        requested_batch_size: int,
    ) -> int:
        # Decouple model buffer sizing from max context length:
        # max_model_len controls API/context limits, while runtime_seq_len controls
        # [B, T, ...] persistent buffers in the trainer. This mirrors vLLM's
        # max_num_batched_tokens behavior and prevents large-context OOM at startup.
        env_raw = os.getenv("SUROGATE_SERVE_MAX_BATCHED_TOKENS", "").strip()
        if env_raw:
            try:
                max_batched_tokens = int(env_raw)
            except ValueError:
                logger.warning(
                    "Invalid SUROGATE_SERVE_MAX_BATCHED_TOKENS=%r; falling back to config value=%d.",
                    env_raw,
                    int(config.max_num_batched_tokens),
                )
                max_batched_tokens = int(config.max_num_batched_tokens)
        else:
            max_batched_tokens = int(config.max_num_batched_tokens)

        if max_batched_tokens <= 0:
            return max(1, int(requested_seq_len))

        requested_seq_len = max(1, int(requested_seq_len))
        requested_batch_size = max(1, int(requested_batch_size))
        prefill_chunk = max(0, int(config.prefill_chunk_size))
        per_seq_budget = max(
            1,
            (max_batched_tokens + requested_batch_size - 1) // requested_batch_size,
        )
        # Runtime buffer sizing: use per_seq_budget (not prefill_chunk).
        # Prefill compiles its own graph at B=1,T=chunk_size which doesn't
        # need the full [max_num_seqs, chunk_size] persistent buffer.
        runtime_seq_len = min(
            requested_seq_len,
            max(2, per_seq_budget),
        )
        # Prefill chunks are capped by the runtime static T capacity in the
        # continuous engine. This keeps prefill copy/graph shapes within the
        # trainer's allocated input buffers even when max_model_len is large.
        if runtime_seq_len < requested_seq_len:
            logger.info(
                "Runtime seq_len budgeted to %d (context_len=%d, batch_size=%d, max_batched_tokens=%d, per_seq_budget=%d, prefill_chunk_size=%d).",
                runtime_seq_len,
                requested_seq_len,
                requested_batch_size,
                max_batched_tokens,
                per_seq_budget,
                prefill_chunk,
            )
        return int(runtime_seq_len)

    @staticmethod
    def _load_model_generation_config(model_dir: str) -> tuple[Optional[str], dict[str, Any]]:
        generation_cfg_path = os.path.join(model_dir, "generation_config.json")
        if not os.path.isfile(generation_cfg_path):
            return None, {}
        try:
            with open(generation_cfg_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning(
                "Failed to load generation_config.json at %s: %s",
                generation_cfg_path,
                exc,
            )
            return generation_cfg_path, {}
        if not isinstance(payload, dict):
            logger.warning(
                "Ignoring generation_config.json at %s: expected JSON object.",
                generation_cfg_path,
            )
            return generation_cfg_path, {}
        return generation_cfg_path, payload

    @staticmethod
    def _apply_model_generation_config_defaults(model_dir: str, config: ServeConfig) -> None:
        generation_cfg_path, generation_cfg = NativeServingRuntime._load_model_generation_config(
            model_dir
        )
        if not generation_cfg:
            return
        applied = config.apply_generation_config_defaults(generation_cfg)
        if applied:
            applied_str = ", ".join(f"{k}={v}" for k, v in sorted(applied.items()))
            logger.info(
                "Loaded generation defaults from %s (%s).",
                generation_cfg_path,
                applied_str,
            )

    @staticmethod
    def _resolve_model_dir(model: str) -> str:
        if os.path.isdir(model):
            return model
        if os.path.isfile(model):
            return model

        from huggingface_hub import snapshot_download

        logger.info(f"Resolving model from HF: {model}")
        return snapshot_download(model)

    def ensure_auth(self, request: Request):
        api_key = self.config.api_key
        if not api_key:
            return
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {api_key}"
        if auth != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def _messages_to_prompt_text(self, messages: list[ChatMessage]) -> str:
        normalized: list[dict[str, str]] = []
        for m in messages:
            content = m.content
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        if "text" in part and isinstance(part["text"], str):
                            parts.append(part["text"])
                        elif part.get("type") == "text" and isinstance(
                            part.get("text"), str
                        ):
                            parts.append(part["text"])
                text = "\n".join(parts)
            else:
                text = str(content)
            normalized.append({"role": m.role, "content": text})

        try:
            return self.tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return "\n".join(
                f"{m['role']}: {m['content']}" for m in normalized
            ) + "\nassistant:"

    def _build_choice(
        self,
        token_row: Sequence[int],
        prompt_len: int,
        completion_len: int,
        *,
        index: int,
        max_tokens: int,
        stop_strings: Sequence[str],
    ) -> GeneratedChoice:
        pl = int(prompt_len)
        cl = int(completion_len)
        raw_ids = [int(t) for t in token_row[pl : pl + cl]]
        raw_text = self.tokenizer.decode(raw_ids, skip_special_tokens=True)

        trimmed_text, stop_hit = _apply_stop(raw_text, stop_strings)
        if stop_hit:
            out_ids = self.tokenizer.encode(trimmed_text, add_special_tokens=False)
        else:
            out_ids = raw_ids

        if stop_hit:
            finish_reason = "stop"
        elif raw_ids and raw_ids[-1] == self.eos_id:
            finish_reason = "stop"
        elif cl >= max_tokens:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        return GeneratedChoice(
            index=index,
            text=trimmed_text,
            token_ids=[int(t) for t in out_ids],
            finish_reason=finish_reason,
            prompt_tokens=pl,
            completion_tokens=len(out_ids),
        )

    # (Old session-based methods removed — continuous engine is the only path.)

    def _fit_prompt_and_max_tokens(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        *,
        param_name: str,
    ) -> tuple[list[int], int]:
        runtime_seq_len = max(2, int(self._runtime_seq_len))
        out_prompt = [int(t) for t in prompt_ids]
        requested_max_tokens = max(1, int(max_tokens))
        prompt_tokens = len(out_prompt)
        total = prompt_tokens + requested_max_tokens
        if total > runtime_seq_len:
            raise ContextLengthExceededError(
                (
                    f"This model's maximum context length is {runtime_seq_len} tokens, "
                    f"but you requested {total} tokens "
                    f"({prompt_tokens} in the prompt; {requested_max_tokens} for the completion). "
                    "Please reduce the prompt or max_tokens."
                ),
                param=param_name,
            )
        return out_prompt, requested_max_tokens

    # Old session-based methods (_run_generation_batch, _run_continuous_step,
    # _open_persistent_continuous_session, _run_persistent_continuous_step)
    # have been removed — the continuous engine is the only serving path.

    def _finalize_continuous_request(self, item: ActiveGeneration) -> None:
        raw_text = self.tokenizer.decode(item.generated_ids, skip_special_tokens=True)
        trimmed_text, stop_hit = _apply_stop(raw_text, item.pending.stop_strings)
        if stop_hit:
            out_ids = self.tokenizer.encode(trimmed_text, add_special_tokens=False)
            finish_reason = "stop"
        else:
            out_ids = item.generated_ids
            finish_reason = item.finish_reason

        item.pending.result = [
            GeneratedChoice(
                index=0,
                text=trimmed_text,
                token_ids=[int(t) for t in out_ids],
                finish_reason=finish_reason,
                prompt_tokens=len(item.pending.prompt_ids),
                completion_tokens=len(out_ids),
            )
        ]

    # ------------------------------------------------------------------
    # Continuous engine loop (iteration-level continuous batching)
    # ------------------------------------------------------------------

    def _pop_one_pending(self) -> Optional[ActiveGeneration]:
        """Pop one pending request from the queue, or None."""
        with self._pending_cv:
            if self._pending:
                return ActiveGeneration(pending=self._pending.pop(0))
        return None

    def _continuous_engine_loop(self) -> None:
        """Iteration-level continuous batching loop.

        Each iteration:
          1. Admit at most ONE new request (prefill it).
          2. Decode all active sequences.
          3. Finalize finished sequences.

        By admitting only 1 request per iteration, prefill is interleaved
        with decode — early requests start generating immediately while
        later requests are still being prefilled.
        """
        engine_id: Optional[int] = None
        slot_map: dict[int, ActiveGeneration] = {}

        try:
            engine_max_seq_len = min(
                self._runtime_seq_len,
                self.config.max_model_len
                if self.config.max_model_len is not None
                else 8192,
            )
            engine_id = int(
                self.trainer.create_continuous_engine(
                    max_num_seqs=self._max_batch_sequences,
                    max_seq_len=engine_max_seq_len,
                    gpu_memory_utilization=self.config.gpu_memory_utilization,
                    use_cuda_graphs=bool(self.config.use_cuda_graphs),
                )
            )
            logger.info(
                "Continuous engine created (max_seqs=%d, max_seq_len=%d, engine_id=%d)",
                self._max_batch_sequences,
                engine_max_seq_len,
                engine_id,
            )

            while not self._shutdown:
                # --- Phase 1: Admit all pending requests ---
                # Prefill all available requests as fast as possible to
                # maximize decode batch size and throughput.
                free_slots = self._max_batch_sequences - len(slot_map)
                while free_slots > 0:
                    new_item = self._pop_one_pending()
                    if new_item is None:
                        break
                    try:
                        prompt_ids = new_item.pending.prompt_ids
                        max_tokens = new_item.pending.params.max_tokens
                        if len(prompt_ids) + max_tokens > self._runtime_seq_len:
                            max_tokens = max(
                                1, self._runtime_seq_len - len(prompt_ids)
                            )
                        slot_id = self.trainer.engine_add_sequence(
                            engine_id,
                            prompt_ids,
                            max_gen_len=max_tokens,
                            temperature=new_item.pending.params.temperature,
                            eos_token_id=self.eos_id,
                            top_k=new_item.pending.params.top_k,
                            top_p=new_item.pending.params.top_p,
                            min_p=new_item.pending.params.min_p,
                            prefill_chunk_size=self.config.prefill_chunk_size,
                        )
                        if slot_id < 0:
                            with self._pending_cv:
                                self._pending.insert(0, new_item.pending)
                            break
                        slot_map[slot_id] = new_item
                        free_slots -= 1
                    except Exception as exc:
                        new_item.pending.error = exc
                        new_item.pending.done.set()

                if not slot_map:
                    # No active work — wait briefly for a request.
                    with self._pending_cv:
                        if not self._pending and not self._shutdown:
                            self._pending_cv.wait(timeout=0.001)
                    continue

                # --- Phase 2: Decode multiple tokens in C++ ---
                # Adaptive step sizing: small when queue has waiters,
                # large when idle (matches old session-based logic).
                # Use small step_tokens only when there are pending requests
                # AND free slots to admit them.  If slots are full, pending
                # requests can't be admitted anyway — maximize decode throughput.
                free_slots_now = self._max_batch_sequences - len(slot_map)
                with self._pending_cv:
                    waiting_and_admittable = bool(self._pending) and free_slots_now > 0
                if waiting_and_admittable:
                    step_tokens = self._continuous_step_tokens
                else:
                    step_tokens = self._continuous_idle_step_tokens
                # Cap by the minimum remaining budget across active seqs.
                min_remaining = min(
                    max(1, item.pending.params.max_tokens - len(item.generated_ids))
                    for item in slot_map.values()
                    if not item.finished
                )
                step_tokens = max(1, min(step_tokens, min_remaining))

                try:
                    slot_ids, tokens_per_slot, finished_flags, comp_lens = (
                        self.trainer.engine_step(engine_id, step_tokens)
                    )
                except Exception as exc:
                    for item in slot_map.values():
                        if not item.finished:
                            item.pending.error = exc
                            item.pending.done.set()
                    slot_map.clear()
                    logger.exception("engine_step failed")
                    continue

                # --- Phase 3: Process multi-token results ---
                release_slots: list[int] = []
                for sid, toks, fin, clen in zip(
                    slot_ids, tokens_per_slot, finished_flags, comp_lens
                ):
                    item = slot_map.get(sid)
                    if item is None or item.finished:
                        continue
                    item.generated_ids.extend(int(t) for t in toks)

                    should_finish = bool(fin)
                    if not should_finish and item.pending.stop_strings:
                        partial_text = self.tokenizer.decode(
                            item.generated_ids, skip_special_tokens=True
                        )
                        _, stop_hit = _apply_stop(
                            partial_text, item.pending.stop_strings
                        )
                        if stop_hit:
                            should_finish = True
                    if not should_finish:
                        if len(item.generated_ids) >= item.pending.params.max_tokens:
                            should_finish = True

                    if should_finish:
                        item.finished = True
                        last_tok = item.generated_ids[-1] if item.generated_ids else -1
                        if last_tok == self.eos_id:
                            item.finish_reason = "stop"
                        elif item.pending.stop_strings:
                            partial_text = self.tokenizer.decode(
                                item.generated_ids, skip_special_tokens=True
                            )
                            _, stop_hit = _apply_stop(
                                partial_text, item.pending.stop_strings
                            )
                            item.finish_reason = "stop" if stop_hit else "length"
                        else:
                            item.finish_reason = "length"

                        self._finalize_continuous_request(item)
                        item.pending.done.set()
                        release_slots.append(sid)

                for sid in release_slots:
                    try:
                        self.trainer.engine_release_slot(engine_id, sid)
                    except Exception:
                        logger.exception("engine_release_slot(%d) failed", sid)
                    slot_map.pop(sid, None)

        except Exception:
            logger.exception("continuous engine loop crashed")
        finally:
            # Cleanup: release all active slots, destroy engine.
            if engine_id is not None:
                for sid in list(slot_map.keys()):
                    try:
                        self.trainer.engine_release_slot(engine_id, sid)
                    except Exception:
                        pass
                try:
                    self.trainer.engine_destroy(engine_id)
                except Exception:
                    pass
            # Signal any remaining pending requests.
            for item in slot_map.values():
                if not item.pending.done.is_set():
                    item.pending.error = RuntimeError("continuous engine shut down")
                    item.pending.done.set()

    def _generate(
        self,
        prompt_ids: list[int],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        stop_strings: Sequence[str],
    ) -> list[GeneratedChoice]:
        pending = PendingGeneration(
            prompt_ids=prompt_ids,
            stop_strings=list(stop_strings),
            params=GenerationParams(
                n=n,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
            ),
        )
        with self._pending_cv:
            self._pending.append(pending)
            self._pending_cv.notify()

        pending.done.wait()
        if pending.error is not None:
            raise pending.error
        if pending.result is None:
            raise RuntimeError("generation batcher produced no result")
        return pending.result

    def generate_for_chat(self, req: ChatCompletionsRequest) -> list[GeneratedChoice]:
        if req.model and req.model not in {self.model_id, self.config.model}:
            raise ValueError(f"Unknown model '{req.model}'")
        if req.n <= 0:
            raise ValueError("`n` must be > 0")

        prompt_text = self._messages_to_prompt_text(req.messages)
        # apply_chat_template() already injects BOS/role/control markers.
        # Re-adding special tokens here diverges from HF/vLLM chat behavior,
        # especially on Qwen3.5.
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        max_tokens = int(
            req.max_completion_tokens
            if req.max_completion_tokens is not None
            else (req.max_tokens if req.max_tokens is not None else self.config.max_gen_len)
        )
        temperature = (
            float(req.temperature)
            if req.temperature is not None
            else self.config.temperature
        )
        top_p = float(req.top_p) if req.top_p is not None else self.config.top_p
        top_k = int(req.top_k) if req.top_k is not None else self.config.top_k
        min_p = float(req.min_p) if req.min_p is not None else self.config.min_p
        repetition_penalty = (
            float(req.repetition_penalty)
            if req.repetition_penalty is not None
            else self.config.repetition_penalty
        )
        prompt_ids, max_tokens = self._fit_prompt_and_max_tokens(
            prompt_ids, max(1, max_tokens), param_name="messages"
        )

        return self._generate(
            prompt_ids,
            n=req.n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=max(0, top_k),
            top_p=top_p,
            min_p=max(0.0, min_p),
            repetition_penalty=max(1e-6, repetition_penalty),
            stop_strings=_normalize_stop(req.stop),
        )

    def generate_for_completion(self, req: CompletionsRequest) -> list[GeneratedChoice]:
        if req.model and req.model not in {self.model_id, self.config.model}:
            raise ValueError(f"Unknown model '{req.model}'")
        if req.n <= 0:
            raise ValueError("`n` must be > 0")

        if isinstance(req.prompt, list):
            if len(req.prompt) == 0:
                raise ValueError("`prompt` cannot be empty")
            if len(req.prompt) > 1:
                raise ValueError("This phase-1 server supports one prompt per request")
            prompt_text = req.prompt[0]
        else:
            prompt_text = req.prompt
        prompt_ids = self.tokenizer.encode(prompt_text)

        max_tokens = int(req.max_tokens or self.config.max_gen_len)
        temperature = (
            float(req.temperature)
            if req.temperature is not None
            else self.config.temperature
        )
        top_p = float(req.top_p) if req.top_p is not None else self.config.top_p
        top_k = int(req.top_k) if req.top_k is not None else self.config.top_k
        min_p = float(req.min_p) if req.min_p is not None else self.config.min_p
        repetition_penalty = (
            float(req.repetition_penalty)
            if req.repetition_penalty is not None
            else self.config.repetition_penalty
        )
        prompt_ids, max_tokens = self._fit_prompt_and_max_tokens(
            prompt_ids, max(1, max_tokens), param_name="prompt"
        )

        return self._generate(
            prompt_ids,
            n=req.n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=max(0, top_k),
            top_p=top_p,
            min_p=max(0.0, min_p),
            repetition_penalty=max(1e-6, repetition_penalty),
            stop_strings=_normalize_stop(req.stop),
        )


def _usage_from_choices(choices: list[GeneratedChoice]) -> dict[str, int]:
    prompt_tokens = choices[0].prompt_tokens if choices else 0
    completion_tokens = sum(c.completion_tokens for c in choices)
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
    }


def _chat_stream_chunks(
    request_id: str,
    created: int,
    model_id: str,
    tokenizer: Any,
    choices: list[GeneratedChoice],
    include_usage: bool,
):
    usage = _usage_from_choices(choices) if include_usage else None

    async def _gen():
        for c in choices:
            yield _sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": c.index,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            for piece in _token_text_chunks(tokenizer, c.token_ids):
                yield _sse(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": c.index,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            yield _sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": c.index,
                            "delta": {},
                            "finish_reason": c.finish_reason,
                        }
                    ],
                }
            )
        if include_usage and usage is not None:
            yield _sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [],
                    "usage": usage,
                }
            )
        yield b"data: [DONE]\n\n"

    return _gen()


def _completion_stream_chunks(
    request_id: str,
    created: int,
    model_id: str,
    tokenizer: Any,
    choices: list[GeneratedChoice],
    include_usage: bool,
):
    usage = _usage_from_choices(choices) if include_usage else None

    async def _gen():
        for c in choices:
            for piece in _token_text_chunks(tokenizer, c.token_ids):
                yield _sse(
                    {
                        "id": request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": c.index,
                                "text": piece,
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            yield _sse(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": c.index,
                            "text": "",
                            "finish_reason": c.finish_reason,
                        }
                    ],
                }
            )
        if include_usage and usage is not None:
            yield _sse(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [],
                    "usage": usage,
                }
            )
        yield b"data: [DONE]\n\n"

    return _gen()


def create_app(runtime: NativeServingRuntime) -> FastAPI:
    app = FastAPI(title="Surogate Native OpenAI-Compatible Server", version="0.1.0")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
    async def v1_root():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request):
        runtime.ensure_auth(request)
        created = int(time.time())
        model_id = runtime.model_id
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "surogate",
                    "root": model_id,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionsRequest, request: Request):
        runtime.ensure_auth(request)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        try:
            choices = await asyncio.to_thread(runtime.generate_for_chat, req)
        except ContextLengthExceededError as e:
            return _openai_context_length_error(str(e), e.param)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error("chat completion failed", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

        if req.stream:
            include_usage = bool(
                req.stream_options and req.stream_options.include_usage
            )
            return StreamingResponse(
                _chat_stream_chunks(
                    request_id,
                    created,
                    runtime.model_id,
                    runtime.tokenizer,
                    choices,
                    include_usage,
                ),
                media_type="text/event-stream",
            )

        usage = _usage_from_choices(choices)
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": runtime.model_id,
            "choices": [
                {
                    "index": c.index,
                    "message": {"role": "assistant", "content": c.text},
                    "finish_reason": c.finish_reason,
                }
                for c in choices
            ],
            "usage": usage,
        }
        return JSONResponse(content=response)

    @app.post("/v1/completions")
    async def completions(req: CompletionsRequest, request: Request):
        runtime.ensure_auth(request)
        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        try:
            choices = await asyncio.to_thread(runtime.generate_for_completion, req)
        except ContextLengthExceededError as e:
            return _openai_context_length_error(str(e), e.param)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error("completion failed", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

        if req.stream:
            include_usage = bool(
                req.stream_options and req.stream_options.include_usage
            )
            return StreamingResponse(
                _completion_stream_chunks(
                    request_id,
                    created,
                    runtime.model_id,
                    runtime.tokenizer,
                    choices,
                    include_usage,
                ),
                media_type="text/event-stream",
            )

        usage = _usage_from_choices(choices)
        response = {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": runtime.model_id,
            "choices": [
                {
                    "index": c.index,
                    "text": c.text,
                    "finish_reason": c.finish_reason,
                }
                for c in choices
            ],
            "usage": usage,
        }
        return JSONResponse(content=response)

    return app


def serve(config: ServeConfig):
    runtime = NativeServingRuntime(config)
    app = create_app(runtime)
    logger.info(f"Starting native inference server on {config.host}:{config.port}")
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        access_log=False,
    )
