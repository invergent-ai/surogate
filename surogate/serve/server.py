from __future__ import annotations

import asyncio
import gc
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
        qlora_config, prequant_method, is_moe_model = self._resolve_prequant_qlora_config(
            model_dir
        )
        self._is_moe_model = bool(is_moe_model)

        logger.info(f"Loading tokenizer for {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=config.trust_remote_code
        )
        self.eos_id = self.tokenizer.eos_token_id or 151643

        logger.info(f"Building DSL IR for {model_dir}")
        ir_json = build_dsl_ir_for_model(model_dir)
        jit_manifests = compile_jit_kernels(ir_json)
        requested_batch_size = max(1, int(config.batch_size))
        requested_seq_len = self._resolve_effective_sequence_len(config)

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
        options.use_cuda_graphs = bool(config.use_cuda_graphs)
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
            batch_size=requested_batch_size,
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
            self._runtime_seq_len,
            self.trainer,
        ) = self._build_trainer_with_oom_retries(
            model_dir=model_dir,
            config=config,
            options=options,
            qlora_config=qlora_config,
            model_weights_path=model_weights_path,
            initial_batch_size=requested_batch_size,
            initial_seq_len=requested_seq_len,
        )
        if (
            self._runtime_batch_size != requested_batch_size
            or self._runtime_seq_len != requested_seq_len
        ):
            logger.warning(
                "Serve startup auto-sized capacity from batch_size=%d seq_len=%d to "
                "batch_size=%d seq_len=%d after CUDA OOM retries.",
                requested_batch_size,
                requested_seq_len,
                self._runtime_batch_size,
                self._runtime_seq_len,
            )
        else:
            logger.info(
                "Serve startup capacity: batch_size=%d seq_len=%d",
                self._runtime_batch_size,
                self._runtime_seq_len,
            )
        self._pending_lock = threading.Lock()
        self._pending_cv = threading.Condition(self._pending_lock)
        self._pending: list[PendingGeneration] = []
        self._shutdown = False
        self._batch_wait_s = max(
            0.0,
            float(os.getenv("SUROGATE_SERVE_BATCH_WAIT_MS", "2")) / 1000.0,
        )
        self._continuous_enabled = os.getenv("SUROGATE_SERVE_CONTINUOUS_BATCHING", "1") != "0"
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
        # Trainer capacity is bounded by loaded runtime batch size (total decode sequences B).
        configured_cap = max(1, int(self._runtime_batch_size))
        env_cap = int(os.getenv("SUROGATE_SERVE_MAX_BATCH_SEQUENCES", str(configured_cap)))
        self._max_batch_sequences = max(1, min(configured_cap, env_cap))
        if self._max_batch_sequences <= 1:
            logger.warning(
                "Serve runtime running with batch_size=1; concurrent requests will be serialized. "
                "Set --batch_size>1 to improve throughput."
            )
        self._batch_worker = threading.Thread(
            target=self._generation_worker_loop,
            name="surogate-generate-batcher",
            daemon=True,
        )
        self._batch_worker.start()
        logger.info(
            "Native inference runtime ready (continuous_batching=%s step_tokens=%d idle_step_tokens=%d)",
            bool(self._continuous_enabled),
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
    def _clear_cuda_allocator() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _next_capacity_after_oom(
        *,
        batch_size: int,
        seq_len: int,
        min_seq_len: int,
    ) -> Optional[tuple[int, int]]:
        # First preserve completion headroom (>= max_gen_len) while shrinking seq.
        safe_min_seq = max(128, int(min_seq_len))
        if seq_len > safe_min_seq:
            next_seq = max(safe_min_seq, seq_len // 2)
            if next_seq < seq_len:
                return batch_size, next_seq

        # Then shrink parallel decode width.
        if batch_size > 1:
            next_batch = max(1, batch_size // 2)
            if next_batch < batch_size:
                return next_batch, seq_len

        # Last-resort shrink below max_gen_len to allow startup on constrained VRAM.
        if seq_len > 128:
            next_seq = max(128, seq_len // 2)
            if next_seq < seq_len:
                return batch_size, next_seq
        return None

    def _build_trainer_with_oom_retries(
        self,
        *,
        model_dir: str,
        config: ServeConfig,
        options: Any,
        qlora_config: Any,
        model_weights_path: str,
        initial_batch_size: int,
        initial_seq_len: int,
    ) -> tuple[int, int, Any]:
        batch_size = max(1, int(initial_batch_size))
        seq_len = max(1, int(initial_seq_len))
        min_seq_len = max(128, int(config.max_gen_len))
        attempt = 0

        while True:
            options.lmhead_chunks = self._resolve_lmhead_chunks(
                config,
                effective_seq_len=seq_len,
                batch_size=batch_size,
            )
            logger.info(
                "Serve runtime init attempt %d with batch_size=%d seq_len=%d lmhead_chunks=%d",
                attempt + 1,
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
                if not self._is_cuda_oom_error(exc):
                    raise

                next_capacity = self._next_capacity_after_oom(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    min_seq_len=min_seq_len,
                )
                if next_capacity is None:
                    raise RuntimeError(
                        "Serve startup failed due to CUDA OOM even after capacity "
                        f"reductions (batch_size={batch_size}, seq_len={seq_len}). "
                        "Reduce --batch_size/--sequence_len or lower --gpu-memory-utilization."
                    ) from exc
                next_batch, next_seq = next_capacity
                logger.warning(
                    "Serve startup OOM at batch_size=%d seq_len=%d. Retrying with "
                    "batch_size=%d seq_len=%d.",
                    batch_size,
                    seq_len,
                    next_batch,
                    next_seq,
                )
                self._clear_cuda_allocator()
                batch_size, seq_len = next_batch, next_seq
                attempt += 1

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
    def _resolve_prequant_qlora_config(model_dir: str):
        """Auto-detect pre-quantized models and return matching QLoRAConfig."""
        try:
            model_info = ModelInfo.create(model_dir)
            quant_info = model_info.quant_info or {}
            quant_method = (model_info.quant_method or "").lower()
            modules_to_not_convert = quant_info.get("modules_to_not_convert") or []

            if quant_method == "prequant_fp8":
                qlora = _surogate.QLoRAConfig.prequant_fp8()
            elif quant_method == "prequant_nvfp4":
                qlora = _surogate.QLoRAConfig.prequant_nvfp4()
            elif quant_method == "prequant_mxfp4":
                qlora = _surogate.QLoRAConfig.prequant_mxfp4()
            else:
                return None, None, bool(model_info.is_moe_model)

            if modules_to_not_convert:
                qlora.modules_to_not_convert = list(modules_to_not_convert)
            logger.info(
                "Detected pre-quantized model (%s); enabling prequant serve loading path.",
                quant_method,
            )
            return qlora, quant_method, bool(model_info.is_moe_model)
        except Exception as exc:
            logger.warning(
                "Prequant auto-detection failed for %s (%s). Proceeding without prequant config.",
                model_dir,
                exc,
            )
            return None, None, False

    @staticmethod
    def _resolve_lmhead_chunks(
        config: ServeConfig,
        *,
        effective_seq_len: int,
        batch_size: Optional[int] = None,
    ) -> int:
        # Training runtime persists an output buffer shaped [B*T/lmhead_chunks, V].
        # Keep per-chunk token count small for inference startup memory stability.
        tokens_per_chunk = max(
            128,
            int(os.getenv("SUROGATE_SERVE_LMHEAD_TOKENS_PER_CHUNK", "1024")),
        )
        bsz = max(1, int(config.batch_size if batch_size is None else batch_size))
        total_tokens = max(1, bsz * int(effective_seq_len))
        chunks = max(1, math.ceil(total_tokens / tokens_per_chunk))
        return int(chunks)

    @staticmethod
    def _resolve_effective_sequence_len(config: ServeConfig) -> int:
        # Serve allocates static [B, T, ...] training-era buffers. Keep B*T within
        # a conservative startup budget to avoid OOM, especially on single-GPU runs.
        requested_t = max(1, int(config.sequence_len))
        bsz = max(1, int(config.batch_size))
        base_budget = int(os.getenv("SUROGATE_SERVE_STATIC_TOKEN_BUDGET", "8192"))
        util_scale = float(config.gpu_memory_utilization) / 0.8
        token_budget = max(1024, int(base_budget * max(0.5, util_scale)))
        max_t = max(1, token_budget // bsz)
        if requested_t > max_t:
            logger.warning(
                "Reducing serve seq_len from %d to %d to respect static token budget "
                "(batch_size=%d, budget_tokens=%d). Override with --sequence_len and/or "
                "SUROGATE_SERVE_STATIC_TOKEN_BUDGET if needed.",
                requested_t,
                max_t,
                bsz,
                token_budget,
            )
            return int(max_t)
        return int(requested_t)

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

    def _prompt_capacity(self, params: GenerationParams) -> int:
        return max(1, self._max_batch_sequences // max(1, params.n))

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

    def _run_generation_batch(self, batch: list[PendingGeneration]) -> None:
        params = batch[0].params
        prompts = [item.prompt_ids for item in batch]
        tokens, _, prompt_lens, completion_lens = self.trainer.generate(
            prompts=prompts,
            num_completions=params.n,
            max_gen_len=params.max_tokens,
            temperature=params.temperature,
            eos_token_id=self.eos_id,
            use_lora=self.config.use_lora,
            use_cuda_graphs=self.config.use_cuda_graphs,
            top_k=params.top_k,
            top_p=params.top_p,
            min_p=params.min_p,
            prefill_chunk_size=self.config.prefill_chunk_size,
            repetition_penalty=params.repetition_penalty,
        )

        for b_idx, item in enumerate(batch):
            choices: list[GeneratedChoice] = []
            for c_idx in range(params.n):
                row_idx = b_idx * params.n + c_idx
                choices.append(
                    self._build_choice(
                        tokens[row_idx],
                        int(prompt_lens[row_idx]),
                        int(completion_lens[row_idx]),
                        index=c_idx,
                        max_tokens=params.max_tokens,
                        stop_strings=item.stop_strings,
                    )
                )
            item.result = choices

    def _run_continuous_step(
        self,
        active: list[ActiveGeneration],
        params: GenerationParams,
        *,
        step_tokens: int,
    ) -> None:
        prompts = [
            item.pending.prompt_ids + item.generated_ids
            for item in active
        ]
        tokens, _, prompt_lens, completion_lens = self.trainer.generate(
            prompts=prompts,
            num_completions=1,
            max_gen_len=step_tokens,
            temperature=params.temperature,
            eos_token_id=self.eos_id,
            use_lora=self.config.use_lora,
            use_cuda_graphs=self.config.use_cuda_graphs,
            top_k=params.top_k,
            top_p=params.top_p,
            min_p=params.min_p,
            prefill_chunk_size=self.config.prefill_chunk_size,
            repetition_penalty=params.repetition_penalty,
        )

        for row_idx, item in enumerate(active):
            if item.finished:
                continue

            pl = int(prompt_lens[row_idx])
            cl = int(completion_lens[row_idx])
            if cl <= 0:
                # Defensive: avoid spinning if backend returns no new tokens.
                item.finished = True
                item.finish_reason = "stop"
                continue

            new_ids = [int(t) for t in tokens[row_idx][pl : pl + cl]]
            if new_ids:
                item.generated_ids.extend(new_ids)

            raw_text = self.tokenizer.decode(item.generated_ids, skip_special_tokens=True)
            trimmed_text, stop_hit = _apply_stop(raw_text, item.pending.stop_strings)
            if stop_hit:
                item.generated_ids = [
                    int(t)
                    for t in self.tokenizer.encode(trimmed_text, add_special_tokens=False)
                ]
                item.finished = True
                item.finish_reason = "stop"
                continue

            if item.generated_ids and item.generated_ids[-1] == self.eos_id:
                item.finished = True
                item.finish_reason = "stop"
                continue

            if len(item.generated_ids) >= params.max_tokens:
                if len(item.generated_ids) > params.max_tokens:
                    item.generated_ids = item.generated_ids[: params.max_tokens]
                item.finished = True
                item.finish_reason = "length"

    def _run_continuous_step_with_oom_backoff(
        self,
        active: list[ActiveGeneration],
        params: GenerationParams,
        *,
        step_tokens: int,
    ) -> None:
        cur_step_tokens = max(1, int(step_tokens))
        while True:
            try:
                self._run_continuous_step(active, params, step_tokens=cur_step_tokens)
                return
            except RuntimeError as exc:
                if (not self._is_cuda_oom_error(exc)) or cur_step_tokens <= 1:
                    raise
                next_step_tokens = max(1, cur_step_tokens // 2)
                logger.warning(
                    "Continuous decode OOM at step_tokens=%d; retrying with step_tokens=%d.",
                    cur_step_tokens,
                    next_step_tokens,
                )
                self._clear_cuda_allocator()
                cur_step_tokens = next_step_tokens

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

    def _generation_worker_loop(self) -> None:
        while True:
            with self._pending_cv:
                while not self._pending and not self._shutdown:
                    self._pending_cv.wait()
                if self._shutdown:
                    return
                first = self._pending.pop(0)

            # Fallback for n>1: keep the previous one-shot batch behavior.
            if (not self._continuous_enabled) or first.params.n != 1:
                batch: list[PendingGeneration] = [first]
                prompt_cap = self._prompt_capacity(first.params)
                deadline = time.perf_counter() + self._batch_wait_s
                with self._pending_cv:
                    while len(batch) < prompt_cap:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        if not self._pending:
                            self._pending_cv.wait(timeout=remaining)
                            continue

                        i = 0
                        while i < len(self._pending) and len(batch) < prompt_cap:
                            item = self._pending[i]
                            if item.params == first.params:
                                batch.append(self._pending.pop(i))
                            else:
                                i += 1
                        if not self._pending:
                            continue

                err: Optional[Exception] = None
                try:
                    self._run_generation_batch(batch)
                except Exception as e:
                    err = e

                for item in batch:
                    if err is not None:
                        item.error = err
                    item.done.set()
                continue

            params = first.params
            prompt_cap = self._prompt_capacity(params)
            active: list[ActiveGeneration] = [ActiveGeneration(pending=first)]
            deadline = time.perf_counter() + self._batch_wait_s
            with self._pending_cv:
                while len(active) < prompt_cap:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    if not self._pending:
                        self._pending_cv.wait(timeout=remaining)
                        continue
                    i = 0
                    while i < len(self._pending) and len(active) < prompt_cap:
                        queued = self._pending[i]
                        if queued.params == params:
                            active.append(ActiveGeneration(pending=self._pending.pop(i)))
                        else:
                            i += 1

            while True:
                waiting_match = False
                with self._pending_cv:
                    i = 0
                    while i < len(self._pending):
                        queued = self._pending[i]
                        if queued.params != params:
                            i += 1
                            continue
                        if len(active) < prompt_cap:
                            active.append(ActiveGeneration(pending=self._pending.pop(i)))
                            continue
                        waiting_match = True
                        i += 1

                if not active:
                    if not waiting_match:
                        break
                    continue

                unfinished = [a for a in active if not a.finished]
                if not unfinished:
                    active.clear()
                    if not waiting_match:
                        break
                    continue

                max_step_tokens = (
                    self._continuous_step_tokens
                    if waiting_match or len(active) > 1
                    else self._continuous_idle_step_tokens
                )
                remaining_min = min(
                    max(1, params.max_tokens - len(a.generated_ids))
                    for a in unfinished
                )
                step_tokens = max(1, min(max_step_tokens, remaining_min))

                err: Optional[Exception] = None
                try:
                    self._run_continuous_step_with_oom_backoff(
                        active, params, step_tokens=step_tokens
                    )
                except Exception as e:
                    err = e

                if err is not None:
                    for item in active:
                        item.pending.error = err
                        item.pending.done.set()
                    break

                still_active: list[ActiveGeneration] = []
                for item in active:
                    if item.finished:
                        self._finalize_continuous_request(item)
                        item.pending.done.set()
                    else:
                        still_active.append(item)
                active = still_active

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
