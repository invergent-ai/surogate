from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

from transformers import AutoTokenizer

from surogate import _surogate
from surogate.core.config.serve_config import ServeConfig
from surogate.core.model.registry import ModelInfo
from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.kernels.jit_compile import compile_jit_kernels
from surogate.utils.hf import get_model_weights_path
from surogate.utils.logger import get_logger

logger = get_logger()

class NativeServingRuntime:
    def __init__(self, config: ServeConfig):
        self.config = config
        self.model_id = config.model_id or config.model or "native"

        model_dir = self._resolve_model_dir(config.model or "")
        self._model_dir = model_dir
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
        try:
            self._token_id_limit = max(1, int(len(self.tokenizer)))
        except Exception:
            self._token_id_limit = 0
        self._fallback_token_id = getattr(self.tokenizer, "unk_token_id", None)
        if (
            self._fallback_token_id is None
            or self._fallback_token_id < 0
            or (
                self._token_id_limit > 0
                and int(self._fallback_token_id) >= self._token_id_limit
            )
        ):
            if self._token_id_limit > 0 and 0 <= int(self.eos_id) < self._token_id_limit:
                self._fallback_token_id = int(self.eos_id)
            else:
                self._fallback_token_id = 0
        self._invalid_token_count = 0

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
        dynamic_token_capacity = max(
            int(requested_batch_size),
            int(config.prefill_chunk_size) if int(config.prefill_chunk_size) > 0 else int(config.max_num_batched_tokens),
        )
        if hasattr(options, "dynamic_token_buffers"):
            options.dynamic_token_buffers = True
        if hasattr(options, "max_token_capacity"):
            options.max_token_capacity = int(dynamic_token_capacity)
        self._dynamic_token_capacity = int(dynamic_token_capacity)
        options.lmhead_chunks = self._resolve_lmhead_chunks(
            config,
            effective_seq_len=requested_seq_len,
            max_num_seqs=requested_batch_size,
        )
        options.min_stack_mb = self._resolve_min_stack_mb(config)
        self._auto_min_stack_mb = int(options.min_stack_mb)
        min_activation_env = os.getenv("SUROGATE_SERVE_CONTINUOUS_MIN_ACTIVATION_MB", "").strip()
        if min_activation_env:
            try:
                continuous_min_activation_mb = int(min_activation_env)
            except ValueError:
                logger.warning(
                    "Invalid SUROGATE_SERVE_CONTINUOUS_MIN_ACTIVATION_MB=%r; falling back to auto default.",
                    min_activation_env,
                )
                continuous_min_activation_mb = 0
        else:
            continuous_min_activation_mb = 0
        if continuous_min_activation_mb <= 0:
            has_chunk_gdr = (
                '"chunk_gated_delta_rule"' in ir_json
                or '"gated_delta_rule"' in ir_json
            )
            continuous_min_activation_mb = 2048 if has_chunk_gdr else 256
            if has_chunk_gdr:
                logger.info(
                    "Detected gated-delta-rule ops in IR; using higher continuous activation reserve=%d MiB. "
                    "Override with SUROGATE_SERVE_CONTINUOUS_MIN_ACTIVATION_MB if needed.",
                    continuous_min_activation_mb,
                )
        self._continuous_min_activation_mb = max(64, int(continuous_min_activation_mb))
        logger.info(
            "Serving runtime stack floor=%d MiB lmhead_chunks=%d offload_grads=%s dynamic_token_capacity=%d (gpu_memory_utilization=%.2f)",
            int(options.min_stack_mb),
            int(options.lmhead_chunks),
            bool(options.offload_grads),
            int(self._dynamic_token_capacity),
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
        configured_cap = max(1, int(self._runtime_batch_size))
        env_cap = int(os.getenv("SUROGATE_SERVE_MAX_BATCH_SEQUENCES", str(configured_cap)))
        self._max_batch_sequences = max(1, min(configured_cap, env_cap))

        if self._max_batch_sequences <= 1:
            logger.warning(
                "Serve runtime running with max_num_seqs=1; concurrent requests will be serialized. "
                "Set --max_num_seqs>1 to improve throughput."
            )

        self._init_cpp_http_server()

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

    def _init_cpp_http_server(self) -> None:
        if not hasattr(_surogate, "NativeHttpServer"):
            raise RuntimeError("NativeHttpServer binding is not available in _surogate.")

        cfg = _surogate.NativeHttpServerConfig()
        cfg.host = self.config.host
        cfg.port = int(self.config.port)
        cfg.model_id = self.model_id
        cfg.api_key = self.config.api_key or ""
        cfg.model_dir = str(self._model_dir)
        cfg.eos_token_id = int(self.eos_id)
        cfg.max_context_len = int(self._max_context_len)
        cfg.runtime_seq_len = int(self._runtime_seq_len)
        cfg.max_batch_sequences = int(self._max_batch_sequences)
        cfg.long_prefill_token_threshold = int(
            os.getenv("SUROGATE_SERVE_LONG_PREFILL_THRESHOLD", "0"))
        cfg.max_num_partial_prefills = int(
            os.getenv("SUROGATE_SERVE_MAX_PARTIAL_PREFILLS", "1"))
        cfg.stream_batch_tokens = max(
            1, int(os.getenv("SUROGATE_SERVE_STREAM_BATCH_TOKENS", "16"))
        )
        cfg.gpu_memory_utilization = float(self.config.gpu_memory_utilization)
        cfg.use_cuda_graphs = bool(self.config.use_cuda_graphs)
        cfg.max_num_batched_tokens = int(self.config.max_num_batched_tokens)
        cfg.continuous_min_activation_mb = int(self._continuous_min_activation_mb)
        cfg.continuous_engine_max_seq_len = int(
            min(
                self._runtime_seq_len,
                self.config.max_model_len
                if self.config.max_model_len is not None
                else 8192,
            )
        )
        cfg.max_gen_len = int(self.config.max_gen_len)
        cfg.temperature = float(self.config.temperature)
        cfg.top_k = int(self.config.top_k)
        cfg.top_p = float(self.config.top_p)
        cfg.min_p = float(self.config.min_p)
        cfg.repetition_penalty = float(self.config.repetition_penalty)
        cfg.max_http_threads = max(
            0, int(os.getenv("SUROGATE_SERVE_HTTP_THREADS", "0"))
        )
        cfg.enable_loop_trace = os.getenv("SUROGATE_SERVE_LOOP_TRACE", "").strip().lower() not in {
            "",
            "0",
            "false",
            "no",
            "off",
        }
        self._cpp_http_server = _surogate.NativeHttpServer(self.trainer, cfg)

    def serve_cpp(self) -> None:
        if getattr(self, "_cpp_http_server", None) is None:
            raise RuntimeError("Native C++ HTTP server is not initialized.")
        self._cpp_http_server.serve()

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

        # Try the requested runtime seq_len first, then progressively smaller
        # capacities on CUDA OOM to keep serving online automatically.
        seq_candidates: list[int] = []
        for frac in (1.0, 0.75, 0.625, 0.5, 0.375, 0.25):
            cand = int(seq_len * frac)
            cand = max(64, (cand // 64) * 64)
            if cand > 0:
                seq_candidates.append(cand)
        seq_candidates.append(64)
        seq_candidates = sorted(set(seq_candidates), reverse=True)

        last_oom: Optional[RuntimeError] = None
        for seq_try in seq_candidates:
            options.lmhead_chunks = self._resolve_lmhead_chunks(
                config,
                effective_seq_len=seq_try,
                max_num_seqs=batch_size,
            )
            logger.info(
                "Serve runtime init with batch_size=%d seq_len=%d lmhead_chunks=%d",
                batch_size,
                seq_try,
                int(options.lmhead_chunks),
            )
            trainer = None
            try:
                trainer = _surogate.SurogateTrainer(
                    ngpu=config.gpus,
                    config=_surogate.PretrainedConfig.from_pretrained(model_dir, config.dtype),
                    options=options,
                    batch_size=batch_size,
                    seq_len=seq_try,
                    grad_accum=1,
                    memcpy_all_gather=True,
                    memcpy_send_recv=True,
                    qlora_config=qlora_config,
                )
                logger.info(f"Importing weights from {model_weights_path}")
                trainer.import_weights(model_weights_path)
                if seq_try != seq_len:
                    logger.warning(
                        "Serve startup auto-reduced runtime seq_len from %d to %d after CUDA OOM.",
                        seq_len,
                        seq_try,
                    )
                return batch_size, seq_try, trainer
            except RuntimeError as exc:
                if trainer is not None:
                    del trainer
                if not self._is_cuda_oom_error(exc):
                    raise
                last_oom = exc
                logger.warning(
                    "CUDA OOM at startup with batch_size=%d seq_len=%d; retrying with smaller seq_len.",
                    batch_size,
                    seq_try,
                )
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        raise RuntimeError(
            "Serve startup failed due to CUDA OOM after runtime seq_len fallback attempts "
            f"(batch_size={batch_size}, requested_seq_len={seq_len}, tried={seq_candidates})."
        ) from last_oom

    @staticmethod
    def _resolve_min_stack_mb(config: ServeConfig) -> int:
        explicit_min = config.min_stack_mb
        if explicit_min is not None:
            return max(64, int(explicit_min))

        # Continuous serving allocates KV pages from the stack arena.
        # Keep a substantial floor by default to avoid under-provisioned page pools.
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
                # Budget from both reserved headroom and actual free memory.
                by_reserve = int(reserve_mb * 0.50)
                by_free = int(max(1024, free_mb * 0.25))
                auto_min_mb = max(1024, min(4096, by_reserve, by_free))
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
            auto_tune_tokens = os.getenv(
                "SUROGATE_SERVE_AUTO_MAX_BATCHED_TOKENS", "1"
            ).strip().lower() not in {"0", "false", "no"}
            if auto_tune_tokens and not config.is_explicit("max_num_batched_tokens"):
                tokens_per_seq = max(
                    64,
                    int(os.getenv("SUROGATE_SERVE_AUTO_TOKENS_PER_SEQ", "512")),
                )
                auto_budget = requested_batch_size * tokens_per_seq
                auto_budget = max(4096, min(32768, auto_budget))
                if auto_budget > max_batched_tokens:
                    logger.info(
                        "Auto-tuned runtime buffer capacity from %d to %d tokens "
                        "(batch_size=%d, tokens_per_seq=%d). ",
                        max_batched_tokens,
                        auto_budget,
                        requested_batch_size,
                        tokens_per_seq,
                    )
                    max_batched_tokens = auto_budget

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


def serve(config: ServeConfig):
    runtime = NativeServingRuntime(config)
    logger.info(f"Starting native inference server on {config.host}:{config.port}")
    runtime.serve_cpp()
