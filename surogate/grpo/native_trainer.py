"""Native GRPO trainer — single-process RL training with built-in generation.

Replaces the 3-component system (vLLM inference + orchestrator + trainer)
with a synchronous loop:
    1. Generate completions via trainer.generate()
    2. Score with reward provider (verifiers or callback)
    3. Compute advantages (per-problem baseline subtraction)
    4. Train with forward_for_grpo() + backward_grpo()
    5. Optimizer step

Supports: W&B monitoring, online evaluation, rollout filtering,
checkpointing with resume, temperature scheduling, LR scheduling.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
from transformers import AutoTokenizer

from surogate import _surogate
from surogate.grpo.config import GRPOLossConfig
from surogate.grpo.native_config import NativeGRPOConfig
from surogate.grpo.orchestrator.advantage import compute_advantages
from surogate.grpo.reward_provider import setup_reward_provider
from surogate.train.lr_schedule import LRSchedule
from surogate.utils.hf import get_model_weights_path
from surogate.utils.logger import get_logger
from surogate.utils.tensor import to_surogate_dtype

logger = get_logger()


class NativeGRPOTrainer:
    """Production-grade native GRPO trainer with single-process generation."""

    def __init__(
        self,
        config: NativeGRPOConfig,
        reward_fn: Callable[[list[str], list[str]], list[float]] | None = None,
    ):
        """
        Args:
            config: Native GRPO training configuration.
            reward_fn: Optional reward function for callback mode.
                Signature: fn(prompts, completions) -> rewards.
                If provided, overrides config.reward settings.
        """
        self.config = config

        # --- Tokenizer ---
        logger.info(f"Loading tokenizer for {config.model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_dir, trust_remote_code=True
        )
        self.eos_id = self.tokenizer.eos_token_id or 151643

        # --- Build DSL IR ---
        from surogate.dsl.ir_builder import build_dsl_ir_for_model

        dsl_extra = {}
        if getattr(config, "ep_size", 1) > 1:
            dsl_extra["ep_size"] = config.ep_size
        ir_json = build_dsl_ir_for_model(
            config.model_dir, extra_config=dsl_extra or None
        )

        # --- Compile JIT kernels ---
        from surogate.kernels.jit_compile import compile_jit_kernels

        jit_manifests = compile_jit_kernels(ir_json)

        # --- Runtime options ---
        # Start from the fully configured SFT runtime options (production parity),
        # then inject native-specific IR/JIT and hard requirements.
        rc = config.runtime_config
        options = rc
        options.dsl_ir_json = ir_json
        if jit_manifests:
            options.jit_kernel_manifests = jit_manifests
        options.use_cuda_graphs = False
        options.doc_masking = getattr(rc, "doc_masking", True)
        options.recompute = "true"

        # Ensure the DeviceMemoryStack is large enough for paged decode and
        # backward graph temps. Use RuntimeOptions (not process env vars).
        if int(getattr(options, "min_stack_mb", 0)) <= 0:
            gen = config.generation
            page_block_size = 256
            total_seqs = config.problems_per_step * gen.num_completions
            # Estimate KV cache bytes (conservative: full max_total_len per seq)
            try:
                from surogate.core.model.hf_config import HfConfigFactory

                hf_cfg = config.model_info.config
                n_layers = HfConfigFactory.get_config_attr(
                    hf_cfg, "num_hidden_layers"
                )
                n_kv_heads = HfConfigFactory.get_config_attr(
                    hf_cfg, "num_key_value_heads"
                )
                head_dim = HfConfigFactory.get_config_attr(
                    hf_cfg, "hidden_size"
                ) // HfConfigFactory.get_config_attr(hf_cfg, "num_attention_heads")
                max_prompt = 100
                max_total = max_prompt + gen.max_gen_len  # prompt + gen
                # Contiguous upper-bound estimate (kept as fallback)
                kv_bytes_contig = (
                    2 * n_kv_heads * head_dim * n_layers * 2 * total_seqs * max_total
                )
                # Paged GRPO estimate (matches GenerationEngine::generate_grpo budgeting).
                M = config.problems_per_step
                N = gen.num_completions
                shared_prefix_pages = (max_prompt + page_block_size - 1) // page_block_size
                private_partial_extra = (N - 1) if (max_prompt > 0 and N > 1) else 0
                prefix_pages_per_prompt = shared_prefix_pages + private_partial_extra
                suffix_pages_per_seq = (gen.max_gen_len + page_block_size - 1) // page_block_size
                total_pages = M * prefix_pages_per_prompt + M * N * suffix_pages_per_seq
                page_elems = page_block_size * n_kv_heads * head_dim
                k_pool_bytes = n_layers * total_pages * page_elems * 2  # bf16
                kv_bytes_paged = 2 * k_pool_bytes  # K + V
                kv_bytes = max(kv_bytes_contig, kv_bytes_paged)
                # Training activations: ~hidden_size * seq_len * 4 bytes per
                # intermediate tensor, with ~4 such tensors per layer
                hidden_size = HfConfigFactory.get_config_attr(hf_cfg, "hidden_size")
                train_bytes = hidden_size * config.sequence_len * 4 * 4 * n_layers
                total_bytes = kv_bytes * 3.2 + train_bytes * 2.0
                options.min_stack_mb = max(4096, int(total_bytes / (1024 * 1024)))
            except Exception:
                options.min_stack_mb = 8192
            logger.info(
                "Set RuntimeOptions.min_stack_mb=%d for %d sequences, seq_len=%d",
                int(options.min_stack_mb),
                int(total_seqs),
                int(config.sequence_len),
            )
        # --- Build LoRA config ---
        # Use a clean LoRA config with just rank/alpha from the YAML.
        # SFTConfig's lora_config may have target_modules='all' which
        # applies LoRA to all layers including LM head — this causes
        # buffer overflows during large-batch generation (B=128+).
        lora_config = None
        if config.lora:
            lora_config = _surogate.LoRAAdapterConfig()
            lora_config.rank = config.lora_rank
            lora_config.alpha = config.lora_alpha

        # --- Create C++ trainer ---
        logger.info(
            f"Creating native GRPO trainer for {config.model} ({config.gpus} GPUs)"
        )
        self.trainer = _surogate.SurogateTrainer(
            ngpu=config.gpus,
            config=_surogate.PretrainedConfig.from_pretrained(
                config.model_dir, to_surogate_dtype(config.torch_dtype)
            ),
            options=options,
            batch_size=config.per_device_train_batch_size,
            seq_len=config.sequence_len,
            grad_accum=1,  # Set dynamically per step
            memcpy_all_gather=True,
            memcpy_send_recv=True,
            lora_config=lora_config,
            qlora_config=config.qlora_config,
        )
        # Serialize low-level trainer API calls across producer/consumer threads.
        self._trainer_call_lock = threading.Lock()

        # --- Pad logprob for position 0 ---
        self._pad_logprob = None
        try:
            from surogate.core.model.hf_config import HfConfigFactory

            vocab_size = HfConfigFactory.get_config_attr(
                config.model_info.config, "vocab_size"
            )
        except Exception:
            vocab_size = None
        if vocab_size:
            self._pad_logprob = float(np.log(1.0 / float(vocab_size)))
        self._vocab_size = vocab_size or 151936  # Fallback for filter setup

        # --- Import weights ---
        model_weights_path = get_model_weights_path(config.model_dir)
        logger.info(f"Importing weights from {model_weights_path}")
        self.trainer.import_weights(model_weights_path)

        # --- LR schedule ---
        lr_max_steps = config.max_steps if config.max_steps > 0 else 1_000_000
        warmup_steps = config.warmup_steps
        if warmup_steps == 0 and config.warmup_ratio > 0:
            warmup_steps = int(lr_max_steps * config.warmup_ratio)
        self.lr_schedule = LRSchedule(
            base_lr=config.learning_rate,
            max_steps=lr_max_steps,
            warmup_steps=warmup_steps,
            cooldown_steps=config.cooldown_steps,
            final_lr=config.learning_rate * config.final_lr_fraction,
            schedule_type=config.lr_scheduler_type,
            wsd_decay_steps_fraction=getattr(
                config, "wsd_decay_steps_fraction", 0.0
            ),
        )

        # --- Reward provider ---
        self.reward_provider = setup_reward_provider(
            config=config.reward,
            tokenizer=self.tokenizer,
            reward_fn=reward_fn,
        )

        # Wire up the NativeClient for multi-turn verifiers rollouts
        from surogate.grpo.reward_provider import VerifiersRewardProvider

        if isinstance(self.reward_provider, VerifiersRewardProvider):
            self.reward_provider.set_trainer(
                trainer=self.trainer,
                max_gen_len=config.generation.max_gen_len,
                use_lora=config.lora,
                prefill_chunk_size=config.generation.prefill_chunk_size,
            )

        # --- Monitor ---
        self.monitor = None
        if config.report_to is not None:
            from surogate.grpo.utils.monitor import setup_monitor

            self.monitor = setup_monitor(
                wandb_config=config.report_to,
                output_dir=Path(config.output_dir),
                tokenizer=self.tokenizer,
                run_config=config,
            )

        # --- Teacher model (for KL distillation) ---
        self._teacher_trainer = None
        if config.teacher_model:
            from surogate.dsl.ir_builder import build_dsl_ir_for_model
            from huggingface_hub import snapshot_download

            teacher_dir = snapshot_download(config.teacher_model)
            teacher_ir = build_dsl_ir_for_model(teacher_dir)
            teacher_options = _surogate.RuntimeOptions()
            teacher_options.dsl_ir_json = teacher_ir

            teacher_jit = compile_jit_kernels(teacher_ir)
            if teacher_jit:
                teacher_options.jit_kernel_manifests = teacher_jit

            logger.info(f"Loading teacher model: {config.teacher_model}")
            self._teacher_trainer = _surogate.SurogateTrainer(
                ngpu=config.gpus,
                config=_surogate.PretrainedConfig.from_pretrained(
                    teacher_dir, to_surogate_dtype(config.torch_dtype)
                ),
                options=teacher_options,
                batch_size=config.per_device_train_batch_size,
                seq_len=config.sequence_len,
                grad_accum=1,
                memcpy_all_gather=True,
                memcpy_send_recv=True,
            )
            teacher_weights = get_model_weights_path(teacher_dir)
            self._teacher_trainer.import_weights(teacher_weights)
            logger.info("Teacher model loaded")

        # --- Filters ---
        self._filters = []
        if config.filters:
            from surogate.grpo.orchestrator.filters import setup_filters

            self._filters = setup_filters(config.filters, vocab_size=self._vocab_size)
            if self._filters:
                logger.info(
                    f"Initialized {len(self._filters)} rollout filter(s): "
                    f"{[f.name for f in self._filters]}"
                )

        # --- Eval environments ---
        self._eval_envs = []
        self._eval_env_names = []
        if config.eval and config.eval.env:
            import verifiers as vf

            for env_cfg in config.eval.env:
                env_id = (
                    env_cfg.id.split("@")[0] if "@" in env_cfg.id else env_cfg.id
                )
                env = vf.load_environment(env_id, **(env_cfg.args or {}))
                env_name = env_cfg.name or env_id
                self._eval_envs.append(env)
                self._eval_env_names.append(env_name)
                logger.info(f"Loaded eval env '{env_name}'")

    # ========================================================================
    # Generation
    # ========================================================================

    def _get_temperature(self, step: int) -> float:
        """Get temperature for the current step (supports scheduling)."""
        config = self.config
        if config.sampling is not None:
            from surogate.grpo.utils.temp_scheduling import compute_temperature

            return compute_temperature(step, config.sampling, config.max_steps)
        return 1.0

    def _encode_prompt(self, prompt: str) -> list[int]:
        """Encode a prompt into safe in-vocab token ids."""
        token_ids = list(self.tokenizer.encode(prompt, add_special_tokens=True))
        vocab_size = int(self._vocab_size)
        fallback = self.eos_id if 0 <= int(self.eos_id) < vocab_size else 0

        if not token_ids:
            return [fallback]

        safe_ids = []
        for tid in token_ids:
            tid_i = int(tid)
            if 0 <= tid_i < vocab_size:
                safe_ids.append(tid_i)
            else:
                safe_ids.append(fallback)
        return safe_ids

    def generate_rollout(
        self,
        prompts: list[str],
        step: int,
        prompt_token_ids: list[list[int]] | None = None,
    ) -> dict:
        """Generate completions for a batch of prompts.

        Returns dict with keys: prompt_texts, completion_texts, all_tokens,
        logprobs, prompt_lens, completion_lens, num_completions.
        """
        config = self.config
        gen = config.generation

        temperature = self._get_temperature(step)
        num_completions = gen.num_completions

        # Tokenize prompts (or use pre-tokenized IDs from overlap prefetch).
        if prompt_token_ids is None:
            prompt_token_ids = [self._encode_prompt(p) for p in prompts]
        elif len(prompt_token_ids) != len(prompts):
            raise ValueError(
                "generate_rollout: prompt_token_ids length must match prompts length"
            )

        # Generate completions with paged GRPO decode.
        gen_start = time.time()
        with self._trainer_call_lock:
            tokens, logprobs, prompt_lens, completion_lens = self.trainer.generate(
                prompts=prompt_token_ids,
                num_completions=num_completions,
                max_gen_len=gen.max_gen_len,
                temperature=temperature,
                eos_token_id=self.eos_id,
                use_lora=config.lora,
                use_cuda_graphs=True,
                top_k=gen.top_k,
                top_p=gen.top_p,
                prefill_chunk_size=gen.prefill_chunk_size,
            )
        gen_time = time.time() - gen_start

        # Decode completions
        total = len(tokens)
        completion_texts = []
        prompt_texts_expanded = []

        for i in range(total):
            prompt_idx = i // num_completions
            pl = prompt_lens[i]
            cl = completion_lens[i]
            comp_tokens = tokens[i][pl : pl + cl]
            completion_texts.append(
                self.tokenizer.decode(comp_tokens, skip_special_tokens=True)
            )
            prompt_texts_expanded.append(prompts[prompt_idx])

        return {
            "prompt_texts": prompt_texts_expanded,
            "completion_texts": completion_texts,
            "all_tokens": tokens,
            "logprobs": logprobs,
            "prompt_lens": prompt_lens,
            "completion_lens": completion_lens,
            "num_completions": num_completions,
            "temperature": temperature,
            "gen_time": gen_time,
        }

    def _fetch_and_tokenize_prompts(
        self, problems_per_step: int
    ) -> tuple[list[str], list[list[int]]]:
        """Fetch next prompt batch and pre-tokenize it.

        This runs in a background thread to overlap CPU prompt preparation with
        current-step GPU training work.
        """
        prompts = self.reward_provider.get_next_batch(problems_per_step)
        prompt_token_ids = [self._encode_prompt(p) for p in prompts]
        return prompts, prompt_token_ids

    # ========================================================================
    # Filtering
    # ========================================================================

    def _apply_native_filters(
        self,
        completion_token_ids: list[list[int]],
        completion_logprobs: list[list[float]],
    ) -> tuple[list[bool], dict[str, float]]:
        """Apply gibberish/repetition filters on raw token data.

        Returns (mask, metrics) where mask[i]=True means keep the rollout.
        """
        if not self._filters:
            return [True] * len(completion_token_ids), {}

        n = len(completion_token_ids)
        keep = [True] * n
        counts = {f.name: 0 for f in self._filters}

        for i in range(n):
            for filt in self._filters:
                # Build a minimal rollout-like structure for the filter
                rollout = {
                    "trajectory": [
                        {
                            "tokens": {
                                "completion_ids": completion_token_ids[i],
                                "completion_logprobs": completion_logprobs[i],
                                "completion_mask": [1] * len(completion_token_ids[i]),
                            }
                        }
                    ],
                    "metrics": {},
                }
                result = filt.check(rollout)
                if result.detected:
                    counts[filt.name] += 1
                    if filt.enforce:
                        keep[i] = False
                    break

        metrics = {}
        for f in self._filters:
            metrics[f"filter/{f.name}_count"] = float(counts[f.name])
            metrics[f"filter/{f.name}_rate"] = counts[f.name] / n if n > 0 else 0.0
        total_filtered = sum(1 for k in keep if not k)
        metrics["filter/total_enforced_rate"] = total_filtered / n if n > 0 else 0.0

        if total_filtered > 0:
            logger.info(
                f"Filtered {total_filtered}/{n} rollouts "
                f"({', '.join(f'{name}={c}' for name, c in counts.items() if c > 0)})"
            )

        return keep, metrics

    # ========================================================================
    # Training step
    # ========================================================================

    @property
    def _is_multiturn(self) -> bool:
        """True when using multi-turn verifiers rollouts via NativeClient.

        Multi-turn mode runs rollouts sequentially (one generate() per turn)
        which is needed for envs that inject tool outputs between turns.
        Single-turn mode (default) uses batched generate() + rubric scoring
        which is much faster for envs like reverse-text, math, code, etc.
        """
        if not (self.config.reward and self.config.reward.multiturn):
            return False
        from surogate.grpo.reward_provider import VerifiersRewardProvider

        return (
            isinstance(self.reward_provider, VerifiersRewardProvider)
            and self.reward_provider._native_client is not None
        )

    def train_step(
        self,
        prompts: list[str],
        step: int,
        prompt_token_ids: list[list[int]] | None = None,
        start_next_prefetch: Callable[[], None] | None = None,
    ) -> dict:
        """One GRPO training step: generate -> score -> train.

        In multi-turn verifiers mode, generation + scoring happen together
        inside the environment's rollout loop. In callback/single-turn mode,
        generation and scoring are separate steps.

        Returns metrics dict.
        """
        if self._is_multiturn:
            return self._train_step_multiturn(
                prompts,
                step,
                start_next_prefetch=start_next_prefetch,
            )
        return self._train_step_single(
            prompts,
            step,
            prompt_token_ids=prompt_token_ids,
            start_next_prefetch=start_next_prefetch,
        )

    def _train_step_single(
        self,
        prompts: list[str],
        step: int,
        prompt_token_ids: list[list[int]] | None = None,
        start_next_prefetch: Callable[[], None] | None = None,
    ) -> dict:
        """Single-turn training step: generate -> score -> train."""
        prepared = self._prepare_single_turn_payload(
            prompts,
            step,
            prompt_token_ids=prompt_token_ids,
            start_next_prefetch=start_next_prefetch,
        )
        return self._finalize_single_turn_payload(prepared, step)

    def _prepare_single_turn_payload(
        self,
        prompts: list[str],
        step: int,
        prompt_token_ids: list[list[int]] | None = None,
        start_next_prefetch: Callable[[], None] | None = None,
    ) -> dict:
        """Prepare a single-turn batch through rollout + scoring + advantages."""
        config = self.config
        num_completions = config.generation.num_completions

        rollout = self.generate_rollout(
            prompts, step, prompt_token_ids=prompt_token_ids
        )
        all_tokens = rollout["all_tokens"]
        all_logprobs = rollout["logprobs"]
        prompt_lens_list = rollout["prompt_lens"]
        completion_lens_list = rollout["completion_lens"]
        total = len(all_tokens)

        # Reward provider is thread-safe; prefetch can start during scoring.
        if start_next_prefetch is not None:
            start_next_prefetch()

        score_start = time.time()
        rewards = self.reward_provider.score(
            rollout["prompt_texts"],
            rollout["completion_texts"],
            num_completions,
        )
        rewards = np.array(rewards, dtype=np.float32)
        score_time = time.time() - score_start

        comp_token_ids = []
        comp_logprobs = []
        for i in range(total):
            pl = prompt_lens_list[i]
            cl = completion_lens_list[i]
            comp_token_ids.append(list(all_tokens[i][pl : pl + cl]))
            lp = all_logprobs[i] if len(all_logprobs[i]) >= cl else all_logprobs[i]
            comp_logprobs.append(list(lp[:cl]))

        keep_mask, filter_metrics = self._apply_native_filters(
            comp_token_ids, comp_logprobs
        )

        loss_mask_override = np.ones(total, dtype=bool)
        for i in range(total):
            if not keep_mask[i]:
                loss_mask_override[i] = False

        advantages = compute_advantages(
            rewards=rewards.tolist(),
            completion_lengths=[int(cl) for cl in completion_lens_list],
            samples_per_problem=num_completions,
            advantage_config=config.advantage,
        )
        advantages = np.array(advantages, dtype=np.float32)

        return {
            "prompts": prompts,
            "rollout": rollout,
            "rewards": rewards,
            "advantages": advantages,
            "all_tokens": all_tokens,
            "all_logprobs": all_logprobs,
            "prompt_lens": prompt_lens_list,
            "completion_lens": completion_lens_list,
            "loss_mask_override": loss_mask_override,
            "filter_metrics": filter_metrics,
            "score_time": score_time,
        }

    def _finalize_single_turn_payload(self, prepared: dict, step: int) -> dict:
        """Finalize a prepared single-turn batch via train + optimizer + metrics."""
        config = self.config
        train_start = time.time()
        step_metrics, loss_scale = self._train_on_trajectories(
            all_tokens=prepared["all_tokens"],
            all_logprobs=prepared["all_logprobs"],
            prompt_lens=prepared["prompt_lens"],
            completion_lens=prepared["completion_lens"],
            advantages=prepared["advantages"],
            loss_mask_override=prepared["loss_mask_override"],
            temperature=prepared["rollout"]["temperature"],
        )

        lr = self.lr_schedule.get_lr(step)
        result = self._optimizer_step(lr, step)
        train_time = time.time() - train_start

        return self._build_metrics(
            prompts=prepared["prompts"],
            rewards=prepared["rewards"],
            advantages=prepared["advantages"],
            prompt_lens=prepared["prompt_lens"],
            completion_lens=prepared["completion_lens"],
            step_metrics=step_metrics,
            loss_scale=loss_scale,
            grad_norm=result.get("norm", 0.0),
            lr=lr,
            temperature=prepared["rollout"]["temperature"],
            gen_time=prepared["rollout"]["gen_time"],
            score_time=prepared["score_time"],
            train_time=train_time,
            step=step,
            filter_metrics=prepared["filter_metrics"],
            num_completions=config.generation.num_completions,
        )

    def _train_step_multiturn(
        self,
        prompts: list[str],
        step: int,
        start_next_prefetch: Callable[[], None] | None = None,
    ) -> dict:
        """Multi-turn training step: rollouts (generate+score) -> train.

        Generation and scoring happen together inside env.run_rollout()
        via the NativeClient. We then extract trajectory tokens and logprobs
        from the rollout outputs for training.
        """
        from surogate.grpo.orchestrator.trajectories import interleave_rollout

        config = self.config
        num_completions = config.generation.num_completions

        # 1. Run multi-turn rollouts (generation + scoring in one pass)
        # Update NativeClient's generation params with scheduled temperature
        temperature = self._get_temperature(step)
        if hasattr(self.reward_provider, '_native_client') and self.reward_provider._native_client is not None:
            self.reward_provider._native_client.default_temperature = temperature

        gen_start = time.time()
        rewards = self.reward_provider.score(prompts, [], num_completions, use_rollouts=True)
        rewards = np.array(rewards, dtype=np.float32)
        gen_score_time = time.time() - gen_start

        # In multiturn mode, rollout extraction still touches reward-provider
        # state right after score(); defer overlap prefetch to caller fallback.
        _ = start_next_prefetch

        # 2. Extract trajectory data from rollout outputs
        rollout_outputs = self.reward_provider.get_rollout_data() or []

        # Convert rollout outputs to training samples via interleave_rollout
        all_tokens_list = []
        all_logprobs_list = []
        all_completion_masks = []  # Per-token mask from multi-turn interleaving
        prompt_lens_list = []
        completion_lens_list = []
        sample_advantages = []

        for rollout_idx, output in enumerate(rollout_outputs):
            if isinstance(output, Exception):
                continue

            # Use interleave_rollout to handle multi-turn merging
            samples = interleave_rollout(output)
            if samples is None:
                continue

            for sample in samples:
                # Build flat token sequence: prompt_ids + completion_ids
                full_tokens = sample.prompt_ids + sample.completion_ids
                pl = len(sample.prompt_ids)
                cl = len(sample.completion_ids)

                all_tokens_list.append(np.array(full_tokens, dtype=np.int32))
                all_logprobs_list.append(
                    np.array(sample.completion_logprobs, dtype=np.float32)
                )
                # Preserve the per-token completion_mask from interleave_rollout
                # (False for env response tokens between turns)
                all_completion_masks.append(
                    np.array(sample.completion_mask, dtype=bool)
                )
                prompt_lens_list.append(pl)
                completion_lens_list.append(cl)
                sample_advantages.append(rollout_idx)  # Track which rollout this sample came from

        total = len(all_tokens_list)
        if total == 0:
            logger.warning("No valid rollout data for training. Skipping step.")
            return self._build_metrics(
                prompts=prompts,
                rewards=rewards,
                advantages=np.array([], dtype=np.float32),
                prompt_lens=[],
                completion_lens=[],
                step_metrics={"policy_loss": 0.0, "mismatch_kl": 0.0, "is_masked": 0.0, "keep_tokens": 0, "total_tokens": 0},
                loss_scale=1,
                grad_norm=0.0,
                lr=self.lr_schedule.get_lr(step),
                temperature=self._get_temperature(step),
                gen_time=gen_score_time,
                score_time=0.0,
                train_time=0.0,
                step=step,
                filter_metrics={},
                num_completions=config.generation.num_completions,
            )

        # 3. Compute advantages using per-rollout rewards
        advantages_per_rollout = compute_advantages(
            rewards=rewards.tolist(),
            completion_lengths=[1] * len(rewards),  # Per-rollout, not per-token
            samples_per_problem=num_completions,
            advantage_config=config.advantage,
        )
        advantages_per_rollout = np.array(advantages_per_rollout, dtype=np.float32)

        # Map per-rollout advantages to per-sample
        advantages = np.array(
            [advantages_per_rollout[idx] for idx in sample_advantages],
            dtype=np.float32,
        )

        # 4. Build loss masks from completion_mask (respects multi-turn masking)
        loss_mask_override = np.ones(total, dtype=bool)

        # 5. Training forward/backward
        train_start = time.time()
        step_metrics, loss_scale = self._train_on_trajectories(
            all_tokens=all_tokens_list,
            all_logprobs=all_logprobs_list,
            prompt_lens=prompt_lens_list,
            completion_lens=completion_lens_list,
            advantages=advantages,
            loss_mask_override=loss_mask_override,
            completion_masks=all_completion_masks,
            temperature=temperature,
        )

        # 6. Optimizer step
        lr = self.lr_schedule.get_lr(step)
        result = self._optimizer_step(lr, step)
        train_time = time.time() - train_start

        temperature = self._get_temperature(step)

        return self._build_metrics(
            prompts=prompts,
            rewards=rewards,
            advantages=advantages,
            prompt_lens=prompt_lens_list,
            completion_lens=completion_lens_list,
            step_metrics=step_metrics,
            loss_scale=loss_scale,
            grad_norm=result.get("norm", 0.0),
            lr=lr,
            temperature=temperature,
            gen_time=gen_score_time,
            score_time=0.0,  # Included in gen_time for multi-turn
            train_time=train_time,
            step=step,
            filter_metrics={},
            num_completions=config.generation.num_completions,
        )

    # ========================================================================
    # Shared training internals
    # ========================================================================

    def _train_on_trajectories(
        self,
        all_tokens: list,
        all_logprobs: list,
        prompt_lens: list[int],
        completion_lens: list[int],
        advantages: np.ndarray,
        loss_mask_override: np.ndarray,
        temperature: float,
        completion_masks: list[np.ndarray] | None = None,
    ) -> tuple[dict, int]:
        """Pack trajectories and run forward_for_grpo + backward_grpo.

        Uses First Fit Decreasing packing to combine multiple short
        trajectories into packed sequences, dramatically reducing the
        number of forward/backward passes (e.g. 128 trajectories → ~12
        packed micro-batches with seq_len=2048).

        Returns (step_metrics, loss_scale).
        """
        from surogate.grpo.batch import packed_samples_into_micro_bs
        from surogate.grpo.transport.types import MicroBatch

        config = self.config
        seq_len = config.sequence_len
        ngpu = config.gpus
        loss_config = config.loss or GRPOLossConfig()
        total = len(all_tokens)

        # --- Build pre-packed samples directly (avoid intermediate TrainingSample objects) ---
        packed_samples: list[tuple[int, MicroBatch]] = []
        for i in range(total):
            pl = prompt_lens[i]
            cl = completion_lens[i]
            toks = all_tokens[i]

            prompt_ids = list(toks[:pl])
            completion_ids = list(toks[pl : pl + cl])
            if completion_masks is not None and loss_mask_override[i]:
                comp_mask = list(completion_masks[i][:cl].astype(bool))
            elif loss_mask_override[i]:
                comp_mask = [True] * cl
            else:
                comp_mask = [False] * cl

            traj_lp = all_logprobs[i]
            comp_lp = (
                list(traj_lp[:cl])
                if len(traj_lp) >= cl
                else list(traj_lp) + [0.0] * (cl - len(traj_lp))
            )

            input_ids = prompt_ids + completion_ids
            loss_mask_list = ([False] * pl) + comp_mask
            inference_lp = ([0.0] * pl) + comp_lp
            adv_tokens = [float(advantages[i])] * len(input_ids)
            position_ids = list(range(len(input_ids)))
            prompt_temp = temperature if cl > 0 else 1.0
            temperatures = ([prompt_temp] * pl) + ([temperature] * cl)

            if len(input_ids) > seq_len:
                input_ids = input_ids[:seq_len]
                loss_mask_list = loss_mask_list[:seq_len]
                inference_lp = inference_lp[:seq_len]
                adv_tokens = adv_tokens[:seq_len]
                position_ids = position_ids[:seq_len]
                temperatures = temperatures[:seq_len]

            packed_samples.append(
                (
                    0,
                    MicroBatch(
                        input_ids=input_ids,
                        loss_mask=loss_mask_list,
                        advantages=adv_tokens,
                        inference_logprobs=inference_lp,
                        position_ids=position_ids,
                        temperatures=temperatures,
                    ),
                )
            )

        # --- Pack into micro-batches using First Fit Decreasing ---
        micro_batches = packed_samples_into_micro_bs(packed_samples, seq_len, num_loras=1)
        n_mb = len(micro_batches)

        # Compute loss_scale across all micro-batches
        loss_scale = int(sum(sum(1 for m in mb.loss_mask if m) for mb in micro_batches))
        loss_scale = max(loss_scale, 1)

        with self._trainer_call_lock:
            self.trainer.set_grad_accumulation(n_mb)

        step_metrics = {
            "policy_loss": 0.0,
            "mismatch_kl": 0.0,
            "is_masked": 0.0,
            "keep_tokens": 0,
            "total_tokens": 0,
        }

        # Reusable staging buffers (avoid per-micro-batch np.pad/np.tile allocations).
        input_single = np.zeros((1, seq_len), dtype=np.int32)
        masked_targets_single = np.full((1, seq_len), -100, dtype=np.int32)
        position_single = np.zeros((1, seq_len), dtype=np.int32)
        temp_single = np.ones((1, seq_len), dtype=np.float32)
        inf_lp_single = np.zeros((1, seq_len), dtype=np.float32)
        adv_single = np.zeros((1, seq_len), dtype=np.float32)
        loss_mask_single = np.zeros((1, seq_len), dtype=np.uint8)

        if ngpu > 1:
            input_stage = np.zeros((ngpu, seq_len), dtype=np.int32)
            masked_targets_stage = np.full((ngpu, seq_len), -100, dtype=np.int32)
            position_stage = np.zeros((ngpu, seq_len), dtype=np.int32)
            temp_stage = np.ones((ngpu, seq_len), dtype=np.float32)
        else:
            input_stage = input_single
            masked_targets_stage = masked_targets_single
            position_stage = position_single
            temp_stage = temp_single

        for mb in micro_batches:
            T_actual = len(mb.input_ids)

            input_single.fill(0)
            masked_targets_single.fill(-100)
            position_single.fill(0)
            temp_single.fill(1.0)
            inf_lp_single.fill(0.0)
            adv_single.fill(0.0)
            loss_mask_single.fill(0)

            if T_actual > 0:
                input_single[0, :T_actual] = mb.input_ids[:T_actual]
                position_single[0, :T_actual] = mb.position_ids[:T_actual]
                if T_actual < seq_len:
                    last_pos = int(mb.position_ids[T_actual - 1])
                    position_single[0, T_actual:] = np.arange(
                        last_pos + 1,
                        last_pos + 1 + (seq_len - T_actual),
                        dtype=np.int32,
                    )
                temp_single[0, :T_actual] = mb.temperatures[:T_actual]
                inf_lp_single[0, :T_actual] = mb.inference_logprobs[:T_actual]
                adv_single[0, :T_actual] = mb.advantages[:T_actual]
                loss_mask_single[0, :T_actual] = np.asarray(
                    mb.loss_mask[:T_actual], dtype=np.uint8
                )

                if T_actual > 1:
                    masked_targets_single[0, : T_actual - 1] = np.asarray(
                        mb.input_ids[1:T_actual], dtype=np.int32
                    )
                    masked_targets_single[0, : T_actual - 1][
                        loss_mask_single[0, 1:T_actual] == 0
                    ] = -100

            if ngpu > 1:
                input_stage[:] = input_single
                masked_targets_stage[:] = masked_targets_single
                position_stage[:] = position_single
                temp_stage[:] = temp_single

            with self._trainer_call_lock:
                mb_metrics = self.trainer.step_with_native_grpo(
                    input_stage,
                    masked_targets_stage,
                    inference_logprobs=inf_lp_single,
                    advantages=adv_single,
                    loss_mask=loss_mask_single,
                    kl_tau=float(loss_config.kl_tau),
                    adv_tau=float(loss_config.adv_tau),
                    ipo_mask_low=float(loss_config.ipo_mask_low),
                    ipo_mask_high=float(loss_config.ipo_mask_high),
                    loss_scale=float(loss_scale),
                    position_ids=position_stage,
                    temperatures=temp_stage,
                )

            for key in step_metrics:
                if key in mb_metrics:
                    step_metrics[key] += mb_metrics[key]

        # Normalize averaged metrics
        if n_mb > 0:
            for key in ("policy_loss", "mismatch_kl", "is_masked"):
                step_metrics[key] /= max(n_mb, 1)

        return step_metrics, loss_scale

    def _optimizer_step(self, lr: float, step: int) -> dict:
        """Run optimizer step and return result dict."""
        config = self.config
        opt_config = _surogate.OptimizerConfig(
            optimizer=config.optimizer,
            learning_rate=lr,
            weight_decay=config.weight_decay,
            grad_clip=config.max_grad_norm,
            adamw_beta1=config.adamw_beta1,
            adamw_beta2=config.adamw_beta2,
            adamw_epsilon=config.adamw_epsilon,
        )
        with self._trainer_call_lock:
            return self.trainer.update_with_config(opt_config, step + 1)

    @staticmethod
    def _build_metrics(
        prompts: list[str],
        rewards: np.ndarray,
        advantages: np.ndarray,
        prompt_lens: list[int],
        completion_lens: list[int],
        step_metrics: dict,
        loss_scale: int,
        grad_norm: float,
        lr: float,
        temperature: float,
        gen_time: float,
        score_time: float,
        train_time: float,
        step: int,
        filter_metrics: dict,
        num_completions: int,
    ) -> dict:
        """Build the metrics dict returned by train_step."""
        total = len(prompt_lens)
        M = len(prompts)
        N = num_completions

        # Batch statistics (only when rewards align with M*N)
        if len(rewards) == M * N:
            reward_groups = rewards.reshape(M, N)
            solve_all = float((reward_groups.sum(axis=1) == N).mean())
            solve_none = float((reward_groups.sum(axis=1) == 0).mean())
        else:
            solve_all = 0.0
            solve_none = 0.0
        effective_batch_size = 1 - solve_none - solve_all

        num_tokens = sum(
            int(pl + cl) for pl, cl in zip(prompt_lens, completion_lens)
        )
        total_time = gen_time + score_time + train_time

        return {
            # Rewards
            "reward/mean": float(rewards.mean()) if len(rewards) > 0 else 0.0,
            "reward/std": float(rewards.std()) if len(rewards) > 0 else 0.0,
            "reward/min": float(rewards.min()) if len(rewards) > 0 else 0.0,
            "reward/max": float(rewards.max()) if len(rewards) > 0 else 0.0,
            "reward/median": float(np.median(rewards)) if len(rewards) > 0 else 0.0,
            # Advantages
            "advantage/mean": float(advantages.mean()) if len(advantages) > 0 else 0.0,
            "advantage/std": float(advantages.std()) if len(advantages) > 0 else 0.0,
            # Batch
            "batch/solve_none": solve_none,
            "batch/solve_all": solve_all,
            "batch/effective_batch_size": effective_batch_size,
            # Progress
            "progress/tokens": num_tokens,
            "progress/samples": total,
            "progress/problems": M,
            # Sequence lengths
            "seq_len/mean": float(np.mean([pl + cl for pl, cl in zip(prompt_lens, completion_lens)])) if total > 0 else 0.0,
            "decode_len/mean": float(np.mean(completion_lens)) if total > 0 else 0.0,
            # Training
            "train/policy_loss": step_metrics.get("policy_loss", 0.0),
            "train/mismatch_kl": step_metrics.get("mismatch_kl", 0.0),
            "train/is_masked": step_metrics.get("is_masked", 0.0),
            "train/keep_tokens": step_metrics.get("keep_tokens", 0),
            "train/total_tokens": step_metrics.get("total_tokens", 0),
            "train/grad_norm": grad_norm,
            "train/lr": lr,
            "train/loss_scale": loss_scale,
            # Sampling
            "sampling/temperature": temperature,
            # Timing
            "time/generate": gen_time,
            "time/score": score_time,
            "time/train": train_time,
            "time/step": total_time,
            # Throughput
            "perf/throughput": num_tokens / max(total_time, 1e-6),
            # Step
            "step": step,
            # Filters
            **filter_metrics,
        }

    # ========================================================================
    # Evaluation
    # ========================================================================

    def evaluate(self, step: int) -> dict:
        """Run online evaluation against configured eval environments.

        Returns metrics dict with eval/ prefix.
        """
        if not self._eval_envs:
            return {}

        eval_config = self.config.eval
        gen_config = self.config.generation
        max_gen_len = eval_config.max_gen_len or gen_config.max_gen_len
        temperature = eval_config.temperature

        all_metrics = {}

        for env, env_name in zip(self._eval_envs, self._eval_env_names):
            logger.info(f"Evaluating on '{env_name}'")

            # Get eval dataset
            num_examples = eval_config.num_examples
            ds = env.get_dataset(n=num_examples)
            if not ds:
                logger.warning(f"No eval examples for '{env_name}'")
                continue

            # Extract prompts
            eval_prompts = []
            for ex in ds:
                prompt_messages = ex.get("prompt", [])
                if isinstance(prompt_messages, list) and prompt_messages:
                    try:
                        text = self.tokenizer.apply_chat_template(
                            prompt_messages, tokenize=False, add_generation_prompt=True
                        )
                    except Exception:
                        text = "\n".join(
                            m.get("content", "")
                            for m in prompt_messages
                            if isinstance(m, dict)
                        )
                else:
                    text = str(prompt_messages)
                eval_prompts.append(text)

            # Generate completions
            prompt_token_ids = [self._encode_prompt(p) for p in eval_prompts]
            rollouts_per = eval_config.rollouts_per_example

            # Guard temperature: use small value for near-greedy instead of exact 0
            gen_temp = max(temperature, 1e-6)

            tokens, logprobs, prompt_lens, completion_lens = self.trainer.generate(
                prompts=prompt_token_ids,
                num_completions=rollouts_per,
                max_gen_len=max_gen_len,
                temperature=gen_temp,
                eos_token_id=self.eos_id,
                use_lora=self.config.lora,
                use_cuda_graphs=True,
                top_k=0,
                top_p=1.0,
                prefill_chunk_size=gen_config.prefill_chunk_size,
            )

            # Decode completions
            total = len(tokens)
            completions = []
            prompt_texts = []
            for i in range(total):
                prompt_idx = i // rollouts_per
                pl = prompt_lens[i]
                cl = completion_lens[i]
                comp = self.tokenizer.decode(
                    tokens[i][pl : pl + cl], skip_special_tokens=True
                )
                completions.append(comp)
                prompt_texts.append(eval_prompts[prompt_idx])

            # Score against the EVAL environment's rubric (not training env)
            import asyncio

            eval_states = []
            for i, (prompt_text, completion) in enumerate(zip(prompt_texts, completions)):
                example_idx = i // rollouts_per
                example = ds[example_idx]
                eval_states.append({
                    "example_id": example.get("example_id", example_idx),
                    "task": example.get("task", env_name),
                    "prompt": example.get("prompt", [{"role": "user", "content": prompt_text}]),
                    "completion": [{"role": "assistant", "content": completion}],
                    "answer": example.get("answer", ""),
                    "info": example.get("info", {}),
                    "reward": 0.0,
                    "metrics": {},
                    "timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0},
                    "is_completed": True,
                    "is_truncated": False,
                    "trajectory": [],
                    "stop_condition": None,
                    "error": None,
                })

            async def _eval_score():
                await env.rubric.score_group(eval_states)
                return [s.get("reward", 0.0) for s in eval_states]

            rewards = np.array(asyncio.run(_eval_score()), dtype=np.float32)

            # Compute eval metrics
            all_metrics[f"eval/{env_name}/reward_mean"] = float(rewards.mean())
            all_metrics[f"eval/{env_name}/reward_std"] = float(rewards.std())
            all_metrics[f"eval/{env_name}/accuracy"] = float((rewards > 0).mean())
            all_metrics[f"eval/{env_name}/num_examples"] = len(ds)

            logger.info(
                f"Eval '{env_name}': reward={rewards.mean():.4f} "
                f"accuracy={(rewards > 0).mean():.2%} "
                f"({len(ds)} examples)"
            )

        return all_metrics

    # ========================================================================
    # Checkpointing
    # ========================================================================

    def save_checkpoint(self, step: int):
        """Save training checkpoint."""
        config = self.config
        ckpt_dir = config.checkpoint_dir or str(
            Path(config.output_dir) / "checkpoints"
        )
        step_dir = Path(ckpt_dir) / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save C++ trainer state
        self.trainer.save_checkpoint(str(step_dir), step)

        # Save Python state
        state = {
            "step": step,
            "config": {
                "model": config.model,
                "learning_rate": config.learning_rate,
                "max_steps": config.max_steps,
            },
        }
        with open(step_dir / "native_grpo_state.json", "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved checkpoint at step {step} to {step_dir}")

    def _resume_from_checkpoint(self) -> int:
        """Resume training from a checkpoint. Returns the step to start from."""
        ckpt_path = self.config.resume_from_checkpoint
        if not ckpt_path:
            return 0

        ckpt_dir = Path(ckpt_path)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

        # Load Python state
        state_file = ckpt_dir / "native_grpo_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            resume_step = state.get("step", 0)
        else:
            # Try to infer step from directory name (step_N)
            try:
                resume_step = int(ckpt_dir.name.split("_")[-1])
            except (ValueError, IndexError):
                resume_step = 0

        # Load C++ trainer state
        self.trainer.load_checkpoint(str(ckpt_dir), resume_step)
        logger.info(f"Resumed from checkpoint at step {resume_step} ({ckpt_dir})")

        return resume_step + 1  # Start from the next step

    def _export_final(self):
        """Export final adapter or model."""
        config = self.config
        output_path = Path(config.output_dir)

        if config.lora:
            adapter_dir = output_path / "final_adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            self.trainer.export_adapter(str(adapter_dir))
            logger.info(f"Final LoRA adapter saved to {adapter_dir}")
        else:
            model_dir = output_path / "final_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            self.trainer.export_model(str(model_dir))
            # Copy tokenizer files
            self._copy_tokenizer_files(config.model_dir, str(model_dir))
            logger.info(f"Final model saved to {model_dir}")

    @staticmethod
    def _copy_tokenizer_files(src_dir: str, dst_dir: str):
        """Copy tokenizer and config files from source model to output."""
        tokenizer_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "chat_template.jinja",
            "generation_config.json",
        ]
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        for filename in tokenizer_files:
            src = src_path / filename
            if src.exists():
                shutil.copy(src, dst_path / filename)

    # ========================================================================
    # Main training loop
    # ========================================================================

    def train(self):
        """Main GRPO training loop."""
        config = self.config
        max_steps = config.max_steps or int(1e9)
        save_steps = config.save_steps
        eval_interval = config.eval.interval if config.eval else 0
        problems_per_step = config.problems_per_step

        logger.info("Starting native GRPO training")
        logger.info(f"  Model: {config.model}")
        logger.info(f"  GPUs: {config.gpus}")
        logger.info(f"  Sequence length: {config.sequence_len}")
        logger.info(f"  Problems per step: {problems_per_step}")
        logger.info(
            f"  Completions per problem: {config.generation.num_completions}"
        )
        logger.info(f"  Max gen length: {config.generation.max_gen_len}")
        logger.info(f"  Prefill chunk size: {config.generation.prefill_chunk_size}")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(
            f"  LoRA: enabled={config.lora}, "
            f"rank={config.lora_rank}, alpha={config.lora_alpha}"
        )
        logger.info(f"  Doc masking: {getattr(config, 'doc_masking', True)}")
        logger.info(f"  Optimizer: {config.optimizer}")
        logger.info(f"  Max steps: {max_steps}")
        if config.loss:
            logger.info(
                f"  Loss: kl_tau={config.loss.kl_tau}, adv_tau={config.loss.adv_tau}, "
                f"ipo_mask_low={config.loss.ipo_mask_low}, "
                f"ipo_mask_high={config.loss.ipo_mask_high}"
            )

        # Ensure output dir exists
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Resume from checkpoint if configured
        start_step = self._resume_from_checkpoint()
        if start_step > 0:
            logger.info(f"  Resuming from step {start_step}")

        total_tokens = 0
        total_samples = 0

        def _handle_step_metrics(step: int, metrics: dict):
            nonlocal total_tokens, total_samples
            total_tokens += metrics.get("progress/tokens", 0)
            total_samples += metrics.get("progress/samples", 0)
            metrics["progress/total_tokens"] = total_tokens
            metrics["progress/total_samples"] = total_samples

            if self.monitor is not None:
                self.monitor.log(metrics, step=step)
                self.monitor.flush(step=step)

            logger.info(
                f"Step {step} | "
                f"Reward: {metrics['reward/mean']:.4f} | "
                f"Loss: {metrics['train/policy_loss']:.4f} | "
                f"KL: {metrics['train/mismatch_kl']:.4f} | "
                f"Grad: {metrics['train/grad_norm']:.4f} | "
                f"LR: {metrics['train/lr']:.2e} | "
                f"Temp: {metrics['sampling/temperature']:.2f} | "
                f"Time: {metrics['time/step']:.2f}s"
            )

            if eval_interval > 0 and step % eval_interval == 0:
                eval_metrics = self.evaluate(step)
                if eval_metrics and self.monitor is not None:
                    self.monitor.log(eval_metrics, step=step)
                    self.monitor.flush(step=step)

            if save_steps > 0 and step > 0 and step % save_steps == 0:
                self.save_checkpoint(step)

        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="native-grpo-prefetch"
        ) as prefetch_executor:
            next_batch_future: Future[
                tuple[list[str], list[list[int]]]
            ] = prefetch_executor.submit(
                self._fetch_and_tokenize_prompts, problems_per_step
            )

            for step in range(start_step, max_steps):
                wait_start = time.time()
                prompts, prompt_token_ids = next_batch_future.result()
                prefetch_wait = time.time() - wait_start

                next_holder: dict[
                    str, Future[tuple[list[str], list[list[int]]]] | None
                ] = {"future": None}

                def _start_next_prefetch():
                    if step + 1 >= max_steps:
                        return
                    if next_holder["future"] is None:
                        next_holder["future"] = prefetch_executor.submit(
                            self._fetch_and_tokenize_prompts, problems_per_step
                        )

                metrics = self.train_step(
                    prompts,
                    step,
                    prompt_token_ids=prompt_token_ids,
                    start_next_prefetch=_start_next_prefetch,
                )
                metrics["time/prefetch_wait"] = prefetch_wait

                if step + 1 < max_steps:
                    if next_holder["future"] is None:
                        next_holder["future"] = prefetch_executor.submit(
                            self._fetch_and_tokenize_prompts, problems_per_step
                        )
                    next_batch_future = next_holder["future"]

                _handle_step_metrics(step, metrics)

        # Final export
        self._export_final()

        # Final checkpoint
        if save_steps > 0:
            self.save_checkpoint(max_steps)

        logger.info(
            f"Native GRPO training complete after {max_steps} steps "
            f"({total_tokens} tokens, {total_samples} samples)"
        )


def native_grpo_train(
    config: NativeGRPOConfig,
    reward_fn: Callable[[list[str], list[str]], list[float]] | None = None,
):
    """Entry point for native GRPO training.

    Args:
        config: Native GRPO configuration.
        reward_fn: Optional reward function. If provided, overrides config.reward.
    """
    trainer = NativeGRPOTrainer(config, reward_fn=reward_fn)
    trainer.train()
