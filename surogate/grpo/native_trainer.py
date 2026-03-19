"""Native GRPO trainer — uses the built-in generation engine instead of vLLM.

Replaces the orchestrator + vLLM + transport chain with a single-process loop:
    1. Generate completions via trainer.generate()
    2. Score with reward function
    3. Compute advantages
    4. Train with forward_for_grpo() + backward_grpo()

Usage:
    from surogate.grpo.native_trainer import NativeGRPOTrainer
    trainer = NativeGRPOTrainer(config)
    trainer.train()
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

import numpy as np

from surogate import _surogate
from surogate.grpo.config import GRPOTrainConfig
from surogate.grpo.loss import compute_grpo_per_token_grads
from surogate.grpo.orchestrator.advantage import compute_advantages
from surogate.train.lr_schedule import LRSchedule
from surogate.utils.hf import get_model_weights_path
from surogate.utils.logger import get_logger

logger = get_logger()


class NativeGRPOTrainer:
    """Self-contained GRPO trainer with native generation (no vLLM)."""

    def __init__(
        self,
        config: GRPOTrainConfig,
        reward_fn: Callable[[List[str], List[str]], List[float]],
        tokenizer,
    ):
        """
        Args:
            config: GRPO training configuration.
            reward_fn: Callable(prompts, completions) → rewards.
                       prompts: list of prompt strings.
                       completions: list of completion strings.
                       Returns: list of float rewards.
            tokenizer: HuggingFace tokenizer for encoding/decoding.
        """
        self.config = config
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer

        # Build DSL IR
        from surogate.dsl.ir_builder import build_dsl_ir_for_model
        dsl_extra = {}
        if getattr(config, "ep_size", 1) > 1:
            dsl_extra["ep_size"] = config.ep_size
        ir_json = build_dsl_ir_for_model(config.model_dir, extra_config=dsl_extra or None)
        config.runtime_config.dsl_ir_json = ir_json

        # Compile JIT kernels
        from surogate.kernels.jit_compile import compile_jit_kernels
        jit_manifests = compile_jit_kernels(ir_json)
        if jit_manifests:
            config.runtime_config.jit_kernel_manifests = jit_manifests

        # Create C++ trainer
        self.trainer = _surogate.SurogateTrainer(
            ngpu=config.gpus,
            config=_surogate.PretrainedConfig.from_pretrained(
                config.model_dir, config.torch_dtype or "bf16"),
            options=config.runtime_config,
            batch_size=config.per_device_train_batch_size,
            seq_len=config.sequence_len,
            grad_accum=1,
            memcpy_all_gather=config.memcpy_all_gather,
            memcpy_send_recv=config.memcpy_send_recv,
            lora_config=config.lora_config,
            qlora_config=config.qlora_config,
        )

        # Import weights
        model_weights_path = get_model_weights_path(config.model_dir)
        logger.info(f"Importing weights from {model_weights_path}")
        self.trainer.import_weights(model_weights_path)

        # LR schedule
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
        )

    def generate_rollout(
        self,
        prompts: List[str],
        num_completions: int = 4,
        max_gen_len: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> dict:
        """Generate completions for a batch of prompts.

        Returns dict with keys:
            prompt_texts: List[str]
            completion_texts: List[str]
            prompt_token_ids: List[List[int]]
            completion_token_ids: List[List[int]]
            logprobs: List[List[float]]
            prompt_lens: List[int]
            completion_lens: List[int]
        """
        # Tokenize prompts
        prompt_token_ids = [
            self.tokenizer.encode(p) for p in prompts
        ]

        eos_id = self.tokenizer.eos_token_id or 151643

        # Generate via native engine
        tokens, logprobs, prompt_lens, completion_lens = self.trainer.generate(
            prompts=prompt_token_ids,
            num_completions=num_completions,
            max_gen_len=max_gen_len,
            temperature=temperature,
            eos_token_id=eos_id,
            use_lora=True,  # Generate with policy (LoRA)
            top_k=top_k,
            top_p=top_p,
        )

        # Decode completions
        total = len(tokens)
        completion_texts = []
        completion_token_ids_list = []
        prompt_texts_expanded = []

        for i in range(total):
            prompt_idx = i // num_completions
            pl = prompt_lens[i]
            cl = completion_lens[i]
            comp_tokens = tokens[i][pl:pl + cl]
            completion_token_ids_list.append(comp_tokens)
            completion_texts.append(self.tokenizer.decode(comp_tokens, skip_special_tokens=True))
            prompt_texts_expanded.append(prompts[prompt_idx])

        return {
            "prompt_texts": prompt_texts_expanded,
            "completion_texts": completion_texts,
            "all_tokens": tokens,
            "logprobs": logprobs,
            "prompt_lens": prompt_lens,
            "completion_lens": completion_lens,
            "num_completions": num_completions,
        }

    def train_step(
        self,
        prompts: List[str],
        step: int,
        num_completions: int = 4,
        max_gen_len: int = 512,
        temperature: float = 1.0,
    ) -> dict:
        """One GRPO training step: generate → score → compute advantages → train.

        Returns metrics dict.
        """
        config = self.config

        # 1. Generate rollout
        rollout = self.generate_rollout(
            prompts=prompts,
            num_completions=num_completions,
            max_gen_len=max_gen_len,
            temperature=temperature,
        )

        # 2. Score with reward function
        rewards = self.reward_fn(
            rollout["prompt_texts"],
            rollout["completion_texts"],
        )
        rewards = np.array(rewards, dtype=np.float32)

        # 3. Compute advantages (GRPO: normalize within each prompt group)
        M = len(prompts)
        N = num_completions
        advantages = compute_advantages(
            rewards.reshape(M, N),
            method="grpo",
        ).flatten()

        # 4. Build training batch from rollout
        # Pack all trajectories into a single sequence for the training forward
        all_tokens = rollout["all_tokens"]
        all_logprobs = rollout["logprobs"]
        prompt_lens_list = rollout["prompt_lens"]
        completion_lens_list = rollout["completion_lens"]

        seq_len = config.sequence_len
        ngpu = config.gpus

        # Process each trajectory as a micro-batch
        self.trainer.set_grad_accumulation(len(all_tokens))
        loss_scale = max(sum(completion_lens_list), 1)

        for i, (traj_tokens, traj_lp) in enumerate(zip(all_tokens, all_logprobs)):
            pl = prompt_lens_list[i]
            cl = completion_lens_list[i]
            T_actual = pl + cl

            # Pad to seq_len
            input_padded = np.zeros((1, seq_len), dtype=np.int32)
            input_padded[0, :min(T_actual, seq_len)] = traj_tokens[:seq_len]

            # Targets: shifted input_ids
            targets_padded = np.full((1, seq_len), -100, dtype=np.int32)
            for t in range(min(T_actual - 1, seq_len - 1)):
                if t >= pl - 1:  # Only loss on completion tokens
                    targets_padded[0, t] = traj_tokens[t + 1]

            pos_padded = np.zeros((1, seq_len), dtype=np.int32)
            for t in range(min(T_actual, seq_len)):
                pos_padded[0, t] = t

            if ngpu > 1:
                input_padded = np.tile(input_padded, (ngpu, 1))
                targets_padded = np.tile(targets_padded, (ngpu, 1))
                pos_padded = np.tile(pos_padded, (ngpu, 1))

            # Forward: get policy logprobs
            raw_lp = self.trainer.forward_for_grpo(
                input_padded, targets_padded, position_ids=pos_padded)
            raw_lp = np.asarray(raw_lp[0, :T_actual], dtype=np.float32)

            # Shift logprobs to align with token positions
            trainer_logprobs = np.zeros(T_actual, dtype=np.float32)
            if T_actual > 1:
                trainer_logprobs[1:T_actual] = raw_lp[:T_actual - 1]

            # Build loss mask (1 for completion tokens, 0 for prompt)
            loss_mask = np.zeros(T_actual, dtype=np.float32)
            loss_mask[pl:pl + cl] = 1.0

            # Inference logprobs from rollout
            inference_lp = np.zeros(T_actual, dtype=np.float32)
            if len(traj_lp) > 0:
                inference_lp[pl:pl + min(cl, len(traj_lp))] = np.array(traj_lp[:cl], dtype=np.float32)

            # Compute per-token GRPO gradients
            per_token_grads, _ = compute_grpo_per_token_grads(
                trainer_logprobs=trainer_logprobs,
                inference_logprobs=inference_lp[:T_actual],
                advantages=np.full(T_actual, advantages[i], dtype=np.float32),
                loss_mask=loss_mask[:T_actual],
                loss_config=config.loss,
                sample_ranges=[(0, T_actual)],
            )
            per_token_grads = per_token_grads / float(loss_scale)

            # Shift gradients (align with training targets)
            surogate_grads = np.zeros(T_actual, dtype=np.float32)
            if T_actual > 1:
                surogate_grads[:T_actual - 1] = per_token_grads[1:T_actual]

            grads_padded = np.zeros((1, seq_len), dtype=np.float32)
            grads_padded[0, :T_actual] = surogate_grads
            if ngpu > 1:
                grads_padded = np.tile(grads_padded, (ngpu, 1))

            # Backward
            self.trainer.backward_grpo(grads_padded)

        # 5. Optimizer step
        lr = self.lr_schedule.get_lr(step)
        opt_config = _surogate.OptimizerConfig(
            optimizer=config.optimizer,
            learning_rate=lr,
            weight_decay=config.weight_decay,
            grad_clip=config.max_grad_norm,
            adamw_beta1=config.adamw_beta1,
            adamw_beta2=config.adamw_beta2,
            adamw_epsilon=config.adamw_epsilon,
        )
        self.trainer.update_with_config(opt_config, step + 1)

        return {
            "rewards_mean": float(rewards.mean()),
            "rewards_std": float(rewards.std()),
            "advantages_mean": float(advantages.mean()),
            "num_trajectories": len(all_tokens),
            "avg_completion_len": float(np.mean(completion_lens_list)),
        }
