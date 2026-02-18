# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# GRPO (Group Relative Policy Optimization) trainer with native inference.
#
# Architecture:
#   For each training step:
#     1. [Inference]  enter_inference_mode → allocate KV-cache
#     2. [Inference]  For each prompt: prefill + generate G completions
#     3. [Inference]  exit_inference_mode → free KV-cache
#     4. [Training]   compute_logprobs (policy, use_lora=True)
#     5. [Training]   compute_logprobs (reference, use_lora=False)
#     6. [Python]     Compute rewards → advantages → GRPO per-token gradients
#     7. [Training]   step_with_custom_loss + update_with_config
#
# Memory time-sharing: gradient buffers and optimizer states are live during
# training phases; KV-cache is live during inference phases. They never
# co-exist, so GPU memory stays bounded.

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from surogate import _surogate
from surogate.core.config.grpo_config import GRPOConfig
from surogate.train.grpo_dataset import GRPODataset, GRPOSample, load_grpo_dataset
from surogate.train.lr_schedule import LRSchedule
from surogate.train.rewards import RewardFunction, build_reward_functions
from surogate.train.sampling import sample_token
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger
from surogate.utils.tensor import to_surogate_dtype

logger = get_logger()


class SurogateGRPO:
    """GRPO training loop with native Surogate inference.

    Args:
        config: GRPO configuration.
        args:   CLI arguments (hub_token, etc.).
    """

    def __init__(self, config: GRPOConfig, args: DictDefault):
        self.config = config
        self.args = args

        # Finalize config: resolve model_dir, tokenizer, chat template, etc.
        # (SFT does this inside TokenizeDatasets.__init__; GRPO bypasses that path.)
        config.__post_init__()

        # Number of prompts per update step and completions per prompt
        self.B = config.per_device_train_batch_size  # prompts per step
        self.G = config.num_generations               # completions per prompt

        # The C++ training batch processes B*G sequences simultaneously
        self.train_batch = self.B * self.G

        if not config.lora:
            raise ValueError(
                "GRPO requires lora=true. "
                "The reference model is the base model (use_lora=False); "
                "the policy model adds LoRA on top (use_lora=True)."
            )

        # ------------------------------------------------------------------ #
        # Build the C++ trainer
        # ------------------------------------------------------------------ #
        from surogate.dsl.ir_builder import build_dsl_ir_for_model
        from surogate.utils.hf import get_model_weights_path

        dsl_extra = {}
        if getattr(config, "ep_size", 1) > 1:
            dsl_extra["ep_size"] = config.ep_size
        ir_json = build_dsl_ir_for_model(config.model_dir, extra_config=dsl_extra or None)
        config.runtime_config.dsl_ir_json = ir_json

        model_weights_path = get_model_weights_path(config.model_dir)

        def _make_trainer():
            return _surogate.SurogateTrainer(
                ngpu=config.gpus,
                config=_surogate.PretrainedConfig.from_pretrained(
                    config.model_dir, to_surogate_dtype(config.torch_dtype)
                ),
                options=config.runtime_config,
                batch_size=self.train_batch,
                seq_len=config.sequence_len,
                grad_accum=1,  # GRPO manages its own accumulation
                memcpy_all_gather=config.memcpy_all_gather,
                memcpy_send_recv=config.memcpy_send_recv,
                lora_config=config.lora_config,
                qlora_config=config.qlora_config,
            )

        # Handle checkpoint resume
        self.start_step = 0
        if config.resume_from_checkpoint:
            found_step = _surogate.find_latest_checkpoint(config.checkpoint_dir)
            if found_step >= 0:
                self.start_step = found_step
                self.trainer = _make_trainer()
                if config.adapter_path:
                    logger.info(f"Merging adapter from {config.adapter_path} into base weights...")
                    self.trainer.set_adapter_path(config.adapter_path)
                self.trainer.import_weights(model_weights_path)
                logger.info(f"Loading checkpoint from step {self.start_step}...")
                self.trainer.load_checkpoint(str(config.checkpoint_dir), self.start_step)
            else:
                logger.warning("No checkpoint found to resume from. Starting training from beginning.")

        if not hasattr(self, 'trainer'):
            # Fresh start: LoRA training requires explicit constructor + import_weights
            self.trainer = _make_trainer()
            if config.adapter_path:
                logger.info(f"Merging adapter from {config.adapter_path} into base weights...")
                self.trainer.set_adapter_path(config.adapter_path)
            self.trainer.import_weights(model_weights_path)

        logger.info(
            f"GRPO: {self.B} prompts × {self.G} completions = {self.train_batch} sequences/step, "
            f"seq_len={config.sequence_len}"
        )

        # ------------------------------------------------------------------ #
        # Dataset, reward functions, LR schedule
        # ------------------------------------------------------------------ #
        if not config.grpo_datasets:
            raise ValueError(
                "grpo_datasets must be set to a JSONL file path. "
                "Format: {\"prompt\": \"...\", \"ground_truth\": \"...\"}"
            )
        self.dataset: GRPODataset = load_grpo_dataset(
            config.grpo_datasets, seed=config.train_seed
        )
        self.reward_fns: List[RewardFunction] = build_reward_functions(
            config.reward_functions
        )

        # Max steps
        if config.max_steps > 0:
            self.max_steps = config.max_steps
        else:
            steps_per_epoch = max(1, len(self.dataset) // self.B)
            self.max_steps = steps_per_epoch * config.num_epochs
            logger.info(
                f"Derived {self.max_steps} steps from {config.num_epochs} epoch(s) "
                f"({len(self.dataset)} prompts, batch={self.B})"
            )

        # Warmup steps
        warmup_steps = config.warmup_steps
        if warmup_steps == 0 and config.warmup_ratio > 0:
            warmup_steps = int(self.max_steps * config.warmup_ratio)

        self.lr_schedule = LRSchedule(
            base_lr=config.learning_rate,
            max_steps=self.max_steps,
            warmup_steps=warmup_steps,
            cooldown_steps=config.cooldown_steps,
            final_lr=config.learning_rate * config.final_lr_fraction,
            schedule_type=config.lr_scheduler_type,
            wsd_decay_steps_fraction=config.wsd_decay_steps_fraction,
        )

        # RNG for sampling
        self._rng = np.random.default_rng(config.train_seed)

    # ---------------------------------------------------------------------- #
    # Main training entry point
    # ---------------------------------------------------------------------- #

    def run(self):
        """Run the full GRPO training loop."""
        tokenizer = self.config.tokenizer

        logger.info(
            f"Starting GRPO training: {self.max_steps} steps, "
            f"G={self.G}, β={self.config.grpo_beta}, ε={self.config.grpo_epsilon}"
        )
        logger.info(f"Reward functions: {self.config.reward_functions}")
        logger.info(f"Max completion length: {self.config.max_completion_length}")

        for step in range(self.start_step, self.max_steps):
            step_start = time.time()

            # 1. Sample B prompts
            samples: List[GRPOSample] = self.dataset.sample_batch(self.B)

            # 2. Generate G completions per prompt (inference mode)
            all_prompt_ids: List[np.ndarray] = []       # [B] arrays of int32
            all_completions: List[List[List[int]]] = []  # [B][G] token ID lists
            all_old_lp: List[np.ndarray] = []            # [B*G, seq_len] float32

            max_seq = self.config.sequence_len

            self.trainer.enter_inference_mode(max_seq)
            try:
                for sample in samples:
                    prompt_ids = _tokenize_prompt(tokenizer, sample.prompt)
                    all_prompt_ids.append(prompt_ids)

                    # Prefill: fill KV-cache at positions [0, prompt_len)
                    self.trainer.inference_prefill(prompt_ids)

                    group_completions: List[str] = []
                    group_old_lp: List[np.ndarray] = []  # each [seq_len] float32

                    for g in range(self.G):
                        # Reset decode position to end of prompt
                        self.trainer.set_kv_pos(len(prompt_ids))

                        tokens, lps, eos_hit = self._decode_completion(
                            tokenizer, prompt_ids, max_seq
                        )
                        group_completions.append(tokens)

                        # Build per-position old_lp array aligned to seq_len
                        old_lp_row = _align_old_lp(
                            len(prompt_ids), tokens, lps, max_seq
                        )
                        group_old_lp.append(old_lp_row)

                    all_completions.append(group_completions)
                    all_old_lp.extend(group_old_lp)  # B*G rows
            finally:
                self.trainer.exit_inference_mode()

            # 3. Build training batch [B*G, seq_len]
            input_ids, targets = _prepare_training_batch(
                all_prompt_ids, all_completions, self.G, max_seq, tokenizer.pad_token_id
            )

            # 4. Compute per-sequence rewards → advantages
            flat_prompts: List[str] = []   # B*G
            flat_completions: List[str] = []
            flat_gt: List[Optional[str]] = []
            completion_strs_by_prompt: List[List[str]] = []

            for i, (sample, comps) in enumerate(zip(samples, all_completions)):
                decoded = [
                    tokenizer.decode(c, skip_special_tokens=True) if isinstance(c, list)
                    else c
                    for c in comps
                ]
                completion_strs_by_prompt.append(decoded)
                for comp_str in decoded:
                    flat_prompts.append(sample.prompt)
                    flat_completions.append(comp_str)
                    flat_gt.append(sample.ground_truth)

            # Sum rewards from all reward functions
            rewards = np.zeros(self.B * self.G, dtype=np.float32)
            for fn in self.reward_fns:
                r = fn(flat_prompts, flat_completions, flat_gt)
                rewards += np.array(r, dtype=np.float32)

            # Group-relative advantages: normalize within each prompt's G completions
            advantages = _compute_advantages(rewards, self.B, self.G)

            # 5. Policy and reference log-probs [B*G, seq_len]
            policy_lp = self.trainer.compute_logprobs(input_ids, targets, use_lora=True)
            ref_lp = self.trainer.compute_logprobs(input_ids, targets, use_lora=False)

            old_lp_arr = np.stack(all_old_lp, axis=0)   # [B*G, seq_len]

            # 6. GRPO per-token gradient seeds
            per_token_grads = _compute_grpo_gradients(
                policy_lp=policy_lp,
                ref_lp=ref_lp,
                old_lp=old_lp_arr,
                advantages=advantages,
                targets=targets,
                beta=self.config.grpo_beta,
                epsilon=self.config.grpo_epsilon,
            )

            # 7. Backward + optimizer step
            self.trainer.step_with_custom_loss(input_ids, targets, per_token_grads)

            lr = self.lr_schedule.get_lr(step)
            opt_config = _surogate.OptimizerConfig(
                optimizer=self.config.optimizer,
                learning_rate=lr,
                weight_decay=self.config.weight_decay,
                grad_clip=self.config.max_grad_norm,
                adamw_beta1=self.config.adamw_beta1,
                adamw_beta2=self.config.adamw_beta2,
                adamw_epsilon=self.config.adamw_epsilon,
                normuon_momentum=self.config.normuon_momentum,
                normuon_beta2=self.config.normuon_beta2,
                normuon_lr=lr,
                normuon_cautious_wd=self.config.normuon_cautious_wd,
            )
            update_result = self.trainer.update_with_config(opt_config, step + 1)

            # 8. Logging
            elapsed = time.time() - step_start
            mean_reward = float(rewards.mean())
            grad_norm = float(update_result.get("norm", 0.0))

            logger.info(
                f"step {step:5d}/{self.max_steps} | "
                f"reward={mean_reward:.4f} | "
                f"grad_norm={grad_norm:.4f} | "
                f"lr={lr:.2e} | "
                f"time={elapsed:.1f}s"
            )

            # Periodic checkpoint
            if (self.config.save_steps > 0 and step > 0
                    and step % self.config.save_steps == 0):
                self._save_checkpoint(step)

        # Final save
        self._save_final()

    # ---------------------------------------------------------------------- #
    # Inference helpers
    # ---------------------------------------------------------------------- #

    def _decode_completion(
        self,
        tokenizer,
        prompt_ids: np.ndarray,
        max_seq: int,
    ) -> Tuple[List[int], List[float], bool]:
        """Autoregressively decode one completion, returning tokens and log-probs.

        Returns:
            tokens:  List of generated token IDs (completion only, not prompt).
            lps:     Per-token log-probabilities at generation time (old_log_prob).
            eos_hit: Whether generation stopped due to EOS.
        """
        eos_id = tokenizer.eos_token_id
        max_new = min(
            self.config.max_completion_length,
            max_seq - len(prompt_ids),
        )
        if max_new <= 0:
            return [], [], False

        last_token = int(prompt_ids[-1])
        tokens: List[int] = []
        lps: List[float] = []
        eos_hit = False

        for pos in range(len(prompt_ids), len(prompt_ids) + max_new):
            logits = self.trainer.inference_decode(last_token, pos)
            tok, lp = sample_token(
                logits,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                rng=self._rng,
            )
            tokens.append(tok)
            lps.append(lp)
            last_token = tok
            if eos_id is not None and tok == eos_id:
                eos_hit = True
                break

        return tokens, lps, eos_hit

    # ---------------------------------------------------------------------- #
    # Checkpoint helpers
    # ---------------------------------------------------------------------- #

    def _save_checkpoint(self, step: int):
        try:
            logger.info(f"Saving checkpoint at step {step}...")
            self.trainer.save_checkpoint(self.config.checkpoint_dir, step)
            logger.info(f"Checkpoint saved at step {step}")

            if self.config.save_total_limit > 0:
                _surogate.clean_old_checkpoints(
                    self.config.checkpoint_dir,
                    self.config.save_total_limit,
                    -1,
                )
        except Exception as exc:
            logger.error(f"Failed to save checkpoint at step {step}: {exc}")

    def _save_final(self):
        if self.config.lora:
            adapter_dir = Path(self.config.output_dir)
            adapter_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving LoRA adapter to {adapter_dir}...")
            self.trainer.export_adapter(str(adapter_dir))
            logger.info(f"LoRA adapter saved to {adapter_dir}")
        else:
            logger.info(f"Saving model to {self.config.output_dir}...")
            self.trainer.export_model(str(self.config.output_dir))
            logger.info("Model saved.")


# --------------------------------------------------------------------------- #
# Pure-Python helpers (no C++ calls)
# --------------------------------------------------------------------------- #

def _tokenize_prompt(tokenizer, prompt: str) -> np.ndarray:
    """Tokenize a prompt string and return int32 numpy array."""
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    return np.array(ids, dtype=np.int32)


def _align_old_lp(
    prompt_len: int,
    tokens: List[int],
    lps: List[float],
    seq_len: int,
) -> np.ndarray:
    """Build a seq_len-length old_lp row aligned to the training sequence.

    Prompt positions get 0.0 (masked by targets=-100 anyway).
    Completion positions get the sampled log-probability.
    Padding positions get 0.0.
    """
    row = np.zeros(seq_len, dtype=np.float32)
    for i, lp in enumerate(lps):
        pos = prompt_len + i
        if pos < seq_len:
            row[pos] = float(lp)
    return row


def _prepare_training_batch(
    all_prompt_ids: List[np.ndarray],
    all_completions: List[List],  # [B][G] — each element is List[int] of token IDs
    G: int,
    seq_len: int,
    pad_id: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build [B*G, seq_len] input_ids and targets arrays.

    For each (prompt, completion) pair:
    - input_ids:  [BOS + prompt_tokens + completion_tokens] padded to seq_len
    - targets:    -100 at prompt positions, completion token IDs at completion positions

    Args:
        all_prompt_ids:  [B] arrays of prompt token IDs.
        all_completions: [B][G] lists of generated completion token ID lists.
        G:               Number of completions per prompt.
        seq_len:         Target sequence length.
        pad_id:          Padding token ID (used in input_ids; targets remain -100).

    Returns:
        input_ids: np.ndarray[int32, (B*G, seq_len)]
        targets:   np.ndarray[int32, (B*G, seq_len)]
    """
    if pad_id is None:
        pad_id = 0

    BG = len(all_prompt_ids) * G
    input_ids = np.full((BG, seq_len), fill_value=pad_id, dtype=np.int32)
    targets = np.full((BG, seq_len), fill_value=-100, dtype=np.int32)

    row = 0
    for prompt_ids, comps in zip(all_prompt_ids, all_completions):
        for comp_tokens in comps:
            if isinstance(comp_tokens, str):
                # Safety: already decoded — shouldn't happen in normal flow
                raise TypeError("comp_tokens must be a list of int, not str")

            # Full sequence: prompt + completion
            full = list(prompt_ids) + list(comp_tokens)
            n = min(len(full), seq_len)

            input_ids[row, :n] = full[:n]

            # Targets: -100 for prompt (first len(prompt_ids) tokens), completion IDs after
            prompt_len = len(prompt_ids)
            for i in range(prompt_len, n):
                targets[row, i] = full[i]

            row += 1

    return input_ids, targets


def _compute_advantages(
    rewards: np.ndarray,   # [B*G] float32
    B: int,
    G: int,
) -> np.ndarray:
    """Group-relative advantage normalization.

    For each of the B prompts, normalize its G reward values:
        advantage = (r - mean(r_group)) / (std(r_group) + 1e-8)

    Returns:
        advantages: np.ndarray[float32, (B*G,)]
    """
    advantages = np.zeros_like(rewards)
    for b in range(B):
        start = b * G
        end = start + G
        group = rewards[start:end]
        mean_r = float(group.mean())
        std_r = float(group.std()) + 1e-8
        advantages[start:end] = (group - mean_r) / std_r
    return advantages.astype(np.float32)


def _compute_grpo_gradients(
    policy_lp: np.ndarray,   # [B*G, seq_len] float32
    ref_lp: np.ndarray,      # [B*G, seq_len] float32
    old_lp: np.ndarray,      # [B*G, seq_len] float32
    advantages: np.ndarray,  # [B*G] float32
    targets: np.ndarray,     # [B*G, seq_len] int32
    beta: float,
    epsilon: float,
) -> np.ndarray:
    """Compute per-token gradient seeds for GRPO.

    The gradient seed at each (b, t) position is:
        dL/dCE[b,t]  =  -(dL/d_policy_lp[b,t])

    where:
        r = exp(policy_lp - old_lp)                     # importance ratio
        indicator = (adv > 0 and r <= 1+ε) or           # not clipped
                    (adv < 0 and r >= 1-ε)
        grad_policy = -r * adv * indicator               # policy gradient
        grad_kl = beta * (1 - exp(ref_lp - policy_lp))  # KL gradient
        dL/d_policy_lp = grad_policy + grad_kl

    Gradient seed:
        per_token_grads = -(dL/d_policy_lp) / n_valid   # normalized

    Args:
        policy_lp:  Log-probs under the current policy (LoRA enabled).
        ref_lp:     Log-probs under the reference model (LoRA disabled).
        old_lp:     Log-probs at generation time (old_log_prob).
        advantages: Per-completion scalar advantages, shape [B*G].
        targets:    Training targets; -100 marks masked (prompt) positions.
        beta:       KL penalty coefficient.
        epsilon:    PPO clip range.

    Returns:
        per_token_grads: np.ndarray[float32, (B*G, seq_len)]
    """
    BG, seq_len = policy_lp.shape

    # Valid token mask
    valid = (targets != -100).astype(np.float32)  # [BG, T]
    n_valid = float(valid.sum()) + 1e-8

    # Broadcast advantages [BG] → [BG, T]
    adv = advantages[:, np.newaxis] * np.ones((1, seq_len), dtype=np.float32)

    # Importance ratio (clamp to avoid overflow in exp)
    log_ratio = np.clip(policy_lp - old_lp, -10.0, 10.0)
    r = np.exp(log_ratio)

    # Clipping indicator (PPO-style)
    not_clipped = np.where(
        adv > 0,
        r <= (1.0 + epsilon),
        r >= (1.0 - epsilon),
    )

    # Policy gradient: -(r * adv * not_clipped)
    # We negate because per_token_grads = -(dL/d_policy_lp)
    grad_policy = r * adv * not_clipped.astype(np.float32)  # positive = increase policy_lp

    # KL gradient: beta * (exp(ref_lp - policy_lp) - 1)
    # KL = exp(ref_lp - policy_lp) - (ref_lp - policy_lp) - 1
    # dKL/d_policy_lp = -exp(ref_lp - policy_lp) + 1 = 1 - exp(ref_lp - policy_lp)
    # We want to minimize beta * KL, so:
    # dL_kl/d_policy_lp = beta * (1 - exp(ref_lp - policy_lp))
    # per_token_grads contribution = -(dL_kl/d_policy_lp) = beta * (exp(ref_lp - policy_lp) - 1)
    if beta != 0.0:
        log_ref_ratio = np.clip(ref_lp - policy_lp, -10.0, 10.0)
        grad_kl = beta * (np.exp(log_ref_ratio) - 1.0)
    else:
        grad_kl = 0.0

    # Total: per_token_grads = (grad_policy + grad_kl) / n_valid
    per_token_grads = (grad_policy + grad_kl) / n_valid

    # Zero out masked positions (safety — C++ already ignores target==-100 in backward,
    # but keeping this explicit makes the gradient math auditable)
    per_token_grads = per_token_grads * valid

    return per_token_grads.astype(np.float32)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def grpo_main(config: GRPOConfig, args: DictDefault):
    """Entry point for `surogate grpo config.yaml`."""
    SurogateGRPO(config, args).run()
