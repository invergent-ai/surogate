#!/usr/bin/env python3
"""Minimal working example of native GRPO training (no vLLM).

Usage:
    python examples/grpo/native_grpo_minimal.py

Requires: Qwen/Qwen3-0.6B cached (auto-downloads on first run).
"""

import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

import surogate._surogate as _surogate
from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.utils.hf import get_model_weights_path


def main():
    # ========================================================================
    # 1. Setup model
    # ========================================================================
    model_path = snapshot_download("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ir_json = build_dsl_ir_for_model(model_path)

    config = _surogate.PretrainedConfig.from_pretrained(model_path, "bf16")
    options = _surogate.RuntimeOptions()
    options.dsl_ir_json = ir_json

    # Create trainer with LoRA
    lora_config = _surogate.LoRAAdapterConfig()
    lora_config.rank = 8
    lora_config.alpha = 16

    SEQ_LEN = 256

    trainer = _surogate.SurogateTrainer(
        ngpu=1,
        config=config,
        options=options,
        batch_size=1,
        seq_len=SEQ_LEN,
        grad_accum=1,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
        lora_config=lora_config,
    )
    trainer.import_weights(get_model_weights_path(model_path))
    eos_id = tokenizer.eos_token_id or 151643

    # ========================================================================
    # 2. Define reward function
    # ========================================================================
    def reward_fn(completions: list[str]) -> list[float]:
        """Simple reward: prefer longer, non-repetitive completions."""
        rewards = []
        for c in completions:
            words = c.split()
            unique_ratio = len(set(words)) / max(len(words), 1)
            length_bonus = min(len(words) / 20.0, 1.0)
            rewards.append(unique_ratio * length_bonus)
        return rewards

    # ========================================================================
    # 3. Training loop
    # ========================================================================
    prompts = [
        "The meaning of life is",
        "To build a great AI system, you need",
        "The most important scientific discovery was",
        "In a galaxy far far away,",
    ]
    NUM_COMPLETIONS = 4
    MAX_GEN_LEN = 64
    NUM_STEPS = 5

    opt_config = _surogate.OptimizerConfig(
        optimizer="adamw_8bit",
        learning_rate=2e-5,
        weight_decay=0.01,
        grad_clip=1.0,
    )

    for step in range(NUM_STEPS):
        print(f"\n{'='*60}")
        print(f"Step {step}")
        print(f"{'='*60}")

        # --- Generate rollout ---
        prompt_ids = [tokenizer.encode(p) for p in prompts]
        tokens, logprobs, prompt_lens, completion_lens = trainer.generate(
            prompts=prompt_ids,
            num_completions=NUM_COMPLETIONS,
            max_gen_len=MAX_GEN_LEN,
            temperature=0.8,
            eos_token_id=eos_id,
            use_lora=True,
            top_p=0.9,
        )

        total = len(tokens)
        completions = []
        for i in range(total):
            pl = prompt_lens[i]
            cl = completion_lens[i]
            comp = tokenizer.decode(tokens[i][pl:pl + cl], skip_special_tokens=True)
            completions.append(comp)

        # --- Score ---
        rewards = np.array(reward_fn(completions), dtype=np.float32)
        print(f"  Rewards: mean={rewards.mean():.3f}, std={rewards.std():.3f}")

        # Show a sample
        print(f"  Sample: {prompts[0]!r} → {completions[0][:80]!r}...")

        # --- Compute advantages (GRPO: normalize within each prompt group) ---
        M = len(prompts)
        N = NUM_COMPLETIONS
        advantages = np.zeros(total, dtype=np.float32)
        for m in range(M):
            group = rewards[m * N:(m + 1) * N]
            mean, std = group.mean(), group.std()
            if std > 0:
                advantages[m * N:(m + 1) * N] = (group - mean) / std

        # --- Train on each trajectory ---
        trainer.set_grad_accumulation(total)

        for i in range(total):
            pl = prompt_lens[i]
            cl = completion_lens[i]
            T_actual = pl + cl

            # Pad to SEQ_LEN
            inp = np.zeros((1, SEQ_LEN), dtype=np.int32)
            inp[0, :min(T_actual, SEQ_LEN)] = tokens[i][:SEQ_LEN]

            tgt = np.full((1, SEQ_LEN), -100, dtype=np.int32)
            for t in range(pl - 1, min(T_actual - 1, SEQ_LEN - 1)):
                tgt[0, t] = tokens[i][t + 1]

            pos = np.arange(SEQ_LEN, dtype=np.int32).reshape(1, SEQ_LEN)

            # Forward: get policy logprobs
            raw_lp = trainer.forward_for_grpo(inp, tgt, position_ids=pos)
            raw_lp = np.asarray(raw_lp[0, :T_actual], dtype=np.float32)

            # Build per-token GRPO gradient
            trainer_lp = np.zeros(T_actual, dtype=np.float32)
            if T_actual > 1:
                trainer_lp[1:] = raw_lp[:T_actual - 1]

            # Simple GRPO gradient: advantage * (1 - exp(lp_new - lp_old))
            # Approximate: advantage * sign(lp_new - lp_old) for tokens in completion
            grads = np.zeros(T_actual, dtype=np.float32)
            for t in range(pl, pl + cl):
                if t < T_actual:
                    grads[t] = advantages[i] / max(total, 1)

            # Shift for surogate format
            shifted_grads = np.zeros(T_actual, dtype=np.float32)
            if T_actual > 1:
                shifted_grads[:T_actual - 1] = grads[1:]

            grads_padded = np.zeros((1, SEQ_LEN), dtype=np.float32)
            grads_padded[0, :T_actual] = shifted_grads

            trainer.backward_grpo(grads_padded)

        # --- Optimizer step ---
        result = trainer.update_with_config(opt_config, step + 1)
        print(f"  Optimizer step done.")

    # ========================================================================
    # 4. Final generation to see improvement
    # ========================================================================
    print(f"\n{'='*60}")
    print("Final generation (greedy):")
    print(f"{'='*60}")
    for prompt in prompts[:2]:
        ids = tokenizer.encode(prompt)
        tokens, _, pl, cl = trainer.generate(
            prompts=[ids], num_completions=1,
            max_gen_len=64, temperature=0.0,
            eos_token_id=eos_id, use_lora=True,
        )
        text = tokenizer.decode(tokens[0][pl[0]:], skip_special_tokens=True)
        print(f"  {prompt!r} → {text[:100]}")


if __name__ == "__main__":
    main()
