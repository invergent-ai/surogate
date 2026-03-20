#!/usr/bin/env python3
"""Benchmark for the native generation engine.

Measures prefill and decode throughput for GRPO-style generation
(M prompts × N completions with shared prefix KV cache).

Usage:
    python examples/grpo/bench_generate.py [--model MODEL] [--num-prompts M] [--num-completions N] [--max-gen-len L] [--prompt-len P]
"""

import argparse
import time

import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

import surogate._surogate as _surogate
from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.utils.hf import get_model_weights_path


def run_benchmark(
    model_path: str,
    num_prompts: int,
    num_completions: int,
    max_gen_len: int,
    prompt_len: int,
    seq_len: int,
    warmup: int,
    iterations: int,
    lora_rank: int,
    use_lora: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ir_json = build_dsl_ir_for_model(model_path)

    options = _surogate.RuntimeOptions()
    options.dsl_ir_json = ir_json
    options.use_cuda_graphs = False

    lora_config = None
    if lora_rank > 0:
        lora_config = _surogate.LoRAAdapterConfig()
        lora_config.rank = lora_rank
        lora_config.alpha = lora_rank * 2

    trainer = _surogate.SurogateTrainer(
        ngpu=1,
        config=_surogate.PretrainedConfig.from_pretrained(model_path, "bf16"),
        options=options,
        batch_size=1,
        seq_len=seq_len,
        grad_accum=1,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
        lora_config=lora_config,
    )
    trainer.import_weights(get_model_weights_path(model_path))
    eos_id = tokenizer.eos_token_id or 151643

    # Build synthetic prompts of target length
    filler_text = "The quick brown fox jumps over the lazy dog. " * 20
    filler_ids = tokenizer.encode(filler_text)
    prompts = []
    for _ in range(num_prompts):
        ids = filler_ids[:prompt_len]
        if len(ids) < prompt_len:
            ids = ids + [ids[-1]] * (prompt_len - len(ids))
        prompts.append(ids)

    total_completions = num_prompts * num_completions
    print(f"\n{'=' * 70}")
    print(f"  Generation Benchmark")
    print(f"{'=' * 70}")
    print(f"  Model:            {model_path}")
    print(f"  Prompts:          {num_prompts}")
    print(f"  Completions/prompt: {num_completions}")
    print(f"  Total sequences:  {total_completions}")
    print(f"  Prompt length:    {prompt_len} tokens")
    print(f"  Max gen length:   {max_gen_len} tokens")
    print(f"  LoRA:             rank={lora_rank}, use_lora={use_lora}")
    print(f"  Seq len (train):  {seq_len}")
    print(f"{'=' * 70}\n")

    # --- Warmup ---
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        trainer.generate(
            prompts=prompts[:1],
            num_completions=num_completions,
            max_gen_len=min(max_gen_len, 16),
            temperature=1.0,
            eos_token_id=eos_id,
            use_lora=use_lora,
            use_cuda_graphs=True,
        )

    # --- Benchmark: per-prompt (how the native trainer calls it) ---
    print(f"\nBenchmark: per-prompt generation ({iterations} iterations)")
    per_prompt_times = []
    per_prompt_gen_tokens = []

    for it in range(iterations):
        total_gen = 0
        t0 = time.perf_counter()
        for pid in prompts:
            tokens, logprobs, pl, cl = trainer.generate(
                prompts=[pid],
                num_completions=num_completions,
                max_gen_len=max_gen_len,
                temperature=1.0,
                eos_token_id=eos_id,
                use_lora=use_lora,
                use_cuda_graphs=True,
            )
            total_gen += sum(cl)
        elapsed = time.perf_counter() - t0
        per_prompt_times.append(elapsed)
        per_prompt_gen_tokens.append(total_gen)

    avg_time = np.mean(per_prompt_times)
    avg_tokens = np.mean(per_prompt_gen_tokens)
    prefill_tokens = num_prompts * prompt_len
    decode_tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0
    total_tokens_per_sec = (prefill_tokens + avg_tokens) / avg_time if avg_time > 0 else 0

    print(f"  Avg time:         {avg_time:.3f}s (std={np.std(per_prompt_times):.3f}s)")
    print(f"  Avg gen tokens:   {avg_tokens:.0f}")
    print(f"  Prefill tokens:   {prefill_tokens}")
    print(f"  Decode tok/s:     {decode_tokens_per_sec:.1f}")
    print(f"  Total tok/s:      {total_tokens_per_sec:.1f}")
    print(f"  Time/prompt:      {avg_time / num_prompts * 1000:.1f}ms")
    print(f"  Time/completion:  {avg_time / total_completions * 1000:.1f}ms")

    # --- Benchmark: batched (if possible with small configs) ---
    if num_prompts <= 4 and num_completions <= 4:
        print(f"\nBenchmark: batched generation ({iterations} iterations)")
        batched_times = []
        batched_gen_tokens = []

        for it in range(iterations):
            t0 = time.perf_counter()
            tokens, logprobs, pl, cl = trainer.generate(
                prompts=prompts,
                num_completions=num_completions,
                max_gen_len=max_gen_len,
                temperature=1.0,
                eos_token_id=eos_id,
                use_lora=use_lora,
            )
            elapsed = time.perf_counter() - t0
            batched_times.append(elapsed)
            batched_gen_tokens.append(sum(cl))

        avg_time_b = np.mean(batched_times)
        avg_tokens_b = np.mean(batched_gen_tokens)
        print(f"  Avg time:         {avg_time_b:.3f}s")
        print(f"  Avg gen tokens:   {avg_tokens_b:.0f}")
        print(f"  Decode tok/s:     {avg_tokens_b / avg_time_b:.1f}")
        print(f"  Speedup vs per-prompt: {avg_time / avg_time_b:.2f}x")

    print(f"\n{'=' * 70}")
    print(f"  Summary")
    print(f"{'=' * 70}")
    print(f"  Per-prompt decode throughput: {decode_tokens_per_sec:.1f} tok/s")
    print(f"  Per-step wall time:          {avg_time:.3f}s")
    print(f"  (This is the generate() portion of one GRPO training step)")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark native generation engine")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-prompts", type=int, default=8, help="Number of distinct prompts (M)")
    parser.add_argument("--num-completions", type=int, default=16, help="Completions per prompt (N)")
    parser.add_argument("--max-gen-len", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--prompt-len", type=int, default=50, help="Prompt length in tokens")
    parser.add_argument("--seq-len", type=int, default=256, help="Training sequence length")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=5, help="Benchmark iterations")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (0 to disable)")
    parser.add_argument("--no-lora", action="store_true", help="Generate without LoRA")
    args = parser.parse_args()

    model_path = snapshot_download(args.model)
    run_benchmark(
        model_path=model_path,
        num_prompts=args.num_prompts,
        num_completions=args.num_completions,
        max_gen_len=args.max_gen_len,
        prompt_len=args.prompt_len,
        seq_len=args.seq_len,
        warmup=args.warmup,
        iterations=args.iterations,
        lora_rank=args.lora_rank,
        use_lora=not args.no_lora and args.lora_rank > 0,
    )


if __name__ == "__main__":
    main()
