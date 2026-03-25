#!/usr/bin/env python3
"""
Analyze fusion opportunities across DSL models (inference forward pass).

Usage:
    python -m surogate.compiler.analyze_fusion [--verbose] [--model MODEL]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from surogate.dsl import models as _models_module  # noqa: F401
from surogate.dsl.py_compiler import compile_model_for_hf
from surogate.compiler.partitioner import analyze_model_ir, FusionReport


REPRESENTATIVE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "LlamaForCausalLM": {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
    },
    "Qwen3ForCausalLM": {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "use_qk_norm": True,
    },
    "Qwen3MoeForCausalLM": {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "intermediate_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "attention_bias": False,
        "use_qk_norm": True,
        "norm_topk_prob": True,
        "shared_expert_intermediate_size": 0,
    },
    "Qwen3MoeForCausalLM_shared": {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "intermediate_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "attention_bias": False,
        "use_qk_norm": True,
        "norm_topk_prob": True,
        "shared_expert_intermediate_size": 2048,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Analyze fusion opportunities (inference forward)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--dump-ir", action="store_true")
    args = parser.parse_args()

    configs = REPRESENTATIVE_CONFIGS
    if args.model:
        if args.model not in configs:
            print(f"Unknown model: {args.model}\nAvailable: {', '.join(configs.keys())}")
            sys.exit(1)
        configs = {args.model: configs[args.model]}

    reports: Dict[str, FusionReport] = {}

    for name, config in configs.items():
        arch = config["architectures"][0]
        print(f"\nCompiling {name} ({arch})...")
        try:
            ir_json = compile_model_for_hf(arch, config)
            if args.dump_ir:
                print(json.dumps(json.loads(ir_json), indent=2))
                continue
            report = analyze_model_ir(ir_json, verbose=args.verbose)
            reports[name] = report
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if args.dump_ir or len(reports) < 2:
        return

    # Cross-model comparison
    print(f"\n{'='*76}")
    print(f"  CROSS-MODEL COMPARISON")
    print(f"{'='*76}\n")

    header = (f"  {'Model':<30s} {'Kernel':>7s} {'Epilogue':>9s} {'Prologue':>9s}"
              f" {'Total Saved':>12s} {'Remaining':>10s}")
    print(header)
    print(f"  {'-'*74}")

    for name, r in reports.items():
        epi_saved = sum(len(ef.epilogue_ops) for ef in r.epilogue_fusions)
        pro_saved = sum(len(pf.prologue_ops) for pf in r.prologue_fusions)
        total_saved = epi_saved + pro_saved
        remaining = r.total_kernel_launches - total_saved
        pct = total_saved / r.total_kernel_launches * 100 if r.total_kernel_launches > 0 else 0

        short = name.replace("ForCausalLM", "").replace("MoeForCausalLM", "-MoE")
        print(f"  {short:<30s} {r.total_kernel_launches:>7d} {epi_saved:>9d} {pro_saved:>9d}"
              f" {total_saved:>8d} ({pct:4.1f}%) {remaining:>10d}")

    print()


if __name__ == "__main__":
    main()
