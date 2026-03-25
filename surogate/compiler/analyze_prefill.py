#!/usr/bin/env python3
"""
Analyze prefill memory bandwidth savings from kernel fusion.

For prefill (large T), the bottleneck is memory bandwidth, not kernel launch
overhead. Fusing ops eliminates intermediate tensor reads/writes.

Usage:
    python -m surogate.compiler.analyze_prefill [--seq SEQ] [--model MODEL]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple

from surogate.dsl import models as _models_module  # noqa: F401
from surogate.dsl.py_compiler import compile_model_for_hf


# Bytes per element by dtype
DTYPE_BYTES = {"bf16": 2, "fp32": 4, "fp8": 1, "int32": 4}


def estimate_tensor_bytes(shape: List[Any], dtype: str = "bf16") -> int:
    """Estimate bytes for a tensor given symbolic shape and dtype."""
    elem = DTYPE_BYTES.get(dtype, 2)
    total = elem
    for dim in shape:
        if isinstance(dim, (int, float)):
            total *= int(dim)
        # Skip symbolic dims — they'll be filled in by caller
    return total


# Op classification for prefill bandwidth analysis
COMPUTE_BOUND_OPS = {
    "matmul", "matmul_bias", "matmul_swiglu", "batched_matmul",
    "flash_attention", "fused_lm_head_loss",
    "moe_grouped_gemm", "moe_grouped_gemm_gate_up", "moe_grouped_gemm_down",
    "mamba_conv1d", "mamba_selective_scan", "mamba_combine_scan",
}

MEMORY_BOUND_OPS = {
    "swiglu", "silu", "silu_mul", "sigmoid", "relu", "relu2", "gelu", "geglu",
    "add", "mul", "scale", "add3", "bias_add",
    "rmsnorm", "layernorm", "fused_residual_rmsnorm",
    "rope", "mrope", "qkv_qk_norm_rope", "qkv_qk_norm",
    "softmax", "moe_softmax", "moe_sigmoid",
    "embedding",
}

ZERO_COST_OPS = {
    "view", "transpose", "permute", "contiguous", "copy", "split", "concat",
    "zeros", "ones", "fill", "fill_normal",
    "repeat_interleave_heads",
}

ROUTING_OPS = {
    "moe_topk", "moe_permute", "moe_unpermute",
    "ep_dispatch", "ep_combine",
}


def classify_prefill_op(kernel_type: str) -> str:
    if kernel_type in COMPUTE_BOUND_OPS:
        return "compute"
    if kernel_type in MEMORY_BOUND_OPS:
        return "memory"
    if kernel_type in ZERO_COST_OPS:
        return "zero_cost"
    if kernel_type in ROUTING_OPS:
        return "routing"
    return "unknown"


def estimate_op_bandwidth(
    kernel_type: str,
    B: int, T: int,
    d_model: int, d_ff: int,
    num_heads: int, num_kv_heads: int, head_dim: int,
) -> Tuple[int, int, str]:
    """
    Estimate read and write bytes for one op in prefill.
    Returns (read_bytes, write_bytes, description).
    """
    BT = B * T
    elem = 2  # bf16

    if kernel_type == "swiglu":
        # Reads [BT, 2*M], writes [BT, M] where M = d_ff // 2
        M = d_ff // 2
        r = BT * 2 * M * elem
        w = BT * M * elem
        return r, w, f"[{BT}, {2*M}] → [{BT}, {M}]"

    if kernel_type in ("silu", "relu2", "gelu", "sigmoid"):
        # Reads and writes [BT, d_ff] (or d_model depending on context)
        # For MLP activation: operates on d_ff-sized intermediate
        M = d_ff // 2  # after gate split
        r = BT * M * elem
        w = BT * M * elem
        return r, w, f"[{BT}, {M}] → [{BT}, {M}]"

    if kernel_type == "rope":
        # Reads [B, T, (Hq+2*Hkv)*D], writes same
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        r = BT * qkv_dim * elem
        w = BT * qkv_dim * elem
        return r, w, f"[{B}, {T}, {qkv_dim}] RoPE"

    if kernel_type in ("qkv_qk_norm_rope", "qkv_qk_norm"):
        # Same as rope but includes norm
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        r = BT * qkv_dim * elem + num_heads * head_dim * elem  # + norm weights
        w = BT * qkv_dim * elem
        return r, w, f"[{B}, {T}, {qkv_dim}] QKNorm+RoPE"

    if kernel_type == "fused_residual_rmsnorm":
        # Reads 2x [B, T, C] + weight [C], writes [B, T, C] residual + [B, T, C] normed + rstd
        r = 2 * BT * d_model * elem + d_model * elem
        w = 2 * BT * d_model * elem + BT * 4  # rstd is fp32
        return r, w, f"2x[{B},{T},{d_model}] → res+norm+rstd"

    if kernel_type in ("rmsnorm", "layernorm"):
        r = BT * d_model * elem + d_model * elem
        w = BT * d_model * elem + BT * 4
        return r, w, f"[{B},{T},{d_model}] → norm+rstd"

    if kernel_type in ("add", "mul"):
        r = 2 * BT * d_model * elem
        w = BT * d_model * elem
        return r, w, f"2x[{B},{T},{d_model}] → [{B},{T},{d_model}]"

    if kernel_type == "moe_softmax" or kernel_type == "moe_sigmoid":
        # Small: [BT, num_experts]
        r = BT * 64 * elem  # assume 64 experts
        w = BT * 64 * elem
        return r, w, f"[{BT}, E] router activation"

    if kernel_type == "embedding":
        r = BT * 4  # token ids (int32)
        w = BT * d_model * elem
        return r, w, f"[{BT}] → [{BT}, {d_model}]"

    return 0, 0, "unknown"


def analyze_prefill_bandwidth(ir_json: str, B: int, T: int, verbose: bool = False):
    """Analyze memory bandwidth for prefill forward pass."""
    data = json.loads(ir_json) if isinstance(ir_json, str) else ir_json
    if not data.get("success"):
        raise RuntimeError("IR compilation failed")

    module = data["modules"][0]
    model_name = module["name"]
    config = module.get("config", {})

    # Extract model dimensions from config
    d_model = config.get("d_model", 4096)
    d_ff = config.get("d_ff", 11008)
    num_heads = config.get("num_query_heads", 32)
    num_kv_heads = config.get("num_kv_heads", 8)
    head_dim = config.get("head_size", config.get("head_dim", 128))
    n_layers = config.get("n_layers", 32)

    forward = module["forward"]
    ops = forward["operations"]

    print(f"\n{'='*76}")
    print(f"  PREFILL BANDWIDTH ANALYSIS: {model_name}")
    print(f"  B={B}, T={T}, d_model={d_model}, d_ff={d_ff}, layers={n_layers}")
    print(f"{'='*76}\n")

    # Classify ops and estimate bandwidth
    total_memory_read = 0
    total_memory_write = 0
    total_compute_ops = 0
    total_memory_ops = 0
    total_zero_cost = 0

    # Track fusible memory-bound ops between compute-bound ops
    fusible_segments: List[List[dict]] = []
    current_segment: List[dict] = []
    last_compute_op = None

    per_op_bw: Dict[str, Tuple[int, int, int]] = {}  # type -> (total_read, total_write, count)

    for op in ops:
        kt = op.get("kernel_type", op.get("name", ""))
        cat = classify_prefill_op(kt)

        if cat == "compute":
            total_compute_ops += 1
            if current_segment:
                fusible_segments.append(current_segment)
                current_segment = []
            last_compute_op = kt
        elif cat == "memory":
            total_memory_ops += 1
            r, w, desc = estimate_op_bandwidth(
                kt, B, T, d_model, d_ff, num_heads, num_kv_heads, head_dim)
            total_memory_read += r
            total_memory_write += w

            entry = per_op_bw.get(kt, (0, 0, 0))
            per_op_bw[kt] = (entry[0] + r, entry[1] + w, entry[2] + 1)

            current_segment.append({
                "kernel_type": kt,
                "read_bytes": r,
                "write_bytes": w,
                "description": desc,
            })
        elif cat == "zero_cost":
            total_zero_cost += 1
            # Views don't affect bandwidth, but are part of fusible segments
            if current_segment:
                current_segment.append({
                    "kernel_type": kt,
                    "read_bytes": 0,
                    "write_bytes": 0,
                    "description": "metadata",
                })
        elif cat == "routing":
            if current_segment:
                fusible_segments.append(current_segment)
                current_segment = []

    if current_segment:
        fusible_segments.append(current_segment)

    # Filter to segments with actual memory-bound ops
    bw_segments = [s for s in fusible_segments if any(o["read_bytes"] > 0 for o in s)]

    total_bw = total_memory_read + total_memory_write
    GB = 1024**3

    print(f"  Op breakdown:")
    print(f"    Compute-bound (matmul, attn):  {total_compute_ops}")
    print(f"    Memory-bound (norm, act, rope): {total_memory_ops}")
    print(f"    Zero-cost (view, etc):          {total_zero_cost}")
    print()

    print(f"  Memory-bound op bandwidth (per forward pass):")
    print(f"    Total reads:   {total_memory_read / GB:.2f} GB")
    print(f"    Total writes:  {total_memory_write / GB:.2f} GB")
    print(f"    Total traffic: {total_bw / GB:.2f} GB")
    print()

    print(f"  Per-op-type breakdown:")
    print(f"    {'Op Type':<30s} {'Count':>5s} {'Read (MB)':>10s} {'Write (MB)':>10s} {'Total (MB)':>10s}")
    print(f"    {'-'*65}")
    MB = 1024**2
    for kt, (r, w, count) in sorted(per_op_bw.items(), key=lambda x: -(x[1][0]+x[1][1])):
        print(f"    {kt:<30s} {count:>5d} {r/MB:>10.1f} {w/MB:>10.1f} {(r+w)/MB:>10.1f}")
    print()

    # Fusion analysis: segments between compute ops
    print(f"  ─── FUSIBLE SEGMENTS (memory-bound ops between compute ops) ───\n")
    print(f"  Total fusible segments: {len(bw_segments)}\n")

    # Group by pattern
    patterns: Dict[str, Tuple[int, int, int]] = {}  # pattern -> (count, total_read_saved, total_write_saved)
    for seg in bw_segments:
        mem_ops = [o for o in seg if o["read_bytes"] > 0]
        pattern = " → ".join(o["kernel_type"] for o in mem_ops)
        if not pattern:
            continue

        # When fused: eliminate intermediate writes/reads between ops
        # First op reads input, last op writes output.
        # Intermediate read+write for each internal tensor is eliminated.
        if len(mem_ops) <= 1:
            saved_read = 0
            saved_write = 0
        else:
            # Each intermediate tensor is written by one op and read by the next.
            # Fusion eliminates these intermediate materializations.
            saved_read = sum(o["read_bytes"] for o in mem_ops[1:])  # intermediate reads
            saved_write = sum(o["write_bytes"] for o in mem_ops[:-1])  # intermediate writes

        entry = patterns.get(pattern, (0, 0, 0))
        patterns[pattern] = (entry[0] + 1, entry[1] + saved_read, entry[2] + saved_write)

    if patterns:
        total_saved_read = 0
        total_saved_write = 0
        for pattern, (count, sr, sw) in sorted(patterns.items(), key=lambda x: -(x[1][1]+x[1][2])):
            total_saved_read += sr
            total_saved_write += sw
            saved_mb = (sr + sw) / MB
            if saved_mb > 0:
                print(f"    {count:>3d}x  {pattern}")
                print(f"         Saved: {saved_mb:.1f} MB/step ({sr/MB:.1f} read + {sw/MB:.1f} write)")
                print()

        total_saved = total_saved_read + total_saved_write
        pct = total_saved / total_bw * 100 if total_bw > 0 else 0

        print(f"  ─── FUSION BANDWIDTH SAVINGS ───\n")
        print(f"    Current memory traffic:     {total_bw / GB:.2f} GB")
        print(f"    Saved by fusion:            {total_saved / GB:.3f} GB ({pct:.1f}%)")
        print(f"    Remaining after fusion:     {(total_bw - total_saved) / GB:.2f} GB")
        print()

        # Time estimate at typical HBM bandwidth
        hbm_bw_gbps = 3350  # H100 HBM3 bandwidth in GB/s
        current_time_us = total_bw / (hbm_bw_gbps * 1e9) * 1e6
        saved_time_us = total_saved / (hbm_bw_gbps * 1e9) * 1e6
        print(f"    At H100 HBM bandwidth ({hbm_bw_gbps} GB/s):")
        print(f"      Memory-bound ops time: ~{current_time_us:.0f} us")
        print(f"      Time saved by fusion:  ~{saved_time_us:.0f} us")
        print()

        # Per-layer view
        per_layer_saved = total_saved / n_layers
        per_layer_total = total_bw / n_layers
        print(f"    Per layer: {per_layer_total/MB:.1f} MB → {(per_layer_total-per_layer_saved)/MB:.1f} MB memory traffic")
    else:
        print(f"    No multi-op fusible segments found.")

    print(f"\n{'='*76}")
    # The real question: how does memory-bound op time compare to compute-bound time?
    # Compute: matmul FLOPs
    # For one Llama layer: QKV proj + O proj + gate+up + down = 4 matmuls
    # Each matmul: 2 * B * T * d_model * output_dim FLOPs
    qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
    attn_dim = num_heads * head_dim
    flops_per_layer = 2 * B * T * (
        d_model * qkv_dim +      # QKV proj
        attn_dim * d_model +     # O proj
        d_model * d_ff +         # gate+up (d_ff = 2*M for SwiGLU)
        (d_ff // 2) * d_model    # down proj
    )
    # FlashAttention: ~2 * B * T * T * Hq * D (approximate)
    attn_flops = 2 * B * T * T * num_heads * head_dim
    total_flops = n_layers * (flops_per_layer + attn_flops)

    h100_tflops = 989  # H100 BF16 peak TFLOPS
    compute_time_us = total_flops / (h100_tflops * 1e12) * 1e6

    print(f"  COMPUTE vs MEMORY BOUND ANALYSIS")
    print(f"  (rough estimates, H100 SXM)")
    print(f"    Compute (matmul+attn):  {total_flops/1e12:.1f} TFLOP → ~{compute_time_us:.0f} us at peak")
    if patterns:
        print(f"    Memory-bound ops:       {total_bw/GB:.2f} GB → ~{current_time_us:.0f} us at HBM BW")
        ratio = current_time_us / compute_time_us * 100 if compute_time_us > 0 else 0
        print(f"    Memory / Compute ratio: {ratio:.0f}%")
        if ratio > 20:
            print(f"\n  VERDICT: Memory-bound ops are {ratio:.0f}% of compute time.")
            print(f"           Fusion would save ~{saved_time_us:.0f} us/step — meaningful for prefill.")
        else:
            print(f"\n  VERDICT: Memory-bound ops are only {ratio:.0f}% of compute.")
            print(f"           Prefill is compute-bound — fusion has limited impact.")
    print(f"{'='*76}\n")


CONFIGS: Dict[str, Dict[str, Any]] = {
    "Qwen3-1.7B": {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 2048, "intermediate_size": 6144,
        "num_hidden_layers": 28, "num_attention_heads": 16,
        "num_key_value_heads": 4, "vocab_size": 151936,
        "max_position_embeddings": 40960, "head_dim": 128,
        "rms_norm_eps": 1e-6, "attention_bias": False, "use_qk_norm": True,
    },
    "LlamaForCausalLM": {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096, "intermediate_size": 11008,
        "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 8, "vocab_size": 32000,
        "max_position_embeddings": 131072, "head_dim": 128,
        "rms_norm_eps": 1e-6,
    },
    "Llama-70B": {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 8192, "intermediate_size": 28672,
        "num_hidden_layers": 80, "num_attention_heads": 64,
        "num_key_value_heads": 8, "vocab_size": 128256,
        "max_position_embeddings": 131072, "head_dim": 128,
        "rms_norm_eps": 1e-5,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", "-T", type=int, default=4096)
    parser.add_argument("--batch", "-B", type=int, default=1)
    parser.add_argument("--model", "-m", type=str, default="LlamaForCausalLM")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.model not in CONFIGS:
        print(f"Available: {', '.join(CONFIGS.keys())}")
        sys.exit(1)

    config = CONFIGS[args.model]
    arch = config["architectures"][0]
    ir_json = compile_model_for_hf(arch, config)
    analyze_prefill_bandwidth(ir_json, args.batch, args.seq, verbose=args.verbose)


if __name__ == "__main__":
    main()
