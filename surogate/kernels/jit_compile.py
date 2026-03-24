# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# JIT kernel compilation entry point for the training pipeline.
#
# Called from trainer.py at model init time. Inspects the DSL IR to determine
# which JIT kernels are needed, then compiles them (with caching) and returns
# manifest paths for the C++ runtime.

from __future__ import annotations

import json
from pathlib import Path
from surogate.utils.logger import get_logger

logger = get_logger()

def compile_jit_kernels(ir_json: str) -> dict[str, str]:
    """Compile all JIT kernels required by the model.

    Inspects the DSL IR JSON to determine which kernel families are needed,
    then compiles each (with disk caching) and returns manifest paths.

    Args:
        ir_json: The DSL IR JSON string (as produced by build_dsl_ir_for_model).

    Returns:
        Dict mapping kernel name -> manifest JSON path.
        Empty dict if no JIT kernels are needed.
    """
    manifests: dict[str, str] = {}

    ir = json.loads(ir_json)

    # Check if the model uses gated delta rule
    if _ir_uses_op(ir, "chunk_gated_delta_rule"):
        H, K, V = _extract_gdr_dims(ir)
        logger.info(
            "Model uses gated delta rule (H=%d, K=%d, V=%d) — compiling Triton kernels...",
            H, K, V,
        )
        gdr_manifests = _compile_gated_delta_rule(H, K, V)
        manifests.update(gdr_manifests)

    # Check if the model declares gdn_qkvzba_fuse inference optimization
    if _ir_requests_fusion(ir, "gdn_qkvzba_fuse"):
        num_heads_qk, num_heads_v, head_qk, head_v = _extract_gdn_proj_dims(ir)
        logger.info(
            "Model requests GDN fused projection (Hqk=%d, Hv=%d, Kd=%d, Vd=%d) "
            "— compiling Triton kernel...",
            num_heads_qk, num_heads_v, head_qk, head_v,
        )
        gdn_manifests = _compile_gdn_fused_proj(
            num_heads_qk, num_heads_v, head_qk, head_v,
        )
        manifests.update(gdn_manifests)

    if manifests:
        logger.info("Compiled %d JIT kernels total.", len(manifests))

    return manifests


def _ir_uses_op(ir: dict, op_name: str) -> bool:
    """Check if the IR contains a specific custom op."""
    ir_str = json.dumps(ir)
    return op_name in ir_str


def _extract_gdr_dims(ir: dict) -> tuple[int, int, int]:
    """Extract H, K, V dimensions from the IR for gated delta rule.

    The IR has modules[0].config with linear_num_key_heads, linear_key_head_dim,
    linear_value_head_dim for Qwen3.5-style models.
    """
    # The DSL IR stores model config in modules[0].config
    config = {}
    for module in ir.get("modules", []):
        if isinstance(module, dict) and "config" in module:
            config = module["config"]
            break

    # Also check top-level config as fallback
    if not config:
        config = ir.get("config", {})

    # Qwen3.5 linear attention dims
    H = config.get("linear_num_key_heads", 0)
    K = config.get("linear_key_head_dim", 0)
    V = config.get("linear_value_head_dim", 0)

    # Fallback to standard attention dims
    if H == 0:
        H = config.get("num_query_heads", config.get("num_attention_heads", 0))
    if K == 0:
        K = config.get("head_size", config.get("head_dim", 0))
    if V == 0:
        V = K

    if H == 0 or K == 0:
        raise ValueError(
            "Cannot determine H, K, V dimensions for gated delta rule from IR. "
            f"Config keys: {list(config.keys())}"
        )

    return H, K, V


def _compile_gated_delta_rule(H: int, K: int, V: int) -> dict[str, str]:
    """Compile gated delta rule kernels with caching."""
    from surogate.kernels.cache import KernelCache
    from surogate.kernels.triton.gated_delta_rule import compile_gated_delta_rule

    cache = KernelCache()
    src_files = [
        Path(__file__).parent / "triton" / "gated_delta_rule.py",
        Path(__file__).parent / "compiler.py",
    ]

    return cache.get_or_compile(
        name="gated_delta_rule",
        src_files=src_files,
        dims={"H": H, "K": K, "V": V},
        sm=_detect_sm(),
        compile_fn=lambda output_dir: compile_gated_delta_rule(
            H=H, K=K, V=V, output_dir=output_dir,
        ),
    )


def _ir_requests_fusion(ir: dict, fusion_id: str) -> bool:
    """Check if any model in the IR declares a specific inference fusion pass."""
    for module in ir.get("modules", []):
        if not isinstance(module, dict):
            continue
        opts = module.get("inference_opts", {})
        for _mode, passes in opts.items():
            if isinstance(passes, list) and fusion_id in passes:
                return True
    return False


def _extract_gdn_proj_dims(ir: dict) -> tuple[int, int, int, int]:
    """Extract GDN projection dimensions from the IR.

    Returns (num_heads_qk, num_heads_v, head_qk, head_v) for the
    linear-attention blocks in Qwen3.5-style models.
    """
    config = {}
    for module in ir.get("modules", []):
        if isinstance(module, dict) and "config" in module:
            config = module["config"]
            break
    if not config:
        config = ir.get("config", {})

    # Qwen3.5 linear-attention dimensions
    # These come from the HF config fields mapped through @hf_config:
    #   linear_num_key_heads  -> NUM_HEADS_QK (QK head groups)
    #   linear_num_value_heads -> NUM_HEADS_V (V heads, must be >= NUM_HEADS_QK)
    #   linear_key_head_dim   -> HEAD_QK (per-head QK dimension)
    #   linear_value_head_dim -> HEAD_V (per-head V dimension)
    num_heads_qk = config.get("linear_num_key_heads", 0)
    num_heads_v = config.get("linear_num_value_heads", 0)
    head_qk = config.get("linear_key_head_dim", 0)
    head_v = config.get("linear_value_head_dim", 0)

    # Fallback for non-Qwen3.5 models
    if num_heads_qk == 0:
        num_heads_qk = config.get("num_query_heads", 0)
    if num_heads_v == 0:
        num_heads_v = num_heads_qk  # Default: V heads = QK heads
    if head_qk == 0:
        head_qk = config.get("head_size", config.get("head_dim", 0))
    if head_v == 0:
        head_v = head_qk

    if num_heads_qk == 0 or num_heads_v == 0 or head_qk == 0:
        raise ValueError(
            "Cannot determine GDN projection dims from IR. "
            f"Config keys: {list(config.keys())}"
        )

    return num_heads_qk, num_heads_v, head_qk, head_v


def _compile_gdn_fused_proj(
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
) -> dict[str, str]:
    """Compile GDN fused projection kernels with caching."""
    from surogate.kernels.cache import KernelCache
    from surogate.kernels.triton.gdn_fused_proj import compile_gdn_fused_proj

    cache = KernelCache()
    src_files = [
        Path(__file__).parent / "triton" / "gdn_fused_proj.py",
        Path(__file__).parent / "compiler.py",
    ]

    return cache.get_or_compile(
        name="gdn_fused_proj",
        src_files=src_files,
        dims={
            "num_heads_qk": num_heads_qk,
            "num_heads_v": num_heads_v,
            "head_qk": head_qk,
            "head_v": head_v,
        },
        sm=_detect_sm(),
        compile_fn=lambda output_dir: compile_gdn_fused_proj(
            num_heads_qk=num_heads_qk,
            num_heads_v=num_heads_v,
            head_qk=head_qk,
            head_v=head_v,
            output_dir=output_dir,
        ),
    )


def _detect_sm() -> int:
    """Detect SM version from current GPU."""
    import torch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        return cap[0] * 10 + cap[1]
    from surogate.kernels.compiler import _detect_sm
    return _detect_sm()
