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
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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

    if manifests:
        logger.info("Compiled %d JIT kernels total.", len(manifests))

    return manifests


def _ir_uses_op(ir: dict, op_name: str) -> bool:
    """Check if the IR contains a specific custom op."""
    ir_str = json.dumps(ir)
    return op_name in ir_str


def _extract_gdr_dims(ir: dict) -> tuple[int, int, int]:
    """Extract H, K, V dimensions from the IR for gated delta rule.

    The IR has modules[0].config with linear_num_key_heads, linear_num_value_heads,
    linear_key_head_dim, and linear_value_head_dim for Qwen3.5-style models.

    For Qwen3.5/GatedDeltaNet, ``chunk_gated_delta_rule`` runs after ``query`` and
    ``key`` have been repeated from ``linear_num_key_heads`` to
    ``linear_num_value_heads``. The Triton kernels therefore need to be compiled
    for the runtime head count seen by the custom op, which is
    ``linear_num_value_heads`` when present.
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

    # Qwen3.5 linear attention dims. The GDR op consumes repeated q/k heads,
    # so compile for the runtime head count seen by the op, not the pre-repeat
    # key-head count.
    H = config.get("linear_num_value_heads", 0)
    if H == 0:
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


def _detect_sm() -> int:
    """Detect SM version from current GPU."""
    import torch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        return cap[0] * 10 + cap[1]
    from surogate.kernels.compiler import _detect_sm
    return _detect_sm()
