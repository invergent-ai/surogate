"""Mixture of Experts (MoE) primitives."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import primitive


@primitive(impl="kernels.moe_softmax")
def moe_softmax(logits: Tensor["BT", "E"]) -> Tensor["BT", "E"]:
    """MoE router softmax."""
    ...


@primitive(impl="kernels.moe_sigmoid")
def moe_sigmoid(logits: Tensor["BT", "E"]) -> Tensor["BT", "E"]:
    """MoE router sigmoid (for Qwen3 style)."""
    ...


@primitive(impl="kernels.moe_topk")
def moe_topk(
    probs: Tensor["BT", "E"],
    *,
    top_k: int,
    normalize: bool = True,
) -> tuple[Tensor["BT", "top_k"], Tensor["BT", "top_k", "int32"]]:
    """MoE top-k selection. Returns (weights, indices)."""
    ...


@primitive(impl="kernels.moe_permute")
def moe_permute(
    x: Tensor["BT", "C"],
    indices: Tensor["BT", "top_k", "int32"],
    *,
    top_k: int,
) -> Tensor["BT * top_k", "C"]:
    """Permute inputs for expert computation."""
    ...


@primitive(impl="kernels.moe_unpermute")
def moe_unpermute(
    expert_out: Tensor["BT * top_k", "C"],
    weights: Tensor["BT", "top_k"],
    scatter_indices: Tensor["BT * top_k", "int32"],
    *,
    top_k: int,
) -> Tensor["BT", "C"]:
    """Unpermute and combine expert outputs."""
    ...


@primitive(impl="kernels.moe_grouped_gemm_gate_up")
def moe_grouped_gemm_gate_up(
    x: Tensor["BT", "C"],
    weights: Tensor["E", "2 * D", "C"],
    offsets: Tensor["E + 1", "int32"],
) -> Tensor["BT", "2 * D"]:
    """MoE grouped GEMM for gate+up projection."""
    ...


@primitive(impl="kernels.moe_grouped_gemm_down")
def moe_grouped_gemm_down(
    x: Tensor["BT", "D"],
    weights: Tensor["E", "C", "D"],
    offsets: Tensor["E + 1", "int32"],
) -> Tensor["BT", "C"]:
    """MoE grouped GEMM for down projection."""
    ...
