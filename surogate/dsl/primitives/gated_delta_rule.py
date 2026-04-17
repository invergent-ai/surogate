"""Gated Delta Rule primitives used by Qwen3.5 linear-attention layers."""

from __future__ import annotations

from ..decorators import primitive, save
from ..tensor_type import Tensor


@primitive(impl="kernels.chunk_gated_delta_rule")
def chunk_gated_delta_rule(
    query: Tensor[B, T, H, K],
    key: Tensor[B, T, H, K],
    value: Tensor[B, T, H, V],
    g: Tensor[B, T, H],
    beta: Tensor[B, T, H],
    initial_state: Tensor[B, H, K, V] | None = None,
    *,
    scale: float = 0.0,
    chunk_size: int = 64,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor[B, T, H, V], Tensor[B, H, K, V] | None]:
    """Chunked gated delta rule forward pass.

    Mirrors FLA/HF `chunk_gated_delta_rule` API.
    """
    ...


@chunk_gated_delta_rule.backward
@save("query", "key", "value", "g", "beta", "initial_state")
def chunk_gated_delta_rule_backward(
    d_out: Tensor[B, T, H, V],
    d_final_state: Tensor[B, H, K, V] | None,
    query: Tensor[B, T, H, K],
    key: Tensor[B, T, H, K],
    value: Tensor[B, T, H, V],
    g: Tensor[B, T, H],
    beta: Tensor[B, T, H],
    initial_state: Tensor[B, H, K, V] | None = None,
) -> tuple[
    Tensor[B, T, H, K],  # d_query
    Tensor[B, T, H, K],  # d_key
    Tensor[B, T, H, V],  # d_value
    Tensor[B, T, H],  # d_g
    Tensor[B, T, H],  # d_beta
    Tensor[B, H, K, V] | None,  # d_initial_state
]:
    """Backward pass for chunk gated delta rule."""
    ...
