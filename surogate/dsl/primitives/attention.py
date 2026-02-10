"""Attention primitives."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import primitive, save


@primitive(impl="kernels.flash_attention")
def flash_attention(
    qkv: Tensor["B", "T", "Hq + 2 * Hkv", "D"],
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
    window_size: int | None = None,
) -> tuple[Tensor["B", "T", "Hq", "D"], Tensor["B", "Hq", "T"]]:
    """FlashAttention with packed QKV. Returns (out, lse)."""
    ...


@flash_attention.backward
@save("qkv", "out", "lse")
def flash_attention_backward(
    d_out: Tensor["B", "T", "Hq", "D"],
    out: Tensor["B", "T", "Hq", "D"],
    lse: Tensor["B", "Hq", "T"],
    qkv: Tensor["B", "T", "Hq + 2 * Hkv", "D"],
) -> Tensor["B", "T", "Hq + 2 * Hkv", "D"]:
    """Backward pass for FlashAttention."""
    ...


@primitive(impl="kernels.rope")
def rope(
    qkv: Tensor["B", "T", "H", "D"],
    freqs: Tensor["T", "D // 2", 2, "fp32"],
    position_ids: Tensor["T", "int32"],
    *,
    rotary_dim: int | None = None,
) -> Tensor["B", "T", "H", "D"]:
    """Apply rotary position embedding."""
    ...


@primitive(impl="kernels.mrope")
def mrope(
    qkv: Tensor["B", "T", "H", "D"],
    freqs: Tensor["T", "D // 2", 2, "fp32"],
    position_ids: Tensor[3, "B", "T", "int32"],
    *,
    rotary_dim: int | None = None,
    mrope_section: list[int] | tuple[int, int, int] | None = None,
) -> Tensor["B", "T", "H", "D"]:
    """Apply multimodal rotary position embedding (MRoPE)."""
    ...


@primitive(impl="kernels.qkv_qk_norm")
def qkv_qk_norm(
    qkv: Tensor["B", "T", "Hq + 2 * Hkv", "D"],
    q_norm_weight: Tensor["D"],
    k_norm_weight: Tensor["D"],
    *,
    eps: float = 1e-6,
) -> tuple[Tensor["B", "T", "Hq + 2 * Hkv", "D"], Tensor["*"], Tensor["*"]]:
    """QK norm (no RoPE). Returns (qkv_out, q_rstd, k_rstd)."""
    ...


@primitive(impl="kernels.qkv_qk_norm_rope")
def qkv_qk_norm_rope(
    qkv: Tensor["B", "T", "Hq + 2 * Hkv", "D"],
    q_norm_weight: Tensor["D"],
    k_norm_weight: Tensor["D"],
    freqs: Tensor["T", "D // 2", 2, "fp32"],
    position_ids: Tensor["T", "int32"],
    *,
    eps: float = 1e-6,
) -> tuple[Tensor["B", "T", "Hq + 2 * Hkv", "D"], Tensor["*"], Tensor["*"]]:
    """Fused QK norm + RoPE. Returns (qkv_out, q_rstd, k_rstd)."""
    ...
