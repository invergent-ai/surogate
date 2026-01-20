"""Attention Modules."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import module, param, forward, save
from ..graph_builder import graph


@module
class GQAAttention:
    """Grouped-Query Attention with RoPE and FlashAttention."""

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        use_qkv_bias: bool = False,
    ):
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq
        self.use_qkv_bias = use_qkv_bias

        # Derived constants
        self.C = d_model
        self.Hq = num_query_heads
        self.Hkv = num_kv_heads
        self.D = head_size
        self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.AttnDim = num_query_heads * head_size

    @param
    def qkv_weight(self) -> Tensor["QKV", "C"]:
        """Combined QKV projection weight."""
        ...

    @param(condition=lambda self: self.use_qkv_bias)
    def qkv_bias(self) -> Tensor["QKV"]:
        """Optional QKV projection bias."""
        ...

    @param
    def out_weight(self) -> Tensor["C", "AttnDim"]:
        """Output projection weight."""
        ...

    @param(frozen=True)
    def rope_freqs(self) -> Tensor["max_seq", "D // 2", 2, "fp32"]:
        """Precomputed RoPE frequencies."""
        ...

    @forward
    @save("qkv", "out", "lse")
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        position_ids: Tensor["T", "int32"],
    ) -> Tensor["B", "T", "C"]:
        with graph() as g:
            # QKV projection
            x_flat = g.view(x, shape=["B * T", "C"])
            qkv_flat = g.matmul(x_flat, "qkv_weight", transpose="NT")

            if self.use_qkv_bias:
                qkv_tmp = g.view(qkv_flat, shape=["B", "T", "QKV"])
                qkv_biased = g.bias_add(qkv_tmp, "qkv_bias")
                qkv_packed = g.view(qkv_biased, shape=["B", "T", "Hq + 2 * Hkv", "D"])
            else:
                qkv_packed = g.view(qkv_flat, shape=["B", "T", "Hq + 2 * Hkv", "D"])

            # Apply RoPE
            qkv_rope = g.rope(qkv_packed, "rope_freqs", position_ids, rotary_dim="D")

            # FlashAttention
            attn_out, _ = g.flash_attention(qkv_rope, causal=True)

            # Output projection
            attn_flat = g.view(attn_out, shape=["B * T", "AttnDim"])
            out_flat = g.matmul(attn_flat, "out_weight", transpose="NT")
            out = g.view(out_flat, shape=["B", "T", "C"])

            return out


@module
class Qwen3Attention:
    """Qwen3-style attention with QK-Norm."""

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        max_seq: int,
        use_qkv_bias: bool = False,
        use_qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.max_seq = max_seq
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm
        self.eps = eps

        # Derived constants
        self.C = d_model
        self.Hq = num_query_heads
        self.Hkv = num_kv_heads
        self.D = head_size
        self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.AttnDim = num_query_heads * head_size

    @param
    def qkv_weight(self) -> Tensor["QKV", "C"]:
        ...

    @param(condition=lambda self: self.use_qkv_bias)
    def qkv_bias(self) -> Tensor["QKV"]:
        ...

    @param
    def out_weight(self) -> Tensor["C", "AttnDim"]:
        ...

    @param(condition=lambda self: self.use_qk_norm)
    def q_norm_weight(self) -> Tensor["D"]:
        ...

    @param(condition=lambda self: self.use_qk_norm)
    def k_norm_weight(self) -> Tensor["D"]:
        ...

    @param(frozen=True)
    def rope_freqs(self) -> Tensor["max_seq", "D // 2", 2, "fp32"]:
        ...

    @forward
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        position_ids: Tensor["T", "int32"],
    ) -> Tensor["B", "T", "C"]:
        with graph() as g:
            x_flat = g.view(x, shape=["B * T", "C"])
            qkv_flat = g.matmul(x_flat, "qkv_weight", transpose="NT")

            if self.use_qkv_bias:
                qkv_tmp = g.view(qkv_flat, shape=["B", "T", "QKV"])
                qkv_biased = g.bias_add(qkv_tmp, "qkv_bias")
                qkv_packed = g.view(qkv_biased, shape=["B", "T", "Hq + 2 * Hkv", "D"])
            else:
                qkv_packed = g.view(qkv_flat, shape=["B", "T", "Hq + 2 * Hkv", "D"])

            if self.use_qk_norm:
                qkv_rope, _, _ = g.qkv_qk_norm_rope(
                    qkv_packed,
                    "q_norm_weight",
                    "k_norm_weight",
                    "rope_freqs",
                    position_ids,
                    eps=self.eps,
                )
            else:
                qkv_rope = g.rope(qkv_packed, "rope_freqs", position_ids)

            attn_out, _ = g.flash_attention(qkv_rope, causal=True)

            attn_flat = g.view(attn_out, shape=["B * T", "AttnDim"])
            out_flat = g.matmul(attn_flat, "out_weight", transpose="NT")
            out = g.view(out_flat, shape=["B", "T", "C"])

            return out
