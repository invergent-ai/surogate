"""Attention Modules."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import module, forward, save, Param
from ..graph_builder import graph
from ..dim import Dim, B, T


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

        # Typed dimensions - use short symbolic names that C++ ShapeEnv expects
        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.MaxSeq = Dim("MaxSeq")

        # Derived dimensions (DimExpr)
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D

    # Attention weights
    qkv_weight = Param(Tensor["QKV", "C"])
    qkv_bias = Param(Tensor["QKV"], when="use_qkv_bias")
    out_weight = Param(Tensor["C", "AttnDim"])
    rope_freqs = Param(Tensor["MaxSeq", "D // 2", 2, "fp32"], frozen=True)

    @forward
    @save("qkv", "out", "lse")
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        position_ids: Tensor["T", "int32"],
    ) -> Tensor["B", "T", "C"]:
        with graph() as g:
            # QKV projection
            x_flat = g.view(x, shape=[B * T, self.C])
            if self.use_qkv_bias:
                qkv_flat = g.matmul_bias(x_flat, "qkv_weight", "qkv_bias", transpose="NT")
            else:
                qkv_flat = g.matmul(x_flat, "qkv_weight", transpose="NT")
            qkv_packed = g.view(qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D])

            # Apply RoPE
            qkv_rope = g.rope(qkv_packed, "rope_freqs", position_ids, rotary_dim="D")

            # FlashAttention
            attn_out, _ = g.flash_attention(qkv_rope, causal=True)

            # Output projection
            attn_flat = g.view(attn_out, shape=[B * T, self.AttnDim])
            out_flat = g.matmul(attn_flat, "out_weight", transpose="NT")
            out = g.view(out_flat, shape=[B, T, self.C])

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

        # Typed dimensions - use short symbolic names that C++ ShapeEnv expects
        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.MaxSeq = Dim("MaxSeq")

        # Derived dimensions (DimExpr)
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D

    # Attention weights
    qkv_weight = Param(Tensor["QKV", "C"])
    qkv_bias = Param(Tensor["QKV"], when="use_qkv_bias")
    out_weight = Param(Tensor["C", "AttnDim"])
    q_norm_weight = Param(Tensor["D"], when="use_qk_norm")
    k_norm_weight = Param(Tensor["D"], when="use_qk_norm")
    rope_freqs = Param(Tensor["MaxSeq", "D // 2", 2, "fp32"], frozen=True)

    @forward
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        position_ids: Tensor["T", "int32"],
    ) -> Tensor["B", "T", "C"]:
        with graph() as g:
            x_flat = g.view(x, shape=[B * T, self.C])
            if self.use_qkv_bias:
                qkv_flat = g.matmul_bias(x_flat, "qkv_weight", "qkv_bias", transpose="NT")
            else:
                qkv_flat = g.matmul(x_flat, "qkv_weight", transpose="NT")
            qkv_packed = g.view(qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D])

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

            attn_flat = g.view(attn_out, shape=[B * T, self.AttnDim])
            out_flat = g.matmul(attn_flat, "out_weight", transpose="NT")
            out = g.view(out_flat, shape=[B, T, self.C])

            return out
