"""Qwen3 Transformer Block."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import block, param, forward
from ..graph_builder import graph


@block
class Qwen3Block:
    """Qwen3 transformer block with QK-Norm."""

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        d_ff: int,
        max_seq: int,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        use_qk_norm: bool = True,
    ):
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.eps = eps
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm

        # Derived
        self.C = d_model
        self.Hq = num_query_heads
        self.Hkv = num_kv_heads
        self.D = head_size
        self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.AttnDim = num_query_heads * head_size
        self.M = d_ff
        self.MUp = 2 * d_ff

    @param
    def ln1_weight(self) -> Tensor["C"]:
        ...

    @param
    def ln2_weight(self) -> Tensor["C"]:
        ...

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

    @param
    def mlp_up_weight(self) -> Tensor["MUp", "C"]:
        ...

    @param
    def mlp_down_weight(self) -> Tensor["C", "M"]:
        ...

    @forward
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        residual: Tensor["B", "T", "C"],
        position_ids: Tensor["T", "int32"],
    ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
        with graph() as g:
            # Pre-attention norm
            res_ffn, ln1_out, ln1_rstd = g.fused_residual_rmsnorm(
                residual, x, "ln1_weight", eps=self.eps,
                res_out_name="res_ffn",
                y_name="ln1",
                rstd_name="ln1_rstd",
            )

            # QKV
            ln1_flat = g.view(ln1_out, shape=["B * T", "C"])
            qkv_flat = g.matmul(ln1_flat, "qkv_weight", transpose="NT", out_name="qkv_flat")

            if self.use_qkv_bias:
                qkv_tmp = g.view(qkv_flat, shape=["B", "T", "QKV"])
                qkv_biased = g.bias_add(qkv_tmp, "qkv_bias", out_name="qkv_biased")
                qkv_packed = g.view(qkv_biased, shape=["B", "T", "Hq + 2 * Hkv", "D"], out_name="qkv")
            else:
                qkv_packed = g.view(qkv_flat, shape=["B", "T", "Hq + 2 * Hkv", "D"], out_name="qkv")

            # QK-Norm + RoPE (fused)
            if self.use_qk_norm:
                qkv_rope, q_rstd, k_rstd = g.qkv_qk_norm_rope(
                    qkv_packed,
                    "q_norm_weight",
                    "k_norm_weight",
                    "rope_freqs",
                    position_ids,
                    eps=self.eps,
                    out_name="qkv_rope",
                    q_rstd_name="q_rstd",
                    k_rstd_name="k_rstd",
                )
            else:
                qkv_rope = g.rope(qkv_packed, "rope_freqs", position_ids, out_name="qkv_rope")

            # Attention
            attn_out, lse = g.flash_attention(qkv_rope, causal=True, out_name="att", lse_name="lse")

            # Output projection
            attn_flat = g.view(attn_out, shape=["B * T", "AttnDim"])
            att_out_flat = g.matmul(attn_flat, "out_weight", transpose="NT", out_name="att_out_flat")
            att_out = g.view(att_out_flat, shape=["B", "T", "C"], out_name="att_out")

            # Pre-MLP norm
            res_att, ln2_out, ln2_rstd = g.fused_residual_rmsnorm(
                res_ffn, att_out, "ln2_weight", eps=self.eps,
                res_out_name="res_att",
                y_name="ln2",
                rstd_name="ln2_rstd",
            )

            # MLP (SwiGLU)
            ln2_flat = g.view(ln2_out, shape=["B * T", "C"])
            mlp_up_flat = g.matmul(ln2_flat, "mlp_up_weight", transpose="NT", out_name="mlp_up_flat")
            mlp_up = g.view(mlp_up_flat, shape=["B", "T", "MUp"], out_name="mlp_up")
            mlp_act = g.swiglu(mlp_up, out_name="swiglu")
            mlp_act_flat = g.view(mlp_act, shape=["B * T", "M"])
            out_flat = g.matmul(mlp_act_flat, "mlp_down_weight", transpose="NT", out_name="mlp_down_flat")
            out = g.view(out_flat, shape=["B", "T", "C"], out_name="mlp_down")

            return out, res_att
