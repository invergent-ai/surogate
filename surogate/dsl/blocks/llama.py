"""LLaMA Transformer Block."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import block, forward, Param, Activation, Gradient
from ..graph_builder import graph
from ..dim import Dim, B, T


@block
class LlamaBlock:
    """LLaMA-style transformer block (no QK-Norm, GQA, SwiGLU)."""

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        d_ff: int,
        max_seq: int,
        eps: float = 1e-6,
    ):
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.eps = eps

        # Typed dimensions - use short symbolic names that C++ ShapeEnv expects
        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.M = Dim("M")
        self.MaxSeq = Dim("MaxSeq")

        # Derived dimensions (DimExpr)
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D
        self.MUp = 2 * self.M

    # LayerNorm weights
    ln1_weight = Param(Tensor["C"])
    ln2_weight = Param(Tensor["C"])

    # Attention weights
    qkv_weight = Param(Tensor["QKV", "C"])
    out_weight = Param(Tensor["C", "AttnDim"])
    rope_freqs = Param(Tensor["MaxSeq", "D // 2", 2, "fp32"], frozen=True)

    # MLP weights
    mlp_up_weight = Param(Tensor["MUp", "C"])
    mlp_down_weight = Param(Tensor["C", "M"])

    # =========================================================================
    # Activation slots (forward pass intermediate tensors)
    # =========================================================================

    # Pre-attention normalization
    ln1 = Activation(
        Tensor["B", "T", "C"],
        aliases=["ln1_flat"],
        recompute=True,
        # Recompute LN1 directly from saved res_ffn + rstd to match implementation.
        recompute_from=["res_ffn", "ln1_rstd", "@param:ln1_weight"],
        recompute_op="rmsnorm_apply_saved",
        recompute_policy="fft_only",
    )
    ln1_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True,
                          description="RMSNorm reciprocal std for LN1")

    # QKV projection and RoPE
    qkv = Activation(Tensor["B", "T", "QKV"], aliases=["qkv_flat"])
    qkv_rope = Activation(Tensor["B", "T", "QKV"],
                          description="QKV after RoPE")

    # Attention
    att = Activation(Tensor["B", "T", "AttnDim"], aliases=["att_flat", "attn"],
                     description="Attention output (pre out-proj)")
    lse = Activation(Tensor["B", "Hq", "T"], dtype="fp32", save=True,
                     description="Log-sum-exp from flash attention")
    att_out = Activation(Tensor["B", "T", "C"], aliases=["att_out_flat"],
                         description="After output projection")

    # First residual
    res_att = Activation(Tensor["B", "T", "C"], aliases=["residual_att"],
                         description="Residual + attention")

    # Pre-MLP normalization
    ln2 = Activation(Tensor["B", "T", "C"], aliases=["ln2_flat"])
    ln2_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True,
                          description="RMSNorm reciprocal std for LN2")

    # MLP
    mlp_up = Activation(Tensor["B", "T", "MUp"], aliases=["mlp_up_flat"])
    swiglu = Activation(Tensor["B", "T", "M"], aliases=["swiglu_flat"],
                        description="SwiGLU activation output")
    mlp_down = Activation(Tensor["B", "T", "C"], aliases=["mlp_down_flat"],
                          description="MLP down projection output")

    # Second residual
    res_ffn = Activation(Tensor["B", "T", "C"], aliases=["residual_ffn"],
                         description="Residual + MLP (block output)")

    # =========================================================================
    # Gradient slots (backward pass)
    # =========================================================================

    d_ln1 = Gradient(Tensor["B", "T", "C"], gradient_of="ln1")
    d_qkv = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv")
    d_qkv_rope = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv_rope")
    d_att = Gradient(Tensor["B", "T", "AttnDim"], gradient_of="att")
    d_ln2 = Gradient(Tensor["B", "T", "C"], gradient_of="ln2")
    d_mlp_up = Gradient(Tensor["B", "T", "MUp"], gradient_of="mlp_up")
    d_swiglu = Gradient(Tensor["B", "T", "M"], gradient_of="swiglu")
    d_mlp_down = Gradient(Tensor["B", "T", "C"], gradient_of="mlp_down")
    d_res_att = Gradient(Tensor["B", "T", "C"], gradient_of="res_att")
    d_res_ffn = Gradient(Tensor["B", "T", "C"], gradient_of="res_ffn")

    @forward
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        residual: Tensor["B", "T", "C"],
        position_ids: Tensor["T", "int32"],
    ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
        with graph() as g:
            res_ffn, ln1_out, ln1_rstd = g.fused_residual_rmsnorm(
                residual, x, "ln1_weight", eps=self.eps,
                res_out_name="res_ffn",
                y_name="ln1",
                rstd_name="ln1_rstd",
            )

            ln1_flat = g.view(ln1_out, shape=[B * T, self.C], out_name="ln1_flat")
            qkv_flat = g.matmul(ln1_flat, "qkv_weight", transpose="NT", out_name="qkv_flat")
            qkv_packed = g.view(qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D], out_name="qkv")

            qkv_rope = g.rope(qkv_packed, "rope_freqs", position_ids, rotary_dim="D", out_name="qkv_rope")
            attn_out, lse = g.flash_attention(qkv_rope, causal=True, out_name="att", lse_name="lse")

            attn_flat = g.view(attn_out, shape=[B * T, self.AttnDim], out_name="att_flat")
            att_out_flat = g.matmul(attn_flat, "out_weight", transpose="NT", out_name="att_out_flat")
            att_out = g.view(att_out_flat, shape=[B, T, self.C], out_name="att_out")

            res_att, ln2_out, ln2_rstd = g.fused_residual_rmsnorm(
                res_ffn, att_out, "ln2_weight", eps=self.eps,
                res_out_name="res_att",
                y_name="ln2",
                rstd_name="ln2_rstd",
            )

            ln2_flat = g.view(ln2_out, shape=[B * T, self.C], out_name="ln2_flat")
            mlp_up_flat = g.matmul(ln2_flat, "mlp_up_weight", transpose="NT", out_name="mlp_up_flat")
            mlp_up = g.view(mlp_up_flat, shape=[B, T, self.MUp], out_name="mlp_up")
            mlp_act = g.swiglu(mlp_up, out_name="swiglu")
            mlp_act_flat = g.view(mlp_act, shape=[B * T, self.M], out_name="swiglu_flat")
            out_flat = g.matmul(mlp_act_flat, "mlp_down_weight", transpose="NT", out_name="mlp_down_flat")
            out = g.view(out_flat, shape=[B, T, self.C], out_name="mlp_down")

            return out, res_att
