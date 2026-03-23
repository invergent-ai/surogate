"""Qwen3-VL Transformer Block (text)."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import block, forward, Param, Activation, Gradient
from ..graph_builder import graph
from ..dim import Dim, B, T

@block
class Qwen3VLBlock:
    """Qwen3-VL text transformer block with QK-Norm + MRoPE."""

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
        mrope_section: tuple[int, int, int] | list[int] = (24, 20, 20),
    ):
        self.d_model = d_model  # Hidden dimension of the model
        self.num_query_heads = num_query_heads  # Number of query attention heads
        self.num_kv_heads = num_kv_heads  # Number of key/value heads (GQA); may be < num_query_heads
        self.head_size = head_size  # Feature dimension per attention head (D)
        self.d_ff = d_ff  # Feed-forward intermediate dimension (before SwiGLU split)
        self.max_seq = max_seq  # Maximum sequence length for RoPE frequency cache
        self.eps = eps  # Epsilon for RMSNorm numerical stability
        self.use_qkv_bias = use_qkv_bias  # Whether Q/K/V/out projections include bias terms
        self.mrope_section = list(mrope_section)  # MRoPE split sizes for (temporal, height, width) axes

        # Typed dimensions - use short symbolic names that C++ ShapeEnv expects
        self.C = Dim("C")  # Hidden / channel dimension (= d_model)
        self.Hq = Dim("Hq")  # Number of query heads (= num_query_heads)
        self.Hkv = Dim("Hkv")  # Number of KV heads (= num_kv_heads)
        self.D = Dim("D")  # Per-head feature dimension (= head_size)
        self.M = Dim("M")  # MLP intermediate dimension after SwiGLU split (= d_ff // 2)
        self.MaxSeq = Dim("MaxSeq")  # Max sequence length (= max_seq)

        # Derived dimensions (DimExpr)
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D  # Packed QKV projection dimension: (Hq + 2*Hkv) * D
        self.AttnDim = self.Hq * self.D  # Total attention output dimension: Hq * D
        self.MUp = 2 * self.M  # Up-projection output size: gate + up channels for SwiGLU

    # Parameters
    ln1_weight = Param(Tensor["C"])
    ln2_weight = Param(Tensor["C"])
    qkv_weight = Param(Tensor["QKV", "C"])
    qkv_bias = Param(Tensor["QKV"], when="use_qkv_bias")
    out_weight = Param(Tensor["C", "AttnDim"])
    out_bias = Param(Tensor["C"], when="use_qkv_bias")
    q_norm_weight = Param(Tensor["D"], quantizable=False)
    k_norm_weight = Param(Tensor["D"], quantizable=False)
    rope_freqs = Param(Tensor["MaxSeq", "D // 2", 2, "fp32"], frozen=True)
    mlp_up_weight = Param(Tensor["MUp", "C"])
    mlp_down_weight = Param(Tensor["C", "M"])

    # =========================================================================
    # Activation slots (forward pass intermediate tensors)
    # =========================================================================

    # Pre-attention normalization
    ln1 = Activation(
        Tensor["B", "T", "C"],
        aliases=["ln1_flat"],
        share_policy="when_recomputed",
    )
    ln1_rstd = Activation(
        Tensor["B", "T"], dtype="fp32", save=True,
        share_policy="per_layer",
        description="RMSNorm reciprocal std for LN1",
    )

    # QKV projection
    qkv = Activation(
        Tensor["B", "T", "QKV"],
        aliases=["qkv_flat", "qkv_biased"],
        save=True,
        share_policy="when_recomputed",
    )

    # QK-Norm (no RoPE)
    qkv_norm = Activation(
        Tensor["B", "T", "QKV"],
        save=True,
        share_policy="when_recomputed",
        description="QKV after QK-Norm",
    )

    q_rstd = Activation(
        Tensor["B", "T", "Hq"],
        dtype="fp32",
        save=True,
        share_policy="when_recomputed",
        description="Q head RMSNorm rstd",
    )
    k_rstd = Activation(
        Tensor["B", "T", "Hkv"],
        dtype="fp32",
        save=True,
        share_policy="when_recomputed",
        description="K head RMSNorm rstd",
    )

    # MRoPE
    qkv_rope = Activation(
        Tensor["B", "T", "QKV"],
        save=True,
        share_policy="when_recomputed",
        description="QKV after QK-Norm + MRoPE",
    )

    # Attention
    att = Activation(
        Tensor["B", "T", "AttnDim"],
        aliases=["att_flat", "attn"],
        save=True,
        share_policy="always_recompute",
        description="Attention output (pre out-proj)",
    )
    lse = Activation(
        Tensor["B", "Hq", "T"],
        dtype="fp32",
        save=True,
        share_policy="always_recompute",
        description="Log-sum-exp from flash attention",
    )
    att_out = Activation(
        Tensor["B", "T", "C"],
        aliases=["att_out_flat"],
        share_policy="when_recomputed",
        description="After output projection",
    )

    # First residual
    res_att = Activation(
        Tensor["B", "T", "C"],
        aliases=["residual_att"],
        share_policy="when_recomputed",
        description="Residual + attention",
    )

    # Pre-MLP normalization
    ln2 = Activation(
        Tensor["B", "T", "C"],
        aliases=["ln2_flat"],
        share_policy="when_recomputed",
    )
    ln2_rstd = Activation(
        Tensor["B", "T"], dtype="fp32", save=True,
        share_policy="per_layer",
        description="RMSNorm reciprocal std for LN2",
    )

    # MLP
    mlp_up = Activation(
        Tensor["B", "T", "MUp"],
        aliases=["mlp_up_flat"],
        share_policy="when_recomputed",
    )
    swiglu = Activation(
        Tensor["B", "T", "M"],
        aliases=["swiglu_flat"],
        share_policy="when_recomputed",
        description="SwiGLU activation output",
    )
    mlp_down = Activation(
        Tensor["B", "T", "C"],
        aliases=["mlp_down_flat"],
        share_policy="when_recomputed",
        description="MLP down projection output",
    )

    # Second residual
    res_ffn = Activation(
        Tensor["B", "T", "C"],
        aliases=["residual_ffn"],
        share_policy="per_layer",
        description="Residual + MLP (block output)",
    )

    # =========================================================================
    # Gradient slots (backward pass)
    # =========================================================================

    d_ln1 = Gradient(Tensor["B", "T", "C"], gradient_of="ln1")
    d_qkv = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv")
    d_qkv_norm = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv_norm")
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
        position_ids: Tensor[3, "B", "T", "int32"],
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
            ln1_flat = g.view(ln1_out, shape=[B * T, self.C], out_name="ln1_flat")
            if self.use_qkv_bias:
                qkv_flat = g.matmul_bias(ln1_flat, "qkv_weight", "qkv_bias", transpose="NT", out_name="qkv_flat")
            else:
                qkv_flat = g.matmul(ln1_flat, "qkv_weight", transpose="NT", out_name="qkv_flat")
            qkv_packed = g.view(qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D], out_name="qkv")

            # QK-Norm
            qkv_norm, q_rstd, k_rstd = g.qkv_qk_norm(
                qkv_packed,
                "q_norm_weight",
                "k_norm_weight",
                eps=self.eps,
                out_name="qkv_norm",
                q_rstd_name="q_rstd",
                k_rstd_name="k_rstd",
            )

            # MRoPE
            qkv_rope = g.mrope(
                qkv_norm,
                "rope_freqs",
                position_ids,
                rotary_dim="D",
                mrope_section=self.mrope_section,
                out_name="qkv_rope",
            )

            # Attention
            attn_out, lse = g.flash_attention(qkv_rope, causal=True, out_name="att", lse_name="lse")

            # Output projection
            attn_flat = g.view(attn_out, shape=[B * T, self.AttnDim], out_name="att_flat")
            if self.use_qkv_bias:
                att_out_flat = g.matmul_bias(attn_flat, "out_weight", "out_bias", transpose="NT", out_name="att_out_flat")
            else:
                att_out_flat = g.matmul(attn_flat, "out_weight", transpose="NT", out_name="att_out_flat")
            att_out = g.view(att_out_flat, shape=[B, T, self.C], out_name="att_out")

            # Pre-MLP norm
            res_att, ln2_out, ln2_rstd = g.fused_residual_rmsnorm(
                res_ffn, att_out, "ln2_weight", eps=self.eps,
                res_out_name="res_att",
                y_name="ln2",
                rstd_name="ln2_rstd",
            )

            # MLP (SwiGLU)
            ln2_flat = g.view(ln2_out, shape=[B * T, self.C], out_name="ln2_flat")
            mlp_up_flat = g.matmul(ln2_flat, "mlp_up_weight", transpose="NT", out_name="mlp_up_flat")
            mlp_up = g.view(mlp_up_flat, shape=[B, T, self.MUp], out_name="mlp_up")
            mlp_act = g.swiglu(mlp_up, out_name="swiglu")
            mlp_act_flat = g.view(mlp_act, shape=[B * T, self.M], out_name="swiglu_flat")
            out_flat = g.matmul(mlp_act_flat, "mlp_down_weight", transpose="NT", out_name="mlp_down_flat")
            out = g.view(out_flat, shape=[B, T, self.C], out_name="mlp_down")

            return out, res_att
