"""Qwen3 Transformer Block."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import block, forward, Param, Activation, Gradient
from ..graph_builder import graph
from ..dim import Dim, B, T


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

    # Parameters using class attribute style
    ln1_weight = Param(Tensor["C"])
    ln2_weight = Param(Tensor["C"])
    qkv_weight = Param(Tensor["QKV", "C"])
    qkv_bias = Param(Tensor["QKV"], when="use_qkv_bias")
    out_weight = Param(Tensor["C", "AttnDim"])
    q_norm_weight = Param(Tensor["D"], when="use_qk_norm")
    k_norm_weight = Param(Tensor["D"], when="use_qk_norm")
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
        recompute=True,
        # Recompute LN1 directly from saved res_ffn + rstd.
        # This matches implementation (res_ffn is retrieved from residual storage, not re-derived).
        recompute_from=["res_ffn", "ln1_rstd", "@param:ln1_weight"],
        recompute_op="rmsnorm_apply_saved",
        recompute_policy="always",  # Recompute in both FFT and LoRA modes
    )
    ln1_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True,
                          description="RMSNorm reciprocal std for LN1")

    # QKV projection and RoPE
    # NOTE: qkv is safe to recompute in FFT mode (cuDNN attention is deterministic in latest release).
    # We still save qkv for backward fallbacks.
    qkv = Activation(
        Tensor["B", "T", "QKV"],
        aliases=["qkv_flat", "qkv_biased"],
        save=True,  # Save in FFT mode for consistency
        recompute=True,
        recompute_from=["ln1", "@param:qkv_weight", "?@param:qkv_bias"],
        recompute_op="matmul",
        recompute_attrs={"matmul_op": "qkv", "transpose": "NT"},
        recompute_policy="fft_only",
        lora_targets=["q", "k", "v"],
    )
    # NOTE: qkv_rope is safe to recompute in FFT mode (cuDNN attention is deterministic).
    qkv_rope = Activation(
        Tensor["B", "T", "QKV"],
        save=True,  # Save in FFT mode for consistency with att/lse
        recompute=True,
        recompute_group="qk_norm_rope",
        recompute_outputs=["qkv_rope", "q_rstd", "k_rstd"],
        recompute_from=[
            "qkv",
            "@global:freq_cis",
            "@input:position_ids",
            "?@param:q_norm_weight",
            "?@param:k_norm_weight",
        ],
        recompute_op="qkv_qk_norm_rope",
        recompute_attrs={"rotary_dim": "D"},
        recompute_policy="fft_only",
        description="QKV after QK-Norm + RoPE",
    )

    # QK-norm RSTDs (conditional on use_qk_norm)
    # NOTE: Match qkv_rope policy - recompute in FFT mode
    q_rstd = Activation(
        Tensor["B", "T", "Hq"],
        dtype="fp32",
        save=True,
        recompute=True,
        recompute_group="qk_norm_rope",
        recompute_policy="fft_only",
        when="use_qk_norm",
        description="Q head RMSNorm rstd",
    )
    k_rstd = Activation(
        Tensor["B", "T", "Hkv"],
        dtype="fp32",
        save=True,
        recompute=True,
        recompute_group="qk_norm_rope",
        recompute_policy="fft_only",
        when="use_qk_norm",
        description="K head RMSNorm rstd",
    )

    # Attention
    # NOTE: cuDNN flash attention is deterministic in latest release, so we can recompute in FFT.
    att = Activation(
        Tensor["B", "T", "AttnDim"],
        aliases=["att_flat", "attn"],
        save=True,  # Save for FFT mode backward (needed by att_out backward AND attention backward)
        recompute=True,
        recompute_group="attn_fwd",
        recompute_outputs=["att", "lse"],
        recompute_from=["qkv_rope"],
        recompute_op="flash_attention",
        recompute_attrs={"attn_impl": "cudnn"},
        recompute_policy="fft_only",
        description="Attention output (pre out-proj)",
    )
    lse = Activation(
        Tensor["B", "Hq", "T"],
        dtype="fp32",
        save=True,
        recompute=True,
        recompute_group="attn_fwd",
        recompute_policy="fft_only",
        description="Log-sum-exp from flash attention",
    )
    att_out = Activation(
        Tensor["B", "T", "C"],
        aliases=["att_out_flat"],
        recompute=True,
        recompute_from=["att", "@param:out_weight"],
        recompute_op="matmul",
        recompute_attrs={"matmul_op": "attn_out", "transpose": "NT"},
        recompute_policy="fft_only",
        lora_targets=["o"],
        description="After output projection",
    )

    # First residual
    res_att = Activation(
        Tensor["B", "T", "C"],
        aliases=["residual_att"],
        recompute=True,
        recompute_group="ln2_fused",
        recompute_outputs=["res_att", "ln2"],
        recompute_from=["res_ffn", "att_out", "ln2_rstd", "@param:ln2_weight"],
        recompute_op="fused_residual_rmsnorm_apply_saved",
        recompute_policy="always",  # Recompute in both FFT and LoRA modes
        description="Residual + attention",
    )

    # Pre-MLP normalization
    ln2 = Activation(
        Tensor["B", "T", "C"],
        aliases=["ln2_flat"],
        recompute=True,
        recompute_group="ln2_fused",
        recompute_policy="always",  # Recompute in both FFT and LoRA modes
    )
    ln2_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True,
                          description="RMSNorm reciprocal std for LN2")

    # MLP
    mlp_up = Activation(
        Tensor["B", "T", "MUp"],
        aliases=["mlp_up_flat"],
        recompute=True,
        recompute_from=["ln2", "@param:mlp_up_weight"],
        recompute_op="matmul",
        recompute_attrs={"matmul_op": "mlp_up", "transpose": "NT"},
        recompute_policy="fft_only",
        lora_targets=["up", "gate"],
    )
    swiglu = Activation(
        Tensor["B", "T", "M"],
        aliases=["swiglu_flat"],
        recompute=True,
        recompute_from=["mlp_up"],
        recompute_op="swiglu",
        recompute_attrs={"activation": "swiglu"},
        recompute_policy="fft_only",
        description="SwiGLU activation output",
    )
    mlp_down = Activation(Tensor["B", "T", "C"], aliases=["mlp_down_flat"],
                          description="MLP down projection output")

    # Second residual
    res_ffn = Activation(
        Tensor["B", "T", "C"],
        aliases=["residual_ffn"],
        # res_ffn is stored via residual manager; do not mark as recompute.
        description="Residual + MLP (block output)",
    )

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
            attn_flat = g.view(attn_out, shape=[B * T, self.AttnDim], out_name="att_flat")
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
