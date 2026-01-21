"""Dense Transformer Block."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import block, param, forward
from ..graph_builder import graph
from .common import Activation


@block
class DenseTransformerBlock:
    """Pre-norm dense transformer block with attention + MLP.

    Structure:
        residual, x = fused_residual_rmsnorm(residual, x)
        x = attention(x) + residual
        residual, x = fused_residual_rmsnorm(residual, x)
        x = mlp(x) + residual
    """

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
        use_qk_norm: bool = False,
        activation: Activation = Activation.SwiGLU,
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
        self.activation = activation

        # Derived constants
        self.C = d_model
        self.Hq = num_query_heads
        self.Hkv = num_kv_heads
        self.D = head_size
        self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.AttnDim = num_query_heads * head_size
        self.M = d_ff
        self.MUp = 2 * d_ff if activation == Activation.SwiGLU else d_ff

    # LayerNorm weights
    @param
    def ln1_weight(self) -> Tensor["C"]:
        """Pre-attention layer norm weight."""
        ...

    @param
    def ln2_weight(self) -> Tensor["C"]:
        """Pre-MLP layer norm weight."""
        ...

    # Attention weights
    @param
    def qkv_weight(self) -> Tensor["QKV", "C"]:
        """Combined QKV projection."""
        ...

    @param(condition=lambda self: self.use_qkv_bias)
    def qkv_bias(self) -> Tensor["QKV"]:
        """QKV projection bias."""
        ...

    @param
    def out_weight(self) -> Tensor["C", "AttnDim"]:
        """Attention output projection."""
        ...

    @param(condition=lambda self: self.use_qk_norm)
    def q_norm_weight(self) -> Tensor["D"]:
        """Query norm weight for QK-Norm."""
        ...

    @param(condition=lambda self: self.use_qk_norm)
    def k_norm_weight(self) -> Tensor["D"]:
        """Key norm weight for QK-Norm."""
        ...

    @param(frozen=True)
    def rope_freqs(self) -> Tensor["max_seq", "D // 2", 2, "fp32"]:
        """Precomputed RoPE frequencies."""
        ...

    # MLP weights
    @param
    def mlp_up_weight(self) -> Tensor["MUp", "C"]:
        """MLP up (+ gate for SwiGLU) projection."""
        ...

    @param
    def mlp_down_weight(self) -> Tensor["C", "M"]:
        """MLP down projection."""
        ...

    @forward
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        residual: Tensor["B", "T", "C"],
        position_ids: Tensor["T", "int32"],
    ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
        """Returns (out, residual_out)."""
        with graph() as g:
            # Pre-attention LayerNorm (fused with residual)
            res_ffn, ln1_out, ln1_rstd = g.fused_residual_rmsnorm(
                residual, x, "ln1_weight", eps=self.eps,
                res_out_name="res_ffn",
                y_name="ln1",
                rstd_name="ln1_rstd",
            )

            # QKV projection
            ln1_flat = g.view(ln1_out, shape=["B * T", "C"])
            qkv_flat = g.matmul(ln1_flat, "qkv_weight", transpose="NT", out_name="qkv_flat")

            if self.use_qkv_bias:
                qkv_tmp = g.view(qkv_flat, shape=["B", "T", "QKV"])
                qkv_biased = g.bias_add(qkv_tmp, "qkv_bias", out_name="qkv_biased")
                qkv_packed = g.view(qkv_biased, shape=["B", "T", "Hq + 2 * Hkv", "D"], out_name="qkv")
            else:
                qkv_packed = g.view(qkv_flat, shape=["B", "T", "Hq + 2 * Hkv", "D"], out_name="qkv")

            # RoPE (with optional QK-Norm)
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
                qkv_rope = g.rope(qkv_packed, "rope_freqs", position_ids, rotary_dim="D", out_name="qkv_rope")

            # FlashAttention
            attn_out, lse = g.flash_attention(qkv_rope, causal=True, out_name="att", lse_name="lse")

            # Attention output projection
            attn_flat = g.view(attn_out, shape=["B * T", "AttnDim"])
            att_out_flat = g.matmul(attn_flat, "out_weight", transpose="NT", out_name="att_out_flat")
            att_out = g.view(att_out_flat, shape=["B", "T", "C"], out_name="att_out")

            # Pre-MLP LayerNorm (fused with residual)
            res_att, ln2_out, ln2_rstd = g.fused_residual_rmsnorm(
                res_ffn, att_out, "ln2_weight", eps=self.eps,
                res_out_name="res_att",
                y_name="ln2",
                rstd_name="ln2_rstd",
            )

            # MLP
            ln2_flat = g.view(ln2_out, shape=["B * T", "C"])
            mlp_up_flat = g.matmul(ln2_flat, "mlp_up_weight", transpose="NT", out_name="mlp_up_flat")
            mlp_up = g.view(mlp_up_flat, shape=["B", "T", "MUp"], out_name="mlp_up")

            # Activation
            if self.activation == Activation.SwiGLU:
                mlp_act = g.swiglu(mlp_up, out_name="swiglu")
            elif self.activation == Activation.SiLU:
                mlp_act = g.silu(mlp_up, out_name="swiglu")
            elif self.activation == Activation.ReLU2:
                mlp_act = g.relu2(mlp_up, out_name="swiglu")
            else:
                mlp_act = g.gelu(mlp_up, out_name="swiglu")

            # MLP down projection
            mlp_act_flat = g.view(mlp_act, shape=["B * T", "M"])
            out_flat = g.matmul(mlp_act_flat, "mlp_down_weight", transpose="NT", out_name="mlp_down_flat")
            out = g.view(out_flat, shape=["B", "T", "C"], out_name="mlp_down")

            return out, res_att
