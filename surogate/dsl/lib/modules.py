"""
Standard Modules for Python DSL
"""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import module, param, forward, save
from ..graph_builder import graph


# =============================================================================
# Linear
# =============================================================================


@module
class Linear:
    """Linear projection: y = x @ W^T (+ bias)."""

    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = False):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        # Derived constants (like DSL let: section)
        self.C = in_dim
        self.O = out_dim

    @param
    def weight(self) -> Tensor["O", "C"]:
        """Weight matrix [out_dim, in_dim]."""
        ...

    @param(condition=lambda self: self.use_bias)
    def bias(self) -> Tensor["O"]:
        """Optional bias vector [out_dim]."""
        ...

    @forward
    @save("x")
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "O"]:
        with graph() as g:
            # Flatten batch dimensions
            x_flat = g.view(x, shape=["B * T", "C"])

            # Matrix multiply
            y_flat = g.matmul(x_flat, "weight", transpose="NT")

            # Reshape back
            y_tmp = g.view(y_flat, shape=["B", "T", "O"])

            # Optional bias
            if self.use_bias:
                y = g.bias_add(y_tmp, "bias")
            else:
                y = y_tmp

            return y


# =============================================================================
# RMSNorm
# =============================================================================


@module
class RMSNorm:
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        self.C = d_model

    @param
    def weight(self) -> Tensor["C"]:
        """Normalization weight [d_model]."""
        ...

    @forward
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        with graph() as g:
            y, _ = g.rmsnorm(x, "weight", eps=self.eps)
            return y


@module
class FusedResidualRMSNorm:
    """Fused residual addition + RMS normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        self.C = d_model

    @param
    def weight(self) -> Tensor["C"]:
        """Normalization weight [d_model]."""
        ...

    @forward
    def forward(
        self,
        residual: Tensor["B", "T", "C"],
        x: Tensor["B", "T", "C"],
    ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
        """Returns (residual_out, normalized)."""
        with graph() as g:
            residual_out, y, _ = g.fused_residual_rmsnorm(
                residual, x, "weight", eps=self.eps
            )
            return residual_out, y


# =============================================================================
# MLP
# =============================================================================


@module
class SwiGLUMLP:
    """SwiGLU MLP: down(swiglu(up(x)))."""

    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        self.C = d_model
        self.M = d_ff
        self.MUp = 2 * d_ff  # gate + up concatenated

    @param
    def up_weight(self) -> Tensor["MUp", "C"]:
        """Up projection weight [2*d_ff, d_model] (gate+up fused)."""
        ...

    @param
    def down_weight(self) -> Tensor["C", "M"]:
        """Down projection weight [d_model, d_ff]."""
        ...

    @forward
    @save("x", "up")
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        with graph() as g:
            # Flatten
            x_flat = g.view(x, shape=["B * T", "C"])

            # Up projection (gate + up combined)
            up_flat = g.matmul(x_flat, "up_weight", transpose="NT")
            up = g.view(up_flat, shape=["B", "T", "MUp"])

            # SwiGLU activation
            act = g.swiglu(up)

            # Down projection
            act_flat = g.view(act, shape=["B * T", "M"])
            y_flat = g.matmul(act_flat, "down_weight", transpose="NT")
            y = g.view(y_flat, shape=["B", "T", "C"])

            return y


@module
class GatedMLP:
    """Gated MLP with configurable activation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "silu",  # silu, relu, relu2, gelu
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.C = d_model
        self.M = d_ff

    @param
    def gate_weight(self) -> Tensor["M", "C"]:
        """Gate projection weight [d_ff, d_model]."""
        ...

    @param
    def up_weight(self) -> Tensor["M", "C"]:
        """Up projection weight [d_ff, d_model]."""
        ...

    @param
    def down_weight(self) -> Tensor["C", "M"]:
        """Down projection weight [d_model, d_ff]."""
        ...

    @forward
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        with graph() as g:
            x_flat = g.view(x, shape=["B * T", "C"])

            # Gate and up projections
            gate_flat = g.matmul(x_flat, "gate_weight", transpose="NT")
            up_flat = g.matmul(x_flat, "up_weight", transpose="NT")

            # Apply activation to gate
            if self.activation == "silu":
                gate_act = g.silu(gate_flat)
            elif self.activation == "relu":
                gate_act = g.relu(gate_flat)
            elif self.activation == "relu2":
                gate_act = g.relu2(gate_flat)
            elif self.activation == "gelu":
                gate_act = g.gelu(gate_flat)
            else:
                gate_act = g.silu(gate_flat)  # default

            # Gating
            hidden = g.mul(gate_act, up_flat)

            # Down projection
            y_flat = g.matmul(hidden, "down_weight", transpose="NT")
            y = g.view(y_flat, shape=["B", "T", "C"])

            return y


# =============================================================================
# Attention Modules
# =============================================================================


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
