"""Qwen3.5 dense transformer blocks."""

from __future__ import annotations

from .. import nn
from ..nn import (
    QWEN3_5_ATTN_BLOCK_REMAP,
    QWEN3_5_LINEAR_BLOCK_REMAP,
)


class Qwen3_5AttentionBlock(nn.Block):
    """Qwen3.5 full-attention decoder block (token mixer + MLP).

    Uses separate Q/K/V projections (not fused QKV), QK-Norm with weight+1 bias,
    partial MRoPE, sigmoid-gated attention output, and SwiGLU MLP.
    Both norms use the RMSNormPlus1 pattern (weight + 1 before rmsnorm).
    """

    _name_remap_ = QWEN3_5_ATTN_BLOCK_REMAP

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
        partial_rotary_factor: float = 0.25,
        mrope_section: tuple[int, int, int] | list[int] = (11, 11, 10),
    ):
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.eps = eps
        self.use_qkv_bias = use_qkv_bias
        self.partial_rotary_factor = partial_rotary_factor
        if mrope_section is None or len(mrope_section) < 3:
            mrope_section = (11, 11, 10)
        self.mrope_section = list(mrope_section)

        # Derived dimensions for shape resolution
        self.D = head_size
        self.Hq = num_query_heads
        self.Hkv = num_kv_heads
        self.C = d_model
        self.M = d_ff
        self.MUp = 2 * d_ff
        self.MaxSeq = max_seq
        self.AttnDim = num_query_heads * head_size
        self.QProjDim = 2 * self.AttnDim
        self.KVDim = num_kv_heads * head_size
        self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.RotaryDim = nn._resolve_rotary_dim(head_size, partial_rotary_factor)

        self.attn_norm = nn.RMSNormPlus1(d_model, eps=eps)
        self.self_attn = nn.Qwen3_5Attention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            use_qkv_bias=use_qkv_bias,
            eps=eps,
            partial_rotary_factor=partial_rotary_factor,
            mrope_section=mrope_section,
        )
        self.mlp_norm = nn.RMSNormPlus1(d_model, eps=eps)
        self.mlp = nn.SwiGLUMLP(d_model, d_ff)

    def forward(self, x, residual, position_ids):
        # Pre-attention normalization (fused residual + rmsnorm with weight+1)
        residual, h = self.attn_norm(residual, x)
        # Attention
        h = self.self_attn(h, position_ids)
        # Pre-MLP normalization (fused residual + rmsnorm with weight+1)
        residual, h = self.mlp_norm(residual, h)
        # MLP
        h = self.mlp(h)
        return h, residual


class Qwen3_5LinearBlock(nn.Block):
    """Qwen3.5 linear-attention (Gated DeltaNet) decoder block (token mixer + MLP).

    Uses Gated DeltaNet linear attention (NOT Mamba2), with conv1d,
    chunk_gated_delta_rule, and gated rmsnorm. Both norms use the
    RMSNormPlus1 pattern (weight + 1 before rmsnorm).
    """

    _name_remap_ = QWEN3_5_LINEAR_BLOCK_REMAP

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        chunk_size: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.chunk_size = chunk_size
        self.eps = eps

        if linear_num_value_heads % linear_num_key_heads != 0:
            raise ValueError(
                "Qwen3_5LinearBlock requires linear_num_value_heads to be divisible by linear_num_key_heads"
            )

        # Derived dimensions for shape resolution
        self.C = d_model
        self.M = d_ff
        self.MUp = 2 * d_ff
        self.Hk = linear_num_key_heads
        self.Hv = linear_num_value_heads
        self.Kd = linear_key_head_dim
        self.Vd = linear_value_head_dim
        self.KeyDim = self.Hk * self.Kd
        self.ValueDim = self.Hv * self.Vd
        self.ConvK = linear_conv_kernel_dim
        self.ConvDim = self.KeyDim * 2 + self.ValueDim
        self.HeadRepeat = self.Hv // self.Hk

        self.attn_norm = nn.RMSNormPlus1(d_model, eps=eps)
        self.mixer = nn.GatedDeltaNetMixer(
            d_model,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            chunk_size=chunk_size,
            eps=eps,
        )
        self.mlp_norm = nn.RMSNormPlus1(d_model, eps=eps)
        self.mlp = nn.SwiGLUMLP(d_model, d_ff)

    def forward(self, x, residual, position_ids):
        del position_ids  # Unused in linear-attention layers.

        # Pre-attention normalization (fused residual + rmsnorm with weight+1)
        residual, h = self.attn_norm(residual, x)
        # Linear attention (Gated DeltaNet)
        h = self.mixer(h)
        # Pre-MLP normalization (fused residual + rmsnorm with weight+1)
        residual, h = self.mlp_norm(residual, h)
        # MLP
        h = self.mlp(h)
        return h, residual
