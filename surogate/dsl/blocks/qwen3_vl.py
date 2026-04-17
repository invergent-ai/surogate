"""Qwen3-VL Transformer Block (text)."""

from __future__ import annotations

from .. import nn
from ..modules import GenericMLP, Qwen3VLAttention, RMSNorm
from .common import VL_DENSE_BLOCK_NAME_REMAP


class Qwen3VLBlock(nn.Block):
    """Qwen3-VL text transformer block with QK-Norm + MRoPE."""

    _name_remap_ = VL_DENSE_BLOCK_NAME_REMAP

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
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=eps)
        self.self_attn = Qwen3VLAttention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            use_qkv_bias=use_qkv_bias,
            eps=eps,
            mrope_section=mrope_section,
        )
        self.mlp_norm = RMSNorm(d_model, eps=eps)
        self.mlp = GenericMLP(d_model, d_ff)

    def forward(self, x, residual, position_ids):
        # Pre-attention normalization (fused residual + rmsnorm)
        residual, h = self.attn_norm(residual, x)
        # Attention (QK-Norm + MRoPE)
        h = self.self_attn(h, position_ids)
        # Pre-MLP normalization (fused residual + rmsnorm)
        residual, h = self.mlp_norm(residual, h)
        # MLP (SwiGLU)
        h = self.mlp(h)
        return h, residual
