"""Qwen3 Transformer Block."""

from __future__ import annotations

from .. import nn
from ..nn import DENSE_BLOCK_NAME_REMAP


class Qwen3Block(nn.Block):
    """Qwen3 transformer block with QK-Norm."""

    _name_remap_ = DENSE_BLOCK_NAME_REMAP

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
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model, eps=eps)
        self.self_attn = nn.Qwen3Attention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            use_qkv_bias=use_qkv_bias,
            use_qk_norm=use_qk_norm,
            eps=eps,
        )
        self.mlp_norm = nn.RMSNorm(d_model, eps=eps)
        self.mlp = nn.SwiGLUMLP(d_model, d_ff)

    def forward(self, x, residual, position_ids):
        # Pre-attention normalization (fused residual + rmsnorm)
        residual, h = self.attn_norm(residual, x)
        # Attention
        h = self.self_attn(h, position_ids)
        # Pre-MLP normalization (fused residual + rmsnorm)
        residual, h = self.mlp_norm(residual, h)
        # MLP
        h = self.mlp(h)
        return h, residual
