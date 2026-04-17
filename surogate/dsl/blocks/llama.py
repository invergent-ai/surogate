"""LLaMA Transformer Block."""

from __future__ import annotations

from .. import nn
from ..nn import DENSE_BLOCK_NAME_REMAP


class LlamaBlock(nn.Block):
    """LLaMA transformer block with GQA attention and SwiGLU MLP."""

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
    ):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model, eps=eps)
        self.self_attn = nn.GQAAttention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
        )
        self.mlp_norm = nn.RMSNorm(d_model, eps=eps)
        self.mlp = nn.SwiGLUMLP(d_model, d_ff)

    def forward(self, x, residual, position_ids):
        residual, h = self.attn_norm(residual, x)
        h = self.self_attn(h, position_ids)
        residual, h = self.mlp_norm(residual, h)
        h = self.mlp(h)
        return h, residual
