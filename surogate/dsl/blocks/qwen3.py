"""Qwen3 Transformer Block."""

from __future__ import annotations

from .. import nn
from ..modules import GenericGQAttention, GenericMLP, RMSNorm
from ..attention import AttentionConfig
from ..mlp import MLPConfig
from .common import DENSE_BLOCK_NAME_REMAP


class Qwen3Block(nn.Block):
    """Qwen3 transformer block: QK-norm GQA + SwiGLU MLP."""

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
        self.attn_norm = RMSNorm(d_model, eps=eps)
        self.self_attn = GenericGQAttention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            config=AttentionConfig(
                qk_norm=use_qk_norm,
                qkv_bias=use_qkv_bias,
                eps=eps,
            ),
        )
        self.mlp_norm = RMSNorm(d_model, eps=eps)
        self.mlp = GenericMLP(d_model, d_ff, config=MLPConfig())

    def forward(self, x, residual, position_ids):
        residual, h = self.attn_norm(residual, x)
        h = self.self_attn(h, position_ids)
        residual, h = self.mlp_norm(residual, h)
        h = self.mlp(h)
        return h, residual
