"""GPT-OSS Transformer Block."""

from __future__ import annotations

from .. import nn
from ..dim import B, Dim, T
from ..nn import GPT_OSS_BLOCK_NAME_REMAP


class GptOssBlock(nn.Block):
    """GPT-OSS transformer block with sinks and MoE experts."""

    _name_remap_ = GPT_OSS_BLOCK_NAME_REMAP

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        d_ff: int,
        max_seq: int,
        num_experts: int,
        num_experts_per_tok: int,
        eps: float = 1e-5,
        use_qkv_bias: bool = True,
        ep_size: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.C = Dim("C")

        self.attn_norm = nn.RMSNorm(d_model, eps=eps)
        self.self_attn = nn.GptOssAttention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            use_qkv_bias=use_qkv_bias,
        )
        self.mlp_norm = nn.RMSNorm(d_model, eps=eps)
        self.moe = nn.GptOssMoEExperts(
            d_model,
            d_ff,
            num_experts,
            num_experts_per_tok,
            ep_size=ep_size,
        )

    def forward(self, x, residual, position_ids):
        # Pre-attention normalization
        residual, h = self.attn_norm(residual, x)
        # Attention (with sinks)
        h = self.self_attn(h, position_ids)
        # Pre-MoE normalization
        residual, h = self.mlp_norm(residual, h)
        # Flatten for MoE
        h_flat = self._view(h, [B * T, self.C], name="ln2_flat")
        # MoE experts
        moe_out = self.moe(h_flat)
        # Register output slot and reshape back
        self._register_activation(
            "mlp_down",
            ("B", "T", "C"),
            aliases=["mlp_down_flat"],
            share_policy="per_layer",
            description="MoE output (block output)",
        )
        out = self._view(moe_out, [B, T, self.C], name="mlp_down")
        return out, residual
