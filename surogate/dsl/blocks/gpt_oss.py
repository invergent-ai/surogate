"""GPT-OSS Transformer Block."""

from __future__ import annotations

from .. import nn
from ..modules import GenericGQAttention, GptOssMoEExperts, RMSNorm
from ..attention import AttentionConfig
from ..dim import B, Dim, T
from .common import DENSE_BLOCK_NAME_REMAP


GPT_OSS_BLOCK_NAME_REMAP: dict[str, str] = {
    # Inherit attn_norm, self_attn, mlp_norm mappings from dense
    **{k: v for k, v in DENSE_BLOCK_NAME_REMAP.items() if k.startswith(("attn_norm_", "self_attn_", "mlp_norm_"))},
    # GPT-OSS attention extra: sinks
    "self_attn_sinks": "sinks",
    # --- moe (GptOssMoEExperts) -> strip moe_ prefix ---
    # Params
    "moe_router_weight": "router_weight",
    "moe_router_bias": "router_bias",
    "moe_experts_gate_up": "experts_gate_up",
    "moe_experts_gate_up_bias": "experts_gate_up_bias",
    "moe_experts_down": "experts_down",
    "moe_experts_down_bias": "experts_down_bias",
    # Activations
    "moe_router_logits": "router_logits",
    "moe_routing_weights": "routing_weights",
    "moe_routing_indices": "routing_indices",
    "moe_permuted_input": "permuted_input",
    "moe_scatter_indices": "scatter_indices",
    "moe_ep_recv_input": "ep_recv_input",
    "moe_ep_recv_scatter": "ep_recv_scatter",
    "moe_expert_gate_up": "expert_gate_up",
    "moe_expert_gate_up_bias": "expert_gate_up_bias",
    "moe_expert_act": "expert_act",
    "moe_expert_down": "expert_down",
    "moe_expert_down_bias": "expert_down_bias",
    "moe_ep_combined": "ep_combined",
    # moe_out / moe_out_flat already match canonical names
}


class GptOssBlock(nn.Block):
    """GPT-OSS transformer block: sink-token attention + MoE experts."""

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

        self.attn_norm = RMSNorm(d_model, eps=eps)
        self.self_attn = GenericGQAttention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            config=AttentionConfig(
                qkv_bias=use_qkv_bias,
                has_sinks=True,
                eps=eps,
            ),
        )
        self.mlp_norm = RMSNorm(d_model, eps=eps)
        self.moe = GptOssMoEExperts(
            d_model,
            d_ff,
            num_experts,
            num_experts_per_tok,
            ep_size=ep_size,
        )

    def forward(self, x, residual, position_ids):
        residual, h = self.attn_norm(residual, x)
        h = self.self_attn(h, position_ids)
        residual, h = self.mlp_norm(residual, h)
        h_flat = self._view(h, [B * T, self.C], name="ln2_flat")
        moe_out = self.moe(h_flat)
        self._register_activation(
            "mlp_down",
            ("B", "T", "C"),
            aliases=["mlp_down_flat"],
            share_policy="per_layer",
            description="MoE output (block output)",
        )
        out = self._view(moe_out, [B, T, self.C], name="mlp_down")
        return out, residual
