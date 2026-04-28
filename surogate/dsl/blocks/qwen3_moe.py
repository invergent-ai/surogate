"""Qwen3 MoE Transformer Block."""

from __future__ import annotations

from .. import nn
from ..attention import AttentionConfig
from ..block_schema import BlockSchema, DistributionDecl, EPTopology, RoutingSchema, SlotDecl, StreamingHint
from ..dim import B, Dim, T
from ..modules import GenericGQAttention, MoEExpertsGated, MoESharedExpert, RMSNorm
from .common import MOE_BLOCK_NAME_REMAP


class Qwen3MoEBlock(nn.Block):
    """Qwen3 MoE transformer block with QK-Norm and Mixture of Experts."""

    _name_remap_ = MOE_BLOCK_NAME_REMAP
    schema = BlockSchema(
        slots=(
            SlotDecl(
                "router_weight", kind="param", shape=("E", "C"), distribution=DistributionDecl.router_replicated()
            ),
            SlotDecl(
                "experts_gate_up",
                kind="param",
                shape=("E", "2M", "C"),
                residency="auto",
                distribution=DistributionDecl.expert_parallel(global_experts="num_experts"),
                grouped=True,
                streaming_hint=StreamingHint(prefetch_distance=1),
            ),
            SlotDecl(
                "experts_down",
                kind="param",
                shape=("E", "C", "M"),
                residency="auto",
                distribution=DistributionDecl.expert_parallel(global_experts="num_experts"),
                grouped=True,
                streaming_hint=StreamingHint(prefetch_distance=1),
            ),
            SlotDecl(
                "permuted_input", shape=("dispatched_tokens", "C"), distribution=DistributionDecl.expert_parallel()
            ),
        ),
        routing=RoutingSchema(
            kind="topk_softmax",
            topk="num_experts_per_tok",
            norm_topk_prob="norm_topk_prob",
            shared_experts="use_shared_expert",
        ),
        ep_topology=EPTopology(ep_size_param="ep_size"),
        attrs={"block_family": "qwen3_moe"},
    )

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
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        use_qk_norm: bool = True,
        norm_topk_prob: bool = True,
        use_shared_expert: bool = False,
        shared_expert_intermediate: int = 0,
        ep_size: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_shared_expert = use_shared_expert
        self.shared_expert_intermediate = shared_expert_intermediate if shared_expert_intermediate > 0 else d_ff
        self.C = Dim("C")

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
        self.moe = MoEExpertsGated(
            d_model,
            d_ff,
            num_experts,
            num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
            ep_size=ep_size,
        )
        if use_shared_expert:
            self.shared_expert = MoESharedExpert(
                d_model,
                self.shared_expert_intermediate,
            )

    def forward(self, x, residual, position_ids):
        # Pre-attention normalization
        residual, h = self.attn_norm(residual, x)
        # Attention
        h = self.self_attn(h, position_ids)
        # Pre-MoE normalization
        residual, h = self.mlp_norm(residual, h)
        # Flatten for MoE
        h_flat = self._view(h, [B * T, self.C], name="ln2_flat")
        # MoE experts
        moe_out = self.moe(h_flat)
        # Optional shared expert
        if self.use_shared_expert:
            shared_out = self.shared_expert(h_flat)
            moe_out = self._add(moe_out, shared_out, name="moe_combined")
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
