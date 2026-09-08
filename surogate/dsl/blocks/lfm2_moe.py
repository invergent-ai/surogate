"""LFM2-MoE hybrid decoder blocks.

LFM2-MoE keeps LFM2's operator axis -- full attention or short convolution per
layer -- and replaces the dense feed-forward with a sparse mixture of experts on
every layer past ``num_dense_layers``. The two axes are independent, so a model
carries up to four block shapes; the dense pair is LFM2's own
(``Lfm2AttentionBlock`` / ``Lfm2ConvBlock``) and the two here are their MoE
counterparts.

The router is sigmoid-gated with an aux-loss-free selection bias: the bias moves
which experts win top-k, never the weight a winner is applied with. That is the
same routing Laguna uses, so the expert module is shared -- the alternative,
``MoEExpertsGated``, emits a softmax router and would have trained this model
with routing arithmetic its checkpoint was never fitted to.
"""

from __future__ import annotations

from .. import nn
from ..attention import AttentionConfig
from ..block_schema import BlockSchema, DistributionDecl, EPTopology, RoutingSchema, ServeObject, SlotDecl, StreamingHint
from ..dim import B, Dim, T
from ..modules import GenericGQAttention, LagunaMoEExperts, Lfm2ShortConv, RMSNorm
from .common import MOE_BLOCK_NAME_REMAP
from .lfm2 import (
    LFM2_ATTENTION_BLOCK_REMAP, LFM2_CONV_BLOCK_REMAP,
    _LFM2_ATTENTION_OBJECTS, _LFM2_CONV_OBJECTS,
)

_LFM2_MOE_OBJECTS = (
    ServeObject("input_norm", "bf16", ("C",), ("operator_norm_weight",)),
    ServeObject("post_attention_norm", "bf16", ("C",), ("ffn_norm_weight",)),
    ServeObject("moe/router", "bf16", ("E", "C"), ("router_weight",)),
    ServeObject("moe/router_bias", "fp32", ("E",), ("e_score_correction_bias",)),
    ServeObject("moe/routed_gate_up", "quantised", ("RoutedGateUpRows", "C"),
                ("experts_gate_up",), transform="flatten_experts", residency="auto"),
    ServeObject("moe/routed_down", "quantised", ("RoutedDownRows", "MoeM"),
                ("experts_down",), transform="flatten_experts", residency="auto"),
)

# The MoE half of the name remap, lifted off the dense-MoE block so the two
# families cannot drift apart.
_MOE_REMAP_TAIL: dict[str, str] = {
    **{k: v for k, v in MOE_BLOCK_NAME_REMAP.items() if k.startswith("moe_")},
    "moe_e_score_correction_bias": "e_score_correction_bias",
}

# Operator halves of the LFM2 remaps, with the dense MLP entries dropped.
_LFM2_ATTENTION_OPERATOR_REMAP: dict[str, str] = {
    k: v for k, v in LFM2_ATTENTION_BLOCK_REMAP.items() if not k.startswith("mlp_")
}
_LFM2_CONV_OPERATOR_REMAP: dict[str, str] = {
    k: v for k, v in LFM2_CONV_BLOCK_REMAP.items() if not k.startswith("mlp_")
}

LFM2_MOE_ATTENTION_BLOCK_REMAP: dict[str, str] = {**_LFM2_ATTENTION_OPERATOR_REMAP, **_MOE_REMAP_TAIL}
LFM2_MOE_CONV_BLOCK_REMAP: dict[str, str] = {**_LFM2_CONV_OPERATOR_REMAP, **_MOE_REMAP_TAIL}


def _moe_slots() -> tuple[SlotDecl, ...]:
    return (
        SlotDecl("router_weight", kind="param", shape=("E", "C"), distribution=DistributionDecl.router_replicated()),
        SlotDecl(
            "e_score_correction_bias",
            kind="param",
            shape=("E",),
            distribution=DistributionDecl.router_replicated(),
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
        SlotDecl("permuted_input", shape=("dispatched_tokens", "C"), distribution=DistributionDecl.expert_parallel()),
        SlotDecl("mlp_down", shape=("B", "T", "C")),
    )


def _moe_routing() -> RoutingSchema:
    # Sigmoid scores, top-k chosen after adding the load-balancing bias, then
    # renormalised over the winners and scaled. LFM2-MoE has no shared expert.
    return RoutingSchema(
        kind="topk_sigmoid",
        topk="num_experts_per_tok",
        norm_topk_prob="norm_topk_prob",
        scoring_bias=True,
        shared_experts=0,
    )


class Lfm2MoeAttentionBlock(nn.Block):
    """LFM2-MoE full-attention layer with a sparse feed-forward."""

    _name_remap_ = LFM2_MOE_ATTENTION_BLOCK_REMAP
    schema = BlockSchema(
        slots=(
            SlotDecl("qkv_weight", kind="param", shape=("QKV", "C")),
            SlotDecl("out_weight", kind="param", shape=("C", "AttnDim")),
            SlotDecl("q_norm_weight", kind="param", shape=("D",)),
            SlotDecl("k_norm_weight", kind="param", shape=("D",)),
            SlotDecl("qkv_rope", shape=("B", "T", "QKV"), save_for_backward=True),
            SlotDecl("operator_out", shape=("B", "T", "C")),
            SlotDecl("res_ffn", shape=("B", "T", "C")),
            *_moe_slots(),
        ),
        routing=_moe_routing(),
        ep_topology=EPTopology(ep_size_param="ep_size"),
        attrs={"block_family": "lfm2_moe_attention"},
        serve_objects=(*_LFM2_ATTENTION_OBJECTS, *_LFM2_MOE_OBJECTS),
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
        eps: float = 1e-5,
        routed_scaling_factor: float = 1.0,
        ep_size: int = 1,
    ):
        super().__init__()
        self.use_qk_norm = True
        self.use_qkv_bias = False
        self.use_out_bias = False
        self.d_model = d_model
        self.M = d_ff
        self.MUp = 2 * d_ff
        self.C = Dim("C")
        self.operator_norm = RMSNorm(d_model, eps=eps)
        self.self_attn = GenericGQAttention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            config=AttentionConfig(qk_norm=True, eps=eps),
        )
        self.ffn_norm = RMSNorm(d_model, eps=eps)
        self.moe = LagunaMoEExperts(
            d_model,
            d_ff,
            num_experts,
            num_experts_per_tok,
            routed_scaling_factor=routed_scaling_factor,
            ep_size=ep_size,
        )

    def forward(self, x, residual, position_ids):
        residual, h = self.operator_norm(residual, x)
        h = self.self_attn(h, position_ids)
        residual, h = self.ffn_norm(residual, h)
        h_flat = self._view(h, [B * T, self.C], name="ffn_ln_flat")
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


class Lfm2MoeConvBlock(nn.Block):
    """LFM2-MoE short-convolution layer with a sparse feed-forward."""

    _name_remap_ = LFM2_MOE_CONV_BLOCK_REMAP
    schema = BlockSchema(
        slots=(
            SlotDecl("conv_in_proj_weight", kind="param", shape=("3C", "C"), residency="auto"),
            SlotDecl("conv_weight", kind="param", shape=("C", 1, "K")),
            SlotDecl("conv_out_proj_weight", kind="param", shape=("C", "C"), residency="auto"),
            SlotDecl("conv_in_proj", shape=("B", "T", "3C"), save_for_backward=True),
            SlotDecl("conv_bx", shape=("B", "C", "T"), save_for_backward=True),
            SlotDecl("operator_out_cf", shape=("B", "C", "T"), save_for_backward=True),
            SlotDecl("operator_out", shape=("B", "T", "C")),
            SlotDecl("res_ffn", shape=("B", "T", "C")),
            *_moe_slots(),
        ),
        routing=_moe_routing(),
        ep_topology=EPTopology(ep_size_param="ep_size"),
        attrs={"block_family": "lfm2_moe_conv"},
        serve_objects=(*_LFM2_CONV_OBJECTS, *_LFM2_MOE_OBJECTS),
    )

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_tok: int,
        conv_kernel: int = 3,
        eps: float = 1e-5,
        conv_bias: bool = False,
        routed_scaling_factor: float = 1.0,
        ep_size: int = 1,
    ):
        super().__init__()
        self.use_bias = conv_bias
        self.d_model = d_model
        self.M = d_ff
        self.MUp = 2 * d_ff
        self.C = Dim("C")
        self.operator_norm = RMSNorm(d_model, eps=eps)
        self.short_conv = Lfm2ShortConv(d_model, conv_kernel=conv_kernel, use_bias=conv_bias)
        self.ffn_norm = RMSNorm(d_model, eps=eps)
        self.moe = LagunaMoEExperts(
            d_model,
            d_ff,
            num_experts,
            num_experts_per_tok,
            routed_scaling_factor=routed_scaling_factor,
            ep_size=ep_size,
        )

    def forward(self, x, residual, position_ids):
        residual, h = self.operator_norm(residual, x)
        h = self.short_conv(h)
        residual, h = self.ffn_norm(residual, h)
        h_flat = self._view(h, [B * T, self.C], name="ffn_ln_flat")
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
