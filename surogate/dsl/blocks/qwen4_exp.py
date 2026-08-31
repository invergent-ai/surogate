"""Qwen3.8-Flash-Next (qwen4_exp) decoder blocks.

A qwen3_5_moe core wrapped in hyper-connections:

* the residual is ``hc_count`` parallel streams (4 x d_model wide); every sublayer is
  wrapped in a HyperConnection mix/combine pair, which REPLACES all layer norms — there
  is no ``attn_norm``/``mlp_norm`` here and no final norm in the model;
* full-attention layers use :class:`Qwen3_5Attention` unchanged (gated output, partial
  MRoPE, QK-Norm with weight+1);
* linear layers use :class:`GatedDeltaNetMixer` with ``gate_activation="sigmoid"`` — the
  one numerical difference from Qwen3.5's GDN (SiLU there);
* the MoE is softmax top-k WITH renormalisation (``norm_topk_prob=True`` — note
  Qwen3.5-MoE uses False) plus a sigmoid-gated shared expert, identical layout.

Deliberately NOT expressed here (v1 scope, tracked in the model docstring): the PLE
n-gram memory of layer 1, the QSA indexer (training runs dense attention, which is the
exact semantics — the indexer is an inference-time selection), and the MTP head.
"""

from __future__ import annotations

from .. import nn
from ..block_schema import (
    BlockSchema,
    DistributionDecl,
    EPTopology,
    RoutingSchema,
    ServeObject,
    SlotDecl,
    StreamingHint,
)
from ..dim import B, Dim, T
from ..modules import (
    GatedDeltaNetMixer,
    HyperConnection,
    HyperConnectionCombine,
    MoEExpertsGated,
    MoESharedExpert,
    Qwen3_5Attention,
    _resolve_rotary_dim,
)
from .qwen3_5_moe import QWEN3_5_MOE_ATTN_BLOCK_REMAP, QWEN3_5_MOE_LINEAR_BLOCK_REMAP


# Attention / mixer / MoE names reuse the Qwen3.5-MoE canon (full_*, lin_*, bare moe
# names). Hyper-connection names need no remap: the auto-generated ``hc_attn_*`` /
# ``hc_ffn_*`` prefixes are already the canonical (and artifact) names.
QWEN4_EXP_ATTN_BLOCK_REMAP: dict[str, str] = {
    k: v for k, v in QWEN3_5_MOE_ATTN_BLOCK_REMAP.items() if k.startswith(("self_attn_", "moe_"))
}

QWEN4_EXP_LINEAR_BLOCK_REMAP: dict[str, str] = {
    k: v for k, v in QWEN3_5_MOE_LINEAR_BLOCK_REMAP.items() if k.startswith(("mixer_", "moe_"))
}


def _hyper_connection_objects(prefix: str) -> tuple[ServeObject, ...]:
    """The four tensors one hyper-connection mix needs, as a serve artifact stores
    them. The gamma is FP32 because it is a norm; the projections are BF16 because
    they are small and quantising them buys nothing measurable."""

    return (
        ServeObject(f"{prefix}/norm", "fp32", ("HcWidth",), (f"{prefix}_norm",)),
        ServeObject(f"{prefix}/down", "bf16", ("HcLowRank", "HcWidth"), (f"{prefix}_down",)),
        ServeObject(f"{prefix}/up", "bf16", ("HcWidth", "HcLowRank"), (f"{prefix}_up",)),
        ServeObject(f"{prefix}/inject", "bf16", ("HcCount", "HcWidth"), (f"{prefix}_inject",)),
    )


#: The MoE tail, identical on every layer. `router_shared_gate` fuses the routed
#: router with the shared expert's single gate row — 512 + 1 — which is why the
#: serve object has one more row than the declaration's router.
_MOE_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject(
        "mlp/router_shared_gate", "bf16", ("RouterRows", "C"),
        ("router_weight", "shared_expert_gate_proj_weight"),
    ),
    ServeObject(
        "mlp/routed_gate_up", "w8", ("RoutedGateUpRows", "C"), ("experts_gate_up",),
        residency="auto",
    ),
    ServeObject(
        "mlp/routed_down", "w8", ("RoutedDownRows", "M"), ("experts_down",),
        residency="auto",
    ),
    ServeObject(
        "mlp/shared_gate_up", "w8", ("SharedGateUpRows", "C"),
        ("shared_expert_gate", "shared_expert_up"),
    ),
    ServeObject("mlp/shared_down", "w8", ("C", "SharedM"), ("shared_expert_down",)),
)

#: Full attention. The fused projection is not a concatenation: the declaration's
#: query projection carries query and gate interleaved per head, and the engine
#: wants them as contiguous q | k | gate | v blocks, so the composition names a
#: transform the converter implements.
_ATTENTION_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject(
        "attention/query_key_gate_value", "w8", ("AttnFusedRows", "C"),
        ("full_q_proj_weight", "full_k_proj_weight", "full_v_proj_weight"),
        transform="split_interleaved_query_gate",
    ),
    # The checkpoint stores these gammas zero-centred and the kernel applies the
    # unit offset itself, so the converter unfolds the +1 rather than folding it.
    ServeObject("attention/query_norm", "bf16", ("HeadDim",), ("q_norm_weight",),
                transform="unfold_unit_offset"),
    ServeObject("attention/key_norm", "bf16", ("HeadDim",), ("k_norm_weight",),
                transform="unfold_unit_offset"),
    ServeObject("attention/output", "w8", ("C", "QuerySize"), ("full_out_weight",)),
    # QSA indexer: served, but not yet part of the training graph, so these objects
    # have no declared components. They are listed because the artifact carries them.
    ServeObject("attention/indexer/query", "bf16", ("IndexerQueryRows", "C")),
    ServeObject("attention/indexer/key", "bf16", ("IndexerDim", "C")),
    ServeObject("attention/indexer/query_norm", "bf16", ("IndexerDim",)),
    ServeObject("attention/indexer/key_norm", "bf16", ("IndexerDim",)),
)

#: Gated delta net. `query_key_value_z` fuses the qkv projection with the output
#: gate's z projection; `a_b_projection` fuses the two scalar-per-head projections.
_GDN_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("gdn/a_log", "fp32", ("Hv",), ("lin_A_log",), transform="log_negate"),
    ServeObject("gdn/dt_bias", "fp32", ("Hv",), ("lin_dt_bias",)),
    ServeObject("gdn/convolution", "bf16", ("ConvK", "ConvDim"), ("lin_conv_weight",),
                transform="transpose_taps"),
    ServeObject("gdn/a_b_projection", "bf16", ("TwoHv", "C"),
                ("lin_in_proj_a_weight", "lin_in_proj_b_weight")),
    ServeObject("gdn/query_key_value_z", "w8", ("GdnFusedRows", "C"),
                ("lin_in_proj_qkv_weight", "lin_in_proj_z_weight")),
    ServeObject("gdn/norm", "bf16", ("Vd",), ("lin_norm_weight",)),
    ServeObject("gdn/output", "w8", ("C", "ValueDim"), ("lin_out_weight",)),
)

#: The n-gram PLE memory, on its one layer. Not in the training graph yet; the
#: table itself travels as a raw resource rather than a tensor.
_PLE_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("ple/key", "bf16", ("HcWidth", "PleEmbed")),
    ServeObject("ple/value", "bf16", ("C", "PleEmbed")),
    ServeObject("ple/norm_key", "fp32", ("HcWidth",)),
    ServeObject("ple/norm_query", "fp32", ("HcWidth",)),
    ServeObject("ple/norm_conv", "fp32", ("HcWidth",)),
    ServeObject("ple/convolution", "bf16", ("PleConvKernel", "HcWidth")),
)


def _qwen4_exp_schema(block_family: str, *, mixer: str) -> BlockSchema:
    slots: tuple[SlotDecl, ...] = (
        SlotDecl("router_weight", kind="param", shape=("E", "C"), distribution=DistributionDecl.router_replicated()),
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
        SlotDecl("shared_expert_gate_proj_weight", kind="param", shape=(1, "C")),
        SlotDecl("permuted_input", shape=("dispatched_tokens", "C"), distribution=DistributionDecl.expert_parallel()),
    )
    return BlockSchema(
        slots=slots,
        routing=RoutingSchema(
            kind="topk_softmax",
            topk="num_experts_per_tok",
            norm_topk_prob=True,
            shared_experts="shared_expert_intermediate",
        ),
        ep_topology=EPTopology(ep_size_param="ep_size"),
        serve_objects=(
            *_hyper_connection_objects("hc_attn"),
            *(_ATTENTION_SERVE_OBJECTS if mixer == "attention" else _GDN_SERVE_OBJECTS),
            *_hyper_connection_objects("hc_ffn"),
            *_MOE_SERVE_OBJECTS,
        ),
        attrs={"block_family": block_family},
    )


class _Qwen4ExpMoEMixin:
    """The MoE tail shared by both block types: routed softmax top-k (renormalised)
    plus the sigmoid-gated shared expert, fed by the ffn-side hyper-connection mix."""

    def _moe_tail(self, h_flat):
        moe_out = self.moe(h_flat)
        shared_out = self.shared_expert(h_flat)
        self._register_param("shared_expert_gate_proj_weight", (1, "C"))
        shared_gate = self._matmul(h_flat, "shared_expert_gate_proj_weight", name="shared_expert_gate_proj")
        shared_gate = self._sigmoid(shared_gate, name="shared_expert_gate_sigmoid")
        shared_out = self._mul(shared_out, shared_gate, name="shared_expert_gated")
        moe_out = self._add(moe_out, shared_out, name="moe_combined")
        self._register_activation(
            "mlp_down",
            ("B", "T", "C"),
            aliases=["mlp_down_flat"],
            share_policy="per_layer",
            description="MoE output (block output)",
        )
        return self._view(moe_out, [B, T, self.C], name="mlp_down")


class Qwen4ExpAttentionBlock(nn.Block, _Qwen4ExpMoEMixin):
    """Flash-Next full-attention decoder block (layers 3, 7, ..., 47)."""

    _name_remap_ = QWEN4_EXP_ATTN_BLOCK_REMAP
    schema = _qwen4_exp_schema("qwen4_exp_moe_attention", mixer="attention")

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
        shared_expert_intermediate: int,
        hc_count: int = 4,
        hc_lowrank: int = 320,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        partial_rotary_factor: float = 0.25,
        mrope_section: tuple[int, int, int] | list[int] = (11, 11, 10),
        ep_size: int = 1,
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
        self.shared_expert_intermediate = shared_expert_intermediate
        self.C = Dim("C")

        # Derived dimensions for shape resolution
        self.D = head_size
        self.Hq = num_query_heads
        self.Hkv = num_kv_heads
        self.M = d_ff
        self.MaxSeq = max_seq
        self.AttnDim = num_query_heads * head_size
        self.QProjDim = 2 * self.AttnDim
        self.KVDim = num_kv_heads * head_size
        self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.RotaryDim = _resolve_rotary_dim(head_size, partial_rotary_factor)
        self.S = hc_count
        self.R = hc_lowrank
        self.SC = hc_count * d_model

        self.hc_attn = HyperConnection(d_model, hc_count, hc_lowrank, eps=eps)
        self.hc_attn_combine = HyperConnectionCombine(d_model, hc_count)
        self.self_attn = Qwen3_5Attention(
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
        self.hc_ffn = HyperConnection(d_model, hc_count, hc_lowrank, eps=eps)
        self.hc_ffn_combine = HyperConnectionCombine(d_model, hc_count)
        self.moe = MoEExpertsGated(
            d_model,
            d_ff,
            num_experts,
            num_experts_per_tok,
            norm_topk_prob=True,
            ep_size=ep_size,
        )
        self.shared_expert = MoESharedExpert(d_model, shared_expert_intermediate)

    def forward(self, x, residual, position_ids):
        del x  # Blocks read only the wide residual streams.

        h, inject_a = self.hc_attn(residual)
        a = self.self_attn(h, position_ids)
        residual = self.hc_attn_combine(residual, a, inject_a)

        h2, inject_f = self.hc_ffn(residual)
        h_flat = self._view(h2, [B * T, self.C], name="moe_in_flat")
        out = self._moe_tail(h_flat)
        residual = self.hc_ffn_combine(residual, out, inject_f)
        return out, residual


class Qwen4ExpLinearBlock(nn.Block, _Qwen4ExpMoEMixin):
    """Flash-Next linear-attention (Gated DeltaNet) decoder block — sigmoid output gate."""

    _name_remap_ = QWEN4_EXP_LINEAR_BLOCK_REMAP
    schema = _qwen4_exp_schema("qwen4_exp_moe_linear", mixer="gdn")

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_tok: int,
        shared_expert_intermediate: int,
        hc_count: int = 4,
        hc_lowrank: int = 320,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 48,
        chunk_size: int = 64,
        eps: float = 1e-6,
        gate_activation: str = "sigmoid",
        ep_size: int = 1,
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
        self.shared_expert_intermediate = shared_expert_intermediate
        self.C = Dim("C")

        if linear_num_value_heads % linear_num_key_heads != 0:
            raise ValueError(
                "Qwen4ExpLinearBlock requires linear_num_value_heads to be divisible by linear_num_key_heads"
            )

        # Derived dimensions for shape resolution
        self.M = d_ff
        self.Hk = linear_num_key_heads
        self.Hv = linear_num_value_heads
        self.Kd = linear_key_head_dim
        self.Vd = linear_value_head_dim
        self.KeyDim = self.Hk * self.Kd
        self.ValueDim = self.Hv * self.Vd
        self.ConvK = linear_conv_kernel_dim
        self.ConvDim = self.KeyDim * 2 + self.ValueDim
        self.HeadRepeat = self.Hv // self.Hk
        self.S = hc_count
        self.R = hc_lowrank
        self.SC = hc_count * d_model

        self.hc_attn = HyperConnection(d_model, hc_count, hc_lowrank, eps=eps)
        self.hc_attn_combine = HyperConnectionCombine(d_model, hc_count)
        self.mixer = GatedDeltaNetMixer(
            d_model,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            chunk_size=chunk_size,
            eps=eps,
            gate_activation=gate_activation,
        )
        self.hc_ffn = HyperConnection(d_model, hc_count, hc_lowrank, eps=eps)
        self.hc_ffn_combine = HyperConnectionCombine(d_model, hc_count)
        self.moe = MoEExpertsGated(
            d_model,
            d_ff,
            num_experts,
            num_experts_per_tok,
            norm_topk_prob=True,
            ep_size=ep_size,
        )
        self.shared_expert = MoESharedExpert(d_model, shared_expert_intermediate)

    def forward(self, x, residual, position_ids):
        del x  # Blocks read only the wide residual streams.
        del position_ids  # Unused in linear-attention layers.

        h, inject_a = self.hc_attn(residual)
        m = self.mixer(h)
        residual = self.hc_attn_combine(residual, m, inject_a)

        h2, inject_f = self.hc_ffn(residual)
        h_flat = self._view(h2, [B * T, self.C], name="moe_in_flat")
        out = self._moe_tail(h_flat)
        residual = self.hc_ffn_combine(residual, out, inject_f)
        return out, residual
