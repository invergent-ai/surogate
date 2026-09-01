"""DeepSeek-V4 decoder blocks.

DeepSeek-V4 is **not** MLA (that was V3). Every layer is:

* **shared-KV MQA** — one KV head, and K and V are literally the same tensor, so the
  values carry RoPE and the block strips it off the attention output again;
* a **128-token sliding window** on every layer, plus, on all but ``sliding_attention``
  layers, a long-range *compressor* branch whose compressed KV entries are concatenated
  onto the KV axis (CSA at m=4 with a Lightning Indexer picking ``index_topk`` blocks per
  query, HCA at m'=128 with no indexer);
* a **grouped low-rank output projection** (``o_a_proj`` block-diagonal over ``o_groups``
  head groups, then ``o_b_proj`` mixes to hidden);
* a **sqrt-softplus MoE on every layer** (no ``first_k_dense_replace``), with a shared
  expert, batched experts and — on the first ``mlp_layer_types.count("hash_moe")`` layers
  — a frozen ``tid2eid`` hash router instead of the learned top-k;
* **manifold-constrained hyper-connections (mHC)**: the residual is ``hc_mult`` parallel
  streams, mixed in and out at both sublayer sites by a Sinkhorn-projected doubly-
  stochastic combine matrix.

Six block shapes are declared, the cross product of the attention schedule
(``sliding`` / ``csa`` / ``hca``) and the MoE schedule (``moe`` / ``hash``). Only the
combinations a checkpoint actually uses are instantiated.

WHAT IS DEFERRED (each is a missing DSL primitive, not a modelling choice; every one is
also marked with a ``DEFERRED`` comment at the point it would have been emitted):

1. **mHC.** ``DeepseekV4HyperConnection`` needs a Sinkhorn-Knopp projection —
   ``hc_sinkhorn_iters`` rounds of ``x / x.sum(dim)`` over a per-token ``[hc, hc]``
   matrix — and then a per-token ``comb.T @ residual`` stream mix. The graph builder has
   no reduction-over-a-dim and no division op, so neither the projection nor the combine
   can be written. ``qwen4_exp``'s :class:`HyperConnection` is a *different*
   parameterisation (low-rank down/up + a scalar inject) and reusing it would compute
   something else entirely. The blocks therefore run the ordinary single-stream pre-norm
   residual. ``attn_hc`` / ``ffn_hc`` / ``hc_head`` tensors are named in the HF mapping and
   declared as serve objects, but are not loaded as parameters.
2. **CSA / HCA compressors and the Lightning Indexer.** Running-window buffers, per-window
   softmax pooling with a position bias, the Ca/Cb overlap scheme and a top-k gather over
   compressed blocks. No primitive for any of it; the compressor tensors are declared as
   serve objects only. Attention here is pure sliding-window.
3. **RoPE convention and the inverse rotation** — see :class:`DeepseekV4Attention`.
4. **Hash routing.** ``tid2eid[input_ids]`` selects experts by token id. The MoE lowering
   derives its indices from ``moe_topk``; there is no path to feed externally gathered
   indices, and ``input_ids`` does not even reach the block. The ``tid2eid`` table is
   declared (frozen) so the layer is identifiable and the tensor has a home; selection
   falls back to the learned top-k.
5. **``sqrt`` of the router score and the ``swiglu_limit`` clamps** — see
   :class:`~surogate.dsl.modules.deepseek_v4.DeepseekV4MoEExperts`.
6. **MTP.** ``num_nextn_predict_layers`` draft layers; captured in the config, not
   declared (same call ``qwen4_exp`` made).
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
from ..dim import B, T
from ..modules import DeepseekV4Attention, DeepseekV4MoEExperts, MoESharedExpert, RMSNorm


#: Attention schedule entries, HF ``layer_types`` value -> short block-type stem.
DEEPSEEK_V4_ATTENTION_KINDS: dict[str, str] = {
    "sliding_attention": "sliding",
    "compressed_sparse_attention": "csa",
    "heavily_compressed_attention": "hca",
}

#: MoE schedule entries, HF ``mlp_layer_types`` value -> short block-type stem.
DEEPSEEK_V4_MLP_KINDS: dict[str, str] = {
    "moe": "moe",
    "hash_moe": "hash",
}


_ATTN_REMAP: dict[str, str] = {
    # --- attn_norm (RMSNorm) -> ln1 / res_ffn ---
    "attn_norm_weight": "ln1_weight",
    "attn_norm_res": "res_ffn",
    "attn_norm_y": "ln1",
    "attn_norm_rstd": "ln1_rstd",
    # --- self_attn (DeepseekV4Attention) -> strip the prefix ---
    "self_attn_q_a_proj_weight": "q_a_proj_weight",
    "self_attn_q_a_norm_weight": "q_a_norm_weight",
    "self_attn_q_b_proj_weight": "q_b_proj_weight",
    "self_attn_kv_proj_weight": "kv_proj_weight",
    "self_attn_kv_norm_weight": "kv_norm_weight",
    "self_attn_o_a_proj_weight": "o_a_proj_weight",
    "self_attn_o_b_proj_weight": "o_b_proj_weight",
    "self_attn_sinks": "sinks",
    "self_attn_rope_freqs": "rope_freqs",
    "self_attn_x_flat": "x_flat",
    "self_attn_q_a_proj": "q_a_proj",
    "self_attn_q_residual": "q_residual",
    "self_attn_q_a_rstd": "q_a_rstd",
    "self_attn_q_b_proj": "q_b_proj",
    "self_attn_q_rows": "q_rows",
    "self_attn_q_rstd": "q_rstd",
    "self_attn_q_normed": "q_normed",
    "self_attn_q_4d": "q_4d",
    "self_attn_kv_proj": "kv_proj",
    "self_attn_kv_rstd": "kv_rstd",
    "self_attn_kv_normed": "kv_normed",
    "self_attn_kv_4d": "kv_4d",
    "self_attn_qkv_rope": "qkv_rope",
    "self_attn_att": "att",
    "self_attn_att_flat": "att_flat",
    "self_attn_lse": "lse",
    "self_attn_att_grouped": "att_grouped",
    "self_attn_o_a_proj": "o_a_proj",
    "self_attn_att_out": "att_out",
    "self_attn_att_out_flat": "att_out_flat",
    # --- mlp_norm (RMSNorm) -> ln2 / res_att ---
    "mlp_norm_weight": "ln2_weight",
    "mlp_norm_res": "res_att",
    "mlp_norm_y": "ln2",
    "mlp_norm_rstd": "ln2_rstd",
}

_MOE_REMAP: dict[str, str] = {
    "moe_router_weight": "router_weight",
    "moe_e_score_correction_bias": "e_score_correction_bias",
    "moe_tid2eid": "tid2eid",
    "moe_experts_gate_up": "experts_gate_up",
    "moe_experts_down": "experts_down",
    "moe_router_logits": "router_logits",
    "moe_router_probs": "router_probs",
    "moe_routing_weights": "routing_weights",
    "moe_routing_indices": "routing_indices",
    "moe_permuted_input": "permuted_input",
    "moe_scatter_indices": "scatter_indices",
    "moe_ep_recv_input": "ep_recv_input",
    "moe_ep_recv_scatter": "ep_recv_scatter",
    "moe_expert_gate_up": "expert_gate_up",
    "moe_expert_act": "expert_act",
    "moe_expert_down": "expert_down",
    "moe_ep_combined": "ep_combined",
}

DEEPSEEK_V4_BLOCK_NAME_REMAP: dict[str, str] = {**_ATTN_REMAP, **_MOE_REMAP}


# ---------------------------------------------------------------------------
# Serve objects
# ---------------------------------------------------------------------------

#: The attention half, identical on every layer type. `sinks` and the two low-rank norms
#: stay full precision; the projections are quantisable.
_ATTENTION_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("attention/query_down", "w8", ("QLoraRank", "C"), ("q_a_proj_weight",)),
    ServeObject("attention/query_down_norm", "fp32", ("QLoraRank",), ("q_a_norm_weight",)),
    ServeObject("attention/query_up", "w8", ("AttnDim", "QLoraRank"), ("q_b_proj_weight",)),
    ServeObject("attention/key_value", "w8", ("D", "C"), ("kv_proj_weight",)),
    ServeObject("attention/key_value_norm", "fp32", ("D",), ("kv_norm_weight",)),
    ServeObject("attention/sinks", "fp32", ("Hq",), ("sinks",)),
    ServeObject("attention/output_grouped", "w8", ("OaOut", "OaIn"), ("o_a_proj_weight",)),
    ServeObject("attention/output_mix", "w8", ("C", "OaOut"), ("o_b_proj_weight",)),
)

#: The long-range compressor, on CSA and HCA layers. Not in the training graph (see the
#: module docstring), so these objects have no declared components — the artifact carries
#: them the way `qwen4_exp` carries its QSA indexer.
_HCA_COMPRESSOR_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("attention/compressor/key_value", "bf16", ("D", "C")),
    ServeObject("attention/compressor/gate", "bf16", ("D", "C")),
    ServeObject("attention/compressor/position_bias", "fp32", ("CompressRate", "D")),
    ServeObject("attention/compressor/norm", "fp32", ("D",)),
)

#: CSA doubles the compressor width (the Ca/Cb two-series overlap layout) and adds the
#: Lightning Indexer on top.
_CSA_COMPRESSOR_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("attention/compressor/key_value", "bf16", ("TwoD", "C")),
    ServeObject("attention/compressor/gate", "bf16", ("TwoD", "C")),
    ServeObject("attention/compressor/position_bias", "fp32", ("CompressRate", "TwoD")),
    ServeObject("attention/compressor/norm", "fp32", ("D",)),
    ServeObject("attention/indexer/key_value", "bf16", ("TwoIndexDim", "C")),
    ServeObject("attention/indexer/gate", "bf16", ("TwoIndexDim", "C")),
    ServeObject("attention/indexer/position_bias", "fp32", ("CompressRate", "TwoIndexDim")),
    ServeObject("attention/indexer/norm", "fp32", ("IndexDim",)),
    ServeObject("attention/indexer/query", "bf16", ("IndexerQueryRows", "QLoraRank")),
    ServeObject("attention/indexer/scorer", "bf16", ("IndexHeads", "C")),
)

#: The mHC pair. Deferred like the compressor, so no components.
def _hyper_connection_serve_objects(prefix: str) -> tuple[ServeObject, ...]:
    return (
        ServeObject(f"{prefix}/mix", "fp32", ("HcMix", "HcWidth")),
        ServeObject(f"{prefix}/base", "fp32", ("HcMix",)),
        ServeObject(f"{prefix}/scale", "fp32", (3,)),
    )


def _moe_serve_objects(*, hash_routed: bool) -> tuple[ServeObject, ...]:
    router = (
        ServeObject("mlp/router", "bf16", ("E", "C"), ("router_weight",)),
        ServeObject("mlp/hash_table", "i32", ("Vocab", "K"), ("tid2eid",)),
    )
    if not hash_routed:
        router = (
            ServeObject("mlp/router", "bf16", ("E", "C"), ("router_weight",)),
            ServeObject("mlp/router_bias", "fp32", ("E",), ("e_score_correction_bias",)),
        )
    return (
        *router,
        ServeObject("mlp/routed_gate_up", "quantised", ("E", "2M", "C"), ("experts_gate_up",), residency="auto"),
        ServeObject("mlp/routed_down", "quantised", ("E", "C", "M"), ("experts_down",), residency="auto"),
        ServeObject("mlp/shared_gate_up", "w8", ("SharedGateUpRows", "C"),
                    ("shared_expert_gate", "shared_expert_up")),
        ServeObject("mlp/shared_down", "w8", ("C", "SharedM"), ("shared_expert_down",)),
    )


_COMPRESSOR_SERVE_OBJECTS: dict[str, tuple[ServeObject, ...]] = {
    "sliding": (),
    "csa": _CSA_COMPRESSOR_SERVE_OBJECTS,
    "hca": _HCA_COMPRESSOR_SERVE_OBJECTS,
}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def _deepseek_v4_schema(*, attention: str, hash_routed: bool) -> BlockSchema:
    router_slot = (
        SlotDecl("tid2eid", kind="param", shape=("Vocab", "K"), dtype="int32")
        if hash_routed
        else SlotDecl("e_score_correction_bias", kind="param", shape=("E",), dtype="fp32")
    )
    slots: tuple[SlotDecl, ...] = (
        SlotDecl("q_a_proj_weight", kind="param", shape=("QLoraRank", "C")),
        SlotDecl("q_a_norm_weight", kind="param", shape=("QLoraRank",)),
        SlotDecl("q_b_proj_weight", kind="param", shape=("AttnDim", "QLoraRank")),
        SlotDecl("kv_proj_weight", kind="param", shape=("D", "C")),
        SlotDecl("kv_norm_weight", kind="param", shape=("D",)),
        SlotDecl("o_a_proj_weight", kind="param", shape=("OaOut", "OaIn")),
        SlotDecl("o_b_proj_weight", kind="param", shape=("C", "OaOut")),
        SlotDecl("sinks", kind="param", shape=("Hq",)),
        SlotDecl("router_weight", kind="param", shape=("E", "C"),
                 distribution=DistributionDecl.router_replicated()),
        router_slot,
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
        SlotDecl("permuted_input", shape=("dispatched_tokens", "C"),
                 distribution=DistributionDecl.expert_parallel()),
        SlotDecl("res_att", shape=("B", "T", "C")),
        SlotDecl("qkv_rope", shape=("B", "T", "QKV"), save_for_backward=True),
        SlotDecl("mlp_down", shape=("B", "T", "C")),
    )
    return BlockSchema(
        slots=slots,
        routing=RoutingSchema(
            # What the *model* is, not what the graph below emits: the router score is
            # sqrt(softplus(.)) and the leading layers select by hash. Both lowerings are
            # deferred; see the module docstring.
            kind="hash_topk" if hash_routed else "topk_sqrtsoftplus",
            topk="num_experts_per_tok",
            # DeepseekV4TopKRouter renormalises unconditionally; `norm_topk_prob` exists in
            # the config but the router never reads it.
            norm_topk_prob=True,
            scoring_bias=not hash_routed,
            shared_experts="shared_expert_intermediate",
        ),
        ep_topology=EPTopology(ep_size_param="ep_size"),
        serve_objects=(
            *_hyper_connection_serve_objects("attn_hc"),
            *_ATTENTION_SERVE_OBJECTS,
            *_COMPRESSOR_SERVE_OBJECTS[attention],
            *_hyper_connection_serve_objects("ffn_hc"),
            *_moe_serve_objects(hash_routed=hash_routed),
        ),
        attrs={
            "block_family": f"deepseek_v4_{attention}_{'hash_moe' if hash_routed else 'moe'}",
            "attention_kind": attention,
            "mlp_kind": "hash_moe" if hash_routed else "moe",
        },
    )


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class _DeepseekV4BlockBase(nn.Block):
    """Shared body for all six DeepSeek-V4 block shapes.

    The six subclasses differ only in their :class:`BlockSchema` — which compressor
    objects the layer carries and which router tensor exists. The graph is identical
    because every difference between the attention types lives in the compressor branch,
    which is deferred (see the module docstring).
    """

    _name_remap_ = DEEPSEEK_V4_BLOCK_NAME_REMAP

    #: HF ``layer_types`` stem ("sliding" / "csa" / "hca"); set per subclass.
    attention_kind: str = "sliding"
    #: True for ``mlp_layer_types == "hash_moe"`` layers.
    hash_routed: bool = False

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        head_size: int,
        q_lora_rank: int,
        o_groups: int,
        o_lora_rank: int,
        d_ff: int,
        shared_expert_intermediate: int,
        max_seq: int,
        num_experts: int,
        num_experts_per_tok: int,
        vocab_size: int,
        routed_scaling_factor: float = 1.5,
        swiglu_limit: float = 10.0,
        sliding_window: int = 128,
        rotary_dim: int = 64,
        eps: float = 1e-6,
        ep_size: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.shared_expert_intermediate = shared_expert_intermediate

        # Dim aliases for schema shape resolution.
        self.C = d_model
        self.D = head_size
        self.Hq = num_query_heads
        self.Hkv = 1  # shared-KV MQA: one KV head, K and V are the same tensor
        self.MaxSeq = max_seq
        self.AttnDim = num_query_heads * head_size
        self.QKV = (num_query_heads + 2) * head_size
        self.QLoraRank = q_lora_rank
        self.OGroups = o_groups
        self.OLoraRank = o_lora_rank
        self.OaIn = self.AttnDim // o_groups
        self.OaOut = o_groups * o_lora_rank
        self.RotaryDim = rotary_dim
        self.M = d_ff
        self.MUp = 2 * d_ff
        self.E = num_experts
        self.K = num_experts_per_tok
        self.Vocab = vocab_size
        self.SharedM = shared_expert_intermediate

        self.attn_norm = RMSNorm(d_model, eps=eps)
        self.self_attn = DeepseekV4Attention(
            d_model,
            num_query_heads,
            head_size,
            q_lora_rank,
            o_groups,
            o_lora_rank,
            max_seq,
            sliding_window=sliding_window,
            rotary_dim=rotary_dim,
            eps=eps,
        )
        self.mlp_norm = RMSNorm(d_model, eps=eps)
        self.moe = DeepseekV4MoEExperts(
            d_model,
            d_ff,
            num_experts,
            num_experts_per_tok,
            routed_scaling_factor=routed_scaling_factor,
            swiglu_limit=swiglu_limit,
            hash_routed=type(self).hash_routed,
            ep_size=ep_size,
        )
        self.shared_expert = MoESharedExpert(d_model, shared_expert_intermediate)

    def forward(self, x, residual, position_ids):
        # DEFERRED (mHC): the reference collapses `hc_mult` residual streams here with
        # `attn_hc`, runs the sublayer on the collapsed stream, then scatters the output
        # back through a Sinkhorn-projected doubly-stochastic combine. Neither the
        # projection (needs sum-over-dim + divide) nor the combine (needs a per-token
        # [hc, hc] x [hc, d] batched mix) has a primitive, so this is the plain
        # single-stream pre-norm residual instead.
        residual, h = self.attn_norm(residual, x)
        h = self.self_attn(h, position_ids)
        residual, h = self.mlp_norm(residual, h)

        h_flat = self._view(h, [B * T, self.d_model], name="ln2_flat")
        # Routed experts; routed_scaling_factor is folded into the weights by moe_topk.
        moe_out = self.moe(h_flat)
        # Shared expert, added ungated (V4 has no shared-expert gate row).
        shared_out = self.shared_expert(h_flat)
        moe_out = self._add(moe_out, shared_out, name="moe_combined")

        self._register_activation(
            "mlp_down",
            ("B", "T", "C"),
            aliases=["mlp_down_flat"],
            share_policy="per_layer",
            description="MoE + shared expert output (block output)",
        )
        out = self._view(moe_out, [B, T, self.d_model], name="mlp_down")
        return out, residual


class DeepseekV4SlidingMoEBlock(_DeepseekV4BlockBase):
    """``sliding_attention`` + learned top-k MoE."""

    attention_kind = "sliding"
    hash_routed = False
    schema = _deepseek_v4_schema(attention="sliding", hash_routed=False)


class DeepseekV4SlidingHashBlock(_DeepseekV4BlockBase):
    """``sliding_attention`` + hash-routed MoE."""

    attention_kind = "sliding"
    hash_routed = True
    schema = _deepseek_v4_schema(attention="sliding", hash_routed=True)


class DeepseekV4CsaMoEBlock(_DeepseekV4BlockBase):
    """``compressed_sparse_attention`` (m=4 + Lightning Indexer) + learned top-k MoE."""

    attention_kind = "csa"
    hash_routed = False
    schema = _deepseek_v4_schema(attention="csa", hash_routed=False)


class DeepseekV4CsaHashBlock(_DeepseekV4BlockBase):
    """``compressed_sparse_attention`` + hash-routed MoE."""

    attention_kind = "csa"
    hash_routed = True
    schema = _deepseek_v4_schema(attention="csa", hash_routed=True)


class DeepseekV4HcaMoEBlock(_DeepseekV4BlockBase):
    """``heavily_compressed_attention`` (m'=128, no indexer) + learned top-k MoE."""

    attention_kind = "hca"
    hash_routed = False
    schema = _deepseek_v4_schema(attention="hca", hash_routed=False)


class DeepseekV4HcaHashBlock(_DeepseekV4BlockBase):
    """``heavily_compressed_attention`` + hash-routed MoE — the V4 bootstrap layers."""

    attention_kind = "hca"
    hash_routed = True
    schema = _deepseek_v4_schema(attention="hca", hash_routed=True)


#: block-type stem -> (param name for HybridBlockStack, block class).
DEEPSEEK_V4_BLOCK_CLASSES: dict[str, type[_DeepseekV4BlockBase]] = {
    "sliding_moe": DeepseekV4SlidingMoEBlock,
    "sliding_hash": DeepseekV4SlidingHashBlock,
    "csa_moe": DeepseekV4CsaMoEBlock,
    "csa_hash": DeepseekV4CsaHashBlock,
    "hca_moe": DeepseekV4HcaMoEBlock,
    "hca_hash": DeepseekV4HcaHashBlock,
}
