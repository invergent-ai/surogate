"""GLM-5.3-Flash (``glm5_next``) decoder blocks.

Two independent axes, so up to four block shapes:

* **mixer** — ``layer_types[i]`` is ``"linear_attention"`` (KDA, three layers in
  four) or ``"deepseek_sparse_attention"`` (NoPE MLA, every fourth layer);
* **feed-forward** — ``mlp_layer_types[i]`` is ``"dense"`` for the leading
  ``first_k_dense_replace`` layers and ``"sparse"`` after that.

In the released 45-layer checkpoint the dense layers (0-2) happen to be all-KDA,
so ``mla_dense`` never occurs — but the config allows it, and a scaled-down
config hits it easily, so all four are declared.

Every layer, of every shape, carries the same skeleton::

    collapsed, post, comb = hc_attn(residual_streams)     # mHC mix
    h                     = input_layernorm(collapsed)
    a                     = KDA(h) | MLA(h)
    residual_streams      = hc_attn_combine(residual, a, post, comb)

    collapsed, post, comb = hc_ffn(residual_streams)
    h                     = post_attention_layernorm(collapsed)
    f                     = SwiGLU MLP | (routed MoE + shared expert)
    residual_streams      = hc_ffn_combine(residual, f, post, comb)

Note the difference from Qwen3.8-Flash-Next: hyper-connections here sit *beside*
the layer norms rather than replacing them, and the model keeps a real final
``norm`` after the streams are averaged.

The router is DeepSeek-V3's: sigmoid scores, top-k chosen on
``score + e_score_correction_bias``, renormalised over the winners, then scaled
by ``routed_scaling_factor`` (2.5). ``n_group``/``topk_group`` are both 1 in
every released config, so the group-restricted selection degenerates to a plain
top-k and is not modelled.

The training graph includes the asymmetric SwiGLU clamp and resets recurrent
state and convolution history at packed sequence boundaries. Sparse DSA
selection, vision and MTP are serving-only declarations; text training currently
requires sequences no longer than ``index_topk``.
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
from ..modules import MoESharedExpert, RMSNorm
from ..modules.glm5_next import (
    Glm5NextDenseMLP,
    Glm5NextHyperConnection,
    Glm5NextHyperConnectionCombine,
    Glm5NextKimiDeltaMixer,
    Glm5NextLatentAttention,
    Glm5NextMoEExperts,
)

#: Norms are registered by ``RMSNorm`` under the attribute name; the canonical
#: (and checkpoint-facing) names are ``ln1``/``ln2`` as everywhere else.
_GLM5_NORM_REMAP: dict[str, str] = {
    "attn_norm_weight": "ln1_weight",
    "attn_norm_y": "ln1",
    "attn_norm_rstd": "ln1_rstd",
    "mlp_norm_weight": "ln2_weight",
    "mlp_norm_y": "ln2",
    "mlp_norm_rstd": "ln2_rstd",
}

#: ``LagunaMoEExperts`` registers its params under a ``moe_`` prefix; strip it.
_GLM5_MOE_REMAP: dict[str, str] = {
    "moe_router_weight": "router_weight",
    "moe_e_score_correction_bias": "e_score_correction_bias",
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
    # shared_expert_* and the hc_*/kda_*/mla_* prefixes are already canonical.
}

GLM5_NEXT_DENSE_BLOCK_REMAP: dict[str, str] = dict(_GLM5_NORM_REMAP)
GLM5_NEXT_MOE_BLOCK_REMAP: dict[str, str] = {**_GLM5_NORM_REMAP, **_GLM5_MOE_REMAP}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

#: The mHC pair, identical on every layer and both sublayer sites.
def _hc_slots(site: str) -> tuple[SlotDecl, ...]:
    return (
        SlotDecl(f"hc_{site}_fn", kind="param", shape=("HcMix", "HcWidth")),
        SlotDecl(f"hc_{site}_base", kind="param", shape=("HcMix",), dtype="fp32"),
        SlotDecl(f"hc_{site}_scale", kind="param", shape=(3,), dtype="fp32"),
        SlotDecl(f"hc_{site}_comb", shape=("B * T", "HcCount", "HcCount"), dtype="fp32", save_for_backward=True),
    )


_KDA_SLOTS: tuple[SlotDecl, ...] = (
    SlotDecl("kda_qkv_weight", kind="param", shape=("KdaConvDim", "C"), residency="auto"),
    SlotDecl("kda_conv_weight", kind="param", shape=("KdaConvDim", 1, "KdaConvK")),
    SlotDecl("kda_f_a_weight", kind="param", shape=("KdaHeadDim", "C")),
    SlotDecl("kda_f_b_weight", kind="param", shape=("KdaDim", "KdaHeadDim")),
    SlotDecl("kda_dt_bias", kind="param", shape=("KdaDim",), dtype="fp32"),
    SlotDecl("kda_A_log", kind="param", shape=("KdaHeads",), dtype="fp32"),
    SlotDecl("kda_b_weight", kind="param", shape=("KdaHeads", "C")),
    SlotDecl("kda_g_a_weight", kind="param", shape=("KdaHeadDim", "C")),
    SlotDecl("kda_g_b_weight", kind="param", shape=("KdaDim", "KdaHeadDim")),
    SlotDecl("kda_o_norm_weight", kind="param", shape=("KdaHeadDim",)),
    SlotDecl("kda_out_weight", kind="param", shape=("C", "KdaDim"), residency="auto"),
    SlotDecl("kda_out", shape=("B", "T", "C")),
)

_MLA_SLOTS: tuple[SlotDecl, ...] = (
    SlotDecl("mla_q_a_weight", kind="param", shape=("QRank", "C")),
    SlotDecl("mla_q_a_norm_weight", kind="param", shape=("QRank",)),
    SlotDecl("mla_q_b_weight", kind="param", shape=("QDim", "QRank"), residency="auto"),
    SlotDecl("mla_kv_a_weight", kind="param", shape=("KVRank", "C")),
    SlotDecl("mla_kv_a_norm_weight", kind="param", shape=("KVRank",)),
    SlotDecl("mla_kv_b_weight", kind="param", shape=("KVBDim", "KVRank"), residency="auto"),
    SlotDecl("mla_out_weight", kind="param", shape=("C", "VDim"), residency="auto"),
    SlotDecl("mla_att_out", shape=("B", "T", "C")),
)

_DENSE_FFN_SLOTS: tuple[SlotDecl, ...] = (
    SlotDecl("mlp_up_weight", kind="param", shape=("2M", "C"), residency="auto"),
    SlotDecl("mlp_down_weight", kind="param", shape=("C", "M"), residency="auto"),
)

_MOE_FFN_SLOTS: tuple[SlotDecl, ...] = (
    SlotDecl("router_weight", kind="param", shape=("E", "C"), distribution=DistributionDecl.router_replicated()),
    SlotDecl("e_score_correction_bias", kind="param", shape=("E",), dtype="fp32"),
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
    SlotDecl("shared_expert_gate", kind="param", shape=("SharedM", "C")),
    SlotDecl("shared_expert_up", kind="param", shape=("SharedM", "C")),
    SlotDecl("shared_expert_down", kind="param", shape=("C", "SharedM")),
    SlotDecl("permuted_input", shape=("dispatched_tokens", "C"), distribution=DistributionDecl.expert_parallel()),
)


def _glm5_routing() -> RoutingSchema:
    # DeepSeek-V3 router: sigmoid scores, selection on score + correction bias,
    # renormalised over the winners, scaled by routed_scaling_factor.
    return RoutingSchema(
        kind="topk_sigmoid",
        topk="num_experts_per_tok",
        norm_topk_prob="norm_topk_prob",
        scoring_bias=True,
        shared_experts="shared_expert_intermediate",
    )


#: How a serving artifact stores a GLM-5.3 block.
#:
#: Four kinds -- {KDA, MLA} x {dense feed-forward, mixture} -- built from four lists, because
#: the hyper-connection mix and the two norms are the same on every one of them.
_HC_SERVE_OBJECTS: tuple[ServeObject, ...] = tuple(
    obj
    for site in ("attn", "ffn")
    for obj in (
        # BF16, not quantised: these 24 rows produce every mixing weight the residual is
        # recombined with, at both sites of all 45 layers, and a coarse width there is paid on
        # the whole stream rather than on one projection's output. It is 71 MB.
        ServeObject(f"hc/{site}_mix", "bf16", ("HcMix", "HcWidth"), (f"hc_{site}_fn",)),
        ServeObject(f"hc/{site}_base", "fp32", ("HcMix",), (f"hc_{site}_base",)),
        ServeObject(f"hc/{site}_scale", "fp32", (3,), (f"hc_{site}_scale",)),
    )
)

_GLM5_NORM_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("input_norm", "bf16", ("C",), ("ln1_weight",)),
    ServeObject("post_attention_norm", "bf16", ("C",), ("ln2_weight",)),
)

#: Kimi Delta Attention. q, k and v share one projection and one depthwise convolution, which
#: is what the reference does at runtime and what the engine's convolution reads; the decay is a
#: low-rank pair through a head-width bottleneck, and so is the output gate.
_KDA_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("kda/query_key_value", "quantised", ("KdaConvDim", "C"), ("kda_qkv_weight",)),
    # Stored [K, channels] like the linear-attention convolution's taps, which is the layout its
    # kernel reads; the checkpoint holds them [channels, 1, K].
    ServeObject("kda/convolution", "bf16", ("KdaConvK", "KdaConvDim"), ("kda_conv_weight",),
                transform="transpose_taps"),
    ServeObject("kda/decay_a", "quantised", ("KdaHeadDim", "C"), ("kda_f_a_weight",)),
    ServeObject("kda/decay_b", "quantised", ("KdaDim", "KdaHeadDim"), ("kda_f_b_weight",)),
    ServeObject("kda/decay_bias", "fp32", ("KdaDim",), ("kda_dt_bias",)),
    ServeObject("kda/a_log", "fp32", ("KdaHeads",), ("kda_A_log",)),
    ServeObject("kda/beta", "quantised", ("KdaHeads", "C"), ("kda_b_weight",)),
    ServeObject("kda/gate_a", "quantised", ("KdaHeadDim", "C"), ("kda_g_a_weight",)),
    ServeObject("kda/gate_b", "quantised", ("KdaDim", "KdaHeadDim"), ("kda_g_b_weight",)),
    ServeObject("kda/norm", "bf16", ("KdaHeadDim",), ("kda_o_norm_weight",)),
    ServeObject("kda/output", "quantised", ("C", "KdaDim"), ("kda_out_weight",)),
)

#: Multi-head latent attention, NoPE. Two low-rank projections with a norm inside each; the
#: attention they feed is ordinary, because `kv_b` gives every head its own key and value.
_MLA_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("mla/query_a", "quantised", ("QRank", "C"), ("mla_q_a_weight",)),
    ServeObject("mla/query_a_norm", "bf16", ("QRank",), ("mla_q_a_norm_weight",)),
    ServeObject("mla/query_b", "quantised", ("QDim", "QRank"), ("mla_q_b_weight",)),
    ServeObject("mla/kv_a", "quantised", ("KVRank", "C"), ("mla_kv_a_weight",)),
    ServeObject("mla/kv_a_norm", "bf16", ("KVRank",), ("mla_kv_a_norm_weight",)),
    # The latent expansion, held as its two halves. They are one parameter in training and one
    # tensor in HuggingFace; a serving artifact splits them because the served attention uses
    # them on opposite sides of its scores. The key half is applied to the *query* -- the
    # absorbed form, which is also the orientation llama.cpp stores it in, so it is read where
    # it lies as [latent, nope] per head -- and the value half unfolds the attended latent
    # afterwards. Both stay in the file in the file's own format.
    ServeObject("mla/k_b", "quantised", ("KAbsorbDim", "NopeDim"), ("mla_kv_b_weight",)),
    ServeObject("mla/v_b", "quantised", ("VDim", "KVRank"), ("mla_kv_b_weight",)),
    ServeObject("mla/output", "quantised", ("C", "VDim"), ("mla_out_weight",)),
)

#: The dense feed-forward of the leading layers, at `intermediate_size` rather than the
#: mixture's width.
_GLM5_DENSE_FFN_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("mlp/gate_up", "quantised", ("TwoM", "C"),
                ("mlp_up_weight.gate", "mlp_up_weight.up")),
    ServeObject("mlp/down", "quantised", ("C", "M"), ("mlp_down_weight",)),
)

#: The mixture. Its router is one row per expert and nothing else: the always-on expert is
#: added with weight one, so there is no gate row to fuse on -- unlike every other MoE family
#: here, whose shared expert is weighted by a sigmoid the router carries.
_GLM5_MOE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("moe/router", "bf16", ("E", "C"), ("router_weight",)),
    # Selection is on the score plus this bias; the weight is the score without it.
    ServeObject("moe/router_bias", "fp32", ("E",), ("e_score_correction_bias",)),
    ServeObject("moe/routed_gate_up", "quantised", ("RoutedGateUpRows", "C"),
                ("experts_gate_up",), transform="flatten_experts", residency="auto"),
    ServeObject("moe/routed_down", "quantised", ("RoutedDownRows", "MoeM"),
                ("experts_down",), transform="flatten_experts", residency="auto"),
    ServeObject("moe/shared_gate_up", "quantised", ("SharedGateUpRows", "C"),
                ("shared_expert_gate", "shared_expert_up")),
    ServeObject("moe/shared_down", "quantised", ("C", "SharedM"), ("shared_expert_down",)),
)


#: The NextN draft head's own objects, in the order the artifact stores them. Its block is one
#: latent-attention layer over the mixture *without* hyper-connections -- the head runs on a
#: single-stream residual -- so it replays the MLA-over-MoE block's objects minus the two
#: hyper-connection sites, between the three tensors that fold the next token's embedding in
#: and the norm that reads the result out for the trunk's LM head.
GLM5_NEXT_MTP_LAYER_OBJECTS: tuple[ServeObject, ...] = (
    _GLM5_NORM_OBJECTS[0],
    *_MLA_SERVE_OBJECTS,
    _GLM5_NORM_OBJECTS[1],
    *_GLM5_MOE_OBJECTS,
)


def _glm5_serve_objects(*, mixer: str, sparse: bool) -> tuple[ServeObject, ...]:
    """The objects one block kind holds, in the order the artifact stores them."""
    return (
        *_HC_SERVE_OBJECTS,
        _GLM5_NORM_OBJECTS[0],
        *(_KDA_SERVE_OBJECTS if mixer == "kda" else _MLA_SERVE_OBJECTS),
        _GLM5_NORM_OBJECTS[1],
        *(_GLM5_MOE_OBJECTS if sparse else _GLM5_DENSE_FFN_OBJECTS),
    )


def _glm5_schema(block_family: str, *, mixer: str, sparse: bool) -> BlockSchema:
    mixer_slots = _KDA_SLOTS if mixer == "kda" else _MLA_SLOTS
    slots = (
        *_hc_slots("attn"),
        SlotDecl("ln1_weight", kind="param", shape=("C",)),
        *mixer_slots,
        *_hc_slots("ffn"),
        SlotDecl("ln2_weight", kind="param", shape=("C",)),
        *(_MOE_FFN_SLOTS if sparse else _DENSE_FFN_SLOTS),
    )
    return BlockSchema(
        slots=slots,
        routing=_glm5_routing() if sparse else None,
        ep_topology=EPTopology(ep_size_param="ep_size") if sparse else None,
        attrs={"block_family": block_family},
        serve_objects=_glm5_serve_objects(mixer=mixer, sparse=sparse),
    )


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class _Glm5NextBlockBase(nn.Block):
    """Shared mHC skeleton. Subclasses build ``self.mixer`` and the FFN tail."""

    def _init_common(
        self,
        *,
        d_model: int,
        eps: float,
        hc_mult: int,
        hc_eps: float,
        hc_sinkhorn_iters: int,
        swiglu_limit: float,
    ) -> None:
        self.d_model = d_model
        self.swiglu_limit = swiglu_limit
        self.eps = eps
        self.C = Dim("C")
        self.HcCount = hc_mult
        self.HcWidth = hc_mult * d_model
        self.HcMix = (2 + hc_mult) * hc_mult

        self.hc_attn = Glm5NextHyperConnection(
            d_model, hc_mult=hc_mult, hc_eps=hc_eps, hc_sinkhorn_iters=hc_sinkhorn_iters, eps=eps
        )
        self.hc_attn_combine = Glm5NextHyperConnectionCombine(d_model, hc_mult=hc_mult)
        self.attn_norm = RMSNorm(d_model, eps=eps)
        self.hc_ffn = Glm5NextHyperConnection(
            d_model, hc_mult=hc_mult, hc_eps=hc_eps, hc_sinkhorn_iters=hc_sinkhorn_iters, eps=eps
        )
        self.hc_ffn_combine = Glm5NextHyperConnectionCombine(d_model, hc_mult=hc_mult)
        self.mlp_norm = RMSNorm(d_model, eps=eps)

    def _init_kda(
        self,
        *,
        d_model: int,
        linear_num_heads: int,
        linear_head_dim: int,
        linear_conv_kernel_dim: int,
        linear_lower_bound: float,
        chunk_size: int,
        eps: float,
    ) -> None:
        self.KdaHeads = linear_num_heads
        self.KdaHeadDim = linear_head_dim
        self.KdaDim = linear_num_heads * linear_head_dim
        self.KdaConvDim = 3 * self.KdaDim
        self.KdaConvK = linear_conv_kernel_dim
        self.kda = Glm5NextKimiDeltaMixer(
            d_model,
            num_heads=linear_num_heads,
            head_dim=linear_head_dim,
            conv_kernel=linear_conv_kernel_dim,
            gate_lower_bound=linear_lower_bound,
            chunk_size=chunk_size,
            eps=eps,
        )

    def _init_mla(
        self,
        *,
        d_model: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        qk_rope_head_dim: int,
        eps: float,
    ) -> None:
        self.Hq = num_attention_heads
        self.QRank = q_lora_rank
        self.KVRank = kv_lora_rank
        self.QKHead = qk_nope_head_dim + qk_rope_head_dim
        self.VHead = v_head_dim
        self.QDim = num_attention_heads * self.QKHead
        self.KVBDim = num_attention_heads * (qk_nope_head_dim + v_head_dim)
        self.VDim = num_attention_heads * v_head_dim
        self.mla = Glm5NextLatentAttention(
            d_model,
            num_heads=num_attention_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            eps=eps,
        )

    def _init_dense_ffn(self, *, d_model: int, intermediate_size: int) -> None:
        self.M = intermediate_size
        self.use_shared_expert = False
        self.mlp = Glm5NextDenseMLP(d_model, intermediate_size, self.swiglu_limit)

    def _init_moe_ffn(
        self,
        *,
        d_model: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        shared_expert_intermediate: int,
        routed_scaling_factor: float,
        ep_size: int,
    ) -> None:
        self.M = moe_intermediate_size
        self.SharedM = shared_expert_intermediate
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.shared_expert_intermediate = shared_expert_intermediate
        self.routed_scaling_factor = routed_scaling_factor
        self.ep_size = ep_size
        self.use_shared_expert = shared_expert_intermediate > 0
        self.moe = Glm5NextMoEExperts(
            d_model,
            moe_intermediate_size,
            num_experts,
            num_experts_per_tok,
            routed_scaling_factor=routed_scaling_factor,
            ep_size=ep_size,
        )
        self.moe.swiglu_limit = self.swiglu_limit
        if self.use_shared_expert:
            self.shared_expert = MoESharedExpert(d_model, shared_expert_intermediate)
            self.shared_expert.swiglu_limit = self.swiglu_limit

    # -- forward pieces ------------------------------------------------------

    def _mixer_out(self, h, position_ids):
        raise NotImplementedError

    def _ffn_out(self, h):
        raise NotImplementedError

    def _dense_ffn(self, h):
        return self.mlp(h)

    def _moe_ffn(self, h):
        h_flat = self._view(h, [B * T, self.C], name="moe_in_flat")
        out = self.moe(h_flat)
        if self.use_shared_expert:
            # GLM-5.3's shared expert is ungated: plain `hidden + shared(x)`.
            out = self._add(out, self.shared_expert(h_flat), name="moe_combined")
        self._register_activation(
            "mlp_down",
            ("B", "T", "C"),
            aliases=["mlp_down_flat"],
            share_policy="per_layer",
            description="MoE output (block output)",
        )
        return self._view(out, [B, T, self.C], name="mlp_down")

    def forward(self, x, residual, position_ids):
        del x  # Blocks read only the wide residual streams.
        self._register_activation("position_ids", ("B", "T"), dtype="int32")
        self._register_activation("residual", ("B", "T", self.HcWidth))

        h, post_a, comb_a = self.hc_attn(residual)
        h = self.attn_norm(h)
        a = self._mixer_out(h, position_ids)
        residual = self.hc_attn_combine(residual, a, post_a, comb_a)

        h2, post_f, comb_f = self.hc_ffn(residual)
        h2 = self.mlp_norm(h2)
        out = self._ffn_out(h2)
        residual = self.hc_ffn_combine(residual, out, post_f, comb_f)
        return out, residual


class Glm5NextKdaDenseBlock(_Glm5NextBlockBase):
    """KDA linear-attention layer with a dense SwiGLU MLP (layers < ``first_k_dense_replace``)."""

    _name_remap_ = GLM5_NEXT_DENSE_BLOCK_REMAP
    schema = _glm5_schema("glm5_next_kda_dense", mixer="kda", sparse=False)

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        linear_num_heads: int = 64,
        linear_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        linear_lower_bound: float = -5.0,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        swiglu_limit: float = 10.0,
        chunk_size: int = 64,
        eps: float = 1e-5,
    ):
        super().__init__()
        self._init_common(
            d_model=d_model, eps=eps, hc_mult=hc_mult, hc_eps=hc_eps, hc_sinkhorn_iters=hc_sinkhorn_iters,
            swiglu_limit=swiglu_limit,
        )
        self._init_kda(
            d_model=d_model,
            linear_num_heads=linear_num_heads,
            linear_head_dim=linear_head_dim,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_lower_bound=linear_lower_bound,
            chunk_size=chunk_size,
            eps=eps,
        )
        self._init_dense_ffn(d_model=d_model, intermediate_size=intermediate_size)

    def _mixer_out(self, h, position_ids):
        return self.kda(h, position_ids)

    def _ffn_out(self, h):
        return self._dense_ffn(h)


class Glm5NextKdaMoEBlock(_Glm5NextBlockBase):
    """KDA linear-attention layer with the sparse MoE feed-forward."""

    _name_remap_ = GLM5_NEXT_MOE_BLOCK_REMAP
    schema = _glm5_schema("glm5_next_kda_moe", mixer="kda", sparse=True)

    def __init__(
        self,
        d_model: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        shared_expert_intermediate: int,
        routed_scaling_factor: float = 2.5,
        linear_num_heads: int = 64,
        linear_head_dim: int = 128,
        linear_conv_kernel_dim: int = 4,
        linear_lower_bound: float = -5.0,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        swiglu_limit: float = 10.0,
        chunk_size: int = 64,
        eps: float = 1e-5,
        ep_size: int = 1,
    ):
        super().__init__()
        self._init_common(
            d_model=d_model, eps=eps, hc_mult=hc_mult, hc_eps=hc_eps, hc_sinkhorn_iters=hc_sinkhorn_iters,
            swiglu_limit=swiglu_limit,
        )
        self._init_kda(
            d_model=d_model,
            linear_num_heads=linear_num_heads,
            linear_head_dim=linear_head_dim,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_lower_bound=linear_lower_bound,
            chunk_size=chunk_size,
            eps=eps,
        )
        self._init_moe_ffn(
            d_model=d_model,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            shared_expert_intermediate=shared_expert_intermediate,
            routed_scaling_factor=routed_scaling_factor,
            ep_size=ep_size,
        )

    def _mixer_out(self, h, position_ids):
        return self.kda(h, position_ids)

    def _ffn_out(self, h):
        return self._moe_ffn(h)


class Glm5NextMlaDenseBlock(_Glm5NextBlockBase):
    """NoPE-MLA layer with a dense SwiGLU MLP.

    Unreachable in the released 45-layer checkpoint (its three dense layers are
    all KDA), but the config's two axes are independent, so it is declared.
    """

    _name_remap_ = GLM5_NEXT_DENSE_BLOCK_REMAP
    schema = _glm5_schema("glm5_next_mla_dense", mixer="mla", sparse=False)

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        qk_rope_head_dim: int = 0,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        swiglu_limit: float = 10.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self._init_common(
            d_model=d_model, eps=eps, hc_mult=hc_mult, hc_eps=hc_eps, hc_sinkhorn_iters=hc_sinkhorn_iters,
            swiglu_limit=swiglu_limit,
        )
        self._init_mla(
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            eps=eps,
        )
        self._init_dense_ffn(d_model=d_model, intermediate_size=intermediate_size)

    def _mixer_out(self, h, position_ids):
        return self.mla(h)

    def _ffn_out(self, h):
        return self._dense_ffn(h)


class Glm5NextMlaMoEBlock(_Glm5NextBlockBase):
    """NoPE-MLA layer with the sparse MoE feed-forward (layers 3, 7, 11, ...)."""

    _name_remap_ = GLM5_NEXT_MOE_BLOCK_REMAP
    schema = _glm5_schema("glm5_next_mla_moe", mixer="mla", sparse=True)

    def __init__(
        self,
        d_model: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        shared_expert_intermediate: int,
        num_attention_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        qk_rope_head_dim: int = 0,
        routed_scaling_factor: float = 2.5,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        swiglu_limit: float = 10.0,
        eps: float = 1e-5,
        ep_size: int = 1,
    ):
        super().__init__()
        self._init_common(
            d_model=d_model, eps=eps, hc_mult=hc_mult, hc_eps=hc_eps, hc_sinkhorn_iters=hc_sinkhorn_iters,
            swiglu_limit=swiglu_limit,
        )
        self._init_mla(
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            eps=eps,
        )
        self._init_moe_ffn(
            d_model=d_model,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            shared_expert_intermediate=shared_expert_intermediate,
            routed_scaling_factor=routed_scaling_factor,
            ep_size=ep_size,
        )

    def _mixer_out(self, h, position_ids):
        return self.mla(h)

    def _ffn_out(self, h):
        return self._moe_ffn(h)
