"""Gemma4 Transformer Blocks.

Gemma4 uses a sandwich-norm pattern with 4 RMSNorm layers per block:
  input_layernorm -> attention -> post_attention_layernorm -> residual_add
  pre_feedforward_layernorm -> MLP -> post_feedforward_layernorm -> residual_add
  [optional] per_layer_input_gate -> gelu -> mul -> project -> norm -> residual_add
  layer_scalar

Block variants:
  - Gemma4SlidingBlock: sliding window attention, standard MLP
  - Gemma4FullBlock: full attention (optional k_eq_v), standard MLP
  - Gemma4SlidingMoEBlock: sliding attention + MoE (parallel MLP + experts)
  - Gemma4FullMoEBlock: full attention (k_eq_v) + MoE
  - Gemma4SharedSlidingBlock: shared-KV sliding (Q-only attention)
  - Gemma4SharedFullBlock: shared-KV full (Q-only attention, optional double-wide MLP)
"""

from __future__ import annotations

from .. import nn
from ..modules import (
    Gemma4Attention,
    Gemma4MoEExperts,
    Gemma4SharedKVAttention,
    GenericMLP,
    RMSNorm,
    _resolve_rotary_dim,
)
from ..activations import Activation
from ..block_schema import BlockSchema, DistributionDecl, EPTopology, RoutingSchema, SlotDecl, StreamingHint
from ..dim import B, T
from ..mlp import MLPConfig


GEMMA4_BLOCK_NAME_REMAP: dict[str, str] = {
    # --- input_layernorm (standalone rmsnorm) -> ln1 ---
    "input_layernorm_weight": "ln1_weight",
    "input_layernorm_y": "ln1",
    "input_layernorm_rstd": "ln1_rstd",
    # --- self_attn (Gemma4Attention) -> strip prefix ---
    "self_attn_qkv_weight": "qkv_weight",
    "self_attn_out_weight": "out_weight",
    "self_attn_q_norm_weight": "q_norm_weight",
    "self_attn_k_norm_weight": "k_norm_weight",
    "self_attn_rope_freqs": "rope_freqs",
    "self_attn_qkv": "qkv",
    "self_attn_qkv_flat": "qkv_flat",
    "self_attn_qkv_norm": "qkv_norm",
    "self_attn_q_rstd": "q_rstd",
    "self_attn_k_rstd": "k_rstd",
    "self_attn_qkv_normed_flat": "qkv_normed_flat",
    "self_attn_qk_normed_flat": "qk_normed_flat",
    "self_attn_v_flat_3d": "v_flat_3d",
    "self_attn_qkv_normed_3d": "qkv_normed_3d",
    "self_attn_qkv_normed_4d": "qkv_normed_4d",
    "self_attn_qkv_rope": "qkv_rope",
    "self_attn_att": "att",
    "self_attn_att_flat": "att_flat",
    "self_attn_lse": "lse",
    "self_attn_att_out": "att_out",
    "self_attn_att_out_flat": "att_out_flat",
    "self_attn_x_flat": "x_flat",
    "self_attn_ones_d": "ones_d",
    "self_attn_q_flat": "q_flat",
    "self_attn_q_4d": "q_4d",
    "self_attn_q_rn_flat": "q_rn_flat",
    "self_attn_q_normed_flat": "q_normed_flat",
    "self_attn_q_normed": "q_normed",
    "self_attn_kv_source_flat": "kv_source_flat",
    "self_attn_kv_flat": "kv_flat",
    "self_attn_kv_4d": "kv_4d",
    "self_attn_q_roped": "q_roped",
    "self_attn_v_flat_2d": "v_flat_2d",
    "self_attn_v_normed_2d": "v_normed_2d",
    "self_attn_v_normed": "v_normed",
    # --- post_attn_layernorm (standalone rmsnorm) ---
    "post_attn_layernorm_weight": "ln_post_attn_weight",
    "post_attn_layernorm_y": "ln_post_attn",
    "post_attn_layernorm_rstd": "ln_post_attn_rstd",
    # --- pre_ff_layernorm (fused_residual_rmsnorm: res_att + pre_ff_norm) -> ln2 ---
    "pre_ff_layernorm_weight": "ln2_weight",
    "pre_ff_layernorm_y": "ln2",
    "pre_ff_layernorm_rstd": "ln2_rstd",
    # Fused residual output becomes the canonical res_att slot.
    "pre_ff_layernorm_res": "res_att",
    # --- mlp (GenericMLP with gelu, separate gate/up) ---
    "mlp_gate_weight": "mlp_gate_weight",
    "mlp_up_weight": "mlp_up_weight",
    "mlp_down_weight": "mlp_down_weight",
    "mlp_x_flat": "mlp_x_flat",
    "mlp_gate_flat": "mlp_gate_flat",
    "mlp_up_flat": "mlp_up_flat",
    "mlp_gate_act": "mlp_gate_act",
    "mlp_act_flat": "swiglu_flat",
    "mlp_down_flat": "mlp_down_flat",
    "mlp_down": "mlp_down",
    # --- post_ff_layernorm (standalone rmsnorm) ---
    "post_ff_layernorm_weight": "ln_post_ff_weight",
    "post_ff_layernorm_y": "ln_post_ff",
    "post_ff_layernorm_rstd": "ln_post_ff_rstd",
    # --- res_att (canonical residual-after-attention slot) ---
    "res_att": "res_att",
    # --- per-layer input gating ---
    "pli_gate_weight": "pli_gate_weight",
    "pli_proj_weight": "pli_proj_weight",
    "pli_norm_weight": "pli_norm_weight",
    "pli_gate_out": "pli_gate_out",
    "pli_gate_act": "pli_gate_act",
    "pli_gated": "pli_gated",
    "pli_proj_out": "pli_proj_out",
    "pli_normed": "pli_normed",
    # --- layer_scalar (frozen per-layer scaling buffer) ---
    "layer_scalar": "layer_scalar",
}

# Model-level name remap (used by surogate.dsl.models.gemma4).
GEMMA4_MODEL_NAME_REMAP: dict[str, str] = {
    # --- embedding ---
    "embedding_weight": "embedding",
    "embedding_out": "x0",
    # --- per-layer input embedding ---
    "pli_embedding_weight": "pli_embedding",
    "pli_model_proj": "pli_model_proj",
    "pli_proj_norm": "pli_proj_norm",
    # --- final_norm (RMSNorm) ---
    "final_norm_weight": "final_norm",
    "final_norm_res": "residual_final",
    "final_norm_y": "xF",
    "final_norm_rstd": "ln_final_rstd",
    # --- lm_head ---
    "lm_head_weight": "lm_head",
    "lm_head_loss": "loss",
    "lm_head_x_flat": "xF_flat",
}

# Gemma4 uses a separate-gate GELU MLP (no gate/up fusion, GELU on gate).
_GEMMA4_GELU_MLP_CONFIG = MLPConfig(
    activation=Activation.GELU,
    gated=True,
    fuse_gate_up=False,
)


def _gemma4_dense_schema(block_family: str, *, shared_kv: bool = False) -> BlockSchema:
    attn_weight = (
        SlotDecl("self_attn_q_weight", kind="param", shape=("AttnDim", "C"))
        if shared_kv
        else SlotDecl("qkv_weight", kind="param", shape=("QKV", "C"))
    )
    return BlockSchema(
        slots=(
            attn_weight,
            SlotDecl("out_weight", kind="param", shape=("C", "AttnDim")),
            SlotDecl("mlp_gate_weight", kind="param", shape=("M", "C"), residency="auto"),
            SlotDecl("mlp_up_weight", kind="param", shape=("M", "C"), residency="auto"),
            SlotDecl("mlp_down_weight", kind="param", shape=("C", "M"), residency="auto"),
            SlotDecl("layer_scalar", kind="param", shape=(1,), dtype="bf16"),
            SlotDecl("res_att", shape=("B", "T", "C"), save_for_backward=True),
            SlotDecl("qkv_rope", shape=("B", "T", "QKV"), save_for_backward=True),
        ),
        attrs={"block_family": block_family, "shared_kv": shared_kv},
    )


def _gemma4_moe_schema(block_family: str) -> BlockSchema:
    return BlockSchema(
        slots=(
            SlotDecl("qkv_weight", kind="param", shape=("QKV", "C")),
            SlotDecl("out_weight", kind="param", shape=("C", "AttnDim")),
            SlotDecl("mlp_gate_weight", kind="param", shape=("M", "C"), residency="auto"),
            SlotDecl("mlp_up_weight", kind="param", shape=("M", "C"), residency="auto"),
            SlotDecl("mlp_down_weight", kind="param", shape=("C", "M"), residency="auto"),
            SlotDecl(
                "router_weight", kind="param", shape=("E", "C"), distribution=DistributionDecl.router_replicated()
            ),
            SlotDecl("router_scale", kind="param", shape=("C",), dtype="bf16"),
            SlotDecl("per_expert_scale", kind="param", shape=("E",), dtype="bf16"),
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
            SlotDecl("res_att", shape=("B", "T", "C"), save_for_backward=True),
        ),
        routing=RoutingSchema(kind="topk_softmax", topk="num_experts_per_tok", norm_topk_prob=True),
        ep_topology=EPTopology(ep_size_param="ep_size"),
        attrs={"block_family": block_family},
    )


# ============================================================================
# Helpers shared across block types
# ============================================================================


def _sandwich_attn_phase(block, x, residual, position_ids):
    """Sandwich norm attention phase: input_ln → attn → post_attn_ln.

    Leaves the ``residual + h_post_attn = res_att`` add to the next phase so it
    can fuse with ``pre_ff_layernorm`` via RMSNorm's two-arg fused_residual
    path. Returns ``(residual, h_post_attn)`` — the caller composes the sum.

    Gemma4 uses zero residual between blocks (state flows through x, not residual).
    We still materialize the incoming hidden state into the canonical
    ``res_ffn`` slot so dumps / backward replay can read it, but the norm
    itself must follow HF's standalone RMSNorm path.
    """
    fresh_zeros = block._zeros(["B", "T", "d_model"], name="fresh_zero")
    block._register_activation(
        "res_ffn",
        ("B", "T", "d_model"),
        aliases=["residual_ffn", "res_in"],
        share_policy="when_recomputed",
    )
    residual = block._add(fresh_zeros, x, name="res_ffn")
    h = block.input_layernorm(residual)
    h = block.self_attn(h, position_ids)
    h = block.post_attn_layernorm(h)
    return residual, h


def _sandwich_mlp_phase(block, residual, h_post_attn):
    """Sandwich norm MLP phase.

    Opens with fused ``res_att = residual + h_post_attn; h = pre_ff_rmsnorm(res_att)``
    via RMSNorm's two-arg call — replaces a separate add + standalone rmsnorm,
    removes one HBM round-trip per block.
    """
    residual, h = block.pre_ff_layernorm(residual, h_post_attn)
    h = block.mlp(h)
    h = block.post_ff_layernorm(h)
    residual = block._add(residual, h, name="res_mlp")
    return residual


def _per_layer_input_phase(block, residual, per_layer_input):
    """Per-layer input gating: gate → gelu → mul(pli) → project → norm → residual."""
    # Register activation slots for PLI intermediates so backward replay can
    # allocate output buffers when recomputing the forward pass.
    pli_d = block.PLI_D
    block._register_activation("pli_h_flat", ("B * T", "C"), share_policy="always_recompute")
    block._register_activation("pli_flat", ("B * T", pli_d), share_policy="always_recompute")
    block._register_activation("pli_gate_out", ("B * T", pli_d), share_policy="always_recompute")
    block._register_activation("pli_gate_act", ("B * T", pli_d), share_policy="always_recompute")
    block._register_activation("pli_gated", ("B * T", pli_d), share_policy="always_recompute")
    block._register_activation("pli_proj_out", ("B * T", "C"), share_policy="always_recompute")
    block._register_activation("pli_proj_3d", ("B", "T", "C"), share_policy="always_recompute")

    h_flat = block._view(residual, [B * T, block.C], name="pli_h_flat")
    gate_out = block._matmul(h_flat, "pli_gate_weight", name="pli_gate_out")
    gate_act = block._gelu(gate_out, name="pli_gate_act")
    pli_flat = block._view(per_layer_input, [B * T, block.PLI_D], name="pli_flat")
    gated = block._mul(gate_act, pli_flat, name="pli_gated")
    proj_out = block._matmul(gated, "pli_proj_weight", name="pli_proj_out")
    proj_3d = block._view(proj_out, [B, T, block.C], name="pli_proj_3d")
    normed = block.pli_norm(proj_3d)
    residual = block._add(residual, normed, name="res_pli")
    return residual


def _finalize(block, residual):
    # Block's final output: residual scaled by layer_scalar, written to a
    # dedicated per-layer "h_out" slot. Historically we wrote to "mlp_down"
    # (reusing the MLP's persistent slot as a memory optimization), but
    # that created a name collision with the MLP's own down output — and
    # the autodiff's produced_by map keeps only the last writer, so the
    # MLP → post_ff_ln dependency was silently dropped from the backward
    # graph and every MLP LoRA gradient stayed zero. h_out has its own
    # per-layer buffer (simplified_acts::h_out) and is mapped in
    # try_get_tensor_fuzzy so the next block can read it as input.
    block._register_activation("h_out", ("B", "T", "d_model"), share_policy="per_layer", save=True)
    scaled = block._scale_by_param(residual, "layer_scalar")
    zero_copy = block._zeros(["B", "T", "d_model"], name="copy_zero")
    h_out = block._add(scaled, zero_copy, name="h_out")
    return h_out, h_out


def _register_frozen_and_pli_params(block):
    """Register layer_scalar + PLI gate/proj params if PLI is enabled."""
    block._register_param("layer_scalar", (1,), frozen=True, quantizable=False)
    if block.PLI_D > 0:
        block._register_param("pli_gate_weight", ("PLI_D", "C"), quantizable=False)
        block._register_param("pli_proj_weight", ("C", "PLI_D"), quantizable=False)


def _make_dims(
    block, d_model, head_size, num_query_heads, num_kv_heads, d_ff, max_seq, d_per_layer_input, rotary_dim=None
):
    """Set common derived dimension attributes on a block."""
    block.C = d_model
    block.D = head_size
    block.Hq = num_query_heads
    block.Hkv = num_kv_heads
    block.M = d_ff
    block.MaxSeq = max_seq
    block.AttnDim = num_query_heads * head_size
    block.PLI_D = d_per_layer_input
    if rotary_dim is not None:
        block.RotaryDim = rotary_dim
    else:
        block.RotaryDim = head_size


# ============================================================================
# Standard blocks (non-shared, own K/V projections)
# ============================================================================


class Gemma4SlidingBlock(nn.Block):
    """Sliding-window attention + GeLU-gated MLP. Optional per-layer input gating."""

    _name_remap_ = GEMMA4_BLOCK_NAME_REMAP
    schema = _gemma4_dense_schema("gemma4_sliding")

    def __init__(
        self,
        d_model,
        num_query_heads,
        num_kv_heads,
        head_size,
        d_ff,
        max_seq,
        sliding_window=512,
        d_per_layer_input=0,
        eps=1e-6,
    ):
        super().__init__()
        _make_dims(self, d_model, head_size, num_query_heads, num_kv_heads, d_ff, max_seq, d_per_layer_input)
        self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.input_layernorm = RMSNorm(d_model, eps=eps)
        self.self_attn = Gemma4Attention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            sliding_window=sliding_window,
            partial_rotary_factor=1.0,
            eps=eps,
        )
        self.post_attn_layernorm = RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm = RMSNorm(d_model, eps=eps)
        self.mlp = GenericMLP(
            d_model,
            d_ff,
            config=_GEMMA4_GELU_MLP_CONFIG,
        )
        self.post_ff_layernorm = RMSNorm(d_model, eps=eps)
        if d_per_layer_input > 0:
            self.pli_norm = RMSNorm(d_model, eps=eps)

    def forward(self, x, residual, position_ids, per_layer_input):
        _register_frozen_and_pli_params(self)
        residual, h_post_attn = _sandwich_attn_phase(self, x, residual, position_ids)
        residual = _sandwich_mlp_phase(self, residual, h_post_attn)
        if self.PLI_D > 0:
            residual = _per_layer_input_phase(self, residual, per_layer_input)
        return _finalize(self, residual)


# ============================================================================
# MoE block variants (MLP + experts in parallel)
# ============================================================================


def _sandwich_moe_mlp_phase(block, residual, h_post_attn):
    """MLP + MoE parallel phase for Gemma4 MoE blocks.

    HF: MLP output → post_ff_norm_1, MoE(pre_MLP_residual) → post_ff_norm_2,
    combined = norm_1 + norm_2, then post_ff_norm + residual add.

    Opens with fused ``res_att = residual + h_post_attn; h = pre_ff_rmsnorm(res_att)``
    (matches the dense-MLP variant's fusion). The MoE branch shares the
    same residual stream but runs its own norm, so we still reuse ``residual``
    (now the res_att output) below.
    """
    # Fused residual add + pre_ff_layernorm: produces res_att (= residual +
    # h_post_attn) and the pre_ff_normed output in one kernel.
    residual, h = block.pre_ff_layernorm(residual, h_post_attn)
    mlp_out = block.mlp(h)
    mlp_normed = block.post_ff_layernorm_1(mlp_out)

    # MoE path (routes on pre-MLP residual = res_att)
    residual_flat = block._view(residual, [B * T, block.C], name="moe_residual_flat")
    moe_input = block.pre_ff_layernorm_2(residual_flat)
    moe_out = block.moe(moe_input)
    moe_out_3d = block._view(moe_out, [B, T, block.C], name="moe_out_3d")
    moe_normed = block.post_ff_layernorm_2(moe_out_3d)

    # Combine + final norm + residual
    combined = block._add(mlp_normed, moe_normed, name="moe_mlp_combined")
    combined_normed = block.post_ff_layernorm(combined)
    residual = block._add(residual, combined_normed, name="res_mlp")
    return residual


class Gemma4SlidingMoEBlock(nn.Block):
    """Sliding attention + parallel MLP/MoE. For 26B-A4B sliding layers."""

    _name_remap_ = GEMMA4_BLOCK_NAME_REMAP
    schema = _gemma4_moe_schema("gemma4_sliding_moe")

    def __init__(
        self,
        d_model,
        num_query_heads,
        num_kv_heads,
        head_size,
        d_ff,
        max_seq,
        sliding_window=1024,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=704,
        eps=1e-6,
        ep_size=1,
    ):
        super().__init__()
        _make_dims(self, d_model, head_size, num_query_heads, num_kv_heads, d_ff, max_seq, 0)
        self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.E = num_experts
        self.K_exp = num_experts_per_tok

        self.input_layernorm = RMSNorm(d_model, eps=eps)
        self.self_attn = Gemma4Attention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            sliding_window=sliding_window,
            partial_rotary_factor=1.0,
            eps=eps,
        )
        self.post_attn_layernorm = RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm = RMSNorm(d_model, eps=eps)
        self.mlp = GenericMLP(
            d_model,
            d_ff,
            config=_GEMMA4_GELU_MLP_CONFIG,
        )
        self.post_ff_layernorm = RMSNorm(d_model, eps=eps)
        self.post_ff_layernorm_1 = RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm_2 = RMSNorm(d_model, eps=eps)
        self.post_ff_layernorm_2 = RMSNorm(d_model, eps=eps)
        self.moe = Gemma4MoEExperts(
            d_model,
            moe_intermediate_size,
            num_experts,
            num_experts_per_tok,
            eps=eps,
            ep_size=ep_size,
        )

    def forward(self, x, residual, position_ids, per_layer_input):
        self._register_param("layer_scalar", (1,), frozen=True, quantizable=False)
        residual, h_post_attn = _sandwich_attn_phase(self, x, residual, position_ids)
        residual = _sandwich_moe_mlp_phase(self, residual, h_post_attn)
        return _finalize(self, residual)


class Gemma4FullMoEBlock(nn.Block):
    """Full attention (k_eq_v) + parallel MLP/MoE. For 26B-A4B full layers."""

    _name_remap_ = GEMMA4_BLOCK_NAME_REMAP
    schema = _gemma4_moe_schema("gemma4_full_moe")

    def __init__(
        self,
        d_model,
        num_query_heads,
        num_kv_heads,
        head_size,
        d_ff,
        max_seq,
        partial_rotary_factor=0.25,
        proportional_rope=True,
        k_eq_v=True,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=704,
        eps=1e-6,
        ep_size=1,
    ):
        super().__init__()
        rotary_dim = head_size if proportional_rope else _resolve_rotary_dim(head_size, partial_rotary_factor)
        _make_dims(self, d_model, head_size, num_query_heads, num_kv_heads, d_ff, max_seq, 0, rotary_dim=rotary_dim)
        if k_eq_v:
            self.QKV = (num_query_heads + num_kv_heads) * head_size
        else:
            self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.E = num_experts
        self.K_exp = num_experts_per_tok

        self.input_layernorm = RMSNorm(d_model, eps=eps)
        self.self_attn = Gemma4Attention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            partial_rotary_factor=partial_rotary_factor,
            proportional_rope=proportional_rope,
            k_eq_v=k_eq_v,
            eps=eps,
        )
        self.post_attn_layernorm = RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm = RMSNorm(d_model, eps=eps)
        self.mlp = GenericMLP(
            d_model,
            d_ff,
            config=_GEMMA4_GELU_MLP_CONFIG,
        )
        self.post_ff_layernorm = RMSNorm(d_model, eps=eps)
        self.post_ff_layernorm_1 = RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm_2 = RMSNorm(d_model, eps=eps)
        self.post_ff_layernorm_2 = RMSNorm(d_model, eps=eps)
        self.moe = Gemma4MoEExperts(
            d_model,
            moe_intermediate_size,
            num_experts,
            num_experts_per_tok,
            eps=eps,
            ep_size=ep_size,
        )

    def forward(self, x, residual, position_ids, per_layer_input):
        self._register_param("layer_scalar", (1,), frozen=True, quantizable=False)
        residual, h_post_attn = _sandwich_attn_phase(self, x, residual, position_ids)
        residual = _sandwich_moe_mlp_phase(self, residual, h_post_attn)
        return _finalize(self, residual)


class Gemma4FullBlock(nn.Block):
    """Full attention (optional k_eq_v) + GeLU-gated MLP. Optional per-layer input gating."""

    _name_remap_ = GEMMA4_BLOCK_NAME_REMAP
    schema = _gemma4_dense_schema("gemma4_full")

    def __init__(
        self,
        d_model,
        num_query_heads,
        num_kv_heads,
        head_size,
        d_ff,
        max_seq,
        partial_rotary_factor=0.25,
        proportional_rope=True,
        k_eq_v=False,
        d_per_layer_input=0,
        eps=1e-6,
    ):
        super().__init__()
        rotary_dim = head_size if proportional_rope else _resolve_rotary_dim(head_size, partial_rotary_factor)
        _make_dims(
            self,
            d_model,
            head_size,
            num_query_heads,
            num_kv_heads,
            d_ff,
            max_seq,
            d_per_layer_input,
            rotary_dim=rotary_dim,
        )
        self.k_eq_v = k_eq_v
        if k_eq_v:
            self.QKV = (num_query_heads + num_kv_heads) * head_size
        else:
            self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.input_layernorm = RMSNorm(d_model, eps=eps)
        self.self_attn = Gemma4Attention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            partial_rotary_factor=partial_rotary_factor,
            proportional_rope=proportional_rope,
            k_eq_v=k_eq_v,
            eps=eps,
        )
        self.post_attn_layernorm = RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm = RMSNorm(d_model, eps=eps)
        self.mlp = GenericMLP(
            d_model,
            d_ff,
            config=_GEMMA4_GELU_MLP_CONFIG,
        )
        self.post_ff_layernorm = RMSNorm(d_model, eps=eps)
        if d_per_layer_input > 0:
            self.pli_norm = RMSNorm(d_model, eps=eps)

    def forward(self, x, residual, position_ids, per_layer_input):
        _register_frozen_and_pli_params(self)
        residual, h_post_attn = _sandwich_attn_phase(self, x, residual, position_ids)
        residual = _sandwich_mlp_phase(self, residual, h_post_attn)
        if self.PLI_D > 0:
            residual = _per_layer_input_phase(self, residual, per_layer_input)
        return _finalize(self, residual)


# ============================================================================
# Shared-KV blocks (Q-only attention, reads K/V from source layer)
# ============================================================================


def _sandwich_shared_attn_phase(block, x, residual, position_ids, kv_source):
    """Shared-KV attention: input_ln -> Q-only attn(kv_source) -> post_attn_ln.

    Like _sandwich_attn_phase, leaves the res_att add to the caller so
    _sandwich_mlp_phase can fuse it with pre_ff_layernorm. Returns
    (residual, h_post_attn).
    """
    fresh_zeros = block._zeros(["B", "T", "d_model"], name="fresh_zero")
    block._register_activation(
        "res_ffn",
        ("B", "T", "d_model"),
        aliases=["residual_ffn", "res_in"],
        share_policy="when_recomputed",
    )
    residual = block._add(fresh_zeros, x, name="res_ffn")
    h = block.input_layernorm(residual)
    h = block.self_attn(h, position_ids, kv_source)
    h = block.post_attn_layernorm(h)
    return residual, h


class Gemma4SharedKVBlock(nn.Block):
    """Shared-KV block: Q-only attention + MLP (optional double-wide) + PLI.

    Used by layers that share K,V from an earlier (source) layer.
    Takes ``kv_source`` as an additional input -- the source layer's
    packed QKV-rope tensor from which K,V are extracted.
    """

    _name_remap_ = GEMMA4_BLOCK_NAME_REMAP
    schema = _gemma4_dense_schema("gemma4_shared_kv", shared_kv=True)

    def __init__(
        self,
        d_model,
        num_query_heads,
        num_kv_heads,
        head_size,
        d_ff,
        max_seq,
        sliding_window=None,
        partial_rotary_factor=0.25,
        proportional_rope=None,
        d_per_layer_input=256,
        use_double_wide_mlp=False,
        eps=1e-6,
    ):
        super().__init__()
        if proportional_rope is None:
            proportional_rope = sliding_window is None and partial_rotary_factor < 1.0
        rotary_dim = head_size if proportional_rope else _resolve_rotary_dim(head_size, partial_rotary_factor)
        effective_d_ff = d_ff * 2 if use_double_wide_mlp else d_ff
        _make_dims(
            self,
            d_model,
            head_size,
            num_query_heads,
            num_kv_heads,
            effective_d_ff,
            max_seq,
            d_per_layer_input,
            rotary_dim=rotary_dim,
        )
        self.QDim = num_query_heads * head_size

        self.input_layernorm = RMSNorm(d_model, eps=eps)
        self.self_attn = Gemma4SharedKVAttention(
            d_model,
            num_query_heads,
            num_kv_heads,
            head_size,
            max_seq,
            sliding_window=sliding_window,
            partial_rotary_factor=partial_rotary_factor,
            proportional_rope=proportional_rope,
            eps=eps,
        )
        self.post_attn_layernorm = RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm = RMSNorm(d_model, eps=eps)
        self.mlp = GenericMLP(
            d_model,
            effective_d_ff,
            config=_GEMMA4_GELU_MLP_CONFIG,
        )
        self.post_ff_layernorm = RMSNorm(d_model, eps=eps)
        if d_per_layer_input > 0:
            self.pli_norm = RMSNorm(d_model, eps=eps)

    def forward(self, x, residual, position_ids, per_layer_input, kv_source):
        _register_frozen_and_pli_params(self)
        residual, h_post_attn = _sandwich_shared_attn_phase(
            self,
            x,
            residual,
            position_ids,
            kv_source,
        )
        residual = _sandwich_mlp_phase(self, residual, h_post_attn)
        if self.PLI_D > 0:
            residual = _per_layer_input_phase(self, residual, per_layer_input)
        return _finalize(self, residual)
