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
from ..nn import GEMMA4_BLOCK_NAME_REMAP
from ..dim import B, T


# ============================================================================
# Helpers shared across block types
# ============================================================================

def _sandwich_attn_phase(block, x, residual, position_ids):
    """Sandwich norm attention phase: input_ln → attn → post_attn_ln → residual add.

    Gemma4 uses zero residual between blocks (state flows through x, not residual).
    We create fresh zeros to avoid stale data from cross-layer slot reuse.
    """
    fresh_zeros = block._zeros(["B", "T", "d_model"], name="fresh_zero")
    residual, h = block.input_layernorm(fresh_zeros, x)
    h = block.self_attn(h, position_ids)
    h = block.post_attn_layernorm(h)
    residual = block._add(residual, h, name="res_attn")
    return residual


def _sandwich_mlp_phase(block, residual):
    """Sandwich norm MLP phase: pre_ff_ln → mlp → post_ff_ln → residual add."""
    h = block.pre_ff_layernorm(residual)
    h = block.mlp(h)
    h = block.post_ff_layernorm(h)
    residual = block._add(residual, h, name="res_mlp")
    return residual


def _per_layer_input_phase(block, residual, per_layer_input):
    """Per-layer input gating: gate → gelu → mul(pli) → project → norm → residual."""
    # Register activation slots for PLI intermediates so backward replay can
    # allocate output buffers when recomputing the forward pass.
    pli_d = block.PLI_D
    block._register_activation(
        "pli_gate_out", ("B * T", pli_d), share_policy="always_recompute")
    block._register_activation(
        "pli_gate_act", ("B * T", pli_d), share_policy="always_recompute")
    block._register_activation(
        "pli_gated", ("B * T", pli_d), share_policy="always_recompute")
    block._register_activation(
        "pli_proj_out", ("B * T", "C"), share_policy="always_recompute")

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
    """Apply layer_scalar and return (state, zeros).

    The block output is the scaled hidden state. The residual is always zero
    (Gemma4 doesn't use a running residual sum across blocks).
    We return the output as BOTH values: (h_out, h_out). The next block's
    input_layernorm receives (residual=h_out, x=h_out) and computes
    new_residual = h_out + h_out. We compensate by halving the result
    via the layer_scalar or adjusting the fused_residual_rmsnorm behavior.

    Actually — simpler: return the state in the first position and a zero
    in the second. Use the names that the inliner maps to blocks[N].* for
    persistence: "att_out" and "mlp_down" (both are C-dimensional slots
    in the simplified activations that are available at layer boundaries).
    """
    scaled = block._scale_by_param(residual, "layer_scalar")
    # Copy the scaled output into the mlp_down slot (persists across layer
    # boundaries). Must use _add(scaled, 0) not _view to trigger a real copy —
    # _view creates an alias that dangles when the temp stack is restored.
    zero_copy = block._zeros(["B", "T", "d_model"], name="copy_zero")
    h_out = block._add(scaled, zero_copy, name="mlp_down")
    return h_out, h_out


def _register_frozen_and_pli_params(block):
    """Register layer_scalar + PLI gate/proj params if PLI is enabled."""
    block._register_param("layer_scalar", (1,), frozen=True, quantizable=False)
    if block.PLI_D > 0:
        block._register_param("pli_gate_weight", ("PLI_D", "C"), quantizable=False)
        block._register_param("pli_proj_weight", ("C", "PLI_D"), quantizable=False)


def _make_dims(block, d_model, head_size, num_query_heads, num_kv_heads,
               d_ff, max_seq, d_per_layer_input, rotary_dim=None):
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

    def __init__(self, d_model, num_query_heads, num_kv_heads, head_size,
                 d_ff, max_seq, sliding_window=512, d_per_layer_input=0, eps=1e-6):
        super().__init__()
        _make_dims(self, d_model, head_size, num_query_heads, num_kv_heads,
                   d_ff, max_seq, d_per_layer_input)
        self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.input_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.self_attn = nn.Gemma4Attention(
            d_model, num_query_heads, num_kv_heads, head_size, max_seq,
            sliding_window=sliding_window, partial_rotary_factor=1.0, eps=eps,
        )
        self.post_attn_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.mlp = nn.GatedMLP(d_model, d_ff, activation="gelu")
        self.post_ff_layernorm = nn.RMSNorm(d_model, eps=eps)
        if d_per_layer_input > 0:
            self.pli_norm = nn.RMSNorm(d_model, eps=eps)

    def forward(self, x, residual, position_ids, per_layer_input):
        _register_frozen_and_pli_params(self)
        residual = _sandwich_attn_phase(self, x, residual, position_ids)
        residual = _sandwich_mlp_phase(self, residual)
        if self.PLI_D > 0:
            residual = _per_layer_input_phase(self, residual, per_layer_input)
        return _finalize(self, residual)


# ============================================================================
# MoE block variants (MLP + experts in parallel)
# ============================================================================

def _sandwich_moe_mlp_phase(block, residual):
    """MLP + MoE parallel phase for Gemma4 MoE blocks.

    HF: MLP output → post_ff_norm_1, MoE(pre_MLP_residual) → post_ff_norm_2,
    combined = norm_1 + norm_2, then post_ff_norm + residual add.
    """
    # MLP path
    h = block.pre_ff_layernorm(residual)
    mlp_out = block.mlp(h)
    mlp_normed = block.post_ff_layernorm_1(mlp_out)

    # MoE path (routes on pre-MLP residual)
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

    def __init__(self, d_model, num_query_heads, num_kv_heads, head_size,
                 d_ff, max_seq, sliding_window=1024,
                 num_experts=128, num_experts_per_tok=8,
                 moe_intermediate_size=704, eps=1e-6, ep_size=1):
        super().__init__()
        _make_dims(self, d_model, head_size, num_query_heads, num_kv_heads,
                   d_ff, max_seq, 0)
        self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.E = num_experts
        self.K_exp = num_experts_per_tok

        self.input_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.self_attn = nn.Gemma4Attention(
            d_model, num_query_heads, num_kv_heads, head_size, max_seq,
            sliding_window=sliding_window, partial_rotary_factor=1.0, eps=eps,
        )
        self.post_attn_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.mlp = nn.GatedMLP(d_model, d_ff, activation="gelu")
        self.post_ff_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.post_ff_layernorm_1 = nn.RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm_2 = nn.RMSNorm(d_model, eps=eps)
        self.post_ff_layernorm_2 = nn.RMSNorm(d_model, eps=eps)
        self.moe = nn.Gemma4MoEExperts(
            d_model, moe_intermediate_size, num_experts, num_experts_per_tok,
            eps=eps, ep_size=ep_size,
        )

    def forward(self, x, residual, position_ids, per_layer_input):
        self._register_param("layer_scalar", (1,), frozen=True, quantizable=False)
        residual = _sandwich_attn_phase(self, x, residual, position_ids)
        residual = _sandwich_moe_mlp_phase(self, residual)
        return _finalize(self, residual)


class Gemma4FullMoEBlock(nn.Block):
    """Full attention (k_eq_v) + parallel MLP/MoE. For 26B-A4B full layers."""

    _name_remap_ = GEMMA4_BLOCK_NAME_REMAP

    def __init__(self, d_model, num_query_heads, num_kv_heads, head_size,
                 d_ff, max_seq, partial_rotary_factor=0.25, k_eq_v=True,
                 num_experts=128, num_experts_per_tok=8,
                 moe_intermediate_size=704, eps=1e-6, ep_size=1):
        super().__init__()
        rotary_dim = nn._resolve_rotary_dim(head_size, partial_rotary_factor)
        _make_dims(self, d_model, head_size, num_query_heads, num_kv_heads,
                   d_ff, max_seq, 0, rotary_dim=rotary_dim)
        if k_eq_v:
            self.QKV = (num_query_heads + num_kv_heads) * head_size
        else:
            self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.E = num_experts
        self.K_exp = num_experts_per_tok

        self.input_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.self_attn = nn.Gemma4Attention(
            d_model, num_query_heads, num_kv_heads, head_size, max_seq,
            partial_rotary_factor=partial_rotary_factor, k_eq_v=k_eq_v, eps=eps,
        )
        self.post_attn_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.mlp = nn.GatedMLP(d_model, d_ff, activation="gelu")
        self.post_ff_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.post_ff_layernorm_1 = nn.RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm_2 = nn.RMSNorm(d_model, eps=eps)
        self.post_ff_layernorm_2 = nn.RMSNorm(d_model, eps=eps)
        self.moe = nn.Gemma4MoEExperts(
            d_model, moe_intermediate_size, num_experts, num_experts_per_tok,
            eps=eps, ep_size=ep_size,
        )

    def forward(self, x, residual, position_ids, per_layer_input):
        self._register_param("layer_scalar", (1,), frozen=True, quantizable=False)
        residual = _sandwich_attn_phase(self, x, residual, position_ids)
        residual = _sandwich_moe_mlp_phase(self, residual)
        return _finalize(self, residual)


class Gemma4FullBlock(nn.Block):
    """Full attention (optional k_eq_v) + GeLU-gated MLP. Optional per-layer input gating."""

    _name_remap_ = GEMMA4_BLOCK_NAME_REMAP

    def __init__(self, d_model, num_query_heads, num_kv_heads, head_size,
                 d_ff, max_seq, partial_rotary_factor=0.25,
                 k_eq_v=False, d_per_layer_input=0, eps=1e-6):
        super().__init__()
        rotary_dim = nn._resolve_rotary_dim(head_size, partial_rotary_factor)
        _make_dims(self, d_model, head_size, num_query_heads, num_kv_heads,
                   d_ff, max_seq, d_per_layer_input, rotary_dim=rotary_dim)
        self.k_eq_v = k_eq_v
        if k_eq_v:
            self.QKV = (num_query_heads + num_kv_heads) * head_size
        else:
            self.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
        self.input_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.self_attn = nn.Gemma4Attention(
            d_model, num_query_heads, num_kv_heads, head_size, max_seq,
            partial_rotary_factor=partial_rotary_factor, k_eq_v=k_eq_v, eps=eps,
        )
        self.post_attn_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.mlp = nn.GatedMLP(d_model, d_ff, activation="gelu")
        self.post_ff_layernorm = nn.RMSNorm(d_model, eps=eps)
        if d_per_layer_input > 0:
            self.pli_norm = nn.RMSNorm(d_model, eps=eps)

    def forward(self, x, residual, position_ids, per_layer_input):
        _register_frozen_and_pli_params(self)
        residual = _sandwich_attn_phase(self, x, residual, position_ids)
        residual = _sandwich_mlp_phase(self, residual)
        if self.PLI_D > 0:
            residual = _per_layer_input_phase(self, residual, per_layer_input)
        return _finalize(self, residual)


# ============================================================================
# Shared-KV blocks (Q-only attention, reads K/V from source layer)
# ============================================================================

def _sandwich_shared_attn_phase(block, x, residual, position_ids, kv_source):
    """Shared-KV attention: input_ln -> Q-only attn(kv_source) -> post_attn_ln -> residual."""
    fresh_zeros = block._zeros(["B", "T", "d_model"], name="fresh_zero")
    residual, h = block.input_layernorm(fresh_zeros, x)
    h = block.self_attn(h, position_ids, kv_source)
    h = block.post_attn_layernorm(h)
    residual = block._add(residual, h, name="res_attn")
    return residual


class Gemma4SharedKVBlock(nn.Block):
    """Shared-KV block: Q-only attention + MLP (optional double-wide) + PLI.

    Used by layers that share K,V from an earlier (source) layer.
    Takes ``kv_source`` as an additional input -- the source layer's
    packed QKV-rope tensor from which K,V are extracted.
    """

    _name_remap_ = GEMMA4_BLOCK_NAME_REMAP

    def __init__(self, d_model, num_query_heads, num_kv_heads, head_size,
                 d_ff, max_seq, partial_rotary_factor=0.25,
                 d_per_layer_input=256, use_double_wide_mlp=False, eps=1e-6):
        super().__init__()
        rotary_dim = nn._resolve_rotary_dim(head_size, partial_rotary_factor)
        effective_d_ff = d_ff * 2 if use_double_wide_mlp else d_ff
        _make_dims(self, d_model, head_size, num_query_heads, num_kv_heads,
                   effective_d_ff, max_seq, d_per_layer_input, rotary_dim=rotary_dim)
        self.QDim = num_query_heads * head_size

        self.input_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.self_attn = nn.Gemma4SharedKVAttention(
            d_model, num_query_heads, num_kv_heads, head_size, max_seq,
            partial_rotary_factor=partial_rotary_factor, eps=eps,
        )
        self.post_attn_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.pre_ff_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.mlp = nn.GatedMLP(d_model, effective_d_ff, activation="gelu")
        self.post_ff_layernorm = nn.RMSNorm(d_model, eps=eps)
        if d_per_layer_input > 0:
            self.pli_norm = nn.RMSNorm(d_model, eps=eps)

    def forward(self, x, residual, position_ids, per_layer_input, kv_source):
        _register_frozen_and_pli_params(self)
        residual = _sandwich_shared_attn_phase(
            self, x, residual, position_ids, kv_source,
        )
        residual = _sandwich_mlp_phase(self, residual)
        if self.PLI_D > 0:
            residual = _per_layer_input_phase(self, residual, per_layer_input)
        return _finalize(self, residual)
