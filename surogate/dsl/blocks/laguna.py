"""Laguna transformer blocks (Afmoe-style softplus-gated attention + MoE).

Laguna (``LagunaForCausalLM``) uses a standard pre-norm decoder layout:
  input_layernorm -> attention -> +residual
  post_attention_layernorm -> MLP (dense SwiGLU or sparse MoE + shared expert) -> +residual

Block variants (attention type x MLP type):
  - LagunaDenseBlock: attention + dense SwiGLU MLP (fused up+gate)
  - LagunaSparseBlock: attention + MoE (sigmoid router + correction bias,
    routed scaling) + ungated shared expert

Both variants take ``sliding_window`` / ``partial_rotary_factor`` /
``num_query_heads`` per instance, so the model wires one block config per
(attention type, mlp type) combination present in the checkpoint.
"""

from __future__ import annotations

from .. import nn
from ..block_schema import BlockSchema, DistributionDecl, EPTopology, RoutingSchema, SlotDecl, StreamingHint
from ..dim import B, T
from ..modules import LagunaAttention, LagunaMoEExperts, MoESharedExpert, RMSNorm
from ..specs import LoRATarget


_LAGUNA_ATTN_REMAP: dict[str, str] = {
    # --- attn_norm (RMSNorm) -> ln1 / res_ffn ---
    "attn_norm_weight": "ln1_weight",
    "attn_norm_res": "res_ffn",
    "attn_norm_y": "ln1",
    "attn_norm_rstd": "ln1_rstd",
    # --- self_attn (LagunaAttention) -> strip prefix ---
    "self_attn_q_proj_weight": "q_proj_weight",
    "self_attn_k_proj_weight": "k_proj_weight",
    "self_attn_v_proj_weight": "v_proj_weight",
    "self_attn_g_proj_weight": "g_proj_weight",
    "self_attn_out_weight": "out_weight",
    "self_attn_q_norm_weight": "q_norm_weight",
    "self_attn_k_norm_weight": "k_norm_weight",
    "self_attn_rope_freqs": "rope_freqs",
    "self_attn_x_flat": "x_flat",
    "self_attn_q_proj": "q_proj",
    "self_attn_k_proj": "k_proj",
    "self_attn_v_proj": "v_proj",
    "self_attn_q": "q",
    "self_attn_k": "k",
    "self_attn_v": "v",
    "self_attn_qkv": "qkv",
    "self_attn_qkv_rope": "qkv_rope",
    "self_attn_q_rstd": "q_rstd",
    "self_attn_k_rstd": "k_rstd",
    "self_attn_att": "att",
    "self_attn_att_flat": "att_flat",
    "self_attn_att_2d": "att_2d",
    "self_attn_lse": "lse",
    "self_attn_gate_proj_out": "gate_proj_out",
    "self_attn_gate_act": "gate_act",
    "self_attn_gate_2d": "gate_2d",
    "self_attn_att_out": "att_out",
    "self_attn_att_out_flat": "att_out_flat",
    # --- mlp_norm (RMSNorm) -> ln2 / res_att ---
    "mlp_norm_weight": "ln2_weight",
    "mlp_norm_res": "res_att",
    "mlp_norm_y": "ln2",
    "mlp_norm_rstd": "ln2_rstd",
}

LAGUNA_DENSE_BLOCK_NAME_REMAP: dict[str, str] = {
    **_LAGUNA_ATTN_REMAP,
    # mlp params/activations are registered directly with canonical mlp_* names
}

LAGUNA_SPARSE_BLOCK_NAME_REMAP: dict[str, str] = {
    **_LAGUNA_ATTN_REMAP,
    # --- moe (LagunaMoEExperts) -> strip moe_ prefix ---
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
    # shared_expert (MoESharedExpert): prefixed names are already canonical
}


def _laguna_dense_schema(block_family: str) -> BlockSchema:
    return BlockSchema(
        slots=(
            SlotDecl("q_proj_weight", kind="param", shape=("AttnDim", "C")),
            SlotDecl("k_proj_weight", kind="param", shape=("KVDim", "C")),
            SlotDecl("v_proj_weight", kind="param", shape=("KVDim", "C")),
            SlotDecl("g_proj_weight", kind="param", shape=("GateDim", "C")),
            SlotDecl("out_weight", kind="param", shape=("C", "AttnDim")),
            SlotDecl("mlp_up_weight", kind="param", shape=("2M", "C"), residency="auto"),
            SlotDecl("mlp_down_weight", kind="param", shape=("C", "M"), residency="auto"),
            SlotDecl("res_att", shape=("B", "T", "C")),
            SlotDecl("qkv_rope", shape=("B", "T", "QKV"), save_for_backward=True),
        ),
        attrs={"block_family": block_family},
    )


def _laguna_sparse_schema(block_family: str) -> BlockSchema:
    return BlockSchema(
        slots=(
            SlotDecl("q_proj_weight", kind="param", shape=("AttnDim", "C")),
            SlotDecl("k_proj_weight", kind="param", shape=("KVDim", "C")),
            SlotDecl("v_proj_weight", kind="param", shape=("KVDim", "C")),
            SlotDecl("g_proj_weight", kind="param", shape=("GateDim", "C")),
            SlotDecl("out_weight", kind="param", shape=("C", "AttnDim")),
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
            SlotDecl("res_att", shape=("B", "T", "C")),
            SlotDecl("qkv_rope", shape=("B", "T", "QKV"), save_for_backward=True),
        ),
        routing=RoutingSchema(
            kind="topk_sigmoid",
            topk="num_experts_per_tok",
            # HF Laguna always normalizes routing weights (no config knob).
            norm_topk_prob=True,
            scoring_bias=True,
            shared_experts="shared_expert_intermediate",
        ),
        ep_topology=EPTopology(ep_size_param="ep_size"),
        attrs={"block_family": block_family},
    )


def _make_laguna_attn(block, *, d_model, num_query_heads, num_kv_heads, head_size, max_seq,
                      sliding_window, partial_rotary_factor, gate_per_head, eps):
    """Set derived dims and construct the shared attention/norm modules."""
    # Numeric dims — Laguna is hybrid with per-layer-type query head counts.
    block.C = d_model
    block.D = head_size
    block.Hq = num_query_heads
    block.Hkv = num_kv_heads
    block.MaxSeq = max_seq
    block.AttnDim = num_query_heads * head_size
    block.KVDim = num_kv_heads * head_size
    block.QKV = (num_query_heads + 2 * num_kv_heads) * head_size
    block.GateDim = num_query_heads if gate_per_head else num_query_heads * head_size

    block.attn_norm = RMSNorm(d_model, eps=eps)
    block.self_attn = LagunaAttention(
        d_model,
        num_query_heads,
        num_kv_heads,
        head_size,
        max_seq,
        sliding_window=sliding_window,
        partial_rotary_factor=partial_rotary_factor,
        gate_per_head=gate_per_head,
        eps=eps,
    )
    block.mlp_norm = RMSNorm(d_model, eps=eps)


class LagunaDenseBlock(nn.Block):
    """Laguna decoder block with a dense SwiGLU MLP (e.g. layer 0)."""

    _name_remap_ = LAGUNA_DENSE_BLOCK_NAME_REMAP
    schema = _laguna_dense_schema("laguna_dense")

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        d_ff: int,
        max_seq: int,
        sliding_window: int | None = None,
        partial_rotary_factor: float = 1.0,
        gate_per_head: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_ff = d_ff
        self.M = d_ff
        _make_laguna_attn(
            self,
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            max_seq=max_seq,
            sliding_window=sliding_window,
            partial_rotary_factor=partial_rotary_factor,
            gate_per_head=gate_per_head,
            eps=eps,
        )

    def forward(self, x, residual, position_ids):
        residual, h = self.attn_norm(residual, x)
        h = self.self_attn(h, position_ids)
        residual, h = self.mlp_norm(residual, h)

        # Dense SwiGLU MLP with fused up+gate weight ([up; gate] rows — matches
        # the swiglu kernel and the fuse(up_proj, gate_proj) HF mapping).
        # Registered inline with numeric dims: the dense d_ff (e.g. 8192)
        # differs from the sparse blocks' expert d_ff (e.g. 512), so symbolic
        # M/MUp would collide across block types.
        _ff = self.d_ff
        self._register_param(
            "mlp_up_weight",
            (2 * _ff, "C"),
            lora_targets=[
                LoRATarget(name="up", offset=0, size=_ff),
                LoRATarget(name="gate", offset=_ff, size=_ff),
            ],
        )
        self._register_param(
            "mlp_down_weight",
            ("C", _ff),
            lora_targets=[LoRATarget(name="down", size=self.C)],
        )
        self._register_activation(
            "mlp_up",
            ("B", "T", 2 * _ff),
            aliases=["mlp_up_flat"],
            share_policy="when_recomputed",
        )
        self._register_activation(
            "swiglu",
            ("B", "T", _ff),
            aliases=["swiglu_flat"],
            share_policy="when_recomputed",
        )
        self._register_activation(
            "mlp_down",
            ("B", "T", "C"),
            aliases=["mlp_down_flat"],
            share_policy="per_layer",
        )

        h_flat = self._view(h, [B * T, self.C], name="mlp_x_flat")
        up = self._matmul(h_flat, "mlp_up_weight", name="mlp_up_flat")
        act = self._swiglu(up, name="swiglu_flat")
        down = self._matmul(act, "mlp_down_weight", name="mlp_down_flat")
        out = self._view(down, [B, T, self.C], name="mlp_down")
        return out, residual


class LagunaSparseBlock(nn.Block):
    """Laguna decoder block with sparse MoE + shared expert."""

    _name_remap_ = LAGUNA_SPARSE_BLOCK_NAME_REMAP
    schema = _laguna_sparse_schema("laguna_sparse_moe")

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
        routed_scaling_factor: float = 1.0,
        sliding_window: int | None = None,
        partial_rotary_factor: float = 1.0,
        gate_per_head: bool = True,
        eps: float = 1e-6,
        ep_size: int = 1,
    ):
        super().__init__()
        self.d_ff = d_ff
        self.M = d_ff
        self.MUp = 2 * d_ff
        self.E = num_experts
        self.K = num_experts_per_tok
        self.SharedM = shared_expert_intermediate
        self.shared_expert_intermediate = shared_expert_intermediate
        _make_laguna_attn(
            self,
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            max_seq=max_seq,
            sliding_window=sliding_window,
            partial_rotary_factor=partial_rotary_factor,
            gate_per_head=gate_per_head,
            eps=eps,
        )
        self.moe = LagunaMoEExperts(
            d_model,
            d_ff,
            num_experts,
            num_experts_per_tok,
            routed_scaling_factor=routed_scaling_factor,
            ep_size=ep_size,
        )
        self.shared_expert = MoESharedExpert(d_model, shared_expert_intermediate)

    def forward(self, x, residual, position_ids):
        residual, h = self.attn_norm(residual, x)
        h = self.self_attn(h, position_ids)
        residual, h = self.mlp_norm(residual, h)

        h_flat = self._view(h, [B * T, self.C], name="ln2_flat")
        # Routed experts (routed_scaling_factor is folded into the routing
        # weights inside moe_topk).
        moe_out = self.moe(h_flat)
        # Ungated shared expert added on top of the scaled routed output.
        shared_out = self.shared_expert(h_flat)
        moe_out = self._add(moe_out, shared_out, name="moe_combined")

        self._register_activation(
            "mlp_down",
            ("B", "T", "C"),
            aliases=["mlp_down_flat"],
            share_policy="per_layer",
            description="MoE + shared expert output (block output)",
        )
        out = self._view(moe_out, [B, T, self.C], name="mlp_down")
        return out, residual
