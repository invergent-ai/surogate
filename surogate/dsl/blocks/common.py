"""Name-remap dictionaries shared across multiple block / model files.

Per-architecture remaps (``GPT_OSS_*``, ``GEMMA4_*``, ``NEMOTRON_*``,
``QWEN3_5_*``) live alongside their block definitions. The dicts here
cover the generic dense / MoE / VL block shapes and the shared
``STANDARD_MODEL_NAME_REMAP`` / ``VL_MODEL_NAME_REMAP`` model-level
scopes.
"""

from __future__ import annotations

DENSE_BLOCK_NAME_REMAP: dict[str, str] = {
    # --- attn_norm (RMSNorm) -> ln1 / res_ffn ---
    "attn_norm_weight": "ln1_weight",
    "attn_norm_res": "res_ffn",
    "attn_norm_y": "ln1",
    "attn_norm_rstd": "ln1_rstd",
    # --- self_attn (Attention) -> strip prefix ---
    "self_attn_qkv_weight": "qkv_weight",
    "self_attn_qkv_bias": "qkv_bias",
    "self_attn_out_weight": "out_weight",
    "self_attn_out_bias": "out_bias",
    "self_attn_q_norm_weight": "q_norm_weight",
    "self_attn_k_norm_weight": "k_norm_weight",
    "self_attn_rope_freqs": "rope_freqs",
    "self_attn_qkv": "qkv",
    "self_attn_qkv_flat": "qkv_flat",
    "self_attn_qkv_biased": "qkv_biased",
    "self_attn_qkv_rope": "qkv_rope",
    "self_attn_q_rstd": "q_rstd",
    "self_attn_k_rstd": "k_rstd",
    "self_attn_att": "att",
    "self_attn_att_flat": "att_flat",
    "self_attn_attn": "attn",
    "self_attn_lse": "lse",
    "self_attn_att_out": "att_out",
    "self_attn_att_out_flat": "att_out_flat",
    "self_attn_x_flat": "x_flat",
    # --- mlp_norm (RMSNorm) -> ln2 / res_att ---
    "mlp_norm_weight": "ln2_weight",
    "mlp_norm_res": "res_att",
    "mlp_norm_y": "ln2",
    "mlp_norm_rstd": "ln2_rstd",
    # --- mlp (SwiGLUMLP) ---
    # mlp_up_weight, mlp_down_weight, mlp_up, mlp_down, mlp_up_flat,
    # mlp_down_flat are already correct (mlp_ prefix matches canonical names)
    "mlp_act": "swiglu",
    "mlp_act_flat": "swiglu_flat",
    "mlp_x_flat": "mlp_x_flat",  # intermediate, kept as-is
}

MOE_BLOCK_NAME_REMAP: dict[str, str] = {
    # Inherit all attn_norm and self_attn mappings
    **{k: v for k, v in DENSE_BLOCK_NAME_REMAP.items() if k.startswith(("attn_norm_", "self_attn_", "mlp_norm_"))},
    # --- moe (MoEExpertsGated) -> strip moe_ prefix ---
    # Params
    "moe_router_weight": "router_weight",
    "moe_experts_gate_up": "experts_gate_up",
    "moe_experts_down": "experts_down",
    "moe_experts_up": "experts_up",
    # Activations
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
    # moe_out / moe_out_flat already match canonical names
    # --- shared_expert (MoESharedExpert) ---
    # Params use local names gate/up/down; after shared_expert_ prefix they
    # become shared_expert_gate etc. which are already the canonical names.
    # Activation intermediates keep their prefixed names (no remap needed).
}

VL_DENSE_BLOCK_NAME_REMAP: dict[str, str] = {
    **DENSE_BLOCK_NAME_REMAP,
    "self_attn_qkv_norm": "qkv_norm",
}

STANDARD_MODEL_NAME_REMAP: dict[str, str] = {
    # --- embedding ---
    "embedding_weight": "embedding",
    "embedding_out": "x0",
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

VL_MODEL_NAME_REMAP: dict[str, str] = {
    "embedding_weight": "embedding",
    # No "embedding_out" → "x0" — mask_scatter output takes that name
    "final_norm_weight": "final_norm",
    "final_norm_res": "residual_final",
    "final_norm_y": "xF",
    "final_norm_rstd": "ln_final_rstd",
    "lm_head_weight": "lm_head",
    "lm_head_loss": "loss",
    "lm_head_x_flat": "xF_flat",
}
