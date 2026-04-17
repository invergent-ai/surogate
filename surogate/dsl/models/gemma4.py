"""Gemma4 Model.

Supports Gemma4ForCausalLM (text-only) and Gemma4ForConditionalGeneration
(multimodal, text backbone only — vision/audio encoders are external).

Key architectural features:
  - Mixed attention: 5:1 sliding-to-full pattern (configurable via layer_types)
  - Sandwich norms: 4 RMSNorm per layer (pre/post attention, pre/post MLP)
  - QKV-norm: Q/K with (1+weight) scale, V with RMS-only normalization
  - Different head_dim per layer type: head_dim for sliding, global_head_dim for full
  - Different RoPE per layer type: default@10K for sliding, proportional@1M for full
  - GeLU-gated MLP (gelu_pytorch_tanh activation)
  - Scaled word embedding (scale = sqrt(hidden_size))
  - Per-layer input embeddings (when hidden_size_per_layer_input > 0)
  - attention_k_eq_v: V reuses K projection for full-attention layers
  - final_logit_softcapping: tanh-based logit capping before cross-entropy
"""

from __future__ import annotations

from .. import nn
from ..blocks.gemma4 import (
    Gemma4FullBlock,
    Gemma4FullMoEBlock,
    Gemma4SharedKVBlock,
    Gemma4SlidingBlock,
    Gemma4SlidingMoEBlock,
)
from ..hf import fuse
from ..nn import GEMMA4_MODEL_NAME_REMAP
from ..specs import ActivationScope


def _parse_gemma4_layer_types(
    layer_types: list[str] | None,
    n_layers: int,
) -> list[str]:
    """Convert HF layer_types to DSL HybridBlockStack types."""
    if layer_types is None:
        sliding_window_pattern = 6
        layer_types = [
            "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention" for i in range(n_layers)
        ]
    if layer_types and layer_types[-1] != "full_attention":
        layer_types[-1] = "full_attention"
    if len(layer_types) != n_layers:
        raise ValueError(f"layer_types length ({len(layer_types)}) != n_layers ({n_layers})")
    out: list[str] = []
    for t in layer_types:
        if t == "sliding_attention":
            out.append("sliding")
        elif t == "full_attention":
            out.append("full")
        else:
            raise ValueError(f"Unsupported Gemma4 layer type '{t}'.")
    return out


def _gemma4_layer_mappings(layer_prefix: str, *, k_eq_v: bool = False) -> dict[str, object]:
    """HF weight mappings for per-layer Gemma4 block parameters."""
    if k_eq_v:
        qkv_mapping = fuse(
            f"{layer_prefix}.self_attn.q_proj.weight",
            f"{layer_prefix}.self_attn.k_proj.weight",
            dim=0,
        )
    else:
        qkv_mapping = fuse(
            f"{layer_prefix}.self_attn.q_proj.weight",
            f"{layer_prefix}.self_attn.k_proj.weight",
            f"{layer_prefix}.self_attn.v_proj.weight",
            dim=0,
        )
    return {
        # Norms (4 per layer + optional PLI norm)
        "ln1_weight": f"{layer_prefix}.input_layernorm.weight",
        "ln_post_attn_weight": f"{layer_prefix}.post_attention_layernorm.weight",
        "ln2_weight": f"{layer_prefix}.pre_feedforward_layernorm.weight",
        "ln_post_ff_weight": f"{layer_prefix}.post_feedforward_layernorm.weight",
        # Attention
        "qkv_weight": qkv_mapping,
        "out_weight": f"{layer_prefix}.self_attn.o_proj.weight",
        "q_norm_weight": f"{layer_prefix}.self_attn.q_norm.weight",
        "k_norm_weight": f"{layer_prefix}.self_attn.k_norm.weight",
        # MLP (separate gate/up matmuls; see Gemma4 LoRA hook path in
        # matmul.cpp which fires distinct forward hooks for each).
        "mlp_gate_weight": f"{layer_prefix}.mlp.gate_proj.weight",
        "mlp_up_weight": f"{layer_prefix}.mlp.up_proj.weight",
        "mlp_down_weight": f"{layer_prefix}.mlp.down_proj.weight",
        # Per-layer scaling buffer
        "layer_scalar": f"{layer_prefix}.layer_scalar",
        # Per-layer input gating (PLI)
        "pli_gate_weight": f"{layer_prefix}.per_layer_input_gate.weight",
        "pli_proj_weight": f"{layer_prefix}.per_layer_projection.weight",
        "pli_norm_weight": f"{layer_prefix}.post_per_layer_input_norm.weight",
        # Shared-KV attention (Q-only projection, no K/V)
        "self_attn_q_weight": f"{layer_prefix}.self_attn.q_proj.weight",
        "self_attn_q_norm_weight": f"{layer_prefix}.self_attn.q_norm.weight",
    }


def _build_gemma4_block_mappings(
    layer_prefix: str,
    model_prefix: str,
    *,
    k_eq_v: bool = False,
) -> dict[str, object]:
    """HF weight mappings for a Gemma4 model (layer + model-level params)."""
    # Layer-level mappings. The qkv_weight fuse mapping uses non-k_eq_v
    # (3 projections) for sliding blocks. Full blocks with k_eq_v override
    # at the module level via Gemma4Attention._hf_mapping_k_eq_v_.
    mappings = _gemma4_layer_mappings(layer_prefix, k_eq_v=False)
    return {
        **mappings,
        "embedding": f"{model_prefix}.embed_tokens.weight",
        "pli_embedding": f"{model_prefix}.embed_tokens_per_layer.weight",
        "pli_model_proj": f"{model_prefix}.per_layer_model_projection.weight",
        "pli_proj_norm": f"{model_prefix}.per_layer_projection_norm.weight",
        "final_norm": f"{model_prefix}.norm.weight",
        "lm_head": "lm_head.weight",
    }


def _build_gemma4_model(
    cls,
    vocab_size,
    d_model,
    n_layers,
    num_query_heads,
    num_kv_heads,
    d_ff,
    max_seq,
    head_size,
    eps,
    sliding_window,
    layer_types,
    global_head_dim,
    global_num_kv_heads,
    full_partial_rotary_factor,
    d_per_layer_input,
    vocab_size_per_layer_input,
    k_eq_v,
    final_logit_softcapping,
    enable_moe_block=False,
    num_experts=0,
    top_k_experts=0,
    moe_intermediate_size=0,
    num_kv_shared_layers=0,
    use_double_wide_mlp=False,
):
    """Shared constructor logic for Gemma4 causal and conditional models."""
    cls.vocab_size = vocab_size
    cls.d_model = d_model
    cls.n_layers = n_layers
    cls.num_query_heads = num_query_heads
    cls.num_kv_heads = num_kv_heads
    cls.d_ff = d_ff
    cls.max_seq = max_seq
    cls.head_size = head_size
    cls.eps = eps
    cls.sliding_window = sliding_window
    cls.global_head_dim = global_head_dim
    cls.full_partial_rotary_factor = full_partial_rotary_factor
    cls.d_per_layer_input = d_per_layer_input
    cls.vocab_size_per_layer_input = vocab_size_per_layer_input
    cls.k_eq_v = k_eq_v
    cls.final_logit_softcapping = final_logit_softcapping

    if global_num_kv_heads is None:
        global_num_kv_heads = num_kv_heads
    cls.global_num_kv_heads = global_num_kv_heads

    cls.D = head_size
    cls.PLI_D = d_per_layer_input
    cls.PLI_total = n_layers * d_per_layer_input if d_per_layer_input > 0 else 0

    base_types = _parse_gemma4_layer_types(layer_types, n_layers)

    # KV sharing: last N layers share K,V from earlier layers
    first_shared_idx = n_layers - num_kv_shared_layers if num_kv_shared_layers > 0 else n_layers
    kv_sharing_map = {}
    if num_kv_shared_layers > 0:
        non_shared_types = base_types[:first_shared_idx]
        for i in range(first_shared_idx, n_layers):
            # Find the last non-shared layer of the same attention type
            layer_type = base_types[i]
            for j in range(len(non_shared_types) - 1, -1, -1):
                if non_shared_types[j] == layer_type:
                    kv_sharing_map[i] = j
                    break

    # Build final block_types: shared layers split by base attention type
    cls.block_types = []
    for i, t in enumerate(base_types):
        if i >= first_shared_idx and num_kv_shared_layers > 0:
            if t == "full":
                cls.block_types.append("shared_kv_full")
            else:
                cls.block_types.append("shared_kv_sliding")
        else:
            cls.block_types.append(t)

    cls.n_sliding_blocks = sum(1 for t in cls.block_types if t == "sliding")
    cls.n_full_blocks = sum(1 for t in cls.block_types if t == "full")
    cls.n_shared_kv_sliding_blocks = sum(1 for t in cls.block_types if t == "shared_kv_sliding")
    cls.n_shared_kv_full_blocks = sum(1 for t in cls.block_types if t == "shared_kv_full")

    block_configs = []
    if cls.n_sliding_blocks > 0:
        if enable_moe_block:
            block_configs.append(
                (
                    "sliding_blocks",
                    Gemma4SlidingMoEBlock,
                    cls.n_sliding_blocks,
                    dict(
                        d_model=d_model,
                        num_query_heads=num_query_heads,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        d_ff=d_ff,
                        max_seq=max_seq,
                        sliding_window=sliding_window,
                        num_experts=num_experts,
                        num_experts_per_tok=top_k_experts,
                        moe_intermediate_size=moe_intermediate_size,
                        eps=eps,
                    ),
                )
            )
        else:
            block_configs.append(
                (
                    "sliding_blocks",
                    Gemma4SlidingBlock,
                    cls.n_sliding_blocks,
                    dict(
                        d_model=d_model,
                        num_query_heads=num_query_heads,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        d_ff=d_ff,
                        max_seq=max_seq,
                        sliding_window=sliding_window,
                        d_per_layer_input=d_per_layer_input,
                        eps=eps,
                    ),
                )
            )
    if cls.n_full_blocks > 0:
        if enable_moe_block:
            block_configs.append(
                (
                    "full_blocks",
                    Gemma4FullMoEBlock,
                    cls.n_full_blocks,
                    dict(
                        d_model=d_model,
                        num_query_heads=num_query_heads,
                        num_kv_heads=global_num_kv_heads,
                        head_size=global_head_dim,
                        d_ff=d_ff,
                        max_seq=max_seq,
                        partial_rotary_factor=full_partial_rotary_factor,
                        k_eq_v=k_eq_v,
                        num_experts=num_experts,
                        num_experts_per_tok=top_k_experts,
                        moe_intermediate_size=moe_intermediate_size,
                        eps=eps,
                    ),
                )
            )
        else:
            block_configs.append(
                (
                    "full_blocks",
                    Gemma4FullBlock,
                    cls.n_full_blocks,
                    dict(
                        d_model=d_model,
                        num_query_heads=num_query_heads,
                        num_kv_heads=global_num_kv_heads,
                        head_size=global_head_dim,
                        d_ff=d_ff,
                        max_seq=max_seq,
                        partial_rotary_factor=full_partial_rotary_factor,
                        k_eq_v=k_eq_v,
                        d_per_layer_input=d_per_layer_input,
                        eps=eps,
                    ),
                )
            )

    # Shared-KV sliding blocks (Q-only attention, standard head_dim, double-wide MLP)
    if cls.n_shared_kv_sliding_blocks > 0:
        block_configs.append(
            (
                "shared_kv_sliding_blocks",
                Gemma4SharedKVBlock,
                cls.n_shared_kv_sliding_blocks,
                dict(
                    d_model=d_model,
                    num_query_heads=num_query_heads,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    d_ff=d_ff,
                    max_seq=max_seq,
                    partial_rotary_factor=1.0,
                    d_per_layer_input=d_per_layer_input,
                    use_double_wide_mlp=use_double_wide_mlp,
                    eps=eps,
                ),
            )
        )
    # Shared-KV full blocks (Q-only attention, global_head_dim, double-wide MLP)
    if cls.n_shared_kv_full_blocks > 0:
        block_configs.append(
            (
                "shared_kv_full_blocks",
                Gemma4SharedKVBlock,
                cls.n_shared_kv_full_blocks,
                dict(
                    d_model=d_model,
                    num_query_heads=num_query_heads,
                    num_kv_heads=global_num_kv_heads,
                    head_size=global_head_dim,
                    d_ff=d_ff,
                    max_seq=max_seq,
                    partial_rotary_factor=full_partial_rotary_factor,
                    d_per_layer_input=d_per_layer_input,
                    use_double_wide_mlp=use_double_wide_mlp,
                    eps=eps,
                ),
            )
        )

    cls.embedding = nn.ScaledEmbedding(vocab_size, d_model)

    # Per-layer input embeddings (only when d_per_layer_input > 0)
    if d_per_layer_input > 0:
        cls.pli_embedding = nn.ScaledEmbedding(
            vocab_size_per_layer_input,
            n_layers * d_per_layer_input,
            embed_scale=float(d_per_layer_input) ** 0.5,
            dim_name="PLI_total",
        )

    cls.hybrid_blocks = nn.HybridBlockStack(
        block_configs=block_configs,
        block_types=cls.block_types,
        n_layers=n_layers,
        per_layer_input_name="per_layer_input" if d_per_layer_input > 0 else None,
        kv_sharing_map=kv_sharing_map if kv_sharing_map else None,
    )
    cls.final_norm = nn.RMSNorm(d_model, eps=eps)
    cls.lm_head = nn.LMHead(vocab_size, d_model, softcap=final_logit_softcapping)


def _gemma4_forward(model, token_ids, position_ids, targets):
    """Shared forward logic for Gemma4 causal and conditional models."""
    G = ActivationScope.GLOBAL
    d_model = model.d_model
    n_layers = model.n_layers
    d_pli = model.d_per_layer_input

    # IO slots
    model._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
    model._register_activation("position_ids", ("T",), dtype="int32", scope=G)
    model._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])
    model._register_activation("freq_cis", ("max_seq", "D", 2), dtype="fp32", scope=G, aliases=["rope_freqs"])

    # Global intermediate slots
    _h = ("B", "T", "d_model")
    model._register_activation("residual0", _h, scope=G)
    model._register_activation("x0", _h, aliases=["encoded"], scope=G)
    model._register_activation("xN", _h, scope=G)
    model._register_activation("residualN", _h, scope=G)
    model._register_activation("residual_final", _h, scope=G)
    model._register_activation("xF", _h, aliases=["ln_final"], scope=G)
    model._register_activation("xF_flat", ("B * T", "d_model"), scope=G)
    model._register_activation("ln_final_rstd", ("B", "T"), dtype="fp32", save=True, scope=G)
    model._register_activation("loss", ("B * T",), dtype="fp32", aliases=["losses"], scope=G)

    # Main embedding (scaled by sqrt(hidden_size))
    x = model.embedding(token_ids)

    if d_pli > 0:
        pli_total = model.PLI_total

        # Per-layer input computation
        pli_embeds = model.pli_embedding(token_ids)

        model._register_param("pli_model_proj", ("PLI_total", "d_model"), quantizable=False)
        x_flat_pli = model._view(x, ["B * T", "d_model"], name="x_flat_pli")
        pli_proj_flat = model._matmul(x_flat_pli, "pli_model_proj", name="pli_proj_flat")
        pli_proj_3d = model._view(pli_proj_flat, ["B", "T", pli_total], name="pli_proj_3d")
        pli_proj_scaled = model._scale_by_const(pli_proj_3d, float(d_model) ** -0.5, name="pli_proj_scaled")

        pli_embeds_4d = model._view(pli_embeds, ["B", "T", n_layers, d_pli], name="pli_embeds_4d")
        pli_proj_4d = model._view(pli_proj_scaled, ["B", "T", n_layers, d_pli], name="pli_proj_4d")

        model._register_param("pli_proj_norm", ("PLI_D",), quantizable=False)
        pli_proj_rn_flat = model._view(pli_proj_4d, ["B * T * " + str(n_layers), str(d_pli)], name="pli_proj_rn_flat")
        pli_proj_normed = model._rmsnorm(pli_proj_rn_flat, "pli_proj_norm", eps=model.eps, name="pli_proj_normed")
        pli_proj_normed_4d = model._view(pli_proj_normed, ["B", "T", n_layers, d_pli], name="pli_proj_normed_4d")

        pli_combined = model._add(pli_proj_normed_4d, pli_embeds_4d, name="pli_combined")
        per_layer_inputs = model._scale_by_const(pli_combined, 2.0**-0.5, name="per_layer_inputs")

        # Forward through blocks with per-layer inputs
        residual = model._zeros(["B", "T", "d_model"])
        x, residual = model.hybrid_blocks(x, residual, position_ids, per_layer_inputs)
    else:
        # No per-layer inputs — pass a dummy that the compiler will ignore
        # since per_layer_input_name is None on the HybridBlockStack
        residual = model._zeros(["B", "T", "d_model"])
        x, residual = model.hybrid_blocks(x, residual, position_ids)

    # Blocks return (full_state, zeros) — final norm is NOT fused
    x = model.final_norm(x)
    loss = model.lm_head(x, targets)
    return loss


# ============================================================================
# Model classes
# ============================================================================


@nn.hf_config(
    architecture="Gemma4ForCausalLM",
    model_type="gemma4_text",
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    d_ff="intermediate_size",
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    head_size="head_dim",
    eps="rms_norm_eps",
    sliding_window="sliding_window",
    layer_types="layer_types",
    global_head_dim="global_head_dim",
    global_num_kv_heads="num_global_key_value_heads",
    full_partial_rotary_factor="rope_parameters.full_attention.partial_rotary_factor",
    d_per_layer_input="hidden_size_per_layer_input",
    vocab_size_per_layer_input="vocab_size_per_layer_input",
    k_eq_v="attention_k_eq_v",
    final_logit_softcapping="final_logit_softcapping",
    enable_moe_block="enable_moe_block",
    num_experts="num_experts",
    top_k_experts="top_k_experts",
    moe_intermediate_size="moe_intermediate_size",
    num_kv_shared_layers="num_kv_shared_layers",
    use_double_wide_mlp="use_double_wide_mlp",
)
class Gemma4CausalModel(nn.Model):
    """Gemma4 text model for ``Gemma4ForCausalLM``."""

    _name_remap_ = GEMMA4_MODEL_NAME_REMAP
    _hf_block_mappings_ = _build_gemma4_block_mappings(
        "model.layers.{layer}",
        "model",
    )

    def __init__(
        self,
        vocab_size: int = 262144,
        d_model: int = 2304,
        n_layers: int = 30,
        num_query_heads: int = 8,
        num_kv_heads: int = 4,
        d_ff: int = 9216,
        max_seq: int = 131072,
        head_size: int = 256,
        eps: float = 1e-6,
        sliding_window: int = 512,
        layer_types: list[str] | None = None,
        global_head_dim: int = 512,
        global_num_kv_heads: int | None = None,
        full_partial_rotary_factor: float = 0.25,
        d_per_layer_input: int = 256,
        vocab_size_per_layer_input: int = 262144,
        k_eq_v: bool = False,
        final_logit_softcapping: float | None = None,
        enable_moe_block: bool = False,
        num_experts: int = 0,
        top_k_experts: int = 0,
        moe_intermediate_size: int = 0,
        num_kv_shared_layers: int = 0,
        use_double_wide_mlp: bool = False,
    ):
        super().__init__()
        _build_gemma4_model(
            self,
            vocab_size,
            d_model,
            n_layers,
            num_query_heads,
            num_kv_heads,
            d_ff,
            max_seq,
            head_size,
            eps,
            sliding_window,
            layer_types,
            global_head_dim,
            global_num_kv_heads,
            full_partial_rotary_factor,
            d_per_layer_input,
            vocab_size_per_layer_input,
            k_eq_v,
            final_logit_softcapping,
            enable_moe_block=enable_moe_block,
            num_experts=num_experts or 0,
            top_k_experts=top_k_experts or 0,
            moe_intermediate_size=moe_intermediate_size or 0,
            num_kv_shared_layers=num_kv_shared_layers,
            use_double_wide_mlp=use_double_wide_mlp,
        )

    def forward(self, token_ids, position_ids, targets):
        return _gemma4_forward(self, token_ids, position_ids, targets)


@nn.hf_config(
    architecture="Gemma4ForConditionalGeneration",
    model_type="gemma4",
    d_model="text_config.hidden_size",
    n_layers="text_config.num_hidden_layers",
    num_query_heads="text_config.num_attention_heads",
    num_kv_heads="text_config.num_key_value_heads",
    d_ff="text_config.intermediate_size",
    vocab_size="text_config.vocab_size",
    max_seq="text_config.max_position_embeddings",
    head_size="text_config.head_dim",
    eps="text_config.rms_norm_eps",
    sliding_window="text_config.sliding_window",
    layer_types="text_config.layer_types",
    global_head_dim="text_config.global_head_dim",
    global_num_kv_heads="text_config.num_global_key_value_heads",
    full_partial_rotary_factor="text_config.rope_parameters.full_attention.partial_rotary_factor",
    d_per_layer_input="text_config.hidden_size_per_layer_input",
    vocab_size_per_layer_input="text_config.vocab_size_per_layer_input",
    k_eq_v="text_config.attention_k_eq_v",
    final_logit_softcapping="text_config.final_logit_softcapping",
    enable_moe_block="text_config.enable_moe_block",
    num_experts="text_config.num_experts",
    top_k_experts="text_config.top_k_experts",
    moe_intermediate_size="text_config.moe_intermediate_size",
    num_kv_shared_layers="text_config.num_kv_shared_layers",
    use_double_wide_mlp="text_config.use_double_wide_mlp",
)
class Gemma4ConditionalModel(nn.Model):
    """Gemma4 text backbone for ``Gemma4ForConditionalGeneration`` (multimodal)."""

    _name_remap_ = GEMMA4_MODEL_NAME_REMAP
    _hf_block_mappings_ = _build_gemma4_block_mappings(
        "model.language_model.layers.{layer}",
        "model.language_model",
    )

    def __init__(
        self,
        vocab_size: int = 262144,
        d_model: int = 2304,
        n_layers: int = 30,
        num_query_heads: int = 8,
        num_kv_heads: int = 4,
        d_ff: int = 9216,
        max_seq: int = 131072,
        head_size: int = 256,
        eps: float = 1e-6,
        sliding_window: int = 512,
        layer_types: list[str] | None = None,
        global_head_dim: int = 512,
        global_num_kv_heads: int | None = None,
        full_partial_rotary_factor: float = 0.25,
        d_per_layer_input: int = 256,
        vocab_size_per_layer_input: int = 262144,
        k_eq_v: bool = False,
        final_logit_softcapping: float | None = None,
        enable_moe_block: bool = False,
        num_experts: int = 0,
        top_k_experts: int = 0,
        moe_intermediate_size: int = 0,
        num_kv_shared_layers: int = 0,
        use_double_wide_mlp: bool = False,
    ):
        super().__init__()
        _build_gemma4_model(
            self,
            vocab_size,
            d_model,
            n_layers,
            num_query_heads,
            num_kv_heads,
            d_ff,
            max_seq,
            head_size,
            eps,
            sliding_window,
            layer_types,
            global_head_dim,
            global_num_kv_heads,
            full_partial_rotary_factor,
            d_per_layer_input,
            vocab_size_per_layer_input,
            k_eq_v,
            final_logit_softcapping,
            enable_moe_block=enable_moe_block,
            num_experts=num_experts or 0,
            top_k_experts=top_k_experts or 0,
            moe_intermediate_size=moe_intermediate_size or 0,
            num_kv_shared_layers=num_kv_shared_layers,
            use_double_wide_mlp=use_double_wide_mlp,
        )

    def forward(self, token_ids, position_ids, targets):
        return _gemma4_forward(self, token_ids, position_ids, targets)
