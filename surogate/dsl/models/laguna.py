"""Laguna model (``LagunaForCausalLM``, Poolside).

Key architectural features:
  - Mixed attention: full_attention / sliding_attention layer types with
    different RoPE per type (default@10K full-rotary for sliding, YaRN@500K
    partial-rotary 0.5 for full on Laguna-XS-2.1)
  - Per-layer query head counts (``num_attention_heads_per_layer``; uniform
    within each attention layer type, e.g. 48 full / 64 sliding), shared
    ``num_key_value_heads`` and ``head_dim``
  - Softplus output-gated attention (``g_proj``; per-head or per-element)
  - Plain QK-Norm on head_dim
  - Mixed MLP: ``mlp_layer_types`` dense (SwiGLU) / sparse (MoE); MoE uses a
    sigmoid router with e_score_correction_bias, normalized top-k weights,
    ``moe_routed_scaling_factor``, and an ungated shared expert
  - Untied lm_head
"""

from __future__ import annotations

from .. import nn
from ..blocks.laguna import LagunaDenseBlock, LagunaSparseBlock
from ..hf import build_norm_mappings, expand_module_mapping, fuse
from ..modules import Embedding, LMHead, RMSNorm
from ..blocks.common import STANDARD_MODEL_NAME_REMAP
from ..specs import ActivationScope


def _parse_laguna_layer_structure(
    layer_types: list[str] | None,
    mlp_layer_types: list[str] | None,
    num_attention_heads_per_layer: list[int] | None,
    n_layers: int,
    num_query_heads: int,
) -> tuple[list[str], dict[str, int]]:
    """Resolve per-layer (attention, mlp) structure into hybrid block types.

    Returns (block_types, heads_by_attn_kind) where block_types entries are
    "full_dense" / "full_sparse" / "sliding_dense" / "sliding_sparse" and
    heads_by_attn_kind maps "full"/"sliding" to the query head count of that
    attention type.
    """
    if layer_types is None:
        layer_types = ["full_attention"] * n_layers
    if mlp_layer_types is None:
        mlp_layer_types = ["dense"] + ["sparse"] * (n_layers - 1)
    if num_attention_heads_per_layer is None:
        num_attention_heads_per_layer = [num_query_heads] * n_layers

    if len(layer_types) != n_layers:
        raise ValueError(f"layer_types length ({len(layer_types)}) != n_layers ({n_layers})")
    if len(mlp_layer_types) != n_layers:
        raise ValueError(f"mlp_layer_types length ({len(mlp_layer_types)}) != n_layers ({n_layers})")
    if len(num_attention_heads_per_layer) != n_layers:
        raise ValueError(
            f"num_attention_heads_per_layer length ({len(num_attention_heads_per_layer)}) != n_layers ({n_layers})"
        )

    block_types: list[str] = []
    heads_by_attn_kind: dict[str, int] = {}
    for i in range(n_layers):
        attn_t = layer_types[i]
        if attn_t == "sliding_attention":
            attn_kind = "sliding"
        elif attn_t == "full_attention":
            attn_kind = "full"
        else:
            raise ValueError(f"Unsupported Laguna layer type '{attn_t}'.")

        mlp_t = mlp_layer_types[i]
        if mlp_t not in ("dense", "sparse"):
            raise ValueError(f"Unsupported Laguna mlp layer type '{mlp_t}'.")

        heads = int(num_attention_heads_per_layer[i])
        prev = heads_by_attn_kind.setdefault(attn_kind, heads)
        if prev != heads:
            raise ValueError(
                "Laguna requires a uniform query head count per attention layer type; "
                f"{attn_kind} layers have both {prev} and {heads} heads."
            )

        block_types.append(f"{attn_kind}_{mlp_t}")

    return block_types, heads_by_attn_kind


def _resolve_gate_per_head(gating: bool | str) -> bool:
    # Mirrors HF LagunaAttention: ``True`` or ``"per-head"`` selects per-head
    # gating; anything else (``False``, ``"per-element"``, ...) falls through
    # to per-element, exactly like ``config.gating is True or config.gating ==
    # "per-head"`` in modeling_laguna.py.
    return gating is True or gating in ("per-head", "per_head")


def _build_laguna_block_mappings(layer_prefix: str) -> dict[str, object]:
    """HF weight mappings for the Laguna model (layer + model-level params)."""
    from ..modules.attention import LagunaAttention
    from ..modules.moe import LagunaMoEExperts, MoESharedExpert

    moe_prefix = f"{layer_prefix}.mlp"
    return {
        **build_norm_mappings(layer_prefix),
        # Attention (separate projections + softplus gate proj + QK norms)
        **expand_module_mapping(
            LagunaAttention._hf_mapping_defaults_,
            hf_prefix=f"{layer_prefix}.self_attn",
        ),
        # Dense MLP layers (fused [up; gate] rows — matches the swiglu kernel)
        "mlp_up_weight": fuse(
            f"{moe_prefix}.up_proj.weight",
            f"{moe_prefix}.gate_proj.weight",
            dim=0,
        ),
        "mlp_down_weight": f"{moe_prefix}.down_proj.weight",
        # Sparse MoE layers
        **expand_module_mapping(
            LagunaMoEExperts._hf_mapping_defaults_,
            hf_prefix=moe_prefix,
        ),
        # Shared expert (ungated SwiGLU MLP)
        **expand_module_mapping(
            MoESharedExpert._hf_mapping_defaults_,
            hf_prefix=moe_prefix,
            param_prefix="shared_expert_",
        ),
        # Model-level weight mappings
        "embedding": "model.embed_tokens.weight",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
    }


@nn.hf_config(
    architecture="LagunaForCausalLM",
    model_type="laguna",
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    d_ff="moe_intermediate_size",
    intermediate_size="intermediate_size",
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    head_size="head_dim",
    eps="rms_norm_eps",
    num_experts="num_experts",
    num_experts_per_tok="num_experts_per_tok",
    shared_expert_intermediate="shared_expert_intermediate_size",
    routed_scaling_factor="moe_routed_scaling_factor",
    router_logit_softcapping="moe_router_logit_softcapping",
    apply_router_weight_on_input="moe_apply_router_weight_on_input",
    gating="gating",
    sliding_window="sliding_window",
    layer_types="layer_types",
    mlp_layer_types="mlp_layer_types",
    num_attention_heads_per_layer="num_attention_heads_per_layer",
    tie_word_embeddings="tie_word_embeddings",
    sliding_rope_theta="rope_parameters.sliding_attention.rope_theta",
    sliding_rope_type="rope_parameters.sliding_attention.rope_type",
    sliding_partial_rotary_factor="rope_parameters.sliding_attention.partial_rotary_factor",
    full_rope_theta="rope_parameters.full_attention.rope_theta",
    full_rope_type="rope_parameters.full_attention.rope_type",
    full_partial_rotary_factor="rope_parameters.full_attention.partial_rotary_factor",
    full_rope_factor="rope_parameters.full_attention.factor",
    full_rope_original_max_position_embeddings="rope_parameters.full_attention.original_max_position_embeddings",
    full_rope_beta_fast="rope_parameters.full_attention.beta_fast",
    full_rope_beta_slow="rope_parameters.full_attention.beta_slow",
    full_rope_attention_factor="rope_parameters.full_attention.attention_factor",
)
class LagunaModel(nn.Model):
    """Laguna model for ``LagunaForCausalLM``."""

    _name_remap_ = STANDARD_MODEL_NAME_REMAP
    _hf_block_mappings_ = _build_laguna_block_mappings("model.layers.{layer}")

    def __init__(
        self,
        vocab_size: int = 100352,
        d_model: int = 2048,
        n_layers: int = 40,
        num_query_heads: int = 48,
        num_kv_heads: int = 8,
        d_ff: int = 512,
        intermediate_size: int = 8192,
        max_seq: int = 131072,
        head_size: int = 128,
        eps: float = 1e-6,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        shared_expert_intermediate: int = 512,
        routed_scaling_factor: float = 1.0,
        router_logit_softcapping: float = 0.0,
        apply_router_weight_on_input: bool = False,
        gating: bool | str = True,
        sliding_window: int = 512,
        layer_types: list[str] | None = None,
        mlp_layer_types: list[str] | None = None,
        num_attention_heads_per_layer: list[int] | None = None,
        tie_word_embeddings: bool = False,
        sliding_rope_theta: float = 10000.0,
        sliding_rope_type: str = "default",
        sliding_partial_rotary_factor: float = 1.0,
        full_rope_theta: float = 500000.0,
        full_rope_type: str = "default",
        full_partial_rotary_factor: float = 0.5,
        full_rope_factor: float | None = None,
        full_rope_original_max_position_embeddings: int | None = None,
        full_rope_beta_fast: float | None = None,
        full_rope_beta_slow: float | None = None,
        full_rope_attention_factor: float | None = None,
        ep_size: int = 1,
    ):
        super().__init__()
        # ValueError is the config-validation channel the DSL compiler
        # re-raises with context; other exception types can be swallowed on
        # the non-lazy compile path.
        if router_logit_softcapping:
            raise ValueError("Laguna moe_router_logit_softcapping != 0 is not supported yet.")
        if tie_word_embeddings:
            raise ValueError("Laguna tie_word_embeddings=True is not supported yet.")
        if apply_router_weight_on_input:
            # HF's LagunaConfig.validate_architecture rejects this too; the
            # semantics (routing weights applied to the expert INPUT) differ
            # from what the graph implements.
            raise ValueError("Laguna moe_apply_router_weight_on_input=True is not supported yet.")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_ff = d_ff
        self.intermediate_size = intermediate_size
        self.max_seq = max_seq
        self.head_size = head_size
        self.eps = eps
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.shared_expert_intermediate = shared_expert_intermediate
        self.routed_scaling_factor = routed_scaling_factor
        self.sliding_window = sliding_window
        self.gate_per_head = _resolve_gate_per_head(gating)
        self.sliding_rope_theta = sliding_rope_theta
        self.sliding_rope_type = sliding_rope_type
        self.sliding_partial_rotary_factor = sliding_partial_rotary_factor
        self.full_rope_theta = full_rope_theta
        self.full_rope_type = full_rope_type
        self.full_partial_rotary_factor = full_partial_rotary_factor
        self.full_rope_factor = full_rope_factor
        self.full_rope_original_max_position_embeddings = full_rope_original_max_position_embeddings
        self.full_rope_beta_fast = full_rope_beta_fast
        self.full_rope_beta_slow = full_rope_beta_slow
        self.full_rope_attention_factor = full_rope_attention_factor

        self.D = head_size

        self.block_types, heads_by_attn_kind = _parse_laguna_layer_structure(
            layer_types,
            mlp_layer_types,
            num_attention_heads_per_layer,
            n_layers,
            num_query_heads,
        )
        # Keep the HF layer_types list around for the C++ per-layer rope config.
        self.layer_types = (
            layer_types
            if layer_types is not None
            else ["full_attention"] * n_layers
        )

        self.n_full_dense_blocks = sum(1 for t in self.block_types if t == "full_dense")
        self.n_full_sparse_blocks = sum(1 for t in self.block_types if t == "full_sparse")
        self.n_sliding_dense_blocks = sum(1 for t in self.block_types if t == "sliding_dense")
        self.n_sliding_sparse_blocks = sum(1 for t in self.block_types if t == "sliding_sparse")

        def _attn_kwargs(attn_kind: str) -> dict:
            if attn_kind == "full":
                return dict(
                    num_query_heads=heads_by_attn_kind["full"],
                    sliding_window=None,
                    partial_rotary_factor=full_partial_rotary_factor,
                )
            return dict(
                num_query_heads=heads_by_attn_kind["sliding"],
                sliding_window=sliding_window,
                partial_rotary_factor=sliding_partial_rotary_factor,
            )

        common = dict(
            d_model=d_model,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            max_seq=max_seq,
            gate_per_head=self.gate_per_head,
            eps=eps,
        )
        sparse_kwargs = dict(
            d_ff=d_ff,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            shared_expert_intermediate=shared_expert_intermediate,
            routed_scaling_factor=routed_scaling_factor,
            ep_size=ep_size,
        )

        block_configs = []
        for attn_kind in ("full", "sliding"):
            n_dense = getattr(self, f"n_{attn_kind}_dense_blocks")
            n_sparse = getattr(self, f"n_{attn_kind}_sparse_blocks")
            if n_dense > 0:
                block_configs.append(
                    (
                        f"{attn_kind}_dense_blocks",
                        LagunaDenseBlock,
                        n_dense,
                        dict(**common, **_attn_kwargs(attn_kind), d_ff=intermediate_size),
                    )
                )
            if n_sparse > 0:
                block_configs.append(
                    (
                        f"{attn_kind}_sparse_blocks",
                        LagunaSparseBlock,
                        n_sparse,
                        dict(**common, **_attn_kwargs(attn_kind), **sparse_kwargs),
                    )
                )

        self.embedding = Embedding(vocab_size, d_model)
        self.hybrid_blocks = nn.HybridBlockStack(
            block_configs=block_configs,
            block_types=self.block_types,
            n_layers=n_layers,
        )
        self.final_norm = RMSNorm(d_model, eps=eps)
        self.lm_head = LMHead(vocab_size, d_model)

    def forward(self, token_ids, position_ids, targets):
        G = ActivationScope.GLOBAL

        # IO slots
        self._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("position_ids", ("T",), dtype="int32", scope=G)
        self._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])
        self._register_activation("freq_cis", ("max_seq", "D", 2), dtype="fp32", scope=G, aliases=["rope_freqs"])

        # Global intermediate slots
        _h = ("B", "T", "d_model")
        self._register_activation("residual0", _h, scope=G)
        self._register_activation("x0", _h, aliases=["encoded"], scope=G)
        self._register_activation("xN", _h, scope=G)
        self._register_activation("residualN", _h, scope=G)
        self._register_activation("residual_final", _h, scope=G)
        self._register_activation("xF", _h, aliases=["ln_final"], scope=G)
        self._register_activation("xF_flat", ("B * T", "d_model"), scope=G)
        self._register_activation("ln_final_rstd", ("B", "T"), dtype="fp32", save=True, scope=G)
        self._register_activation("loss", ("B * T",), dtype="fp32", aliases=["losses"], scope=G)

        x = self.embedding(token_ids)
        residual = self._zeros(["B", "T", "d_model"])
        x, residual = self.hybrid_blocks(x, residual, position_ids)
        residual, x = self.final_norm(residual, x)
        loss = self.lm_head(x, targets)
        return loss
