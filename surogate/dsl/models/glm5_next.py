"""GLM-5.3-Flash (``glm5_next``) — hyper-connected hybrid KDA/MLA MoE.

Single source of truth for ``Glm5NextForConditionalGeneration``. The text stack
is 45 decoder layers over two independent axes:

* **mixer** — ``layer_types[i]``: Kimi Delta Attention (KDA, a per-key-channel
  gated delta rule with a short causal conv) on three layers in four, and a
  NoPE DeepSeek MLA with a sparse-attention (DSA) indexer on every fourth;
* **feed-forward** — ``mlp_layer_types[i]``: a dense SwiGLU MLP on the first
  three layers, then a 288-expert top-8 MoE with one shared expert, sigmoid
  routing, an ``e_score_correction_bias`` selection bias and a
  ``routed_scaling_factor`` of 2.5.

Both sublayers of every layer are wrapped in manifold-constrained
hyper-connections (mHC): the residual is ``hc_mult`` = 4 parallel streams, and
each site owns a flat ``hc_{attn,ffn}_{fn,base,scale}`` triple. Unlike
Qwen3.8-Flash-Next, the hyper-connections do NOT replace the layer norms —
``input_layernorm`` and ``post_attention_layernorm`` are still there, applied to
the collapsed stream, and the model still has a real ``norm`` after the streams
are averaged (an unweighted mean, ``Glm5NextTextHyperHead``).

Verified against the released checkpoint (GGUF KV of
``GLM-5.3-Flash-UD-Q4_K_XL``): 46 blocks = 45 + 1 MTP, hidden 4096, vocab
154880, 288 experts / top-8 / 1 shared, moe_intermediate 2048, 3 leading dense
blocks, expert_weights_scale 2.5, hc count 4 / 20 Sinkhorn iterations,
rope.dimension_count 0 (NoPE), kda head_dim 128 / conv kernel 4 /
gate_lower_bound -5, indexer 32 heads x 128 dim / top-k 2048.

DECLARED BUT NOT IN THE TRAINING GRAPH (hyperparameters reach the runtime config
so a serve-spec generator can read them; the mechanism is not lowered):

* **DSA indexer** (``index_*``, ``indexer_types``, and the per-MLA-layer
  ``self_attn.indexer.*`` tensors incl. the k-pool ``index_kpool_compress_ape``
  / ``index_kpool_compress_gate``): inference-time sparse *selection*. Training
  runs dense causal attention, which is the exact semantics whenever the
  sequence fits inside ``index_topk`` (2048) — the same argument, and the same
  treatment, as ``qwen4_exp``'s QSA indexer. The indexer tensors are not
  declared, so they are simply unused at import.
* **MTP / next-token-prediction block** (layer 45, ``nextn_predict_layers``):
  transformers itself drops it (``_keys_to_ignore_on_load_unexpected`` skips
  ``layers.45.`` and ``shared_head.``), so it is not part of the training graph.
  A serving artifact carries it as the ``mtp/`` section below: it is the
  checkpoint's own speculative draft head.
* **Vision tower** (``Glm5NextVisionModel``, ``model.visual.*``): text-only
  declaration, like ``qwen4_exp``'s. Its tensors are unused at import.
Native text training implements mHC, KDA forward/backward, packed convolution,
FP32 routing and SwiGLU clipping. The runtime rejects sequences longer than
``index_topk`` rather than silently replacing sparse DSA with dense attention.
Vision/MTP/indexer weights remain outside the training parameter set.
"""

from __future__ import annotations

from .. import nn
from ..block_schema import ServeObject, ServeSection
from ..blocks.glm5_next import (
    GLM5_NEXT_MTP_LAYER_OBJECTS,
    Glm5NextKdaDenseBlock,
    Glm5NextKdaMoEBlock,
    Glm5NextMlaDenseBlock,
    Glm5NextMlaMoEBlock,
)
from ..hf import fuse, stack_experts
from ..modules import Embedding, LMHead, RMSNorm, StreamBroadcast
from ..modules.glm5_next import Glm5NextHyperHead
from ..specs import ActivationScope

#: The NextN draft head, `blk.N.nextn.*` of the GGUF and `layers.N.*` of the checkpoint for N
#: the trunk's layer count. One latent-attention layer over the mixture on a single-stream
#: residual: the next token's embedding and the trunk's normalised hidden, each under its own
#: norm, are concatenated and projected to the model width, the layer runs, and
#: `shared_head.norm` reads the result out for the trunk's LM head. The embedding table and the
#: LM head are the trunk's. Declared with a count so a trunk-only checkpoint (``nextn_predict_layers``
#: 0) emits nothing.
GLM5_NEXT_MTP_SERVE_SECTION = ServeSection(
    prefix="mtp/",
    repeat="nextn_predict_layers",
    hf_prefix="model.language_model.layers.45.",
    hf_layer="model.language_model.layers.45",
    objects=(
        ServeObject("input_projection", "quantised", ("C", "TwoC"), source="eh_proj.weight"),
        ServeObject("embedding_norm", "bf16", ("C",), source="enorm.weight"),
        ServeObject("hidden_norm", "bf16", ("C",), source="hnorm.weight"),
        *(
            ServeObject("layer/" + o.name, o.format, o.shape, o.components,
                        transform=o.transform, residency=o.residency)
            for o in GLM5_NEXT_MTP_LAYER_OBJECTS
        ),
        ServeObject("final_norm", "bf16", ("C",), source="shared_head.norm.weight"),
    ),
)


GLM5_NEXT_MODEL_NAME_REMAP: dict[str, str] = {
    # --- embedding ---
    "embedding_weight": "embedding",
    "embedding_out": "x0",
    # --- stream_init (StreamBroadcast) ---
    "stream_init_res": "residual0",
    # --- final_norm (RMSNorm, after the unweighted stream mean) ---
    "final_norm_weight": "final_norm",
    "final_norm_y": "xF",
    "final_norm_rstd": "ln_final_rstd",
    # --- lm_head ---
    "lm_head_weight": "lm_head",
    "lm_head_loss": "loss",
    "lm_head_x_flat": "xF_flat",
}


def _build_glm5_next_block_mappings(layer_prefix: str, model_prefix: str) -> dict[str, object]:
    """HF mappings for GLM-5.3-Flash.

    Names are the ones the *checkpoint on disk* uses, which are not always the
    ones the transformers modules use — transformers rewrites them at load via
    ``conversion_mapping.py``'s ``glm5_next`` entry. Cross-checked against
    ``study/colibri/c/tools/check_glm53_checkpoint.py``:

    * the mHC parameters are flat on the layer (``hc_attn_fn``), not under an
      ``attn_hc`` submodule;
    * KDA stores q/k/v projections *and* q/k/v convs separately (transformers
      concatenates the conv triple into one grouped conv, which is what the
      fuse below reproduces);
    * the forget-gate tensors sit directly under ``self_attn`` rather than
      under a ``forget_gate`` submodule;
    * experts are per-expert ``mlp.experts.{i}.{gate,up,down}_proj.weight``
      (transformers batches them into ``gate_up_proj``/``down_proj``), so the
      default ``stack_experts`` derivation applies;
    * the shared expert is ``mlp.shared_experts`` (plural) and, unlike
      Qwen3.8-Flash-Next's, has no gate row.

    A single union dict is fine for a hybrid: an entry is only emitted for a
    parameter that exists on that layer.
    """

    attn = f"{layer_prefix}.self_attn"
    mlp = f"{layer_prefix}.mlp"
    return {
        # --- manifold-constrained hyper-connections (flat, two sites) ---
        "hc_attn_fn": f"{layer_prefix}.hc_attn_fn",
        "hc_attn_base": f"{layer_prefix}.hc_attn_base",
        "hc_attn_scale": f"{layer_prefix}.hc_attn_scale",
        "hc_ffn_fn": f"{layer_prefix}.hc_ffn_fn",
        "hc_ffn_base": f"{layer_prefix}.hc_ffn_base",
        "hc_ffn_scale": f"{layer_prefix}.hc_ffn_scale",
        # --- layer norms (kept, unlike qwen4_exp) ---
        "ln1_weight": f"{layer_prefix}.input_layernorm.weight",
        "ln2_weight": f"{layer_prefix}.post_attention_layernorm.weight",
        # --- KDA linear attention ---
        "kda_qkv_weight": fuse(
            f"{attn}.q_proj.weight",
            f"{attn}.k_proj.weight",
            f"{attn}.v_proj.weight",
            dim=0,
        ),
        "kda_conv_weight": fuse(
            f"{attn}.q_conv1d.weight",
            f"{attn}.k_conv1d.weight",
            f"{attn}.v_conv1d.weight",
            dim=0,
        ),
        "kda_f_a_weight": f"{attn}.f_a_proj.weight",
        "kda_f_b_weight": f"{attn}.f_b_proj.weight",
        "kda_dt_bias": f"{attn}.dt_bias",
        "kda_A_log": f"{attn}.A_log",
        "kda_b_weight": f"{attn}.b_proj.weight",
        "kda_g_a_weight": f"{attn}.g_a_proj.weight",
        "kda_g_b_weight": f"{attn}.g_b_proj.weight",
        "kda_o_norm_weight": f"{attn}.o_norm.weight",
        "kda_out_weight": f"{attn}.o_proj.weight",
        # --- NoPE MLA (indexer tensors deliberately not declared) ---
        "mla_q_a_weight": f"{attn}.q_a_proj.weight",
        "mla_q_a_norm_weight": f"{attn}.q_a_layernorm.weight",
        "mla_q_b_weight": f"{attn}.q_b_proj.weight",
        "mla_kv_a_weight": f"{attn}.kv_a_proj_with_mqa.weight",
        "mla_kv_a_norm_weight": f"{attn}.kv_a_layernorm.weight",
        "mla_kv_b_weight": f"{attn}.kv_b_proj.weight",
        "mla_out_weight": f"{attn}.o_proj.weight",
        # --- dense feed-forward (the leading layers). Fused row order [up; gate] ---
        "mlp_up_weight": fuse(f"{mlp}.up_proj.weight", f"{mlp}.gate_proj.weight", dim=0),
        "mlp_down_weight": f"{mlp}.down_proj.weight",
        # --- sparse feed-forward ---
        "router_weight": f"{mlp}.gate.weight",
        "e_score_correction_bias": f"{mlp}.gate.e_score_correction_bias",
        "experts_gate_up": stack_experts(
            f"{mlp}.experts.{{expert}}.gate_proj.weight",
            fuse_gate_up=True,
        ),
        "experts_down": stack_experts(f"{mlp}.experts.{{expert}}.down_proj.weight"),
        "shared_expert_gate": f"{mlp}.shared_experts.gate_proj.weight",
        "shared_expert_up": f"{mlp}.shared_experts.up_proj.weight",
        "shared_expert_down": f"{mlp}.shared_experts.down_proj.weight",
        # --- model level ---
        "embedding": f"{model_prefix}.embed_tokens.weight",
        "final_norm": f"{model_prefix}.norm.weight",
        "lm_head": "lm_head.weight",
    }


#: KDA geometry defaults, from ``Glm5NextTextConfig``.
_KDA_DEFAULTS = {
    "num_heads": 64,
    "head_dim": 128,
    "short_conv_kernel_size": 4,
    "gate_lower_bound": -5.0,
}


def _resolve_kda_geometry(
    linear_attn_config: dict | None,
    num_heads: int | None,
    head_dim: int | None,
    conv_kernel: int | None,
    lower_bound: float | None,
) -> tuple[int, int, int, float]:
    """Reconcile the two spellings of the KDA geometry.

    The released ``config.json`` nests it under ``linear_attn_config``
    (``head_dim`` / ``num_heads`` / ``short_conv_kernel_size`` /
    ``gate_lower_bound``); ``Glm5NextTextConfig`` declares the flat
    ``linear_*`` fields and folds the dict onto them in ``__post_init__``. Both
    reach here, so both are accepted — but a config that states both and
    disagrees is malformed, and is rejected rather than silently resolved: the
    compiler echoes constructor-argument names straight from the HF config into
    the runtime config, so a silent override would leave the exported config
    contradicting the graph.
    """

    flat = {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "short_conv_kernel_size": conv_kernel,
        "gate_lower_bound": lower_bound,
    }
    nested = dict(linear_attn_config or {})
    if nested and nested.get("safe_gate", True) and nested.get("gate_lower_bound") is None:
        nested["gate_lower_bound"] = -5.0

    resolved: dict[str, object] = {}
    for key, default in _KDA_DEFAULTS.items():
        flat_value = flat[key]
        nested_value = nested.get(key)
        if flat_value is not None and nested_value is not None and flat_value != nested_value:
            raise ValueError(
                f"glm5_next KDA geometry is stated twice and disagrees: "
                f"linear_attn_config['{key}']={nested_value!r} vs the flat key "
                f"={flat_value!r}"
            )
        chosen = nested_value if nested_value is not None else flat_value
        resolved[key] = default if chosen is None else chosen

    return (
        int(resolved["num_heads"]),
        int(resolved["head_dim"]),
        int(resolved["short_conv_kernel_size"]),
        float(resolved["gate_lower_bound"]),
    )


_MIXER_FOR_LAYER_TYPE = {
    "linear_attention": "kda",
    "deepseek_sparse_attention": "mla",
    # Older exports labelled the MLA layers "full_attention"; the HF config
    # rewrites them, so accept both.
    "full_attention": "mla",
}


def _resolve_glm5_next_block_types(
    *,
    n_layers: int,
    layer_types: list[str] | None,
    mlp_layer_types: list[str] | None,
    first_k_dense_replace: int,
) -> tuple[list[str], list[str], list[str]]:
    """One block type per layer, over both axes.

    Returns ``(block_types, layer_types, mlp_layer_types)`` with the two HF
    schedules filled in from their defaults when the config omits them (which
    is what ``Glm5NextTextConfig.__post_init__`` does).
    """

    if layer_types is None:
        layer_types = [
            "linear_attention" if i % 4 != 3 else "deepseek_sparse_attention" for i in range(n_layers)
        ]
    layer_types = [
        "deepseek_sparse_attention" if t == "full_attention" else t for t in layer_types
    ]
    if len(layer_types) != n_layers:
        raise ValueError(f"layer_types length ({len(layer_types)}) must match n_layers ({n_layers})")

    if mlp_layer_types is None:
        dense = max(0, min(first_k_dense_replace, n_layers))
        mlp_layer_types = ["dense"] * dense + ["sparse"] * (n_layers - dense)
    if len(mlp_layer_types) != n_layers:
        raise ValueError(
            f"mlp_layer_types length ({len(mlp_layer_types)}) must match n_layers ({n_layers})"
        )

    block_types = []
    for index, (layer_type, mlp_type) in enumerate(zip(layer_types, mlp_layer_types)):
        mixer = _MIXER_FOR_LAYER_TYPE.get(layer_type)
        if mixer is None:
            raise ValueError(
                f"Unsupported glm5_next layer type '{layer_type}' at layer {index}. "
                "Expected 'linear_attention' or 'deepseek_sparse_attention'"
            )
        if mlp_type not in ("dense", "sparse"):
            raise ValueError(
                f"Unsupported glm5_next mlp layer type '{mlp_type}' at layer {index}. "
                "Expected 'dense' or 'sparse'"
            )
        block_types.append(mixer if mlp_type == "dense" else f"{mixer}_moe")
    return block_types, layer_types, mlp_layer_types


@nn.hf_config(
    architecture="Glm5NextForConditionalGeneration",
    model_type="glm5_next",
    vocab_size="text_config.vocab_size",
    d_model="text_config.hidden_size",
    n_layers="text_config.num_hidden_layers",
    num_attention_heads="text_config.num_attention_heads",
    d_ff="text_config.intermediate_size",
    moe_d_ff="text_config.moe_intermediate_size",
    max_seq="text_config.max_position_embeddings",
    eps="text_config.rms_norm_eps",
    # MoE
    num_experts="text_config.n_routed_experts",
    num_experts_per_tok="text_config.num_experts_per_tok",
    n_shared_experts="text_config.n_shared_experts",
    routed_scaling_factor="text_config.routed_scaling_factor",
    norm_topk_prob="text_config.norm_topk_prob",
    n_group="text_config.n_group",
    topk_group="text_config.topk_group",
    swiglu_limit="text_config.swiglu_limit",
    first_k_dense_replace="text_config.first_k_dense_replace",
    mlp_layer_types="text_config.mlp_layer_types",
    # MLA
    q_lora_rank="text_config.q_lora_rank",
    kv_lora_rank="text_config.kv_lora_rank",
    qk_nope_head_dim="text_config.qk_nope_head_dim",
    qk_rope_head_dim="text_config.qk_rope_head_dim",
    v_head_dim="text_config.v_head_dim",
    layer_types="text_config.layer_types",
    # KDA
    linear_attn_config="text_config.linear_attn_config",
    linear_num_heads="text_config.linear_num_heads",
    linear_head_dim="text_config.linear_head_dim",
    linear_conv_kernel_dim="text_config.linear_conv_kernel_dim",
    linear_lower_bound="text_config.linear_lower_bound",
    # mHC
    hc_mult="text_config.hc_mult",
    hc_eps="text_config.hc_eps",
    hc_sinkhorn_iters="text_config.hc_sinkhorn_iters",
    # The NextN draft head's depth: serving carries the head, training does not.
    nextn_predict_layers="text_config.nextn_predict_layers",
    # Deferred subsystems: captured for the serve-spec generator (see docstring)
    index_topk="text_config.index_topk",
    index_head_dim="text_config.index_head_dim",
    index_n_heads="text_config.index_n_heads",
    index_kpool="text_config.index_kpool",
    index_kpool_always_select_tail="text_config.index_kpool_always_select_tail",
    indexer_types="text_config.indexer_types",
    tie_word_embeddings="tie_word_embeddings",
    use_visual_inputs="vision_config",
)
class Glm5NextConditionalModel(nn.Model):
    """GLM-5.3-Flash text stack for ``Glm5NextForConditionalGeneration``."""

    #: The endpoints. Per-layer objects come from whichever of the four block schemas the
    #: schedule puts on that layer.
    _serve_objects_ = (
        ServeObject("text/token_embedding", "quantised", ("Vocab", "C"), ("embedding",),
                    scope="model"),
        ServeObject("text/final_norm", "bf16", ("C",), ("final_norm",), scope="model"),
        ServeObject("text/output_head", "quantised", ("Vocab", "C"), ("lm_head",), scope="model"),
    )
    #: Two independent axes -- which mixer, and whether the feed-forward is dense or a mixture --
    #: so four block kinds rather than one per layer position.
    _serve_blocks_ = {
        "kda": Glm5NextKdaDenseBlock,
        "kda_moe": Glm5NextKdaMoEBlock,
        "mla": Glm5NextMlaDenseBlock,
        "mla_moe": Glm5NextMlaMoEBlock,
    }
    _serve_sections_ = (GLM5_NEXT_MTP_SERVE_SECTION,)

    @staticmethod
    def _serve_block_schedule_(config: dict) -> list[str]:
        """Which block runs at each layer.

        Both axes come from the checkpoint: `layer_types` says where the attention is -- an
        irregular list, not a period, ending 39, 43, 45 on the released model -- and
        `mlp_layer_types` (or `first_k_dense_replace`) says which layers are dense. The
        resolver is the one training uses, so the two schedules cannot drift apart.
        """
        block_types, _, _ = _resolve_glm5_next_block_types(
            n_layers=int(config["n_layers"]),
            layer_types=config.get("layer_types"),
            mlp_layer_types=config.get("mlp_layer_types"),
            first_k_dense_replace=int(config.get("first_k_dense_replace", 0) or 0),
        )
        return block_types

    _name_remap_ = GLM5_NEXT_MODEL_NAME_REMAP
    _hf_block_mappings_ = _build_glm5_next_block_mappings(
        "model.language_model.layers.{layer}", "model.language_model"
    )

    def __init__(
        self,
        vocab_size: int = 154880,
        d_model: int = 4096,
        n_layers: int = 45,
        num_attention_heads: int = 64,
        d_ff: int = 12288,
        moe_d_ff: int = 2048,
        max_seq: int = 1048576,
        eps: float = 1e-5,
        num_experts: int = 288,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        routed_scaling_factor: float = 2.5,
        norm_topk_prob: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        swiglu_limit: float = 10.0,
        first_k_dense_replace: int = 3,
        mlp_layer_types: list[str] | None = None,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 256,
        qk_rope_head_dim: int = 0,
        v_head_dim: int = 256,
        layer_types: list[str] | None = None,
        linear_attn_config: dict | None = None,
        # Sentinel defaults: absent from the HF config means "not stated", which
        # is what lets the compiler's config export carry the RESOLVED geometry
        # instead of echoing a stale flat key (see ``_resolve_kda_geometry``).
        linear_num_heads: int | None = None,
        linear_head_dim: int | None = None,
        linear_conv_kernel_dim: int | None = None,
        linear_lower_bound: float | None = None,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        nextn_predict_layers: int = 0,
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        index_kpool: int = 16,
        index_kpool_always_select_tail: bool = True,
        indexer_types: list[str] | None = None,
        tie_word_embeddings: bool = False,
        chunk_size: int = 64,
        ep_size: int = 1,
        use_visual_inputs: bool | dict | None = False,
    ):
        super().__init__()

        (
            linear_num_heads,
            linear_head_dim,
            linear_conv_kernel_dim,
            linear_lower_bound,
        ) = _resolve_kda_geometry(
            linear_attn_config,
            linear_num_heads,
            linear_head_dim,
            linear_conv_kernel_dim,
            linear_lower_bound,
        )

        if qk_rope_head_dim != 0:
            raise ValueError(
                "glm5_next is a NoPE architecture: qk_rope_head_dim must be 0, "
                f"got {qk_rope_head_dim}"
            )
        if q_lora_rank is None or q_lora_rank <= 0:
            raise ValueError("glm5_next requires a positive q_lora_rank for its MLA layers")
        if index_kpool and index_topk % index_kpool != 0:
            raise ValueError(
                f"index_topk ({index_topk}) must be divisible by index_kpool ({index_kpool})"
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_attention_heads = num_attention_heads
        # Validated equal by the HF config; kept so generic tooling that asks
        # for a KV-head count gets the right answer.
        self.num_query_heads = num_attention_heads
        self.num_kv_heads = num_attention_heads
        self.d_ff = d_ff
        self.moe_d_ff = moe_d_ff
        self.max_seq = max_seq
        self.eps = eps

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.shared_expert_intermediate = moe_d_ff * n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.swiglu_limit = swiglu_limit
        self.first_k_dense_replace = first_k_dense_replace

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        # Head size in the generic sense: Q/K and V are the same width here.
        self.D = self.qk_head_dim
        self.head_size = self.qk_head_dim

        self.linear_num_heads = linear_num_heads
        self.linear_head_dim = linear_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_lower_bound = linear_lower_bound

        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters

        # The draft head is not in the training graph; the count reaches the serve
        # declaration, which emits the `mtp/` section that many times (0 or 1).
        self.nextn_predict_layers = nextn_predict_layers

        # Deferred subsystems — config only (see module docstring).
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_kpool = index_kpool
        self.index_kpool_always_select_tail = bool(index_kpool_always_select_tail)
        self.indexer_types = list(indexer_types) if indexer_types else None
        self.has_dsa_indexer = index_n_heads > 0

        self.tie_word_embeddings = tie_word_embeddings
        self.chunk_size = chunk_size
        self.ep_size = ep_size
        self.use_visual_inputs = bool(use_visual_inputs)

        block_types, layer_types, mlp_layer_types = _resolve_glm5_next_block_types(
            n_layers=n_layers,
            layer_types=layer_types,
            mlp_layer_types=mlp_layer_types,
            first_k_dense_replace=first_k_dense_replace,
        )
        self.block_types = block_types
        self.layer_types = layer_types
        self.mlp_layer_types = mlp_layer_types
        self.hybrid_pattern = "".join(
            {"kda": "K", "kda_moe": "k", "mla": "A", "mla_moe": "m"}[t] for t in block_types
        )

        self.n_kda_blocks = sum(1 for t in block_types if t == "kda")
        self.n_kda_moe_blocks = sum(1 for t in block_types if t == "kda_moe")
        self.n_mla_blocks = sum(1 for t in block_types if t == "mla")
        self.n_mla_moe_blocks = sum(1 for t in block_types if t == "mla_moe")
        self.has_kda_blocks = (self.n_kda_blocks + self.n_kda_moe_blocks) > 0
        self.has_mla_blocks = (self.n_mla_blocks + self.n_mla_moe_blocks) > 0

        hc_kwargs = dict(hc_mult=hc_mult, hc_eps=hc_eps, hc_sinkhorn_iters=hc_sinkhorn_iters, swiglu_limit=swiglu_limit)
        kda_kwargs = dict(
            linear_num_heads=linear_num_heads,
            linear_head_dim=linear_head_dim,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            linear_lower_bound=linear_lower_bound,
            chunk_size=chunk_size,
        )
        mla_kwargs = dict(
            num_attention_heads=num_attention_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        moe_kwargs = dict(
            moe_intermediate_size=moe_d_ff,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            shared_expert_intermediate=self.shared_expert_intermediate,
            routed_scaling_factor=routed_scaling_factor,
            ep_size=ep_size,
        )

        block_configs = []
        if self.n_kda_blocks:
            block_configs.append(
                (
                    "kda_blocks",
                    Glm5NextKdaDenseBlock,
                    self.n_kda_blocks,
                    dict(d_model=d_model, intermediate_size=d_ff, eps=eps, **hc_kwargs, **kda_kwargs),
                )
            )
        if self.n_kda_moe_blocks:
            block_configs.append(
                (
                    "kda_moe_blocks",
                    Glm5NextKdaMoEBlock,
                    self.n_kda_moe_blocks,
                    dict(d_model=d_model, eps=eps, **hc_kwargs, **kda_kwargs, **moe_kwargs),
                )
            )
        if self.n_mla_blocks:
            block_configs.append(
                (
                    "mla_blocks",
                    Glm5NextMlaDenseBlock,
                    self.n_mla_blocks,
                    dict(d_model=d_model, intermediate_size=d_ff, eps=eps, **hc_kwargs, **mla_kwargs),
                )
            )
        if self.n_mla_moe_blocks:
            block_configs.append(
                (
                    "mla_moe_blocks",
                    Glm5NextMlaMoEBlock,
                    self.n_mla_moe_blocks,
                    dict(d_model=d_model, eps=eps, **hc_kwargs, **mla_kwargs, **moe_kwargs),
                )
            )

        self.embedding = Embedding(vocab_size, d_model)
        self.stream_init = StreamBroadcast(d_model, hc_mult)
        self.hybrid_blocks = nn.HybridBlockStack(
            block_configs=block_configs,
            block_types=block_types,
            n_layers=n_layers,
        )
        self.hc_head = Glm5NextHyperHead(d_model, hc_mult)
        self.final_norm = RMSNorm(d_model, eps=eps)
        self.lm_head = LMHead(vocab_size, d_model)

    def forward(self, token_ids, position_ids, targets):
        G = ActivationScope.GLOBAL

        # No rotary table is needed. Position IDs mark packed sample boundaries
        # for the convolution and KDA recurrence.
        self._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("position_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])

        _h = ("B", "T", "d_model")
        _wide = ("B", "T", "hc_mult * d_model")
        self._register_activation("residual0", _wide, scope=G)
        self._register_activation("x0", _h, aliases=["encoded"], scope=G)
        self._register_activation("xN", _h, scope=G)
        self._register_activation("residualN", _wide, scope=G)
        self._register_activation("xF", _h, aliases=["ln_final"], scope=G)
        self._register_activation("xF_flat", ("B * T", "d_model"), scope=G)
        self._register_activation("ln_final_rstd", ("B", "T"), dtype="fp32", save=True, scope=G)
        self._register_activation("loss", ("B * T",), dtype="fp32", aliases=["losses"], scope=G)

        x = self.embedding(token_ids)
        residual = self.stream_init(x)
        x, residual = self.hybrid_blocks(x, residual, position_ids)
        pooled = self.hc_head(residual)
        xf = self.final_norm(pooled)
        loss = self.lm_head(xf, targets)
        return loss
