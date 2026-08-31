"""Gemma3 text models.

Two checkpoints share this backbone and differ in what sits on top of it:

* ``Gemma3ForCausalLM`` -- the generative 1B/4B/12B/27B models. Tensors live
  under ``model.``, the LM head is tied to the embedding table, attention is
  causal.
* ``Gemma3TextModel`` -- what ``google/embeddinggemma-300m`` publishes. Tensors
  sit at the *root* (``embed_tokens.weight``, ``layers.N.*``, ``norm.weight``),
  there is **no** LM head at all, and ``use_bidirectional_attention`` is true.

So the prefix is a parameter and the head is a choice, not a constant.

Layer schedule: ``_sliding_window_pattern`` (6) makes every 6th layer global and
the rest local, i.e. full attention at 5/11/17/23 for a 24-layer model. The two
kinds also carry *different rope bases* -- ``rope_theta`` (1e6) for global layers
and ``rope_local_base_freq`` (1e4) for local ones -- which reaches the runtime
through the ``full_rope_theta`` / ``sliding_rope_theta`` config attributes that
``dsl_model.cpp`` already reads per layer type.

Gemma4 reads those two bases from a nested ``rope_parameters.*`` block; Gemma 3
publishes them flat, which is why this model needs its own ``@nn.hf_config``
rather than another entry in Gemma4's.
"""

from __future__ import annotations

from .. import nn
from ..block_schema import ServeObject
from ..blocks.gemma3 import GEMMA3_BLOCK_NAME_REMAP, Gemma3FullBlock, Gemma3SlidingBlock
from ..hf import fuse
from ..modules import LMHead, RMSNorm, ScaledEmbedding
from ..specs import ActivationScope


GEMMA3_MODEL_NAME_REMAP: dict[str, str] = {
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


#: Model-level objects as a serving artifact stores them; the per-layer ones are
#: declared on the block schemas.
#:
#: The final norm carries ``unfold_unit_offset`` for the same reason every block
#: norm does: Gemma stores RMSNorm weights zero-centred and uses them as ``1 + w``.
GEMMA3_MODEL_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("text/token_embedding", "quantised", ("Vocab", "C"), ("embedding",), scope="model"),
    ServeObject("text/final_norm", "bf16", ("C",), ("final_norm",), scope="model",
                transform="unfold_unit_offset"),
)

#: The generative variant's head. EmbeddingGemma has none -- its checkpoint stops
#: at ``norm.weight`` -- which is why this is separate rather than a member of the
#: tuple above.
GEMMA3_OUTPUT_HEAD_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("text/output_head", "quantised", ("Vocab", "C"), ("lm_head",), scope="model"),
)

#: EmbeddingGemma's head, as one matrix.
#:
#: The checkpoint ships it as two sentence-transformers modules -- 2_Dense
#: (768->3072) then 3_Dense (3072->768) -- and both declare an *Identity*
#: activation. Two linear maps with nothing between them compose, so the artifact
#: stores their product: one [C, C] instead of 4.7M parameters, and one GEMM per
#: request instead of two.
#:
#: Measured on the real weights, this is not a trade. Folding rounds once where
#: the pair rounds twice, so against an fp32 reference it comes out marginally
#: ahead in bf16 (min cosine 0.99999678 folded, 0.99999672 unfolded) and within
#: 7.6e-08 in fp32.
#:
#: ``components`` is empty because those two matrices are not declared parameters
#: of this model: they live in `2_Dense/` and `3_Dense/` sub-directories that the
#: training weight loader does not reach, so the head is described for serving
#: only and the converter reads them from the checkpoint. Same shape as the
#: n-gram tables on qwen4exp, which are artifact objects with no declared source.
EMBEDDING_GEMMA_HEAD_SERVE_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("text/embedding_head", "bf16", ("C", "C"), (),
                transform="compose_linear", scope="model", capability="embedding"),
)

#: How the pooled vector is produced, for a target that implements ``embedding``.
#:
#: ``include_prompt`` matters and is easy to miss: the task prefix
#: ("task: search result | query: ") is part of the mean, not stripped before it.
#: Matryoshka dimensions truncate *then* renormalise -- the other order gives
#: vectors that are not unit-norm and silently degrades cosine ranking.
#:
#: Set on the instance, which makes the class the source of truth, but note that
#: only the int and the bool currently reach the compiled IR's config dict:
#: ``py_compiler`` serialises non-declared instance attributes through a filter
#: that admits ints and bools and drops strings and tuples. Widening it would
#: also widen what dimension resolution sees, so the serve generator should read
#: these from the class rather than the IR until there is a reason to change it.
EMBEDDING_GEMMA_POOLING = {
    "pooling": "mean",
    "pooling_include_prompt": True,
    "embedding_normalize": "l2",
    "embedding_dim": 768,
    "matryoshka_dims": (768, 512, 256, 128),
}


def _parse_gemma3_layer_types(
    layer_types: list[str] | None,
    n_layers: int,
    sliding_window_pattern: int,
) -> list[str]:
    """Resolve the local/global schedule.

    Gemma 3 states it as a period rather than a list: layer ``i`` is global when
    ``(i + 1) % pattern == 0``. For 24 layers at period 6 that is 5/11/17/23 --
    the schedule embeddinggemma-300m ships. An explicit ``layer_types`` list from
    the config wins when present.
    """
    if not layer_types:
        layer_types = [
            "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
            for i in range(n_layers)
        ]
    if len(layer_types) != n_layers:
        raise ValueError(f"layer_types length ({len(layer_types)}) != n_layers ({n_layers})")
    out: list[str] = []
    for t in layer_types:
        if t == "sliding_attention":
            out.append("sliding")
        elif t == "full_attention":
            out.append("full")
        else:
            raise ValueError(f"Unsupported Gemma3 layer type '{t}'.")
    return out


def _gemma3_layer_mappings(layer_prefix: str) -> dict[str, object]:
    """HF weight mappings for per-layer Gemma3 block parameters."""
    return {
        # Sandwich norms, 4 per layer.
        "ln1_weight": f"{layer_prefix}.input_layernorm.weight",
        "ln_post_attn_weight": f"{layer_prefix}.post_attention_layernorm.weight",
        "ln2_weight": f"{layer_prefix}.pre_feedforward_layernorm.weight",
        "ln_post_ff_weight": f"{layer_prefix}.post_feedforward_layernorm.weight",
        # Attention. Q/K norm only -- Gemma 3 has no V-norm.
        "qkv_weight": fuse(
            f"{layer_prefix}.self_attn.q_proj.weight",
            f"{layer_prefix}.self_attn.k_proj.weight",
            f"{layer_prefix}.self_attn.v_proj.weight",
            dim=0,
        ),
        "out_weight": f"{layer_prefix}.self_attn.o_proj.weight",
        "q_norm_weight": f"{layer_prefix}.self_attn.q_norm.weight",
        "k_norm_weight": f"{layer_prefix}.self_attn.k_norm.weight",
        # MLP: separate gate/up matmuls, gated GELU.
        "mlp_gate_weight": f"{layer_prefix}.mlp.gate_proj.weight",
        "mlp_up_weight": f"{layer_prefix}.mlp.up_proj.weight",
        "mlp_down_weight": f"{layer_prefix}.mlp.down_proj.weight",
    }


def _build_gemma3_mappings(layer_prefix: str, model_prefix: str, *, tied_lm_head: bool) -> dict[str, object]:
    """Layer plus model-level HF mappings.

    ``model_prefix`` is empty for a bare ``Gemma3TextModel`` checkpoint, whose
    tensors sit at the root. Gemma 3 ties its LM head to the embedding table, so
    ``lm_head`` resolves to the same tensor rather than to a ``lm_head.weight``
    the checkpoint does not publish.
    """
    dot = f"{model_prefix}." if model_prefix else ""
    embedding = f"{dot}embed_tokens.weight"
    return {
        **_gemma3_layer_mappings(layer_prefix),
        "embedding": embedding,
        "final_norm": f"{dot}norm.weight",
        "lm_head": embedding if tied_lm_head else "lm_head.weight",
    }


def _build_gemma3_model(
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
    sliding_window_pattern,
    layer_types,
    sliding_rope_theta,
    full_rope_theta,
    query_pre_attn_scalar,
    causal,
):
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
    cls.query_pre_attn_scalar = query_pre_attn_scalar or head_size
    cls.causal = causal
    # Read back per layer type by dsl_model.cpp; Gemma 3's local layers use a
    # 100x smaller rope base than its global ones.
    cls.sliding_rope_theta = sliding_rope_theta
    cls.full_rope_theta = full_rope_theta
    cls.D = head_size

    cls.block_types = _parse_gemma3_layer_types(layer_types, n_layers, sliding_window_pattern)
    cls.n_sliding_blocks = sum(1 for t in cls.block_types if t == "sliding")
    cls.n_full_blocks = sum(1 for t in cls.block_types if t == "full")

    shared = dict(
        d_model=d_model,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        d_ff=d_ff,
        max_seq=max_seq,
        causal=causal,
        query_pre_attn_scalar=cls.query_pre_attn_scalar,
        eps=eps,
    )

    block_configs = []
    if cls.n_sliding_blocks > 0:
        block_configs.append(
            (
                "sliding_blocks",
                Gemma3SlidingBlock,
                cls.n_sliding_blocks,
                dict(sliding_window=sliding_window, **shared),
            )
        )
    if cls.n_full_blocks > 0:
        block_configs.append(("full_blocks", Gemma3FullBlock, cls.n_full_blocks, dict(shared)))

    # Gemma scales the embedding by sqrt(hidden_size). HF computes this in fp32
    # before casting, so a bf16 rounding of the scalar would not match.
    cls.embedding = ScaledEmbedding(vocab_size, d_model, embed_scale=float(d_model) ** 0.5)
    cls.hybrid_blocks = nn.HybridBlockStack(
        block_configs=block_configs,
        block_types=cls.block_types,
        n_layers=n_layers,
    )
    cls.final_norm = RMSNorm(d_model, eps=eps)
    cls.lm_head = LMHead(vocab_size, d_model)


def _gemma3_forward(model, token_ids, position_ids, targets):
    G = ActivationScope.GLOBAL

    model._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
    model._register_activation("position_ids", ("T",), dtype="int32", scope=G)
    model._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])
    model._register_activation("freq_cis", ("max_seq", "D", 2), dtype="fp32", scope=G, aliases=["rope_freqs"])

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

    x = model.embedding(token_ids)
    residual = model._zeros(["B", "T", "d_model"])
    x, residual = model.hybrid_blocks(x, residual, position_ids)
    x = model.final_norm(x)
    return model.lm_head(x, targets)


#: Config keys shared by both entry points. Gemma 3 publishes the two rope bases
#: flat, unlike Gemma4's nested ``rope_parameters`` block.
_GEMMA3_CONFIG_MAPPING = dict(
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
    sliding_window_pattern="_sliding_window_pattern",
    layer_types="layer_types",
    sliding_rope_theta="rope_local_base_freq",
    full_rope_theta="rope_theta",
    query_pre_attn_scalar="query_pre_attn_scalar",
)


class _Gemma3Base(nn.Model):
    _name_remap_ = GEMMA3_MODEL_NAME_REMAP
    #: Per-layer serve objects live on the block schemas; these are outside the
    #: stack. The output head is added only by the variant that has one.
    _serve_objects_ = GEMMA3_MODEL_SERVE_OBJECTS
    _serve_blocks_ = {"sliding": Gemma3SlidingBlock, "full": Gemma3FullBlock}

    def _init(
        self,
        *,
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
        sliding_window_pattern,
        layer_types,
        sliding_rope_theta,
        full_rope_theta,
        query_pre_attn_scalar,
        causal,
    ):
        _build_gemma3_model(
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
            sliding_window_pattern,
            layer_types,
            sliding_rope_theta,
            full_rope_theta,
            query_pre_attn_scalar,
            causal,
        )

    def forward(self, token_ids, position_ids, targets):
        return _gemma3_forward(self, token_ids, position_ids, targets)


@nn.hf_config(
    architecture="Gemma3ForCausalLM",
    model_type="gemma3",
    **_GEMMA3_CONFIG_MAPPING,
)
class Gemma3CausalModel(_Gemma3Base):
    """Generative Gemma 3 (1B/4B/12B/27B). Tensors under ``model.``, tied LM head."""

    _hf_block_mappings_ = _build_gemma3_mappings("model.layers.{layer}", "model", tied_lm_head=True)
    _serve_objects_ = GEMMA3_MODEL_SERVE_OBJECTS + GEMMA3_OUTPUT_HEAD_SERVE_OBJECTS

    def __init__(
        self,
        vocab_size: int = 262144,
        d_model: int = 1152,
        n_layers: int = 26,
        num_query_heads: int = 4,
        num_kv_heads: int = 1,
        d_ff: int = 6912,
        max_seq: int = 32768,
        head_size: int = 256,
        eps: float = 1e-6,
        sliding_window: int = 512,
        sliding_window_pattern: int = 6,
        layer_types: list[str] | None = None,
        sliding_rope_theta: float = 10000.0,
        full_rope_theta: float = 1000000.0,
        query_pre_attn_scalar: int = 256,
    ):
        super().__init__()
        self._init(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            d_ff=d_ff,
            max_seq=max_seq,
            head_size=head_size,
            eps=eps,
            sliding_window=sliding_window,
            sliding_window_pattern=sliding_window_pattern,
            layer_types=layer_types,
            sliding_rope_theta=sliding_rope_theta,
            full_rope_theta=full_rope_theta,
            query_pre_attn_scalar=query_pre_attn_scalar,
            causal=True,
        )


@nn.hf_config(
    architecture="Gemma3TextModel",
    model_type="gemma3_text",
    use_bidirectional_attention="use_bidirectional_attention",
    **_GEMMA3_CONFIG_MAPPING,
)
class Gemma3TextModel(_Gemma3Base):
    """Bare Gemma 3 text backbone -- what ``google/embeddinggemma-300m`` ships.

    Tensors sit at the checkpoint root with no ``model.`` prefix and there is no
    ``lm_head.weight``; the head declared here maps onto the (tied) embedding
    table so the backbone stays trainable, and the embedding model's real head
    -- mean pool, two dense projections, L2 normalise -- sits above it.

    ``use_bidirectional_attention`` is read from the config rather than assumed:
    it is what separates this checkpoint from a generative Gemma 3 of identical
    geometry, and it reaches the attention kernel through
    ``AttentionConfig.causal``.
    """

    _hf_block_mappings_ = _build_gemma3_mappings("layers.{layer}", "", tied_lm_head=True)
    _serve_objects_ = GEMMA3_MODEL_SERVE_OBJECTS + EMBEDDING_GEMMA_HEAD_SERVE_OBJECTS

    def __init__(
        self,
        vocab_size: int = 262144,
        d_model: int = 768,
        n_layers: int = 24,
        num_query_heads: int = 3,
        num_kv_heads: int = 1,
        d_ff: int = 1152,
        max_seq: int = 2048,
        head_size: int = 256,
        eps: float = 1e-6,
        sliding_window: int = 512,
        sliding_window_pattern: int = 6,
        layer_types: list[str] | None = None,
        sliding_rope_theta: float = 10000.0,
        full_rope_theta: float = 1000000.0,
        query_pre_attn_scalar: int = 256,
        use_bidirectional_attention: bool = False,
    ):
        super().__init__()
        self._init(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            d_ff=d_ff,
            max_seq=max_seq,
            head_size=head_size,
            eps=eps,
            sliding_window=sliding_window,
            sliding_window_pattern=sliding_window_pattern,
            layer_types=layer_types,
            sliding_rope_theta=sliding_rope_theta,
            full_rope_theta=full_rope_theta,
            query_pre_attn_scalar=query_pre_attn_scalar,
            causal=not use_bidirectional_attention,
        )
        for key, value in EMBEDDING_GEMMA_POOLING.items():
            setattr(self, key, value)
