"""Where every GLM-5.3-Flash artifact object comes from.

There is no bridged HF checkpoint here: gguf-py carries no `glm5next` name map, so the six
GGUF shards *are* the source and these recipes name their tensors directly. Declared shapes are
the GGUF's logical ones (reversed `ne`). The expression language, the row algebra that reads
it and the planners that turn it into runs are the shared ones.

Almost everything is a row program over one or three sources, which is what lets the 185 GB of
experts stay in the file. Three objects are not, and each is not for its own reason:

- **the convolution taps.** The file holds one `[channels, K]` tensor per projection, ggml's
  channel-major order; the op reads `weight[tap * C + c]`. Concatenating the three and
  transposing is a reorder inside a row.
- ~~the key half of the latent expansion~~ -- no longer. llama.cpp stores `attn_k_b` as
  `[latent, nope]` per head, the orientation it is applied to the *query* in, and since the
  engine serves the attention absorbed it applies it to the query too. What once had to be
  transposed into BF16 is now read where it lies, like the value half always was.
- **the router**, which the file keeps in F32 and the artifact holds in BF16.
- **the decay's `A_log`.** llama.cpp stores `ssm_a = -exp(A_log)` so that its own graph
  multiplies by the tensor as it lies; the engine's gate, like the checkpoint's, is written in
  terms of `A_log`. Undoing the fold is a log of a negation, which no row program says, so the
  driver materialises it (`materialized_objects`). Bound as stored, the gate ran at
  `exp(-exp(A_log))` -- between 6e-6 and 0.4 where the model wants 1 to 12 -- and every
  linear-attention state forgot its past at one rate whatever the token said.

Everything else -- the fused KDA q|k|v, the fused dense and shared gate/up, and every one of
the 288 experts per layer -- is rows drawn from mapped sources, read where they lie.
"""

from __future__ import annotations

from surogate.serve.convert.common.recipe import (
    Concat,
    Expression,
    Reshape,
    Slice,
    SourceTensor,
    TensorRecipe,
    Transpose,
    expression_shape,
    expression_sources,
    materialize_expression,
    materialize_recipe,
    source,
    validate_recipe_coverage,
)

from . import inventory as inv

RECIPE_ID = "glm5-next-v4"

#: Objects whose op reads the row-split W8 planes rather than the file's own block format.
#: Their weights are still read from the GGUF where they lie -- Q8_0 and W8G32_F16S hold the
#: same numbers, so the loader rearranges them on the device -- and this list is what routes
#: them that way. What is left out is the mixture, which llama.cpp stores as K-quants that no
#: plane decoder reproduces and the engine's own K-quant kernels read directly.
NATIVE_EXCLUDE_SUFFIXES = (
    "text/token_embedding",
    "text/output_head",
    "kda/query_key_value",
    "kda/output",
    "kda/decay_a",
    "kda/decay_b",
    "kda/gate_a",
    "kda/gate_b",
    "kda/beta",
    "mla/query_a",
    "mla/query_b",
    "mla/kv_a",
    "mla/k_b",
    "mla/v_b",
    "mla/output",
    "mlp/gate_up",
    "mlp/down",
    "moe/shared_gate_up",
    "moe/shared_down",
    "mtp/input_projection",
)


def materialized_objects(geometry: inv.Geometry) -> frozenset[str]:
    """Objects the expression language cannot say, which the driver computes instead.

    One kind: the KDA decay's `A_log`, stored by llama.cpp as `-exp(A_log)`. Every other value
    transform a GGUF needs is a rearrangement of rows, and the planners read those in place.
    """
    return frozenset(
        f"text/layers/{layer}/kda/a_log"
        for layer in range(geometry.layers)
        if not geometry.is_attention(layer)
    )


def _blk(layer: int) -> str:
    return f"blk.{layer}."


def _hyper_connection(layer: int, prefix: str, geometry: inv.Geometry) -> list[TensorRecipe]:
    blk = _blk(layer)
    out: list[TensorRecipe] = []
    for site in ("attn", "ffn"):
        out += [
            TensorRecipe(
                f"{prefix}hc/{site}_mix",
                source(f"{blk}hc_{site}_fn.weight", (geometry.hc_mix_rows, geometry.residual)),
            ),
            TensorRecipe(
                f"{prefix}hc/{site}_base",
                source(f"{blk}hc_{site}_base.weight", (geometry.hc_mix_rows,)),
            ),
            TensorRecipe(
                f"{prefix}hc/{site}_scale", source(f"{blk}hc_{site}_scale.weight", (3,))
            ),
        ]
    return out


def _kda(layer: int, prefix: str, geometry: inv.Geometry) -> list[TensorRecipe]:
    blk = _blk(layer)
    dim = geometry.kda_dim
    hidden = geometry.hidden
    taps = Concat(
        tuple(
            Reshape(
                source(f"{blk}ssm_conv1d_{leaf}.weight", (dim, 1, geometry.kda_conv_kernel)),
                (dim, geometry.kda_conv_kernel),
            )
            for leaf in ("q", "k", "v")
        ),
        0,
    )
    return [
        TensorRecipe(
            f"{prefix}kda/query_key_value",
            Concat(
                tuple(
                    source(f"{blk}attn_{leaf}.weight", (dim, hidden)) for leaf in ("q", "k", "v")
                ),
                0,
            ),
        ),
        # ggml holds the taps channel-major, one `[channels, K]` plane per projection; the op
        # reads `weight[tap * C + c]`, so the three stack on the channel axis and the result
        # transposes.
        TensorRecipe(f"{prefix}kda/convolution", Transpose(taps, (1, 0))),
        TensorRecipe(
            f"{prefix}kda/decay_a",
            source(f"{blk}ssm_f_a.weight", (geometry.kda_gate_rank, hidden)),
        ),
        TensorRecipe(
            f"{prefix}kda/decay_b",
            source(f"{blk}ssm_f_b.weight", (dim, geometry.kda_gate_rank)),
        ),
        TensorRecipe(f"{prefix}kda/decay_bias", source(f"{blk}ssm_dt.bias", (dim,))),
        # `kda/a_log` is not a row program: see `materialized_objects`.
        TensorRecipe(
            f"{prefix}kda/beta", source(f"{blk}ssm_beta.weight", (geometry.kda_heads, hidden))
        ),
        TensorRecipe(
            f"{prefix}kda/gate_a",
            source(f"{blk}ssm_g_a.weight", (geometry.kda_gate_rank, hidden)),
        ),
        TensorRecipe(
            f"{prefix}kda/gate_b",
            source(f"{blk}ssm_g_b.weight", (dim, geometry.kda_gate_rank)),
        ),
        TensorRecipe(f"{prefix}kda/norm", source(f"{blk}ssm_norm.weight", (geometry.kda_head_dim,))),
        TensorRecipe(f"{prefix}kda/output", source(f"{blk}attn_output.weight", (hidden, dim))),
    ]


def _mla(layer: int, prefix: str, geometry: inv.Geometry) -> list[TensorRecipe]:
    blk = _blk(layer)
    heads = geometry.query_heads
    latent = geometry.kv_lora_rank
    nope = geometry.qk_head_dim
    value = geometry.v_head_dim
    # `attn_k_b` is [latent, nope] per head, the orientation it is applied to the *query* in;
    # the served attention is absorbed and applies it that way, so it is read as stored. The
    # value half is the projection, [value, latent] per head, and unfolds the attended latent.
    # Read as stored: [latent, nope] per head is the orientation the absorbed attention applies
    # it in, so no transpose and no materialisation.
    k_b = source(f"{blk}attn_k_b.weight", (heads, latent, nope))
    v_b = source(f"{blk}attn_v_b.weight", (heads, value, latent))
    return [
        TensorRecipe(
            f"{prefix}mla/query_a", source(f"{blk}attn_q_a.weight", (geometry.q_lora_rank, geometry.hidden))
        ),
        TensorRecipe(
            f"{prefix}mla/query_a_norm", source(f"{blk}attn_q_a_norm.weight", (geometry.q_lora_rank,))
        ),
        TensorRecipe(
            f"{prefix}mla/query_b",
            source(f"{blk}attn_q_b.weight", (heads * nope, geometry.q_lora_rank)),
        ),
        TensorRecipe(
            f"{prefix}mla/kv_a", source(f"{blk}attn_kv_a_mqa.weight", (latent, geometry.hidden))
        ),
        TensorRecipe(f"{prefix}mla/kv_a_norm", source(f"{blk}attn_kv_a_norm.weight", (latent,))),
        TensorRecipe(f"{prefix}mla/k_b", Reshape(k_b, (heads * latent, nope))),
        TensorRecipe(f"{prefix}mla/v_b", Reshape(v_b, (heads * value, latent))),
        TensorRecipe(
            f"{prefix}mla/output",
            source(f"{blk}attn_output.weight", (geometry.hidden, heads * value)),
        ),
    ]


def _dense_ffn(layer: int, prefix: str, geometry: inv.Geometry) -> list[TensorRecipe]:
    blk = _blk(layer)
    width = geometry.dense_intermediate
    return [
        TensorRecipe(
            f"{prefix}mlp/gate_up",
            Concat(
                tuple(
                    source(f"{blk}ffn_{leaf}.weight", (width, geometry.hidden))
                    for leaf in ("gate", "up")
                ),
                0,
            ),
        ),
        TensorRecipe(f"{prefix}mlp/down", source(f"{blk}ffn_down.weight", (geometry.hidden, width))),
    ]


def _moe(layer: int, prefix: str, geometry: inv.Geometry) -> list[TensorRecipe]:
    blk = _blk(layer)
    experts = geometry.experts
    width = geometry.expert_intermediate
    hidden = geometry.hidden
    shared = geometry.shared_intermediate
    # The file stacks the experts on the outermost axis, one `[width, hidden]` plane each. The
    # fused parameter wants an expert's gate rows followed by its own up rows, so the two
    # stacks interleave per expert rather than concatenating whole -- which is a row program,
    # so all 288 stay where they are.
    gate = source(f"{blk}ffn_gate_exps.weight", (experts, width, hidden))
    up = source(f"{blk}ffn_up_exps.weight", (experts, width, hidden))
    per_expert = tuple(
        half
        for expert in range(experts)
        for half in (
            Reshape(Slice(gate, 0, expert, expert + 1), (width, hidden)),
            Reshape(Slice(up, 0, expert, expert + 1), (width, hidden)),
        )
    )
    out = [
        TensorRecipe(f"{prefix}moe/router", source(f"{blk}ffn_gate_inp.weight", (experts, hidden))),
        TensorRecipe(f"{prefix}moe/router_bias", source(f"{blk}exp_probs_b.bias", (experts,))),
        TensorRecipe(f"{prefix}moe/routed_gate_up", Concat(per_expert, 0)),
        TensorRecipe(
            f"{prefix}moe/routed_down",
            Reshape(
                source(f"{blk}ffn_down_exps.weight", (experts, hidden, width)),
                (experts * hidden, width),
            ),
        ),
    ]
    if shared:
        out += [
            TensorRecipe(
                f"{prefix}moe/shared_gate_up",
                Concat(
                    tuple(
                        source(f"{blk}ffn_{leaf}_shexp.weight", (shared, hidden))
                        for leaf in ("gate", "up")
                    ),
                    0,
                ),
            ),
            TensorRecipe(
                f"{prefix}moe/shared_down",
                source(f"{blk}ffn_down_shexp.weight", (hidden, shared)),
            ),
        ]
    return out


def _mtp(geometry: inv.Geometry) -> list[TensorRecipe]:
    """The NextN draft head: the block past the trunk, under `mtp/`.

    Its layer is the trunk's latent-attention-over-mixture layer, so the two builders above
    read it unchanged; only the fold's three tensors and the read-out norm are the head's own.
    The GGUF carries neither an embedding table nor an LM head for it -- both are the trunk's.
    """
    if geometry.nextn_layers == 0:
        return []
    if geometry.nextn_layers != 1:
        raise NotImplementedError(
            f"this checkpoint declares {geometry.nextn_layers} NextN layers; the served draft "
            f"head is one layer deep"
        )
    layer = geometry.layers
    blk = _blk(layer)
    hidden = geometry.hidden
    out = [
        TensorRecipe(
            "mtp/input_projection", source(f"{blk}nextn.eh_proj.weight", (hidden, 2 * hidden))
        ),
        TensorRecipe("mtp/embedding_norm", source(f"{blk}nextn.enorm.weight", (hidden,))),
        TensorRecipe("mtp/hidden_norm", source(f"{blk}nextn.hnorm.weight", (hidden,))),
        TensorRecipe("mtp/layer/input_norm", source(f"{blk}attn_norm.weight", (hidden,))),
        *_mla(layer, "mtp/layer/", geometry),
        TensorRecipe("mtp/layer/post_attention_norm", source(f"{blk}ffn_norm.weight", (hidden,))),
        *_moe(layer, "mtp/layer/", geometry),
        TensorRecipe(
            "mtp/final_norm", source(f"{blk}nextn.shared_head_norm.weight", (hidden,))
        ),
    ]
    return out


def build_recipes(geometry: inv.Geometry) -> dict[str, TensorRecipe]:
    """Every object's source expression, keyed by object name."""
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "text/token_embedding", source("token_embd.weight", (geometry.vocab, geometry.hidden))
        ),
        TensorRecipe("text/final_norm", source("output_norm.weight", (geometry.hidden,))),
        TensorRecipe(
            "text/output_head", source("output.weight", (geometry.vocab, geometry.hidden))
        ),
    ]
    for layer in range(geometry.layers):
        prefix = f"text/layers/{layer}/"
        blk = _blk(layer)
        recipes += _hyper_connection(layer, prefix, geometry)
        recipes.append(
            TensorRecipe(f"{prefix}input_norm", source(f"{blk}attn_norm.weight", (geometry.hidden,)))
        )
        recipes += (
            _mla(layer, prefix, geometry)
            if geometry.is_attention(layer)
            else _kda(layer, prefix, geometry)
        )
        recipes.append(
            TensorRecipe(
                f"{prefix}post_attention_norm", source(f"{blk}ffn_norm.weight", (geometry.hidden,))
            )
        )
        recipes += (
            _dense_ffn(layer, prefix, geometry)
            if geometry.is_dense(layer)
            else _moe(layer, prefix, geometry)
        )
    recipes += _mtp(geometry)
    return {recipe.object_name: recipe for recipe in recipes}


__all__ = [
    "NATIVE_EXCLUDE_SUFFIXES",
    "RECIPE_ID",
    "build_recipes",
    "expression_sources",
    "materialized_objects",
    "materialize_expression",
    "materialize_recipe",
    "validate_recipe_coverage",
]
