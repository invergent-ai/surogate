"""Hugging Face source recipe for the complete interleaved gated-delta inventory."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from surogate.serve.convert.common.recipe import (
    SOURCE_DTYPE,
    Cast,
    Concat,
    DraftHeadTokenIds,
    Expression,
    GatherRows,
    Reshape,
    ShardReader,
    Slice,
    SourcePreflight,
    SourceTensor,
    TensorRecipe,
    Transpose,
    attention_qproj_part,
    build_vision_recipes,
    expression_shape,
    expression_sources,
    materialize_expression,
    materialize_recipe,
    preflight_sources as _preflight_recipe_sources,
    source,
    source_requirements as _recipe_source_requirements,
    validate_recipe_coverage as _validate_recipe_coverage,
)

from . import inventory


DRAFT_ROWS = 131072

#: Sources are named in the flat dialect. `ShardReader` folds the VL-style
#: `model.language_model.*` nesting the official releases use onto these names, so one
#: spelling reads both an official checkpoint and a GGUF-bridged one.
_TEXT_PREFIX = "model."
_EMBEDDING = _TEXT_PREFIX + "embed_tokens.weight"
_OUTPUT_HEAD = "lm_head.weight"


_sources = expression_sources
_source = source


def _output_head_source(g: "inventory.Geometry") -> str:
    """Where the vocabulary projection comes from. The smaller sizes tie it to the embedding
    and carry no `lm_head`; the 27B ships its own."""
    return _OUTPUT_HEAD if inventory.is_27b(g) else _EMBEDDING


def _attention_qproj_part(source_name: str, gate: bool,
                          g: "inventory.Geometry" = None) -> Expression:
    g = g or inventory.GEOMETRY
    return attention_qproj_part(
        source_name,
        gate,
        num_heads=g.query_heads,
        hidden_size=g.hidden,
    )


def _attention_projection_recipes(source_prefix: str, object_prefix: str,
                                  g: "inventory.Geometry") -> tuple[TensorRecipe, ...]:
    q_proj = source_prefix + "self_attn.q_proj.weight"
    query = _attention_qproj_part(q_proj, gate=False, g=g)
    gate = _attention_qproj_part(q_proj, gate=True, g=g)
    key = _source(source_prefix + "self_attn.k_proj.weight", (g.kv_size, g.hidden))
    value = _source(source_prefix + "self_attn.v_proj.weight", (g.kv_size, g.hidden))
    if inventory.is_27b(g):
        return (
            TensorRecipe(object_prefix + "attention/query_key", Concat((query, key), 0)),
            TensorRecipe(object_prefix + "attention/gate_value", Concat((gate, value), 0)),
        )
    return (
        TensorRecipe(
            object_prefix + "attention/query_key_gate_value",
            Concat((query, key, gate, value), 0),
        ),
    )


def _gdn_projection_recipes(source_prefix: str, object_prefix: str,
                            g: "inventory.Geometry") -> tuple[TensorRecipe, ...]:
    qkv = _source(source_prefix + "linear_attn.in_proj_qkv.weight",
                  (g.convolution_dim, g.hidden))
    z = _source(source_prefix + "linear_attn.in_proj_z.weight", (g.value_dim, g.hidden))
    if inventory.is_27b(g):
        return (
            TensorRecipe(object_prefix + "gdn/query_key",
                         Slice(qkv, 0, 0, 2 * g.key_dim)),
            TensorRecipe(
                object_prefix + "gdn/value_z",
                Concat((Slice(qkv, 0, 2 * g.key_dim, g.convolution_dim), z), 0),
            ),
        )
    return (
        TensorRecipe(object_prefix + "gdn/query_key_value_z", Concat((qkv, z), 0)),
    )


def _build_text_recipes(g: "inventory.Geometry" = None) -> tuple[TensorRecipe, ...]:
    g = g or inventory.GEOMETRY
    recipes: list[TensorRecipe] = [
        TensorRecipe("text/token_embedding", _source(_EMBEDDING, (g.vocab, g.hidden)))
    ]

    for layer in range(g.layers):
        source_prefix = f"{_TEXT_PREFIX}layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            TensorRecipe(
                object_prefix + "input_norm",
                _source(source_prefix + "input_layernorm.weight", (g.hidden,)),
            )
        )

        if layer in set(g.full_attention_layers):
            recipes.extend(_attention_projection_recipes(source_prefix, object_prefix, g))
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "attention/query_norm",
                        _source(source_prefix + "self_attn.q_norm.weight", (g.head_dim,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/key_norm",
                        _source(source_prefix + "self_attn.k_norm.weight", (g.head_dim,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/output",
                        _source(source_prefix + "self_attn.o_proj.weight", (g.hidden, g.query_size)),
                    ),
                )
            )
        else:
            convolution = _source(
                source_prefix + "linear_attn.conv1d.weight",
                (g.convolution_dim, 1, g.gdn_conv_kernel),
            )
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "gdn/a_log",
                        Cast(_source(source_prefix + "linear_attn.A_log", (g.gdn_value_heads,)), inventory.FP32),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/dt_bias",
                        Cast(_source(source_prefix + "linear_attn.dt_bias", (g.gdn_value_heads,)), inventory.FP32),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/convolution",
                        Transpose(
                            Reshape(Slice(convolution, 1, 0, 1), (g.convolution_dim, g.gdn_conv_kernel)),
                            (1, 0),
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/a_projection",
                        _source(source_prefix + "linear_attn.in_proj_a.weight", (g.gdn_value_heads, g.hidden)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/b_projection",
                        _source(source_prefix + "linear_attn.in_proj_b.weight", (g.gdn_value_heads, g.hidden)),
                    ),
                )
            )
            recipes.extend(_gdn_projection_recipes(source_prefix, object_prefix, g))
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "gdn/norm",
                        _source(source_prefix + "linear_attn.norm.weight", (g.gdn_key_head_dim,)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/output",
                        _source(source_prefix + "linear_attn.out_proj.weight", (g.hidden, g.value_dim)),
                    ),
                )
            )

        recipes.extend(
            (
                TensorRecipe(
                    object_prefix + "post_attention_norm",
                    _source(source_prefix + "post_attention_layernorm.weight", (g.hidden,)),
                ),
                TensorRecipe(
                    object_prefix + "mlp/gate_up",
                    Concat(
                        (
                            _source(source_prefix + "mlp.gate_proj.weight", (g.intermediate, g.hidden)),
                            _source(source_prefix + "mlp.up_proj.weight", (g.intermediate, g.hidden)),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/down",
                    _source(source_prefix + "mlp.down_proj.weight", (g.hidden, g.intermediate)),
                ),
            )
        )

    recipes.extend(
        (
            TensorRecipe(
                "text/final_norm",
                _source(_TEXT_PREFIX + "norm.weight", (g.hidden,)),
            ),
            TensorRecipe(
                "text/output_head",
                _source(_output_head_source(g), (g.vocab, g.hidden)),
            ),
        )
    )
    return tuple(recipes)


def _build_draft_head_recipes(g: "inventory.Geometry" = None) -> tuple[TensorRecipe, ...]:
    g = g or inventory.GEOMETRY
    return (
        TensorRecipe(
            "text/draft_head",
            GatherRows(
                _source(_output_head_source(g), (g.vocab, g.hidden)),
                token_ids_object="text/draft_head_token_ids",
                rows=DRAFT_ROWS,
            ),
        ),
        TensorRecipe(
            "text/draft_head_token_ids",
            DraftHeadTokenIds(
                ranking_path="freq_corpus/fixtures/ranking/ranking.train.counts.i64",
                tokenizer_resource="frontend/tokenizer_config.json",
                vocab_rows=248320,
                tokenizer_id_count=248077,
                rows=DRAFT_ROWS,
            ),
        ),
    )


def _build_mtp_recipes(g: "inventory.Geometry" = None) -> tuple[TensorRecipe, ...]:
    g = g or inventory.GEOMETRY
    source_prefix = "mtp.layers.0."
    q_proj = source_prefix + "self_attn.q_proj.weight"
    return (
        TensorRecipe("mtp/input_projection", _source("mtp.fc.weight", (g.hidden, 2 * g.hidden))),
        TensorRecipe(
            "mtp/embedding_norm",
            _source("mtp.pre_fc_norm_embedding.weight", (g.hidden,)),
        ),
        TensorRecipe(
            "mtp/hidden_norm",
            _source("mtp.pre_fc_norm_hidden.weight", (g.hidden,)),
        ),
        TensorRecipe(
            "mtp/layer/input_norm",
            _source(source_prefix + "input_layernorm.weight", (g.hidden,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/query_key_gate_value",
            Concat(
                (
                    _attention_qproj_part(q_proj, gate=False, g=g),
                    _source(source_prefix + "self_attn.k_proj.weight", (g.kv_size, g.hidden)),
                    _attention_qproj_part(q_proj, gate=True, g=g),
                    _source(source_prefix + "self_attn.v_proj.weight", (g.kv_size, g.hidden)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            "mtp/layer/attention/query_norm",
            _source(source_prefix + "self_attn.q_norm.weight", (g.head_dim,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/key_norm",
            _source(source_prefix + "self_attn.k_norm.weight", (g.head_dim,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/output",
            _source(source_prefix + "self_attn.o_proj.weight", (g.hidden, g.query_size)),
        ),
        TensorRecipe(
            "mtp/layer/post_attention_norm",
            _source(source_prefix + "post_attention_layernorm.weight", (g.hidden,)),
        ),
        TensorRecipe(
            "mtp/layer/mlp/gate_up",
            Concat(
                (
                    _source(source_prefix + "mlp.gate_proj.weight", (g.intermediate, g.hidden)),
                    _source(source_prefix + "mlp.up_proj.weight", (g.intermediate, g.hidden)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            "mtp/layer/mlp/down",
            _source(source_prefix + "mlp.down_proj.weight", (g.hidden, g.intermediate)),
        ),
        TensorRecipe("mtp/final_norm", _source("mtp.norm.weight", (g.hidden,))),
    )


def _build_vision_recipes(g: "inventory.Geometry" = None) -> tuple[TensorRecipe, ...]:
    g = g or inventory.GEOMETRY
    return build_vision_recipes(g.hidden, **inventory.vision_tower(g))


def build_recipes(g: "inventory.Geometry" = None) -> tuple[TensorRecipe, ...]:
    """Where every artifact object comes from, for a checkpoint of this size."""
    g = g or inventory.GEOMETRY
    return (
        _build_text_recipes(g)
        + _build_draft_head_recipes(g)
        + _build_mtp_recipes(g)
        + _build_vision_recipes(g)
    )


#: The registered size's recipes, for callers that have no checkpoint in hand.
RECIPE_SPECS = build_recipes()
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    _validate_recipe_coverage(RECIPE_SPECS, inventory.TENSOR_SPECS)


def source_requirements(
    recipes: tuple[TensorRecipe, ...] | None = None,
) -> dict[str, SourceTensor]:
    """Every source tensor the given recipes read; the registered size's unless one is passed."""
    return _recipe_source_requirements(RECIPE_SPECS if recipes is None else recipes)


def preflight_sources(
    model_dir: str | Path,
    recipes: tuple[TensorRecipe, ...] | Mapping[str, TensorRecipe] | None = None,
) -> SourcePreflight:
    # surogate vendor patch (PATCHES.md #14): a GGUF repack plan narrows the
    # bridged-checkpoint requirement to the recipes it does not cover.
    if recipes is None:
        recipes = RECIPE_SPECS
    elif isinstance(recipes, Mapping):
        recipes = tuple(recipes.values())
    return _preflight_recipe_sources(model_dir, recipes)


validate_recipe_coverage()
