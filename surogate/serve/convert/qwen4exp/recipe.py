"""GGUF row mappings for a resolved hyper-connected hybrid checkpoint.

Value heads are reordered from tiled GGUF order to grouped order. Attention
query and gate rows are separated using the checkpoint's head width.
"""

from surogate.serve.convert.common.recipe import (
    Concat, Expression, Reshape, Slice, SourceTensor, TensorRecipe, Transpose,
    attention_qproj_part, expression_shape, expression_sources, materialize_expression,
    materialize_recipe, source, source_requirements as _common_source_requirements,
)
from . import inventory as inv


NATIVE_EXCLUDE_SUFFIXES = (
    "text/token_embedding",
    "text/output_head",
    "attention/query_key_gate_value",
    "attention/output",
    "gdn/query_key_value_z",
    "gdn/output",
    "mlp/shared_gate_up",
    "mlp/shared_down",
    "mtp/input_projection",
)

PLE_TABLE_SOURCE = "per_layer_token_embd.weight"


def _untile_value_rows(g, expression: Expression, rows: int, span: int) -> Expression:
    groups = g.gdn_key_heads
    per_group = g.gdn_value_heads // g.gdn_key_heads
    if rows != groups * per_group * span:
        raise ValueError(f"un-tile expects {groups * per_group * span} rows, not {rows}")
    k = expression_shape(expression)[-1]
    return Reshape(
        Transpose(Reshape(expression, (per_group, groups, span, k)), (1, 0, 2, 3)),
        (rows, k),
    )


def _hyper_connection_recipes(g, gguf_prefix: str, object_prefix: str,
                              with_inject: bool) -> tuple[TensorRecipe, ...]:
    recipes = [
        TensorRecipe(object_prefix + "norm",
                     source(gguf_prefix + "norm.weight", (g.residual,))),
        TensorRecipe(object_prefix + "down",
                     source(gguf_prefix + "down.weight", (g.hc_low_rank, g.residual))),
        TensorRecipe(object_prefix + "up",
                     source(gguf_prefix + "up.weight", (g.residual, g.hc_low_rank))),
    ]
    if with_inject:
        recipes.append(
            TensorRecipe(object_prefix + "inject",
                         source(gguf_prefix + "inject.weight", (g.hc_streams, g.residual)))
        )
    return tuple(recipes)


def _ple_recipes(g, blk: str, prefix: str) -> tuple[TensorRecipe, ...]:
    return (
        TensorRecipe(prefix + "key", source(blk + "ple_key.weight", (g.residual, g.ple_embed))),
        TensorRecipe(prefix + "value", source(blk + "ple_value.weight", (g.hidden, g.ple_embed))),
        TensorRecipe(prefix + "norm_key", source(blk + "ple_norm_key.weight", (g.residual,))),
        TensorRecipe(prefix + "norm_query", source(blk + "ple_norm_query.weight", (g.residual,))),
        TensorRecipe(prefix + "norm_conv", source(blk + "ple_norm_conv.weight", (g.residual,))),
        TensorRecipe(
            prefix + "convolution",
            Transpose(source(blk + "ple_conv1d.weight", (g.residual, g.ple_conv_kernel)), (1, 0)),
        ),
    )


def _attention_recipes(g, blk: str, prefix: str) -> tuple[TensorRecipe, ...]:
    q_proj = blk + "attn_q.weight"
    return (
        TensorRecipe(
            prefix + "query_key_gate_value",
            Concat(
                (
                    attention_qproj_part(q_proj, False, num_heads=g.query_heads,
                                         hidden_size=g.hidden, head_dim=g.head_dim),
                    source(blk + "attn_k.weight", (g.kv_size, g.hidden)),
                    attention_qproj_part(q_proj, True, num_heads=g.query_heads,
                                         hidden_size=g.hidden, head_dim=g.head_dim),
                    source(blk + "attn_v.weight", (g.kv_size, g.hidden)),
                ),
                0,
            ),
        ),
        TensorRecipe(prefix + "output",
                     source(blk + "attn_output.weight", (g.hidden, g.query_size))),
        TensorRecipe(
            prefix + "indexer/query",
            source(blk + "indexer.q_proj.weight",
                   (g.indexer_heads * g.indexer_head_dim, g.hidden)),
        ),
        TensorRecipe(prefix + "indexer/key",
                     source(blk + "indexer.k_proj.weight", (g.indexer_head_dim, g.hidden))),
        TensorRecipe(prefix + "indexer/query_norm",
                     source(blk + "indexer.q_norm.weight", (g.indexer_head_dim,))),
        TensorRecipe(prefix + "indexer/key_norm",
                     source(blk + "indexer.k_norm.weight", (g.indexer_head_dim,))),
    )


def _gdn_recipes(g, blk: str, prefix: str) -> tuple[TensorRecipe, ...]:
    key_rows = 2 * g.key_dim
    conv = source(blk + "ssm_conv1d.weight", (g.convolution_dim, g.gdn_conv_kernel))
    qkv = source(blk + "attn_qkv.weight", (g.convolution_dim, g.hidden))
    return (
        TensorRecipe(
            prefix + "convolution",
            Transpose(
                Concat(
                    (
                        Slice(conv, 0, 0, key_rows),
                        _untile_value_rows(g, Slice(conv, 0, key_rows, g.convolution_dim),
                                           g.value_dim, g.gdn_value_head_dim),
                    ),
                    0,
                ),
                (1, 0),
            ),
        ),
        TensorRecipe(
            prefix + "a_b_projection",
            Concat(
                tuple(
                    _untile_value_rows(g,
                        source(blk + f"ssm_{name}.weight", (g.gdn_value_heads, g.hidden)),
                        g.gdn_value_heads,
                        1,
                    )
                    for name in ("alpha", "beta")
                ),
                0,
            ),
        ),
        TensorRecipe(
            prefix + "query_key_value_z",
            Concat(
                (
                    Slice(qkv, 0, 0, key_rows),
                    _untile_value_rows(g, Slice(qkv, 0, key_rows, g.convolution_dim),
                                       g.value_dim, g.gdn_value_head_dim),
                    _untile_value_rows(g,
                        source(blk + "attn_gate.weight", (g.value_dim, g.hidden)),
                        g.value_dim,
                        g.gdn_value_head_dim,
                    ),
                ),
                0,
            ),
        ),
        TensorRecipe(prefix + "norm", source(blk + "ssm_norm.weight", (g.gdn_value_head_dim,))),
        TensorRecipe(prefix + "output",
                     source(blk + "ssm_out.weight", (g.hidden, g.value_dim))),
    )


def _mlp_recipes(g, blk: str, prefix: str) -> tuple[TensorRecipe, ...]:
    experts, ffn = g.experts, g.intermediate
    return (
        TensorRecipe(
            prefix + "router_shared_gate",
            Concat(
                (
                    source(blk + "ffn_gate_inp.weight", (g.experts, g.hidden)),
                    source(blk + "ffn_gate_inp_shexp.weight", (1, g.hidden)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            prefix + "routed_gate_up",
            Reshape(
                Concat(
                    tuple(
                        source(blk + f"ffn_{half}_exps.weight", (experts, ffn, g.hidden))
                        for half in ("gate", "up")
                    ),
                    1,
                ),
                (experts * 2 * ffn, g.hidden),
            ),
        ),
        TensorRecipe(
            prefix + "routed_down",
            Reshape(
                source(blk + "ffn_down_exps.weight", (experts, g.hidden, ffn)),
                (experts * g.hidden, ffn),
            ),
        ),
        TensorRecipe(
            prefix + "shared_gate_up",
            Concat(
                tuple(
                    source(blk + f"ffn_{half}_shexp.weight", (g.shared_intermediate, g.hidden))
                    for half in ("gate", "up")
                ),
                0,
            ),
        ),
        TensorRecipe(prefix + "shared_down",
                     source(blk + "ffn_down_shexp.weight", (g.hidden, g.shared_intermediate))),
    )


def _build_text_recipes(g: inv.Geometry) -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe("text/token_embedding",
                     source("token_embd.weight", (g.vocab, g.hidden))),
    ]
    for layer in range(g.layers):
        blk = f"blk.{layer}."
        prefix = f"text/layers/{layer}/"
        if g.ple_ngram and layer == g.ple_layer:
            recipes.extend(_ple_recipes(g, blk, prefix + "ple/"))
        recipes.extend(_hyper_connection_recipes(g, blk + "hc_attn_", prefix + "hc_attn/", True))
        if layer in g.full_attention_layers:
            recipes.extend(_attention_recipes(g, blk, prefix + "attention/"))
        else:
            recipes.extend(_gdn_recipes(g, blk, prefix + "gdn/"))
        recipes.extend(_hyper_connection_recipes(g, blk + "hc_ffn_", prefix + "hc_ffn/", True))
        recipes.extend(_mlp_recipes(g, blk, prefix + "mlp/"))
    recipes.extend(_hyper_connection_recipes(g, "output_hc_", "text/output_hc/", False))
    recipes.append(TensorRecipe("text/output_head",
                                source("token_embd.weight" if g.tied_embeddings else "output.weight", (g.vocab, g.hidden))))
    return tuple(recipes)


def _build_mtp_recipes(g: inv.Geometry) -> tuple[TensorRecipe, ...]:
    blk = f"blk.{g.layers}."
    prefix = "mtp/"
    layer = prefix + "layer/"
    recipes: list[TensorRecipe] = [
        TensorRecipe(prefix + "embedding_norm",
                     source(blk + "nextn.enorm.weight", (g.hidden,))),
        TensorRecipe(prefix + "hidden_norm",
                     source(blk + "nextn.hnorm.weight", (g.residual,))),
        TensorRecipe(prefix + "input_projection",
                     source(blk + "nextn.eh_proj.weight", (g.hidden, 2 * g.hidden))),
    ]
    recipes.extend(_hyper_connection_recipes(g, blk + "hc_attn_", layer + "hc_attn/", True))
    recipes.extend(_attention_recipes(g, blk, layer + "attention/"))
    recipes.extend(_hyper_connection_recipes(g, blk + "hc_ffn_", layer + "hc_ffn/", True))
    recipes.extend(_mlp_recipes(g, blk, layer + "mlp/"))
    recipes.extend(_hyper_connection_recipes(g, blk + "nextn.hc_head_", prefix + "head_hc/", False))
    return tuple(recipes)


def _materialized_objects(g: inv.Geometry) -> frozenset[str]:
    names: set[str] = set()
    for layer in g.full_attention_layers:
        names.update(f"text/layers/{layer}/attention/{leaf}_norm" for leaf in ("query", "key"))
    if g.mtp_layers:
        names.update(f"mtp/layer/attention/{leaf}_norm" for leaf in ("query", "key"))
    for layer in g.gdn_layers:
        names.update(f"text/layers/{layer}/gdn/{leaf}" for leaf in ("a_log", "dt_bias"))
    if g.ple_ngram:
        names.update(f"text/ple/{leaf}" for leaf in
                     ("multipliers", "head_offsets", "head_vocab_sizes"))
    return frozenset(names)



def ple_table_recipe(g: inv.Geometry) -> TensorRecipe:
    return TensorRecipe(inv.PLE_TABLE_RESOURCE,
                        source(PLE_TABLE_SOURCE, (g.ple_table_rows, g.ple_head_dim)))


def build_recipes(g: inv.Geometry) -> tuple[TensorRecipe, ...]:
    recipes = _build_text_recipes(g)
    if g.mtp_layers:
        recipes += _build_mtp_recipes(g)
    return recipes


def materialized_objects(g: inv.Geometry) -> frozenset[str]:
    return _materialized_objects(g)


def source_requirements(g: inv.Geometry):
    return _common_source_requirements(build_recipes(g) +
                                      ((ple_table_recipe(g),) if g.ple_ngram else ()))


def validate_recipe_coverage(g: inv.Geometry) -> None:
    specs, _ = inv.active_specs(geometry=g, vision=False)
    inventory = {spec.name: spec for spec in specs}
    recipes = build_recipes(g)
    by_name = {item.object_name: item for item in recipes}
    materialized = materialized_objects(g)
    if len(by_name) != len(recipes) or set(by_name) & materialized:
        raise ValueError("duplicate qwen4exp object recipe")
    if set(by_name) | materialized != set(inventory):
        raise ValueError("qwen4exp recipes do not cover the resolved inventory")
    if [name for name in inventory if name in by_name] != list(by_name):
        raise ValueError("qwen4exp recipe order does not follow the inventory")
    for name, item in by_name.items():
        if expression_shape(item.expression) != inventory[name].shape:
            raise ValueError(f"{name}: recipe shape differs from the resolved inventory")
