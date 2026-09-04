"""Where every Qwen3.8-Flash-Next (`qwen4exp`) artifact object comes from.

Unlike the other families there is no bridged HF checkpoint here: the four
Unsloth GGUF shards *are* the source, so the recipes name GGUF tensors
directly and their declared shapes are the GGUF's logical ones (reversed
`ne`).  That is the only difference; the expression language, the row algebra
that reads it, and the planners that turn it into runs are the shared ones.

Two conventions this file inverts, both llama.cpp's:

- the GDN value heads are stored *tiled* (`[G0_v0, G1_v0, ..., G0_v1, ...]`)
  and the engine wants HF *grouped* order.  Expressed as
  `Reshape(Transpose(Reshape(...)))` over row axes it is a row permutation at
  128-row granularity, so the weights are still read where they lie -- one run
  per head, not one per row.  `ssm_out` permutes *columns* instead and travels
  as the source's `col_groups` map.
- `attn_q` interleaves each head's query and output-gate rows; the engine wants
  the two spans apart, which is what `attention_qproj_part` extracts.

Objects with no recipe here are the ones the language cannot say: a `log(-x)`,
a 48-scalar permutation of a rank-1 tensor, an unfolded RMSNorm gamma, and the
PLE hash parameters, which come from GGUF key-values rather than any tensor.
They are named in `MATERIALIZED_OBJECTS` so the coverage check still sees one
list, and `convert.py` builds them itself.
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
    attention_qproj_part,
    expression_shape,
    expression_sources,
    materialize_expression,
    materialize_recipe,
    source,
    source_requirements as _common_source_requirements,
)

from . import inventory as inv


# Objects whose op reads the row-split W8 planes. Their weights are still read from the GGUF --
# Q8_0 and W8G32_F16S hold the same numbers, so the loader rearranges them on the device -- and
# this list is what tells the planner to take that route rather than the native one. Every W8
# object of this model whose GGUF tensor is Q8_0 is here; what is left out is the routed MoE,
# which llama.cpp stores as K-quants (gate/up) and 32-value blocks (down) that no plane decoder
# reproduces, so those are served in the file's own format instead.
NATIVE_EXCLUDE_SUFFIXES = (
    "text/token_embedding",
    "text/output_head",
    "attention/query_key_gate_value",
    "attention/output",
    "gdn/query_key_value_z",
    "gdn/output",
    "mlp/shared_gate_up",
    "mlp/shared_down",
)

#: The PLE n-gram table: one GGUF tensor, IQ4_NL, read where it lies. It is not in
#: `TENSOR_SPECS` -- the engine binds it as one flat host-resident object rather than as part
#: of a layer -- so it carries its own recipe and `convert.py` plans it beside the rest.
PLE_TABLE_SOURCE = "per_layer_token_embd.weight"


def _untile_value_rows(expression: Expression, rows: int, span: int) -> Expression:
    """llama.cpp's tiled value order along the row axis, back to HF grouped order.

    Tiled is `[G0_v0, G1_v0, ...]` -- v-major over the 16 key groups; grouped is `[G0_v0,
    G0_v1, ...]`. Written as a transpose of the two head axes it stays a row program, so a
    projection that needs it is still readable in place: a head is `span` consecutive rows.
    """
    groups = inv.GDN_KEY_HEADS
    per_group = inv.GDN_VALUE_HEADS // inv.GDN_KEY_HEADS
    if rows != groups * per_group * span:
        raise ValueError(f"un-tile expects {groups * per_group * span} rows, not {rows}")
    k = expression_shape(expression)[-1]
    return Reshape(
        Transpose(Reshape(expression, (per_group, groups, span, k)), (1, 0, 2, 3)),
        (rows, k),
    )


def _hyper_connection_recipes(gguf_prefix: str, object_prefix: str,
                              with_inject: bool) -> tuple[TensorRecipe, ...]:
    recipes = [
        TensorRecipe(object_prefix + "norm",
                     source(gguf_prefix + "norm.weight", (inv.HC_WIDTH,))),
        TensorRecipe(object_prefix + "down",
                     source(gguf_prefix + "down.weight", (inv.HC_LOW_RANK, inv.HC_WIDTH))),
        TensorRecipe(object_prefix + "up",
                     source(gguf_prefix + "up.weight", (inv.HC_WIDTH, inv.HC_LOW_RANK))),
    ]
    if with_inject:
        recipes.append(
            TensorRecipe(object_prefix + "inject",
                         source(gguf_prefix + "inject.weight", (inv.HC_COUNT, inv.HC_WIDTH)))
        )
    return tuple(recipes)


def _ple_recipes(blk: str, prefix: str) -> tuple[TensorRecipe, ...]:
    return (
        TensorRecipe(prefix + "key", source(blk + "ple_key.weight", (inv.HC_WIDTH, inv.PLE_EMBED))),
        TensorRecipe(prefix + "value", source(blk + "ple_value.weight", (inv.HIDDEN, inv.PLE_EMBED))),
        TensorRecipe(prefix + "norm_key", source(blk + "ple_norm_key.weight", (inv.HC_WIDTH,))),
        TensorRecipe(prefix + "norm_query", source(blk + "ple_norm_query.weight", (inv.HC_WIDTH,))),
        TensorRecipe(prefix + "norm_conv", source(blk + "ple_norm_conv.weight", (inv.HC_WIDTH,))),
        # The conv op reads weight[tap * C + c] (tap-major, channel fastest); the GGUF holds
        # the channel-major (10240, 4) that ggml wants.
        TensorRecipe(
            prefix + "convolution",
            Transpose(source(blk + "ple_conv1d.weight", (inv.HC_WIDTH, inv.PLE_CONV_KERNEL)), (1, 0)),
        ),
    )


def _attention_recipes(blk: str, prefix: str) -> tuple[TensorRecipe, ...]:
    q_proj = blk + "attn_q.weight"
    return (
        TensorRecipe(
            prefix + "query_key_gate_value",
            Concat(
                (
                    attention_qproj_part(q_proj, False, num_heads=inv.QUERY_HEADS,
                                         hidden_size=inv.HIDDEN),
                    source(blk + "attn_k.weight", (inv.KV_SIZE, inv.HIDDEN)),
                    attention_qproj_part(q_proj, True, num_heads=inv.QUERY_HEADS,
                                         hidden_size=inv.HIDDEN),
                    source(blk + "attn_v.weight", (inv.KV_SIZE, inv.HIDDEN)),
                ),
                0,
            ),
        ),
        TensorRecipe(prefix + "output",
                     source(blk + "attn_output.weight", (inv.HIDDEN, inv.QUERY_SIZE))),
        TensorRecipe(
            prefix + "indexer/query",
            source(blk + "indexer.q_proj.weight",
                   (inv.INDEXER_HEADS * inv.INDEXER_DIM, inv.HIDDEN)),
        ),
        TensorRecipe(prefix + "indexer/key",
                     source(blk + "indexer.k_proj.weight", (inv.INDEXER_DIM, inv.HIDDEN))),
        TensorRecipe(prefix + "indexer/query_norm",
                     source(blk + "indexer.q_norm.weight", (inv.INDEXER_DIM,))),
        TensorRecipe(prefix + "indexer/key_norm",
                     source(blk + "indexer.k_norm.weight", (inv.INDEXER_DIM,))),
    )


def _gdn_recipes(blk: str, prefix: str) -> tuple[TensorRecipe, ...]:
    key_rows = 2 * inv.GDN_KEY_DIM
    conv = source(blk + "ssm_conv1d.weight", (inv.GDN_CONV_DIM, inv.GDN_CONV_KERNEL))
    qkv = source(blk + "attn_qkv.weight", (inv.GDN_CONV_DIM, inv.HIDDEN))
    return (
        TensorRecipe(
            # Un-tiled along the rows first and transposed after: doing it the other way round
            # would reorder within a row, which no row program can say. The two are the same
            # permutation of the same values.
            prefix + "convolution",
            Transpose(
                Concat(
                    (
                        Slice(conv, 0, 0, key_rows),
                        _untile_value_rows(Slice(conv, 0, key_rows, inv.GDN_CONV_DIM),
                                           inv.GDN_VALUE_DIM, inv.GDN_HEAD_DIM),
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
                    _untile_value_rows(
                        source(blk + f"ssm_{name}.weight", (inv.GDN_VALUE_HEADS, inv.HIDDEN)),
                        inv.GDN_VALUE_HEADS,
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
                    _untile_value_rows(Slice(qkv, 0, key_rows, inv.GDN_CONV_DIM),
                                       inv.GDN_VALUE_DIM, inv.GDN_HEAD_DIM),
                    _untile_value_rows(
                        source(blk + "attn_gate.weight", (inv.GDN_VALUE_DIM, inv.HIDDEN)),
                        inv.GDN_VALUE_DIM,
                        inv.GDN_HEAD_DIM,
                    ),
                ),
                0,
            ),
        ),
        TensorRecipe(prefix + "norm", source(blk + "ssm_norm.weight", (inv.GDN_HEAD_DIM,))),
        # Rows are identity; the *columns* are value-tiled, which is not a row program. The
        # inverse travels as the source's `col_groups` map instead -- 128 columns are exactly
        # four quantisation groups, so whole groups move.
        TensorRecipe(prefix + "output",
                     source(blk + "ssm_out.weight", (inv.HIDDEN, inv.GDN_VALUE_DIM))),
    )


def _mlp_recipes(blk: str, prefix: str) -> tuple[TensorRecipe, ...]:
    experts, ffn = inv.EXPERTS, inv.EXPERT_FFN
    return (
        TensorRecipe(
            prefix + "router_shared_gate",
            Concat(
                (
                    source(blk + "ffn_gate_inp.weight", (inv.EXPERTS, inv.HIDDEN)),
                    # 1-D in the GGUF; the object wants it as the router's last row.
                    source(blk + "ffn_gate_inp_shexp.weight", (1, inv.HIDDEN)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            # The experts are two stacked tensors, one per half, not one tensor per expert:
            # concatenating them on the expert's output axis gives exactly
            # stored_row(e, half, r) = e*1280 + half*640 + r, so the fused parent is a row
            # gather over blocks -- one run per expert per half -- and never a dequantisation.
            prefix + "routed_gate_up",
            Reshape(
                Concat(
                    tuple(
                        source(blk + f"ffn_{half}_exps.weight", (experts, ffn, inv.HIDDEN))
                        for half in ("gate", "up")
                    ),
                    1,
                ),
                (experts * 2 * ffn, inv.HIDDEN),
            ),
        ),
        TensorRecipe(
            prefix + "routed_down",
            Reshape(
                source(blk + "ffn_down_exps.weight", (experts, inv.HIDDEN, ffn)),
                (experts * inv.HIDDEN, ffn),
            ),
        ),
        TensorRecipe(
            prefix + "shared_gate_up",
            Concat(
                tuple(
                    source(blk + f"ffn_{half}_shexp.weight", (inv.SHARED_FFN, inv.HIDDEN))
                    for half in ("gate", "up")
                ),
                0,
            ),
        ),
        TensorRecipe(prefix + "shared_down",
                     source(blk + "ffn_down_shexp.weight", (inv.HIDDEN, inv.SHARED_FFN))),
    )


def _build_text_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe("text/token_embedding",
                     source("token_embd.weight", (inv.VOCAB, inv.HIDDEN))),
    ]
    for layer in range(inv.LAYERS):
        blk = f"blk.{layer}."
        prefix = f"text/layers/{layer}/"
        if layer == inv.PLE_LAYER:
            recipes.extend(_ple_recipes(blk, prefix + "ple/"))
        recipes.extend(_hyper_connection_recipes(blk + "hc_attn_", prefix + "hc_attn/", True))
        if layer in inv.FULL_ATTENTION_LAYERS:
            recipes.extend(_attention_recipes(blk, prefix + "attention/"))
        else:
            recipes.extend(_gdn_recipes(blk, prefix + "gdn/"))
        recipes.extend(_hyper_connection_recipes(blk + "hc_ffn_", prefix + "hc_ffn/", True))
        recipes.extend(_mlp_recipes(blk, prefix + "mlp/"))
    recipes.extend(_hyper_connection_recipes("output_hc_", "text/output_hc/", False))
    recipes.append(TensorRecipe("text/output_head",
                                source("output.weight", (inv.VOCAB, inv.HIDDEN))))
    return tuple(recipes)


def _materialized_objects() -> frozenset[str]:
    """Objects the expression language cannot say, and why each one resists it."""
    names: set[str] = set()
    for layer in inv.FULL_ATTENTION_LAYERS:
        # llama.cpp folds the unit offset into the RMSNorm gamma; the runtime re-adds it, so
        # the converter has to unfold it. A subtraction is not a rearrangement of rows.
        names.update(f"text/layers/{layer}/attention/{leaf}_norm" for leaf in ("query", "key"))
    for layer in inv.GDN_LAYERS:
        # log(-x), and a 48-scalar permutation of a rank-1 tensor: the row algebra needs a
        # trailing K axis to keep untouched, and these have none.
        names.update(f"text/layers/{layer}/gdn/{leaf}" for leaf in ("a_log", "dt_bias"))
    # Built from GGUF key-values, not from any tensor.
    names.update(f"text/ple/{leaf}" for leaf in
                 ("multipliers", "head_offsets", "head_vocab_sizes"))
    return frozenset(names)


TEXT_RECIPE_SPECS = _build_text_recipes()
MATERIALIZED_OBJECTS = _materialized_objects()
PLE_TABLE_RECIPE = TensorRecipe(
    inv.PLE_TABLE_RESOURCE,
    source(PLE_TABLE_SOURCE, (inv.PLE_TABLE_ROWS, inv.PLE_HEAD_DIM)),
)

RECIPE_SPECS = TEXT_RECIPE_SPECS
RECIPES_BY_NAME = {item.object_name: item for item in RECIPE_SPECS}


def build_recipes(geometry=None) -> tuple[TensorRecipe, ...]:
    """Where every artifact object comes from.

    This target serves one published model and one export of it, so every width here is that
    model's, written out; the argument exists for a uniform call across the families.
    """
    return RECIPE_SPECS


def source_requirements():
    return _common_source_requirements(RECIPE_SPECS + (PLE_TABLE_RECIPE,))


def validate_recipe_coverage() -> None:
    """Recipes plus the hand-built objects are the text inventory, in its order.

    The vision tower is deliberately absent: this converter's only source is a GGUF, and no
    GGUF export of Flash-Next carries one (the four-shard Q4_K_XL set has 1,224 tensors and
    none of them vision). `inventory.TENSOR_SPECS` still describes the tower, because whether
    an *artifact* has one is a property of the export; a converter that cannot produce it says
    so rather than advertising it.
    """
    inventory_names = [spec.name for spec in inv.TEXT_CORE_TENSOR_SPECS]
    covered = [name for name in inventory_names
               if name in RECIPES_BY_NAME or name in MATERIALIZED_OBJECTS]
    if covered != inventory_names:
        missing = sorted(set(inventory_names) - set(covered))
        raise ValueError(f"qwen4exp recipes do not cover {len(missing)} objects: {missing[:8]}")
    if len(RECIPES_BY_NAME) != len(RECIPE_SPECS):
        raise ValueError("more than one qwen4exp recipe targets the same artifact object")
    if set(RECIPES_BY_NAME) & MATERIALIZED_OBJECTS:
        raise ValueError("an object is both a recipe and hand-built")
    stray = sorted((set(RECIPES_BY_NAME) | MATERIALIZED_OBJECTS) - set(inventory_names))
    if stray:
        raise ValueError(f"recipes name objects outside the inventory: {stray[:8]}")
    recipe_order = [name for name in inventory_names if name in RECIPES_BY_NAME]
    if recipe_order != [item.object_name for item in RECIPE_SPECS]:
        raise ValueError("qwen4exp recipe order does not follow the tensor inventory")
    by_name = {spec.name: spec for spec in inv.TEXT_CORE_TENSOR_SPECS}
    for item in RECIPE_SPECS:
        actual = expression_shape(item.expression)
        if actual != by_name[item.object_name].shape:
            raise ValueError(
                f"{item.object_name}: recipe shape {actual} != inventory "
                f"{by_name[item.object_name].shape}"
            )


validate_recipe_coverage()


__all__ = [
    "Concat",
    "Expression",
    "MATERIALIZED_OBJECTS",
    "NATIVE_EXCLUDE_SUFFIXES",
    "PLE_TABLE_RECIPE",
    "PLE_TABLE_SOURCE",
    "RECIPES_BY_NAME",
    "RECIPE_SPECS",
    "Reshape",
    "Slice",
    "SourceTensor",
    "TensorRecipe",
    "Transpose",
    "build_recipes",
    "expression_shape",
    "expression_sources",
    "materialize_expression",
    "materialize_recipe",
    "source",
    "source_requirements",
    "validate_recipe_coverage",
]
