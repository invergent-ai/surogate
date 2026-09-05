"""Hugging Face source recipe for the complete interleaved gated-delta inventory.

Where each object comes from is not written here: it is derived from the training
declaration, whose `hf_mapping` names every checkpoint tensor and whose `ServeObject`s
name the parameters each artifact object is built from. This module states only what
the declaration does not: how the group-wise export cuts the fused projections at the
27B (a storage decision, applied as a row cut), the speculative draft head (a ranking
policy over the vocabulary), and the vision tower (no trained counterpart).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from surogate.serve.convert.common.declaration import (
    cut_rows,
    declare,
    derive_recipes,
)
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
from surogate.dsl.ir_builder import resolve_architecture

from . import inventory

DRAFT_ROWS = 131072

_sources = expression_sources
_source = source

#: How the group-wise export stores the fused projections: at the 27B it types the halves
#: apart, so the derived fused object is cut into them. The MTP layer replays the same
#: block and is stored fused, which is why the cut is confined to the text layers.
_TEXT_LAYERS = "text/layers/"


def _storage_cuts(g: "inventory.Geometry") -> dict[str, tuple[tuple[str, int], ...]]:
    export = inventory.export_for(inventory.GROUPWISE_INT, g)
    cuts: dict[str, tuple[tuple[str, int], ...]] = {}
    attention = inventory.attention_input_roles(g, export)
    if len(attention) > 1:
        cuts["attention/query_key_gate_value"] = tuple((name, shape[0]) for name, shape in attention)
    gdn = inventory.gdn_input_roles(g, export)
    if len(gdn) > 1:
        cuts["gdn/query_key_value_z"] = tuple((name, shape[0]) for name, shape in gdn)
    return cuts


def _build_declared_recipes(g: "inventory.Geometry", hf_config: Mapping | None) -> tuple[TensorRecipe, ...]:
    """The text stack and the MTP head, from the declaration compiled for this checkpoint
    (or for the config its dimensions imply, when the caller has only a geometry)."""
    config = dict(hf_config) if hf_config is not None else inventory.hf_config_for(g)
    declaration = declare(resolve_architecture(config), config)
    recipes = derive_recipes(declaration, capabilities={"text"})
    return cut_rows(recipes, _storage_cuts(g), prefix=_TEXT_LAYERS)


def _build_draft_head_recipes(output_head: Expression) -> tuple[TensorRecipe, ...]:
    if not isinstance(output_head, SourceTensor):
        raise TypeError("the draft head gathers rows of one checkpoint tensor")
    return (
        TensorRecipe(
            "text/draft_head",
            GatherRows(
                output_head,
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


def _build_vision_recipes(g: "inventory.Geometry" = None) -> tuple[TensorRecipe, ...]:
    g = g or inventory.GEOMETRY
    return build_vision_recipes(g.hidden, **inventory.vision_tower(g))


def build_recipes(g: "inventory.Geometry" = None, *,
                  hf_config: Mapping | None = None) -> tuple[TensorRecipe, ...]:
    """Where every artifact object comes from, for a checkpoint of this size, in the
    inventory's order. `hf_config` is the checkpoint's own `config.json` when the caller
    has it; the declaration is compiled against it, so a nested (VL) release and a flat
    one derive the same flat-dialect recipes."""
    g = g or inventory.GEOMETRY
    declared = _build_declared_recipes(g, hf_config)
    by_name = {recipe.object_name: recipe for recipe in declared}
    recipes = (
        declared
        + _build_draft_head_recipes(by_name["text/output_head"].expression)
        + _build_vision_recipes(g)
    )
    by_name = {recipe.object_name: recipe for recipe in recipes}
    ordered = tuple(by_name.pop(spec.name) for spec in inventory.build_tensor_specs(g))
    if by_name:
        raise ValueError(f"recipes for objects the inventory does not list: {sorted(by_name)[:6]}")
    return ordered


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
