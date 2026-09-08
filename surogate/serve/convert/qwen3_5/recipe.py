"""Hugging Face source recipe for the complete interleaved gated-delta inventory.

Where each object comes from is not written here: it is derived from the training
declaration, whose `hf_mapping` names every checkpoint tensor and whose `ServeObject`s
name the parameters each artifact object is built from. This module states only what
the declaration does not: how a storage profile cuts fused projections, the speculative draft head (a ranking
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
    Cast,
    Concat,
    DraftHeadTokenIds,
    Expression,
    GatherRows,
    Reshape,
    Slice,
    SourcePreflight,
    SourceTensor,
    TensorRecipe,
    Transpose,
    build_vision_recipes,
    expression_shape,
    expression_sources,
    materialize_recipe,
    preflight_sources as _preflight_recipe_sources,
    source,
    source_requirements as _recipe_source_requirements,
    validate_recipe_coverage as _validate_recipe_coverage,
)
from surogate.dsl.ir_builder import resolve_architecture

from . import inventory

_sources = expression_sources
_source = source

#: Fused projections can be stored as typed halves, expressed by row cuts. The MTP layer replays the same
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


def _build_declared_recipes(g: "inventory.Geometry") -> tuple[TensorRecipe, ...]:
    recipes = derive_recipes(g.declared, capabilities={"text"})
    return cut_rows(recipes, _storage_cuts(g), prefix=_TEXT_LAYERS)


def _build_draft_head_recipes(output_head: Expression, g: inventory.Geometry) -> tuple[TensorRecipe, ...]:
    if not isinstance(output_head, SourceTensor):
        raise TypeError("the draft head gathers rows of one checkpoint tensor")
    return (
        TensorRecipe(
            "text/draft_head",
            GatherRows(
                output_head,
                token_ids_object="text/draft_head_token_ids",
                rows=g.draft_vocab,
            ),
        ),
        TensorRecipe(
            "text/draft_head_token_ids",
            DraftHeadTokenIds(
                ranking_path="freq_corpus/fixtures/ranking/ranking.train.counts.i64",
                tokenizer_resource="frontend/tokenizer_config.json",
                vocab_rows=g.vocab,
                tokenizer_id_count=g.token_domain,
                rows=g.draft_vocab,
            ),
        ),
    )


def _build_vision_recipes(g: "inventory.Geometry") -> tuple[TensorRecipe, ...]:
    tower = inventory.vision_tower(g)
    if not tower:
        return ()
    tower.pop("patch_rows")
    vision = g.declared.hf_config["vision_config"]
    return build_vision_recipes(g.hidden, **tower, patch_channels=vision["in_channels"],
                               patch_temporal=vision["temporal_patch_size"],
                               patch_size=vision["patch_size"])


def build_recipes(g: "inventory.Geometry") -> tuple[TensorRecipe, ...]:
    """Derive source recipes from the same resolved checkpoint as the inventory."""
    declared = _build_declared_recipes(g)
    by_name = {recipe.object_name: recipe for recipe in declared}
    recipes = (declared + _build_draft_head_recipes(by_name["text/output_head"].expression, g)
               + _build_vision_recipes(g))
    by_name = {recipe.object_name: recipe for recipe in recipes}
    return tuple(by_name[spec.name] for spec in inventory.build_tensor_specs(g))


def validate_recipe_coverage(g: "inventory.Geometry") -> None:
    _validate_recipe_coverage(build_recipes(g), inventory.build_tensor_specs(g))


def source_requirements(recipes: tuple[TensorRecipe, ...]) -> dict[str, SourceTensor]:
    return _recipe_source_requirements(recipes)


def preflight_sources(
    model_dir: str | Path,
    recipes: tuple[TensorRecipe, ...] | Mapping[str, TensorRecipe],
) -> SourcePreflight:
    if isinstance(recipes, Mapping):
        recipes = tuple(recipes.values())
    return _preflight_recipe_sources(model_dir, recipes)
