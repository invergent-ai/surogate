"""Where every Qwen3-MoE artifact object comes from in the checkpoint.

Nothing here writes a mapping. The declaration says which checkpoint tensor each parameter is
(`hf_mapping`) and which parameters each artifact object is built from, in row order
(`ServeObject.components`); `derive_recipes` joins the two. This module holds the target's own
settled choices around that join.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from surogate.serve.convert.common.declaration import derive_recipes
from surogate.serve.convert.common.recipe import (
    SourcePreflight,
    TensorRecipe,
    # Re-exported: the GGUF repack planner walks a recipe's sources through the module it
    # planned against, so every converter offers this name.
    expression_sources,
    preflight_source_reader,
)
from surogate.serve.convert.common.safetensors import ShardReader

from . import inventory


def build_recipes(geometry: inventory.Geometry) -> tuple[TensorRecipe, ...]:
    """Every object's source, in object order, for one checkpoint's config."""
    declared = geometry.declared
    config = declared.hf_config
    return tuple(
        derive_recipes(
            declared,
            capabilities=set(inventory.CAPABILITIES),
            # Whether the head is tied is the checkpoint's to say, not the family's: the
            # released Qwen3-MoE checkpoints ship an `lm_head.weight`, and a future one that
            # does not would state it in its config.
            tied_output_head=bool(config.get("tie_word_embeddings", False)),
        )
    )


def build_recipes_by_name(geometry: inventory.Geometry) -> dict[str, TensorRecipe]:
    """The same recipes, keyed by object name -- what the GGUF repack planner asks for."""
    return {r.object_name: r for r in build_recipes(geometry)}


def geometry_from_config(config: Mapping[str, object]) -> inventory.Geometry:
    """Resolve once before the GGUF planner builds objects and recipes."""
    return inventory.geometry_from_config(config)


def open_reader(model_dir: str | Path) -> ShardReader:
    return ShardReader.for_directory(Path(model_dir))


def preflight_sources(
    reader: ShardReader,
    recipes: Sequence[TensorRecipe],
) -> SourcePreflight:
    """Every source tensor the recipes name, checked against the checkpoint before a single
    byte is converted."""
    return preflight_source_reader(reader, recipes)


__all__ = [
    "build_recipes",
    "geometry_from_config",
    "build_recipes_by_name",
    "open_reader",
    "preflight_sources",
]
