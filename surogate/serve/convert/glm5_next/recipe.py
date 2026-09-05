"""Where every GLM-5.3-Flash artifact object comes from in the checkpoint.

Nothing here writes a mapping. The declaration says which checkpoint tensor each parameter is
(`hf_mapping`) and which parameters each artifact object is built from, in row order
(`ServeObject.components`); `derive_recipes` joins the two. This module holds the target's own
settled choices around that join.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from surogate.serve.convert.common.declaration import declare, derive_recipes
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


def build_recipes(config: Mapping[str, object]) -> tuple[TensorRecipe, ...]:
    """Every object's source, in object order, for one checkpoint's config."""
    declared = declare(inventory.ARCHITECTURE, dict(config))
    return tuple(
        derive_recipes(
            declared,
            capabilities=set(inventory.CAPABILITIES),
            # Whether the head is tied is the checkpoint's to say, not the family's: the
            # released GLM-5.3-Flash checkpoints ship an `lm_head.weight`, and a future one that
            # does not would state it in its config.
            tied_output_head=bool(config.get("tie_word_embeddings", False)),
        )
    )


def build_recipes_by_name(config: Mapping[str, object]) -> dict[str, TensorRecipe]:
    """The same recipes, keyed by object name -- what the GGUF repack planner asks for."""
    return {r.object_name: r for r in build_recipes(config)}


def geometry_from_config(config: Mapping[str, object]) -> dict[str, object]:
    """What `build_recipes` needs to describe one checkpoint.

    The GGUF repack planner asks a converter for "the geometry" and hands whatever it gets back
    to `build_recipes` and `inventory.build_tensor_specs`. For the converters that predate the
    declaration that is a `Geometry` of resolved dimensions; for this one it is the config
    itself, because the declaration resolves the dimensions and there is nothing to precompute.
    The config is validated on the way through, so a checkpoint the target cannot shape is
    refused here rather than at the first missing tensor.
    """
    inventory.geometry_from_config(config)
    return dict(config)


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
