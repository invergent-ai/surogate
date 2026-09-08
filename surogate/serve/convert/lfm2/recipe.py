"""Where every LFM2 artifact object comes from in the checkpoint.

Nothing here writes a mapping. The declaration already says which checkpoint
tensor each parameter is (`hf_mapping`) and which parameters each artifact object
is built from, in row order (`ServeObject.components`); `derive_recipes` joins the
two. This module exists to hold the target's own settled choices around that join
-- which capabilities it converts, and that LFM2 ties its output head to its
embedding so the head is not a tensor of its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from surogate.serve.convert.common.declaration import derive_recipes
from surogate.serve.convert.common.recipe import (
    SourcePreflight,
    TensorRecipe,
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
            # LFM2 has no `lm_head.weight`: the declaration says the head is tied
            # to the embedding, and the converter reads one weight for both.
            tied_output_head=bool(config.get("tie_word_embeddings", True)),
        )
    )


def open_reader(model_dir: str | Path) -> ShardReader:
    return ShardReader.for_directory(Path(model_dir))


def preflight_sources(
    reader: ShardReader,
    recipes: Sequence[TensorRecipe],
) -> SourcePreflight:
    """Every source tensor the recipes name, checked against the checkpoint before
    a single byte is converted."""
    return preflight_source_reader(reader, recipes)


__all__ = ["build_recipes", "open_reader", "preflight_sources"]
