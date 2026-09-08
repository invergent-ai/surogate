"""Where every E-series Gemma 4 artifact object comes from in the checkpoint.

Nothing here writes a mapping. The declaration already says which checkpoint tensor
each parameter is (`hf_mapping`) and which parameters each artifact object is built
from, in row order (`ServeObject.components`); `derive_recipes` joins the two. This
module holds the target's own settled choices around that join.

There is one such choice and it is the output head. Every published Gemma 4 sets
`tie_word_embeddings`, so the head is the embedding table and is not a tensor of its
own -- but the flag is read from the checkpoint rather than assumed here, because a
fine-tune that unties it is a checkpoint this converter should still convert.

Nothing here mentions the shared layers' vestigial `k_proj`/`v_proj`/`k_norm`, and that is
the point: the declaration gives a shared layer a Q-only attention, so no object names them
and `derive_recipes` never looks for them.

**No norm is unfolded.** Gemma 3's converter subtracts one from every norm weight
(`transform="unfold_unit_offset"`) because Gemma 3 applies `1 + w`. Gemma 4 applies
`w` (`Gemma4UnifiedRMSNorm.forward`), so its norms pass through untouched and no
transform is registered here. Inheriting Gemma 3's would scale every norm in the
model by one plus itself, and nothing downstream would raise.
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


def tied_output_head(config: Mapping[str, object]) -> bool:
    """Whether this checkpoint ties its head to its embedding, as it states."""
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    value = config.get("tie_word_embeddings", text.get("tie_word_embeddings"))
    return bool(value)


def build_recipes(geometry: inventory.Geometry) -> tuple[TensorRecipe, ...]:
    """Every object's source, in object order, for one checkpoint's config."""
    declared = geometry.declared
    tied = tied_output_head(declared.hf_config)
    return tuple(
        recipe for recipe in derive_recipes(
            declared,
            capabilities=set(inventory.CAPABILITIES),
            tied_output_head=tied,
        )
        if not tied or recipe.object_name not in inventory.ALIASED_OBJECT_NAMES
    )


def open_reader(model_dir: str | Path) -> ShardReader:
    return ShardReader.for_directory(Path(model_dir))


def preflight_sources(
    reader: ShardReader,
    recipes: Sequence[TensorRecipe],
) -> SourcePreflight:
    """Every source tensor the recipes name, checked against the checkpoint before a
    single byte is converted."""
    return preflight_source_reader(reader, recipes)


__all__ = ["build_recipes", "open_reader", "preflight_sources", "tied_output_head"]
