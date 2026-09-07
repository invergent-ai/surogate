"""Where every Gemma 4 mixture artifact object comes from in the checkpoint.

Nothing here writes a mapping. The declaration already says which checkpoint tensor
each parameter is (`hf_mapping`) and which parameters each artifact object is built
from, in row order (`ServeObject.components`); `derive_recipes` joins the two.

The two settled choices around that join are the dense target's, and hold here for
the same reasons: the output head is the embedding table wherever the checkpoint says
`tie_word_embeddings` (read, not assumed), and **no norm is unfolded** -- Gemma 4
applies `w` where Gemma 3 applies `1 + w`, so inheriting Gemma 3's transform would
scale every norm in the model by one plus itself and nothing downstream would raise.

The mixture adds one: the routed experts arrive **expert-major**, as a
`[experts, 2 * moe_intermediate, hidden]` `gate_up_proj` and an
`[experts, hidden, moe_intermediate]` `down_proj`, and the artifact stores the same
numbers as plain rows. That is `flatten_experts`, a contiguous reshape rather than a
permutation, and it is the same transform and the same two object names every other
routed target here uses -- so one expert-bank binder reads all of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from surogate.serve.convert.common.declaration import declare, derive_recipes
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


def build_recipes(config: Mapping[str, object]) -> tuple[TensorRecipe, ...]:
    """Every object's source, in object order, for one checkpoint's config."""
    declared = declare(inventory.architecture_of(config), dict(config))
    return tuple(
        derive_recipes(
            declared,
            capabilities=set(inventory.CAPABILITIES),
            tied_output_head=tied_output_head(config),
        )
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
