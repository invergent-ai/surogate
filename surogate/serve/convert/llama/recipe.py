"""Hugging Face source recipe for the dense Llama inventory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from surogate.serve.convert.common.safetensors import ShardReader
from surogate.serve.convert.common.recipe import (
    SourcePreflight,
    TensorRecipe,
    expression_sources,
    preflight_source_reader,
)
from surogate.serve.convert.common.declaration import derive_recipes

from . import inventory


# ---------------------------------------------------------------------------
# checkpoint geometry
# ---------------------------------------------------------------------------


def geometry_from_config(config: Mapping[str, object]) -> inventory.Geometry:
    """Read the artifact-shaping dimensions straight off `config.json`.

    `head_dim` is derived: a Llama config in this dialect has no such key, and
    the architecture fixes the width at `hidden_size // num_attention_heads`.
    """

    from surogate.serve.convert.common.checkpoint import resolve_dense

    declared = resolve_dense("LlamaForCausalLM", config)
    resolved = declared.config
    return inventory.Geometry(
        layers=int(resolved["n_layers"]), hidden=int(resolved["d_model"]),
        intermediate=int(resolved["d_ff"]), vocab=int(resolved["vocab_size"]),
        query_heads=int(resolved["num_query_heads"]), kv_heads=int(resolved["num_kv_heads"]),
        head_dim=int(resolved["head_size"]), declared=declared,
    )


# ---------------------------------------------------------------------------
# source recipe
# ---------------------------------------------------------------------------


def build_recipes(
    geometry: inventory.Geometry,
    *,
    tied_output_head: bool | None = None,
) -> tuple[TensorRecipe, ...]:
    """Derive recipes from the same resolved declaration as the inventory."""
    if tied_output_head is None:
        tied_output_head = bool(geometry.declared.hf_config.get("tie_word_embeddings", False))
    recipes = derive_recipes(
        geometry.declared, capabilities={"text"}, tied_output_head=tied_output_head
    )
    return recipes


def source_requirements(recipes: Sequence[TensorRecipe]) -> dict:
    requirements: dict = {}
    for recipe in recipes:
        for requirement in expression_sources(recipe.expression):
            requirements.setdefault(requirement.name, requirement)
    return requirements



# ---------------------------------------------------------------------------
# checkpoint access
# ---------------------------------------------------------------------------


def open_reader(model_dir: str | Path) -> ShardReader:
    """Open a sharded or single-file safetensors checkpoint.

    TinyLlama is one 2.1 GB `model.safetensors` with no index at all, and a
    reader that only knew how to follow an index could not open it; larger
    Llama releases are sharded, so both doors stay open.
    """

    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    if index.exists():
        return ShardReader(root)
    single = root / "model.safetensors"
    if single.exists():
        return ShardReader.from_file(single)
    raise FileNotFoundError(
        f"{root} holds neither model.safetensors.index.json nor model.safetensors"
    )


def preflight_sources(
    model_dir: str | Path,
    recipes: Sequence[TensorRecipe],
) -> SourcePreflight:
    with open_reader(model_dir) as reader:
        return preflight_source_reader(reader, recipes)
