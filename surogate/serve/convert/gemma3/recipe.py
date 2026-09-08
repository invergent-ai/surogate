"""Hugging Face source recipe for the Gemma 3 inventory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from surogate.serve.convert.common.safetensors import ShardReader
from surogate.serve.convert.common.recipe import (
    SourcePreflight,
    TensorRecipe,
    expression_sources,
    preflight_source_reader,
    source,
)
from surogate.serve.convert.common.declaration import derive_recipes

from . import inventory


# ---------------------------------------------------------------------------
# checkpoint geometry
# ---------------------------------------------------------------------------


# The reader lives beside the recipes because that is where the GGUF repack planner looks
# for it: `serve/ingest.py::_gguf_geometry` asks this module for the checkpoint's dimensions
# before asking it to build against them, and one that cannot answer is planned against the
# registered 270M shapes instead.
def geometry_from_config(config: Mapping[str, object]) -> inventory.Geometry:
    """Read the artifact-shaping dimensions straight off `config.json`.

    `head_dim` is read, never derived. Gemma 3 decouples the head width from the
    hidden size: 270M is 4 heads of 256 against a hidden of 640, so the
    `hidden // heads` shortcut the Llama converter can take would give 160 here.
    """

    from surogate.serve.convert.common.checkpoint import resolve_dense

    if not (config.get("layer_types") or config.get("sliding_window_pattern")
            or config.get("_sliding_window_pattern")):
        raise ValueError("config.json states no attention schedule")
    declared = resolve_dense("Gemma3ForCausalLM", config)
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
    """Derive recipes from the checkpoint declaration used to shape the inventory."""
    if tied_output_head is None:
        tied_output_head = bool(geometry.declared.hf_config.get("tie_word_embeddings", True))
    recipes = derive_recipes(
        geometry.declared, capabilities={"text"}, tied_output_head=tied_output_head
    )
    if tied_output_head:
        recipes = tuple(r for r in recipes if r.object_name != "text/output_head")
    else:
        # The declaration states that this architecture ties, and maps `lm_head` to the
        # embedding tensor accordingly — every published `Gemma3ForCausalLM` does, and
        # gemma-3-270m-it ships no `lm_head.weight` at all. A file that unties stores a
        # head of its own, which is a property of that file rather than of the
        # architecture, so the converter names it here. This is the branch that does not
        # run today.
        head = TensorRecipe(
            "text/output_head",
            source("lm_head.weight", (geometry.vocab, geometry.hidden)),
        )
        recipes = tuple(head if r.object_name == "text/output_head" else r for r in recipes)
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

    Gemma 3 270M is one 536 MB `model.safetensors` with no index at all, and a
    reader that only knew how to follow an index could not open it; the larger
    Gemma 3 releases are sharded, so both doors stay open.
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
