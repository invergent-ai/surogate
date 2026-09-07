"""Hugging Face source recipe for the dense Qwen3 inventory."""

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
from surogate.serve.convert.common.declaration import declare, derive_recipes
from surogate.dsl.ir_builder import resolve_architecture
from surogate.serve.convert.common.recipe import (
    validate_recipe_coverage as _validate_recipe_coverage,
)

from . import inventory


# ---------------------------------------------------------------------------
# checkpoint geometry
# ---------------------------------------------------------------------------


def geometry_from_config(config: Mapping[str, object]) -> inventory.Geometry:
    """Read the artifact-shaping dimensions straight off `config.json`."""

    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim") or hidden // heads)
    return inventory.Geometry(
        layers=int(config["num_hidden_layers"]),
        hidden=hidden,
        intermediate=int(config["intermediate_size"]),
        vocab=int(config["vocab_size"]),
        query_heads=heads,
        kv_heads=int(config["num_key_value_heads"]),
        head_dim=head_dim,
    )


# ---------------------------------------------------------------------------
# source recipe
# ---------------------------------------------------------------------------


def build_recipes(
    geometry: inventory.Geometry = inventory.GEOMETRY,
    *,
    tied_output_head: bool = True,
    hf_config: Mapping[str, object] | None = None,
) -> tuple[TensorRecipe, ...]:
    """Where every artifact object comes from in the checkpoint, in object order.

    Not written here: derived from the training declaration, which maps every
    parameter to its checkpoint tensor (`hf_mapping`) and says which parameters
    each artifact object is built from, in row order (`ServeObject.components`).
    `hf_config` is the checkpoint's own `config.json` when the caller has it, and
    the registered geometry's implied config otherwise.

    `tied_output_head` stays an argument rather than being read from the config:
    it is a property of the file in hand, and the converter has already resolved
    it against what the checkpoint actually ships.
    """
    config = dict(hf_config) if hf_config is not None else inventory.hf_config_for(geometry)
    declaration = declare(resolve_architecture(config), config)
    recipes = derive_recipes(
        declaration, capabilities={"text"}, tied_output_head=tied_output_head
    )
    return recipes


RECIPE_SPECS = build_recipes()
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    _validate_recipe_coverage(RECIPE_SPECS, inventory.TENSOR_SPECS)


def source_requirements(recipes: Sequence[TensorRecipe] = RECIPE_SPECS) -> dict:
    requirements: dict = {}
    for recipe in recipes:
        for requirement in expression_sources(recipe.expression):
            requirements.setdefault(requirement.name, requirement)
    return requirements


validate_recipe_coverage()


# ---------------------------------------------------------------------------
# checkpoint access
# ---------------------------------------------------------------------------


def open_reader(model_dir: str | Path) -> ShardReader:
    """Open a sharded or single-file safetensors checkpoint.

    The hybrid targets in this package are all multi-shard, so their converters
    open by index. Qwen3-0.6B is one 1.4 GB `model.safetensors` with no index at
    all, and a converter that only knew how to follow an index could not read it.
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
    recipes: Sequence[TensorRecipe] = RECIPE_SPECS,
) -> SourcePreflight:
    with open_reader(model_dir) as reader:
        return preflight_source_reader(reader, recipes)
