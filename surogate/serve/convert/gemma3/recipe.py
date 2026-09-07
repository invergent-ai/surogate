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
from surogate.serve.convert.common.declaration import declare, derive_recipes
from surogate.dsl.ir_builder import resolve_architecture
from surogate.serve.convert.common.recipe import (
    validate_recipe_coverage as _validate_recipe_coverage,
)

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

    head_dim = config.get("head_dim")
    if not head_dim:
        raise ValueError(
            "config.json declares no head_dim; Gemma 3 does not derive it from "
            "hidden_size // num_attention_heads and the value cannot be guessed"
        )
    return inventory.Geometry(
        layers=int(config["num_hidden_layers"]),
        hidden=int(config["hidden_size"]),
        intermediate=int(config["intermediate_size"]),
        vocab=int(config["vocab_size"]),
        query_heads=int(config["num_attention_heads"]),
        kv_heads=int(config["num_key_value_heads"]),
        head_dim=int(head_dim),
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

    A tied Gemma 3 has no `text/output_head` object at all — `inventory.ALIAS_SPECS`
    makes the head a role served by `text/token_embedding` — so that recipe is
    dropped rather than given the embedding's expression, which would quantise the
    same 167.8M elements twice and write ~170 MB of byte-identical duplicate.
    """
    config = dict(hf_config) if hf_config is not None else inventory.hf_config_for(geometry)
    declaration = declare(resolve_architecture(config), config)
    recipes = derive_recipes(
        declaration, capabilities={"text"}, tied_output_head=tied_output_head
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


RECIPE_SPECS = build_recipes()
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in RECIPE_SPECS}


def validate_recipe_coverage(
    recipes: Sequence[TensorRecipe] = RECIPE_SPECS,
    *,
    tied_output_head: bool = True,
) -> None:
    stored, _ = inventory.active_specs(tied_output_head=tied_output_head)
    _validate_recipe_coverage(recipes, stored)


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
    recipes: Sequence[TensorRecipe] = RECIPE_SPECS,
) -> SourcePreflight:
    with open_reader(model_dir) as reader:
        return preflight_source_reader(reader, recipes)
