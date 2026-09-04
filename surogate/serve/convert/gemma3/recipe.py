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
) -> tuple[TensorRecipe, ...]:
    """Where every artifact object comes from in the checkpoint, in object order.

    Every entry is a plain read. Gemma 3 fuses nothing, so there is no `Concat`
    anywhere in this recipe — and the norms are copied rather than adjusted,
    because a safetensors checkpoint already stores the unfolded `w` the runtime
    wants (see the module docstring).
    """

    hidden = geometry.hidden
    query, kv = geometry.query_size, geometry.kv_size
    embedding = source("model.embed_tokens.weight", (geometry.vocab, hidden))

    recipes: list[TensorRecipe] = [TensorRecipe("text/token_embedding", embedding)]

    for layer in range(geometry.layers):
        src = f"model.layers.{layer}."
        obj = f"text/layers/{layer}/"
        recipes.extend(
            (
                TensorRecipe(
                    obj + "input_norm",
                    source(src + "input_layernorm.weight", (hidden,)),
                ),
                TensorRecipe(
                    obj + "post_attention_norm",
                    source(src + "post_attention_layernorm.weight", (hidden,)),
                ),
                TensorRecipe(
                    obj + "pre_feedforward_norm",
                    source(src + "pre_feedforward_layernorm.weight", (hidden,)),
                ),
                TensorRecipe(
                    obj + "post_feedforward_norm",
                    source(src + "post_feedforward_layernorm.weight", (hidden,)),
                ),
                TensorRecipe(
                    obj + "attention/query",
                    source(src + "self_attn.q_proj.weight", (query, hidden)),
                ),
                TensorRecipe(
                    obj + "attention/key",
                    source(src + "self_attn.k_proj.weight", (kv, hidden)),
                ),
                TensorRecipe(
                    obj + "attention/value",
                    source(src + "self_attn.v_proj.weight", (kv, hidden)),
                ),
                TensorRecipe(
                    obj + "attention/query_norm",
                    source(src + "self_attn.q_norm.weight", (geometry.head_dim,)),
                ),
                TensorRecipe(
                    obj + "attention/key_norm",
                    source(src + "self_attn.k_norm.weight", (geometry.head_dim,)),
                ),
                TensorRecipe(
                    obj + "attention/output",
                    source(src + "self_attn.o_proj.weight", (hidden, query)),
                ),
                TensorRecipe(
                    obj + "mlp/gate",
                    source(src + "mlp.gate_proj.weight", (geometry.intermediate, hidden)),
                ),
                TensorRecipe(
                    obj + "mlp/up",
                    source(src + "mlp.up_proj.weight", (geometry.intermediate, hidden)),
                ),
                TensorRecipe(
                    obj + "mlp/down",
                    source(src + "mlp.down_proj.weight", (hidden, geometry.intermediate)),
                ),
            )
        )

    recipes.append(TensorRecipe("text/final_norm", source("model.norm.weight", (hidden,))))
    if not tied_output_head:
        # `tie_word_embeddings` is a property of the checkpoint in hand, not of
        # the architecture, so an untied Gemma 3 stores a head of its own and
        # reads it from `lm_head.weight`. Every published `Gemma3ForCausalLM`
        # ties, and gemma-3-270m-it ships no `lm_head.weight` at all, so this is
        # the branch that does not run today.
        recipes.append(
            TensorRecipe(
                "text/output_head", source("lm_head.weight", (geometry.vocab, hidden))
            )
        )
    # The tied case has no `text/output_head` recipe because it has no such
    # object: `inventory.ALIAS_SPECS` makes the head a role served by
    # `text/token_embedding`, and the binder fills both from the one table.
    # Giving it the embedding's expression instead would quantise 167.8M elements
    # a second time and write ~170 MB of byte-identical duplicate.
    return tuple(recipes)


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
