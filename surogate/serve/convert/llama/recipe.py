"""Hugging Face source recipe for the dense Llama inventory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from surogate.serve.convert.common.safetensors import ShardReader
from surogate.serve.convert.common.recipe import (
    Concat,
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


def geometry_from_config(config: Mapping[str, object]) -> inventory.Geometry:
    """Read the artifact-shaping dimensions straight off `config.json`.

    `head_dim` is derived: a Llama config in this dialect has no such key, and
    the architecture fixes the width at `hidden_size // num_attention_heads`.
    """

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
    tied_output_head: bool = False,
) -> tuple[TensorRecipe, ...]:
    """Where every artifact object comes from in the checkpoint, in object order."""

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
                    obj + "attention/query_key_value",
                    # Ungated attention: q | k | v, with no gate block between
                    # k and v. The row order is the one the fused decode kernel
                    # reads and the one `LOGICAL_ROW_VIEW_SPECS` publishes.
                    Concat(
                        (
                            source(src + "self_attn.q_proj.weight", (query, hidden)),
                            source(src + "self_attn.k_proj.weight", (kv, hidden)),
                            source(src + "self_attn.v_proj.weight", (kv, hidden)),
                        ),
                        0,
                    ),
                ),
                # No query_norm / key_norm recipes: Llama has no such tensors.
                TensorRecipe(
                    obj + "attention/output",
                    source(src + "self_attn.o_proj.weight", (hidden, query)),
                ),
                TensorRecipe(
                    obj + "post_attention_norm",
                    source(src + "post_attention_layernorm.weight", (hidden,)),
                ),
                TensorRecipe(
                    obj + "mlp/gate_up",
                    Concat(
                        (
                            source(src + "mlp.gate_proj.weight",
                                   (geometry.intermediate, hidden)),
                            source(src + "mlp.up_proj.weight",
                                   (geometry.intermediate, hidden)),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    obj + "mlp/down",
                    source(src + "mlp.down_proj.weight", (hidden, geometry.intermediate)),
                ),
            )
        )

    recipes.extend(
        (
            TensorRecipe("text/final_norm", source("model.norm.weight", (hidden,))),
            TensorRecipe(
                "text/output_head",
                # tie_word_embeddings=False for TinyLlama: the head is its own
                # tensor and is nothing like the embedding, so reading the
                # embedding here would produce an artifact that loads and is
                # quietly wrong. A tied Llama export ships no `lm_head.weight`
                # at all, hence the other branch.
                embedding
                if tied_output_head
                else source("lm_head.weight", (geometry.vocab, hidden)),
            ),
        )
    )
    return tuple(recipes)


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
    recipes: Sequence[TensorRecipe] = RECIPE_SPECS,
) -> SourcePreflight:
    with open_reader(model_dir) as reader:
        return preflight_source_reader(reader, recipes)
