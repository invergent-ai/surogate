"""Source recipes from the resolved text and optional DFlash declarations."""

from surogate.serve.convert.common.declaration import derive_recipes
from surogate.serve.convert.common.recipe import (
    SourcePreflight, SourceTensor, TensorRecipe, materialize_recipe,
    expression_shape, expression_sources, preflight_sources,
    source_requirements, validate_recipe_coverage as _validate,
)
from surogate.serve.convert.qwen3_5.recipe import _build_draft_head_recipes, _build_vision_recipes
from . import inventory

NATIVE_EXCLUDE_SUFFIXES = (
    "attention/query_key_gate_value", "gdn/query_key_value_z", "gdn/output",
    "moe/shared_gate_up", "moe/shared_down",
)


def build_recipes(g: inventory.Geometry, *, dflash=None) -> tuple[TensorRecipe, ...]:
    declaration = dflash.declaration(g) if dflash is not None else g.declared
    capabilities = {"text", "dflash"} if dflash is not None else {"text"}
    declared = derive_recipes(declaration, capabilities=capabilities)
    by_name = {r.object_name: r for r in declared}
    extra = (_build_draft_head_recipes(by_name["text/output_head"].expression, g)
             + _build_vision_recipes(g))
    by_name.update((r.object_name, r) for r in extra)
    return tuple(by_name[s.name] for s in inventory.build_tensor_specs(g, dflash=dflash))


def validate_recipe_coverage(g: inventory.Geometry, *, dflash=None):
    _validate(build_recipes(g, dflash=dflash), inventory.build_tensor_specs(g, dflash=dflash))


def preflight_base_sources(root, recipes):
    return preflight_sources(root, tuple(r for r in recipes if not r.object_name.startswith("dflash/")))


def preflight_dflash_sources(root, recipes):
    return preflight_sources(root, tuple(r for r in recipes if r.object_name.startswith("dflash/")))
