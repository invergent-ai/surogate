"""Spark checkpoint recipes derived from the training declaration."""

from collections.abc import Mapping

from surogate.dsl.models.spark2_5 import resolve_spark_config
from surogate.serve.convert.common import declaration
from surogate.serve.convert.common.recipe import TensorRecipe
from surogate.serve.convert.llama.recipe import open_reader as open_reader
from surogate.serve.convert.llama.recipe import preflight_sources as preflight_sources

from .inventory import Geometry


def geometry_from_config(config: Mapping[str, object]) -> Geometry:
    dimensions = resolve_spark_config(config)
    declared = declaration.declare("Spark2_5ForCausalLM", dict(config))
    return Geometry(**dimensions, declared=declared)


def build_recipes(g: Geometry) -> tuple[TensorRecipe, ...]:
    return declaration.derive_recipes(g.declared, tied_output_head=g.tied_output_head)
