"""Checkpoint recipes share LFM2's declared tensor mapping machinery."""

from surogate.serve.convert.lfm2.recipe import (
    GGUF_NATIVE, build_recipes, open_reader, preflight_sources,
)
from .inventory import geometry_from_config


def build_recipes_by_name(geometry):
    return {item.object_name: item for item in build_recipes(geometry)}
