"""Serving inventory derived from the resolved checkpoint configuration."""

from collections.abc import Mapping
from typing import Any

from surogate.serve.convert.common import gemma4 as common
from surogate.serve.convert.common.gemma4 import (
    ALIASED_OBJECT_NAMES, ALIAS_SPECS, ARCHITECTURES, BF16, CAPABILITIES,
    Geometry, RESOURCE_SPECS, W8, declared_objects, stored_objects, tensor_specs,
)

TARGET_KEY = 'gemma4_e'
MODEL_ID = TARGET_KEY
WEIGHTS_ID = "groupwise-int"


def architecture_of(config: Mapping[str, Any]) -> str:
    return common.architecture_of(config, TARGET_KEY)


def geometry_from_config(config: Mapping[str, Any]) -> Geometry:
    return common.geometry_from_config(config, TARGET_KEY)
