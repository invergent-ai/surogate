"""E-series Gemma 4 (`google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`) conversion target.

The E-series attends the way the dense sizes do -- 256-wide heads through the window,
512-wide over the whole context -- and adds two things of its own: a second embedding table
whose per-layer slice is mixed into every block, and a tail of layers that hold a query
projection only and read an earlier layer's keys and values.
"""

from .inventory import (
    ARCHITECTURES,
    CAPABILITIES,
    MODEL_ID,
    RESOURCE_SPECS,
    TARGET_KEY,
    WEIGHTS_ID,
    Geometry,
    architecture_of,
    declared_objects,
    geometry_from_config,
    tensor_specs,
)

__all__ = [
    "ARCHITECTURES",
    "CAPABILITIES",
    "Geometry",
    "MODEL_ID",
    "RESOURCE_SPECS",
    "TARGET_KEY",
    "WEIGHTS_ID",
    "architecture_of",
    "declared_objects",
    "geometry_from_config",
    "tensor_specs",
]
