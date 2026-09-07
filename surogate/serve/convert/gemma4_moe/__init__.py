"""Gemma 4 mixture (`google/gemma-4-26B-A4B`) conversion target.

The 26B-A4B runs a dense feed-forward *and* 128 routed experts over the same input on
every one of its 30 layers, and sums them. Its attention is the dense target's exactly
-- two head geometries, `attention_k_eq_v` on the global layers -- so what makes it a
target of its own is entirely the feed-forward: three extra norms, a router with two
learned scales and a weightless normalisation, and an expert bank.

The dense sizes and the E-series are separate architectures with their own targets.
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
    is_mixture,
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
    "is_mixture",
    "tensor_specs",
]
