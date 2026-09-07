"""Dense Gemma 4 (`google/gemma-4-12B`, `google/gemma-4-31B-it`) conversion target.

Gemma 4 alternates windowed attention with global attention *at two different head
geometries* -- 256-wide heads through the window, 512-wide heads over the whole
context -- and on the global layers reuses the key projection as the value, which is
what `attention_k_eq_v` names. The E-series (per-layer input embeddings, cross-layer
KV sharing) and the 26B-A4B mixture are separate architectures with their own
targets.
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
