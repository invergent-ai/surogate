"""LFM2 (`Lfm2ForCausalLM`) artifact conversion target.

LFM2 interleaves full-attention layers with short-convolution layers, the same
hybrid shape the Qwen 3.5 family serves with a different mixer. Which layer is
which comes from the checkpoint's `full_attn_idxs`.
"""

from .inventory import (
    ARCHITECTURE,
    CAPABILITIES,
    MODEL_ID,
    RESOURCE_SPECS,
    TARGET_KEY,
    WEIGHTS_ID,
    Geometry,
    declared_objects,
    geometry_from_config,
    tensor_specs,
)

__all__ = [
    "ARCHITECTURE",
    "CAPABILITIES",
    "Geometry",
    "MODEL_ID",
    "RESOURCE_SPECS",
    "TARGET_KEY",
    "WEIGHTS_ID",
    "declared_objects",
    "geometry_from_config",
    "tensor_specs",
]
