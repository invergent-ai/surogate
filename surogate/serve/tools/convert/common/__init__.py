"""Shared conversion helpers for SInfer converters.

Both halves of the shared code live here: the source-checkpoint readers and
quantisers, and the artifact-side inventory, recipe, and report machinery that
used to sit under a `qwen3_6` package. That package held nothing but this
directory and had no target of its own, while llama, gemma3, qwen3, and
qwen4exp all imported through it -- so its name described none of its callers.
"""

from .quantize import QuantizedMatrix, pick_device, quantize_and_encode, quantize_matrix
from .safetensors import ShardReader, TensorMetadata

__all__ = [
    "QuantizedMatrix",
    "ShardReader",
    "TensorMetadata",
    "pick_device",
    "quantize_and_encode",
    "quantize_matrix",
]
