"""Python reference for the qwen3_5 architecture: an interleaved
gated-delta-net / full-attention decoder with a dense MLP, at any size.
"""

from .bindings import (
    ArtifactBinding,
    AxisView,
    BindingError,
    BoundResource,
    LogicalRowView,
    PhysicalBlock,
    WeightObject,
)
from .config import ModelConfig, VisionConfig
from .model import RefModel
from .weights import MemoryPlan, WeightStore

__all__ = [
    "ArtifactBinding",
    "AxisView",
    "BindingError",
    "BoundResource",
    "LogicalRowView",
    "MemoryPlan",
    "ModelConfig",
    "PhysicalBlock",
    "RefModel",
    "VisionConfig",
    "WeightObject",
    "WeightStore",
]
