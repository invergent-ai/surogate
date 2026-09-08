"""Shared checkpoint-driven vision schedule with MoE weight ownership."""

from ..qwen3_5.vision import VisionEncoder as _VisionEncoder, VisionOutput, VisionStats
from .weights import WeightStore


class VisionEncoder(_VisionEncoder):
    weight_store_type = WeightStore


__all__ = ["VisionEncoder", "VisionOutput", "VisionStats"]
