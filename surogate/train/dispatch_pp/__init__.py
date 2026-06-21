"""Dispatch pipeline-parallelism planner (pure Python, GPU-free)."""

from .types import BlockProfile, StageRange, StagePlan

__all__ = ["BlockProfile", "StageRange", "StagePlan"]
