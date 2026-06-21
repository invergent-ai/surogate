"""Dispatch pipeline-parallelism planner (pure Python, GPU-free)."""

from .planner import plan_stages
from .types import BlockProfile, StageRange, StagePlan

__all__ = ["BlockProfile", "StageRange", "StagePlan", "plan_stages"]
