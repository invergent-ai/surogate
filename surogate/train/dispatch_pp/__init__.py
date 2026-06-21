"""Dispatch pipeline-parallelism planner (pure Python, GPU-free)."""

from .planner import assign_numa, plan_stages
from .types import BlockProfile, StageRange, StagePlan

__all__ = ["BlockProfile", "StageRange", "StagePlan", "plan_stages", "assign_numa"]
