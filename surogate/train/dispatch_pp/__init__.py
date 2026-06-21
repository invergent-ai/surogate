"""Dispatch pipeline-parallelism planner (pure Python, GPU-free)."""

from .planner import assign_numa, plan_stages
from .profile import (
    block_weight_bytes,
    build_block_profiles,
    plan_for_model,
    resolve_vram_budget_bytes,
)
from .types import BlockProfile, StageRange, StagePlan

__all__ = [
    "BlockProfile",
    "StageRange",
    "StagePlan",
    "plan_stages",
    "assign_numa",
    "block_weight_bytes",
    "build_block_profiles",
    "plan_for_model",
    "resolve_vram_budget_bytes",
]
