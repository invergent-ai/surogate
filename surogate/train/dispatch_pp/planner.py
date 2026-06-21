"""Dispatch-PP stage planner.

Pure logic over plain data — no CUDA, no engine import, fully unit-testable. The
planner turns per-block profiles into a :class:`StagePlan`: an asymmetric
forward/backward partition (forward packs more blocks per stage than backward,
which costs ~3x more incl. recompute) chosen to minimize a pipeline-makespan
proxy under a per-stage VRAM ceiling.
"""

from __future__ import annotations

from typing import Sequence

from .types import BlockProfile, StageRange, StagePlan


def pack_stages(
    costs: Sequence[float],
    sizes: Sequence[int],
    max_workload: float,
    mem_ceiling: int,
) -> list[tuple[int, int]]:
    """Greedily pack consecutive blocks into contiguous stages.

    A new stage starts when adding the next block would exceed EITHER the
    per-stage time budget (``max_workload``) OR the per-stage memory ceiling
    (``mem_ceiling``). A single block that exceeds the ceiling still occupies its
    own stage (blocks are indivisible in v1).

    Returns inclusive ``(lo, hi)`` ranges covering ``[0, len(costs))`` exactly once.
    """
    n = len(costs)
    if n == 0:
        return []
    stages: list[tuple[int, int]] = []
    start = 0
    acc_cost = 0.0
    acc_size = 0
    for i in range(n):
        if i > start and (
            acc_cost + costs[i] > max_workload or acc_size + sizes[i] > mem_ceiling
        ):
            stages.append((start, i - 1))
            start = i
            acc_cost = 0.0
            acc_size = 0
        acc_cost += costs[i]
        acc_size += sizes[i]
    stages.append((start, n - 1))
    return stages
