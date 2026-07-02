"""Harness contract + registry (ultra-data2 §6).

A harness runs ONE workflow step (one worker against one task/subtask) and can grade
the terminal state. The rollout executor routes by ``environment.harness`` and never
knows benchmark details. ``direct_qa`` is the lightest harness: call worker → grade text.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import Grade, StepBudget, TaskSpec
from ..workers import Sampling, WorkerPool


class StepInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: TaskSpec
    subtask: str  # the Conductor's subtask; for a direct call, the task prompt
    worker_id: str  # which pool worker executes this step
    step_index: int = 0
    access: list[int] = Field(default_factory=list)
    budget: StepBudget = "medium"
    prior_artifacts: list = Field(default_factory=list)  # access-list artifacts (empty for direct)
    rollout_id: str | None = None
    artifact_dir: str | None = None


class StepResult(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    error: str | None = None
    termination: str = "completed"
    session_ref: str | None = None
    patch_ref: str | None = None
    messages_ref: str | None = None
    tool_events_ref: str | None = None
    workspace_snapshot_ref: str | None = None
    command_log_ref: str | None = None
    artifact_dir: str | None = None


@runtime_checkable
class Harness(Protocol):
    name: str

    async def run_step(
        self, step: StepInput, pool: WorkerPool, sampling: Sampling
    ) -> StepResult: ...

    def grade(self, task: TaskSpec, final: StepResult) -> Grade: ...


HARNESS_REGISTRY: dict[str, type] = {}


BUDGET_WALL_SECONDS: dict[StepBudget, int | None] = {
    "short": 20 * 60,
    "medium": 60 * 60,
    "long": 4 * 60 * 60,
    "max": None,
}


def wall_time_cap_seconds(
    budget: StepBudget,
    *,
    task_cap: int | None = None,
    harness_cap: int | None = None,
) -> int | None:
    """Return the strictest wall-time cap from workflow budget, task spec, and harness config."""

    caps = [BUDGET_WALL_SECONDS[budget], task_cap, harness_cap]
    finite = [cap for cap in caps if cap is not None]
    return min(finite) if finite else None


def register_harness(cls: type) -> type:
    """Class decorator: register a harness under its ``name`` for the executor's router."""
    HARNESS_REGISTRY[cls.name] = cls
    return cls
