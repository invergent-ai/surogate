"""Harness contract + registry (ultra-data2 §6).

A harness runs ONE workflow step (one worker against one task/subtask) and can grade
the terminal state. The rollout executor routes by ``environment.harness`` and never
knows benchmark details. ``direct_qa`` is the lightest harness: call worker → grade text.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import Grade, TaskSpec
from ..workers import Sampling, WorkerPool


class StepInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: TaskSpec
    subtask: str  # the Conductor's subtask; for a direct call, the task prompt
    worker_id: str  # which pool worker executes this step
    step_index: int = 0
    access: list[int] = Field(default_factory=list)
    prior_artifacts: list = Field(default_factory=list)  # access-list artifacts (empty for direct)


class StepResult(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    error: str | None = None
    termination: str = "completed"


@runtime_checkable
class Harness(Protocol):
    name: str

    async def run_step(
        self, step: StepInput, pool: WorkerPool, sampling: Sampling
    ) -> StepResult: ...

    def grade(self, task: TaskSpec, final: StepResult) -> Grade: ...


HARNESS_REGISTRY: dict[str, type] = {}


def register_harness(cls: type) -> type:
    """Class decorator: register a harness under its ``name`` for the executor's router."""
    HARNESS_REGISTRY[cls.name] = cls
    return cls
