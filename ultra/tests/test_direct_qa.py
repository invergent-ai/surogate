"""End-to-end vertical slice, offline (FakeProvider): task → worker → grade → RolloutRecord."""

import pytest

from ultra.config import WorkerSpec
from ultra.rollout import direct_rollout
from ultra.schemas import (
    EnvironmentSpec,
    GraderSpec,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskSpec,
)
from ultra.workers import FakeProvider, Sampling, WorkerPool


def _mc_task(solution="C") -> TaskSpec:
    return TaskSpec(
        task_id="t1",
        capability="factual_qa",
        source=SourceRef(name="existing_bank", version="v1", policy="train_allowed"),
        input=TaskInput(
            messages=[
                {"role": "system", "content": "Answer with 'Answer: X'."},
                {"role": "user", "content": "Which letter is C?"},
            ]
        ),
        environment=EnvironmentSpec(harness="direct_qa"),
        grader=GraderSpec(type="mc_letter", expected_answer=solution),
        splitting=SplittingSpec(group_id="g", split="grpo_train"),
    )


def _pool(answer: str) -> WorkerPool:
    return WorkerPool(
        [WorkerSpec(worker_id="w0", model="fake/model")],
        FakeProvider(lambda model, messages, sampling: answer),
    )


@pytest.mark.asyncio
async def test_direct_rollout_correct():
    rec = await direct_rollout(_mc_task("C"), _pool("Answer: C"), "w0", Sampling(), "rol_1")
    assert rec.grade.success is True
    assert rec.reward == 1.0
    assert rec.workflow.steps[0].worker_id == 0
    assert rec.execution.steps[0].text == "Answer: C"
    assert rec.harness == "direct_qa"


@pytest.mark.asyncio
async def test_direct_rollout_wrong_gets_half_reward():
    rec = await direct_rollout(_mc_task("C"), _pool("Answer: A"), "w0", Sampling(), "rol_2")
    assert rec.grade.success is False
    assert rec.reward == 0.5
