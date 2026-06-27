"""Scaffolds validate, and the cross-fitted step-zero runner reports correctly (offline)."""

import pytest

from ultra.config import WorkerSpec
from ultra.scaffolds import SCAFFOLDS
from ultra.schemas import (
    EnvironmentSpec,
    GraderSpec,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskSpec,
)
from ultra.stepzero import run_stepzero
from ultra.workers import FakeProvider, Sampling, WorkerPool
from ultra.workflow import validate_workflow


def test_all_scaffolds_are_valid_workflows():
    expected = {
        "A_direct",
        "B_plan_execute",
        "C_solve_critique_revise",
        "D_debate_synthesize",
        "E_specialist_plan_execute",
        "F_execute_critic_revise",
    }
    assert set(SCAFFOLDS) == expected
    for build in SCAFFOLDS.values():
        wf = build()
        validate_workflow(wf, worker_count=2)  # E/specialist uses worker index 1
        assert 1 <= len(wf.steps) <= 5


def _task(i, sol="C") -> TaskSpec:
    return TaskSpec(
        task_id=f"t{i}",
        capability="factual_qa",
        source=SourceRef(name="s", version="v", policy="train_allowed"),
        input=TaskInput(messages=[{"role": "user", "content": f"Q{i}"}]),
        environment=EnvironmentSpec(harness="direct_qa"),
        grader=GraderSpec(type="mc_letter", expected_answer=sol),
        splitting=SplittingSpec(group_id="g", split="pool_discovery"),
    )


def _pool(answer_fn, n_workers=2) -> WorkerPool:
    return WorkerPool(
        [WorkerSpec(worker_id=f"w{i}", model="fake") for i in range(n_workers)],
        FakeProvider(answer_fn),
    )


@pytest.mark.asyncio
async def test_no_headroom_when_everything_correct():
    # Every arm always correct → no scaffold beats best-single; oracle signal is pure noise (0).
    pool = _pool(lambda model, messages, sampling: "Answer: C")
    tasks = [_task(i) for i in range(6)]
    report = await run_stepzero(tasks, pool, Sampling(), n_reps=2, n_folds=3)
    assert report.n_tasks == 6 and report.n_reps == 2
    assert report.best_single == 1.0
    assert report.delta_fixed_cv == 0.0
    assert report.oracle_signal == 0.0  # null == observed → no real per-task headroom


@pytest.mark.asyncio
async def test_orchestration_headroom_detected_via_crossfit():
    # Right answer only once a step can see a prior result → direct fails, access-using
    # scaffolds succeed. Cross-fitted Δ_fixed must catch it.
    def fn(model, messages, sampling):
        joined = " ".join(m["content"] for m in messages)
        return "Answer: C" if "Authorized prior-step results" in joined else "Answer: A"

    pool = _pool(fn, n_workers=2)
    tasks = [_task(i) for i in range(6)]
    report = await run_stepzero(tasks, pool, Sampling(), n_reps=2, n_folds=3)
    assert report.best_single == 0.0  # direct never sees a prior → always wrong
    assert report.delta_fixed_cv == 1.0  # an access-using scaffold wins on every held-out fold
    assert report.delta_fixed_ci[0] > 0
