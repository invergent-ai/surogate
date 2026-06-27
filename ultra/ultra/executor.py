"""Multi-step workflow executor (ultra-data §6; ultra-intro execution semantics).

Runs a validated workflow's steps in order, assembling each step's context
``C_i = original query + subtask_i + {subtask_j, response_j : j in access_i}``, routes each
step to the task's harness, grades the final step, and emits a RolloutRecord carrying the
faithful 0 / 0.5 / 1.0 reward. An invalid workflow short-circuits to reward 0.

Steps run sequentially (correctness first); independent steps can be scheduled concurrently
once this is solid.
"""

from __future__ import annotations

import asyncio

from .harness import HARNESS_REGISTRY, StepInput, StepResult
from .schemas import (
    ConductorRecord,
    ExecStep,
    Execution,
    RolloutRecord,
    TaskSpec,
    Workflow,
)
from .workers import Sampling, WorkerPool
from .workflow import WorkflowValidationError, validate_workflow


def faithful_reward(parse_valid: bool, success: bool) -> float:
    """ultra-intro §6: 0 malformed · 0.5 valid+incorrect · 1.0 valid+correct."""
    if not parse_valid:
        return 0.0
    return 1.0 if success else 0.5


async def execute_workflow(
    task: TaskSpec,
    workflow: Workflow,
    pool: WorkerPool,
    sampling: Sampling,
    rollout_id: str,
    *,
    worker_ids: list[str] | None = None,
    worker_harnesses: dict[str, str] | None = None,
    conductor_checkpoint: str | None = None,
    raw_output: str | None = None,
    max_steps: int = 5,
) -> RolloutRecord:
    ids = worker_ids or pool.worker_ids

    try:
        validate_workflow(workflow, worker_count=len(ids), max_steps=max_steps)
    except WorkflowValidationError as e:
        return RolloutRecord(
            rollout_id=rollout_id,
            task_id=task.task_id,
            source_name=task.source.name,
            capability=task.capability,
            harness=task.environment.harness,
            conductor=ConductorRecord(
                checkpoint=conductor_checkpoint, raw_output=raw_output, workflow_parse_valid=False
            ),
            workflow=workflow,
            execution=Execution(steps=[]),
            grade=None,
            reward=0.0,
            valid_for_training=True,  # malformed workflows are training signal (ultra-data2 §12)
            failure_class=f"invalid_workflow: {e}",
        )

    harnesses: dict[str, object] = {}

    def get_harness(name: str):
        if name not in harnesses:
            harnesses[name] = HARNESS_REGISTRY[name]()
        return harnesses[name]

    results: dict[int, StepResult] = {}
    exec_steps: list[ExecStep] = []
    step_harnesses: dict[int, str] = {}

    try:
        for i, step in enumerate(workflow.steps):
            worker_id = ids[step.worker_id]
            harness_name = (worker_harnesses or {}).get(worker_id, task.environment.harness)
            harness = get_harness(harness_name)
            step_harnesses[i] = harness_name
            prior_artifacts = [
                {
                    "step_index": j,
                    "worker_id": workflow.steps[j].worker_id,
                    "worker_name": ids[workflow.steps[j].worker_id],
                    "harness": step_harnesses[j],
                    "subtask": workflow.steps[j].subtask,
                    "response": results[j].text,
                }
                for j in step.access
            ]
            step_input = StepInput(
                task=task,
                subtask=step.subtask,
                worker_id=worker_id,
                step_index=i,
                access=list(step.access),
                prior_artifacts=prior_artifacts,
            )
            res = await harness.run_step(step_input, pool, sampling)
            results[i] = res
            exec_steps.append(
                ExecStep(
                    worker_id=step.worker_id,
                    harness=harness_name,
                    text=res.text,
                    input_tokens=res.input_tokens,
                    output_tokens=res.output_tokens,
                    cost_usd=res.cost_usd,
                    termination=res.termination,
                )
            )

        final = results[len(workflow.steps) - 1]  # last step defines the final answer
        final_harness = get_harness(step_harnesses[len(workflow.steps) - 1])
        grade = await asyncio.to_thread(final_harness.grade, task, final)  # subprocess graders mustn't block the loop

        return RolloutRecord(
            rollout_id=rollout_id,
            task_id=task.task_id,
            source_name=task.source.name,
            capability=task.capability,
            harness=task.environment.harness,
            conductor=ConductorRecord(
                checkpoint=conductor_checkpoint, raw_output=raw_output, workflow_parse_valid=True
            ),
            workflow=workflow,
            execution=Execution(steps=exec_steps),
            grade=grade,
            reward=faithful_reward(True, grade.success),
            valid_for_training=True,
        )
    finally:
        for harness in harnesses.values():
            close = getattr(harness, "close", None)
            if callable(close):
                close()
