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
from pathlib import Path

from .failure_taxonomy import apply_outcome_class
from .harness import HARNESS_REGISTRY, StepInput, StepResult
from .harness.repo_artifacts import safe_slug, write_json
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


def _crash_record(
    *,
    task: TaskSpec,
    workflow: Workflow,
    rollout_id: str,
    conductor_checkpoint: str | None,
    raw_output: str | None,
    exec_steps: list[ExecStep],
    outcome_class: str,
    detail: str,
) -> RolloutRecord:
    record = RolloutRecord(
        rollout_id=rollout_id,
        task_id=task.task_id,
        source_name=task.source.name,
        capability=task.capability,
        harness=task.environment.harness,
        conductor=ConductorRecord(
            checkpoint=conductor_checkpoint,
            raw_output=raw_output,
            workflow_parse_valid=True,
        ),
        workflow=workflow,
        execution=Execution(steps=exec_steps),
        grade=None,
        reward=None,
        outcome_class=outcome_class,
        valid_for_training=False,
        failure_class=f"{outcome_class}: {detail}",
    )
    return record


def _exception_outcome_class(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    provider_needles = (
        "ratelimit",
        "rate limit",
        "429",
        "apierror",
        "api error",
        "apitimeout",
        "timeout error",
        "apiconnection",
        "connection error",
        "permissiondenied",
        "permission denied",
        "authentication",
        "auth",
        "api key",
        "quota",
        "insufficient_quota",
        "403",
        "provider",
        "upstream",
    )
    if any(needle in text for needle in provider_needles):
        return "provider_failure_retry_or_exclude"
    return "harness_crash_exclude"


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
    artifact_dir: Path | None = None,
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
            outcome_class="invalid_workflow_trainable",
            valid_for_training=True,  # malformed workflows are training signal (ultra-data2 §12)
            failure_class=f"invalid_workflow_trainable: {e}",
        )

    harnesses: dict[str, object] = {}

    def get_harness(name: str):
        if name not in harnesses:
            harnesses[name] = HARNESS_REGISTRY[name]()
        return harnesses[name]

    results: dict[int, StepResult] = {}
    exec_steps: list[ExecStep] = []
    step_harnesses: dict[int, str] = {}
    artifact_root = artifact_dir.resolve() if artifact_dir is not None else None
    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)

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
                budget=step.budget,
                prior_artifacts=prior_artifacts,
                rollout_id=rollout_id,
                artifact_dir=str(
                    artifact_root / f"step_{i:02d}_{safe_slug(worker_id)}"
                )
                if artifact_root is not None
                else None,
            )
            try:
                res = await harness.run_step(step_input, pool, sampling)
            except Exception as exc:  # noqa: BLE001 - live harnesses fail heterogeneously
                outcome_class = _exception_outcome_class(exc)
                return _crash_record(
                    task=task,
                    workflow=workflow,
                    rollout_id=rollout_id,
                    conductor_checkpoint=conductor_checkpoint,
                    raw_output=raw_output,
                    exec_steps=exec_steps,
                    outcome_class=outcome_class,
                    detail=f"step {i} {harness_name} raised {type(exc).__name__}: {exc}",
                )
            results[i] = res
            exec_steps.append(
                ExecStep(
                    worker_id=step.worker_id,
                    harness=harness_name,
                    budget=step.budget,
                    session_ref=res.workspace_snapshot_ref or res.session_ref,
                    patch_ref=res.patch_ref,
                    messages_ref=res.messages_ref,
                    tool_events_ref=res.tool_events_ref or res.command_log_ref,
                    text=res.text,
                    input_tokens=res.input_tokens,
                    output_tokens=res.output_tokens,
                    cost_usd=res.cost_usd,
                    termination=res.termination,
                )
            )

        final = results[len(workflow.steps) - 1]  # last step defines the final answer
        final_harness = get_harness(step_harnesses[len(workflow.steps) - 1])
        try:
            grade = await asyncio.to_thread(final_harness.grade, task, final)  # subprocess graders mustn't block the loop
        except Exception as exc:  # noqa: BLE001 - live graders fail heterogeneously
            return _crash_record(
                task=task,
                workflow=workflow,
                rollout_id=rollout_id,
                conductor_checkpoint=conductor_checkpoint,
                raw_output=raw_output,
                exec_steps=exec_steps,
                outcome_class="grader_crash_quarantine",
                detail=f"{type(exc).__name__}: {exc}",
            )
        if artifact_root is not None:
            grade_ref = write_json(artifact_root / "grade.json", grade.model_dump(mode="json"))
            grade = grade.model_copy(update={"grader_ref": grade_ref})

        record = RolloutRecord(
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
        return apply_outcome_class(record)
    finally:
        for harness in harnesses.values():
            close = getattr(harness, "close", None)
            if callable(close):
                close()
