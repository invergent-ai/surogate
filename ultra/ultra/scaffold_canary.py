"""Single-job scaffold canary runner.

This is the narrow live-smoke entrypoint for scaffold-aware repo jobs: load one
TaskSpec, pick one preregistered arm, execute through the normal Ultra executor,
and write one RolloutRecord. It is intentionally small so canary runs are easy to
reproduce before launching the full tournament.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import WorkerSpec
from .executor import execute_workflow
from .scaffold_tournament import canonical_arms, canonical_workers, worker_harness_map
from .schemas import RolloutRecord, StepBudget, TaskSpec, Workflow
from .workers import FakeProvider, Sampling, WorkerPool


def load_taskspecs(path: Path) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                tasks.append(TaskSpec.model_validate(json.loads(line)))
    return tasks


def select_task(tasks: list[TaskSpec], task_id: str | None = None) -> TaskSpec:
    if not tasks:
        raise ValueError("no TaskSpecs loaded")
    if task_id is None:
        return tasks[0]
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise ValueError(f"task_id not found: {task_id}")


def select_arm(name: str):
    for arm in canonical_arms(canonical_workers()):
        if arm.name == name:
            return arm
    raise ValueError(f"unknown scaffold arm: {name}")


async def run_canary(
    *,
    tasks_jsonl: Path,
    arm_name: str,
    task_id: str | None = None,
    rollout_id: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    reasoning: str | None = "high",
    budget: StepBudget | None = None,
    artifact_dir: Path | None = None,
    worker_harnesses: dict[str, str] | None = None,
) -> RolloutRecord:
    workers = canonical_workers()
    worker_ids = [worker.name for worker in workers]
    worker_specs = [WorkerSpec(worker_id=worker.name, model=worker.model) for worker in workers]
    pool = WorkerPool(worker_specs, FakeProvider())
    sampling = Sampling(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        reasoning_effort=reasoning,
    )
    task = select_task(load_taskspecs(tasks_jsonl), task_id)
    arm = select_arm(arm_name)
    workflow = arm.workflow
    if budget is not None:
        workflow = Workflow(steps=[step.model_copy(update={"budget": budget}) for step in workflow.steps])
    return await execute_workflow(
        task,
        workflow,
        pool,
        sampling,
        rollout_id or f"canary-{arm.name}-{task.task_id}",
        worker_ids=worker_ids,
        worker_harnesses=worker_harnesses or worker_harness_map(workers, task_harness=task.environment.harness),
        artifact_dir=artifact_dir,
    )


def rollout_to_json(record: RolloutRecord) -> str:
    return json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True)


async def run_cli(args: Any) -> dict[str, Any]:
    record = await run_canary(
        tasks_jsonl=Path(args.tasks_jsonl),
        task_id=args.task_id,
        arm_name=args.arm,
        rollout_id=args.rollout_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning=args.reasoning,
        budget=args.budget,
        artifact_dir=Path(args.artifact_dir) if getattr(args, "artifact_dir", None) else None,
    )
    text = rollout_to_json(record)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    if getattr(args, "agent_trace_out", None):
        from .trace_export import write_agent_trace

        task = select_task(load_taskspecs(Path(args.tasks_jsonl)), args.task_id)
        workers = canonical_workers()
        write_agent_trace(
            record,
            task,
            worker_models={worker.worker_id: worker.model for worker in workers},
            out=Path(args.agent_trace_out),
        )
    print(text)
    return record.model_dump(mode="json")
