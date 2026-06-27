"""Direct rollout: scaffold A (best single worker) as a one-step workflow.

A thin wrapper over the executor so the single-worker baseline shares the exact same
execution + grading + reward path as every multi-step scaffold and Conductor workflow.
"""

from __future__ import annotations

from .executor import execute_workflow, faithful_reward  # noqa: F401  (re-exported)
from .schemas import RolloutRecord, TaskSpec, Workflow, WorkflowStep
from .workers import Sampling, WorkerPool

_SOLVE = "Solve the task and give the final answer."


async def direct_rollout(
    task: TaskSpec,
    pool: WorkerPool,
    worker_id: str,
    sampling: Sampling,
    rollout_id: str,
) -> RolloutRecord:
    worker_index = pool.worker_ids.index(worker_id)
    workflow = Workflow(steps=[WorkflowStep(worker_id=worker_index, subtask=_SOLVE, access=[])])
    return await execute_workflow(task, workflow, pool, sampling, rollout_id)
