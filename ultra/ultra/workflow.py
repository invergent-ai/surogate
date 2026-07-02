"""Workflow validation + parsing (ultra-data §5, ultra-data2 §5).

A malformed workflow earns reward 0 (ultra-intro §6); this validator is the gate that
decides malformed-vs-valid. Backward-only access lists make cycles impossible by
construction, so there is no separate cycle check.
"""

from __future__ import annotations

import json

from .schemas import Workflow

MAX_STEPS = 5
MAX_SUBTASK_LEN = 4000


class WorkflowValidationError(ValueError):
    pass


def validate_workflow(
    workflow: Workflow,
    worker_count: int,
    *,
    max_steps: int = MAX_STEPS,
    max_subtask_len: int = MAX_SUBTASK_LEN,
) -> None:
    """Raise WorkflowValidationError if the workflow is not executable."""
    steps = workflow.steps
    if not (1 <= len(steps) <= max_steps):
        raise WorkflowValidationError(f"need 1..{max_steps} steps, got {len(steps)}")
    for i, step in enumerate(steps):
        if not (0 <= step.worker_id < worker_count):
            raise WorkflowValidationError(
                f"step {i}: worker_id {step.worker_id} out of range [0,{worker_count})"
            )
        if not step.subtask or not step.subtask.strip():
            raise WorkflowValidationError(f"step {i}: empty subtask")
        if len(step.subtask) > max_subtask_len:
            raise WorkflowValidationError(
                f"step {i}: subtask length {len(step.subtask)} > {max_subtask_len}"
            )
        if len(set(step.access)) != len(step.access):
            raise WorkflowValidationError(f"step {i}: duplicate access ids {step.access}")
        for j in step.access:
            if not (0 <= j < i):  # backward-only: a step may only read STRICTLY earlier steps
                raise WorkflowValidationError(
                    f"step {i}: access {j} is not a prior step (backward-only)"
                )


def parse_workflow(raw: str) -> Workflow:
    """Parse a Conductor's raw JSON output into a Workflow (does not semantically validate)."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        if isinstance(raw, str) and "\\'" in raw:
            try:
                data = json.loads(raw.replace("\\'", "'"))
            except (json.JSONDecodeError, TypeError):
                raise WorkflowValidationError(f"workflow JSON did not parse: {e}") from e
        else:
            raise WorkflowValidationError(f"workflow JSON did not parse: {e}") from e
    try:
        return Workflow(**data)
    except Exception as e:  # pydantic ValidationError, missing keys, wrong types, ...
        raise WorkflowValidationError(f"workflow did not match schema: {e}") from e
