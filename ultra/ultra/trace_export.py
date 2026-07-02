"""Convert artifact-backed rollout records into canonical AgentTrace records."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import (
    AgentTrace,
    RepoStateRef,
    RolloutRecord,
    TaskSpec,
    TraceArtifacts,
    TraceEvent,
    TracePrivacy,
    TracePromptRef,
    TraceUsage,
)


def _user_task(task: TaskSpec) -> str:
    for message in reversed(task.input.messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _origin(record: RolloutRecord) -> str:
    if record.execution.steps:
        harness = record.execution.steps[-1].harness
    else:
        harness = record.harness
    if harness not in {"opencode", "claude_code", "codex"}:
        raise ValueError(f"rollout harness {harness!r} cannot be exported as a coding AgentTrace")
    return harness


def rollout_to_agent_trace(
    record: RolloutRecord,
    task: TaskSpec,
    *,
    worker_models: dict[int, str],
) -> AgentTrace:
    repo = task.input.repo
    steps = record.execution.steps
    final_step = steps[-1] if steps else None
    worker_model = worker_models.get(final_step.worker_id, "unknown") if final_step else "unknown"

    events: list[TraceEvent] = []
    for turn, step in enumerate(steps):
        if step.messages_ref:
            events.append(
                TraceEvent(
                    type="message",
                    agent_turn=turn,
                    content_ref=step.messages_ref,
                    metadata={"worker_id": step.worker_id, "harness": step.harness},
                )
            )
        if step.tool_events_ref:
            events.append(
                TraceEvent(
                    type="command",
                    agent_turn=turn,
                    content_ref=step.tool_events_ref,
                    metadata={"worker_id": step.worker_id, "harness": step.harness},
                )
            )
        if step.patch_ref:
            events.append(
                TraceEvent(
                    type="file_edit",
                    agent_turn=turn,
                    content_ref=step.patch_ref,
                    metadata={"worker_id": step.worker_id, "harness": step.harness},
                )
            )
        if step.termination != "completed":
            events.append(
                TraceEvent(
                    type="error",
                    agent_turn=turn,
                    content_ref=step.tool_events_ref,
                    metadata={"termination": step.termination, "worker_id": step.worker_id},
                )
            )
    if record.grade and record.grade.grader_ref:
        events.append(
            TraceEvent(
                type="test_result",
                agent_turn=max(0, len(steps) - 1),
                content_ref=record.grade.grader_ref,
                metadata={"success": record.grade.success, "score": record.grade.score},
            )
        )

    return AgentTrace(
        trace_id=record.rollout_id,
        origin_harness=_origin(record),  # type: ignore[arg-type]
        harness_version="ultra_repo_trace_v1",
        worker_model=worker_model,
        task_id=record.task_id,
        repo=RepoStateRef(
            url=repo.url if repo is not None else None,
            base_commit=repo.base_commit if repo is not None else None,
        ),
        prompt=TracePromptRef(user_task=_user_task(task)),
        events=events,
        artifacts=TraceArtifacts(
            final_patch_ref=final_step.patch_ref if final_step else None,
            workspace_snapshot_ref=final_step.session_ref if final_step else None,
            public_test_log_ref=final_step.tool_events_ref if final_step else None,
            hidden_grade_ref=record.grade.grader_ref if record.grade else None,
        ),
        grade=record.grade,
        usage=TraceUsage(
            input_tokens=sum(step.input_tokens for step in steps),
            output_tokens=sum(step.output_tokens for step in steps),
            cost_usd=sum(step.cost_usd for step in steps),
        ),
        privacy=TracePrivacy(redacted=True, contains_user_secret=False, license_status="ok_for_internal_training"),
    )


def write_agent_trace(
    record: RolloutRecord,
    task: TaskSpec,
    *,
    worker_models: dict[int, str],
    out: Path,
) -> AgentTrace:
    trace = rollout_to_agent_trace(record, task, worker_models=worker_models)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    return trace
