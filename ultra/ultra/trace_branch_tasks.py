"""Materialize train-ready AgentTrace checkpoints as branch-repair TaskSpecs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import AgentTrace, Grade, RolloutRecord, SourceRef, SplittingSpec, TaskSpec

TRACE_BRANCH_TASKS_VERSION = "trace_state_branch_tasks_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_tasks(path: Path) -> dict[str, TaskSpec]:
    tasks: dict[str, TaskSpec] = {}
    for row in _read_jsonl(path):
        task = TaskSpec.model_validate(row)
        tasks[task.task_id] = task
    return tasks


def _read_trace(path: Path) -> AgentTrace:
    return AgentTrace.model_validate(json.loads(path.read_text()))


def _read_rollout(path: Path) -> RolloutRecord:
    return RolloutRecord.model_validate(json.loads(path.read_text()))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _find_opencode_instance(task: TaskSpec) -> dict[str, Any]:
    for asset in task.input.assets:
        if isinstance(asset, dict) and isinstance(asset.get("opencode_instance"), dict):
            return dict(asset["opencode_instance"])
    raise ValueError(f"task {task.task_id} has no opencode_instance asset")


@dataclass(frozen=True)
class BranchState:
    source_ref: str
    final_patch_ref: str
    workspace_snapshot_ref: str | None
    hidden_grade_ref: str | None
    public_test_log_ref: str | None
    origin_harness: str
    worker_model: str
    grade: Grade | None


def _branch_state_from_trace(candidate: dict[str, Any], trace: AgentTrace) -> BranchState:
    if not trace.artifacts.final_patch_ref:
        raise ValueError(f"trace {trace.trace_id} has no final_patch_ref")
    return BranchState(
        source_ref=str(candidate["source_path"]),
        final_patch_ref=trace.artifacts.final_patch_ref,
        workspace_snapshot_ref=trace.artifacts.workspace_snapshot_ref,
        hidden_grade_ref=trace.artifacts.hidden_grade_ref,
        public_test_log_ref=trace.artifacts.public_test_log_ref,
        origin_harness=trace.origin_harness,
        worker_model=trace.worker_model,
        grade=trace.grade,
    )


def _branch_state_from_rollout(candidate: dict[str, Any], rollout: RolloutRecord) -> BranchState:
    if not rollout.execution.steps:
        raise ValueError(f"rollout {rollout.rollout_id} has no execution steps")
    patch_step = next((step for step in reversed(rollout.execution.steps) if step.patch_ref), None)
    if patch_step is None:
        raise ValueError(f"rollout {rollout.rollout_id} has no patch_ref")
    workspace_step = next((step for step in reversed(rollout.execution.steps) if step.session_ref), patch_step)
    feedback_step = next((step for step in reversed(rollout.execution.steps) if step.tool_events_ref), patch_step)
    grade_details = rollout.grade.details if rollout.grade is not None else {}
    worker_ids = ",".join(str(worker_id) for worker_id in candidate.get("worker_ids", []))
    worker_model = f"workflow_worker_ids:{worker_ids}" if worker_ids else "workflow_worker_unknown"
    return BranchState(
        source_ref=str(candidate["source_path"]),
        final_patch_ref=patch_step.patch_ref or "",
        workspace_snapshot_ref=workspace_step.session_ref,
        hidden_grade_ref=grade_details.get("hidden_grade_ref")
        or (rollout.grade.grader_ref if rollout.grade is not None else None),
        public_test_log_ref=grade_details.get("public_test_log_ref") or feedback_step.tool_events_ref,
        origin_harness=patch_step.harness,
        worker_model=worker_model,
        grade=rollout.grade,
    )


def _branch_prompt(base_problem: str, state: BranchState) -> str:
    previous_grade = "unknown"
    if state.grade is not None:
        previous_grade = "passed" if state.grade.success else "failed"
    return "\n".join(
        [
            "Continue from a captured prior attempt for this train-allowed repository task.",
            "The repository starts with the prior patch already applied.",
            "Review the current source state against the original task, keep correct parts, repair wrong or missing behavior, and do not modify tests.",
            "",
            f"Previous attempt: harness={state.origin_harness}, model={state.worker_model}, grade={previous_grade}.",
            "",
            "Original task:",
            base_problem.strip(),
        ]
    )


def _materialize_one(
    *,
    candidate: dict[str, Any],
    base: TaskSpec,
    state: BranchState,
) -> TaskSpec:
    base_data = base.model_dump(mode="json")
    instance = _find_opencode_instance(base)
    base_problem = str(instance.get("problem_statement") or "")
    branch_problem = _branch_prompt(base_problem, state)
    instance.update(
        {
            "problem_statement": branch_problem,
            "initial_patch_ref": state.final_patch_ref,
            "branch_trace_ref": state.source_ref,
            "workspace_snapshot_ref": state.workspace_snapshot_ref,
            "previous_grade_ref": state.hidden_grade_ref,
            "previous_harness": state.origin_harness,
            "previous_worker_model": state.worker_model,
        }
    )
    branch_asset = {
        "trace_branch": {
            "candidate_id": candidate["candidate_id"],
            "source_trace_ref": state.source_ref,
            "initial_patch_ref": state.final_patch_ref,
            "workspace_snapshot_ref": state.workspace_snapshot_ref,
            "previous_grade_ref": state.hidden_grade_ref,
            "public_test_log_ref": state.public_test_log_ref,
            "previous_success": state.grade.success if state.grade is not None else None,
            "origin_harness": state.origin_harness,
            "worker_model": state.worker_model,
            "source_kind": candidate.get("source_kind"),
            "state_type": candidate.get("state_type") or "trace_checkpoint",
        }
    }

    assets: list[Any] = []
    replaced = False
    for asset in base_data["input"]["assets"]:
        if isinstance(asset, dict) and isinstance(asset.get("opencode_instance"), dict):
            assets.append({"opencode_instance": instance})
            replaced = True
        else:
            assets.append(asset)
    if not replaced:
        assets.insert(0, {"opencode_instance": instance})
    assets.append(branch_asset)

    task_id = f"trace_state_branch__{candidate['candidate_id']}"
    base_data.update(
        {
            "task_id": task_id,
            "source": SourceRef(
                name="trace_state_branches",
                version="v1",
                policy="train_allowed",
                url_or_ref=str(candidate["source_path"]),
                license=base.source.license,
                source_commit=base.source.source_commit,
            ).model_dump(mode="json"),
            "input": {
                **base_data["input"],
                "messages": [{"role": "user", "content": branch_problem}],
                "assets": assets,
            },
            "splitting": SplittingSpec(
                group_id=f"trace_state_branch/{base.task_id}",
                split="grpo_train",
                contamination_group=f"trace_state_branch/{candidate['candidate_id']}",
            ).model_dump(mode="json"),
        }
    )
    metadata = dict(base_data.get("metadata") or {})
    tags = list(metadata.get("tags") or [])
    tags.extend(
        [
            "trace_state_branch",
            "after_first_patch",
            f"origin_harness:{state.origin_harness}",
            "previous_success" if state.grade and state.grade.success else "previous_failure",
        ]
    )
    metadata.update(
        {
            "domain": metadata.get("domain") or "software_engineering",
            "subdomain": "repo_repair_branch",
            "tags": sorted(set(tags)),
            "requires_tools": True,
        }
    )
    base_data["metadata"] = metadata
    return TaskSpec.model_validate(base_data)


def materialize_trace_branch_tasks(
    *,
    branch_candidates_jsonl: Path,
    base_tasks_jsonl: Path,
    out_jsonl: Path,
    report_out: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Write branch-repair TaskSpecs from train-ready AgentTrace candidates."""

    candidates = [row for row in _read_jsonl(branch_candidates_jsonl) if row.get("train_ready")]
    if limit is not None:
        candidates = candidates[:limit]
    base_tasks = _read_tasks(base_tasks_jsonl)

    tasks: list[TaskSpec] = []
    skipped: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for candidate in candidates:
        try:
            task_id = str(candidate["task_id"])
            base = base_tasks.get(task_id)
            if base is None:
                raise ValueError(f"base task not found: {task_id}")
            source_kind = candidate.get("source_kind")
            if source_kind == "agent_trace":
                state = _branch_state_from_trace(candidate, _read_trace(Path(str(candidate["source_path"]))))
            elif source_kind == "rollout_record":
                state = _branch_state_from_rollout(
                    candidate,
                    _read_rollout(Path(str(candidate["source_path"]))),
                )
            else:
                raise ValueError(f"unsupported source_kind {source_kind!r}")
            task = _materialize_one(candidate=candidate, base=base, state=state)
            tasks.append(task)
            counts[state.origin_harness] += 1
        except Exception as exc:  # noqa: BLE001 - keep materialization report useful
            skipped.append(
                {
                    "candidate_id": str(candidate.get("candidate_id")),
                    "task_id": str(candidate.get("task_id")),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    _write_jsonl(out_jsonl, [task.model_dump(mode="json") for task in tasks])
    report = {
        "version": TRACE_BRANCH_TASKS_VERSION,
        "branch_candidates_jsonl": str(branch_candidates_jsonl.resolve()),
        "base_tasks_jsonl": str(base_tasks_jsonl.resolve()),
        "out_jsonl": str(out_jsonl.resolve()),
        "candidate_count": len(candidates),
        "materialized": len(tasks),
        "skipped": len(skipped),
        "skipped_examples": skipped[:20],
        "by_origin_harness": dict(sorted(counts.items())),
        "live_calls": False,
    }
    _write_json(report_out, report)
    return report
