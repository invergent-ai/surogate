"""Convert one audited successful Harbor trajectory into live-control labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .live_control import parse_control_action


MAX_OBSERVATION_CHARS = 12_000
EARLY_COMPLETION_AUDIT_VERSION = "fugu_live_control_early_completion_v1"


class LiveControlTrajectoryError(ValueError):
    """A Harbor trajectory cannot support trusted live-control labels."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LiveControlTrajectoryError(
                    f"invalid route JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise LiveControlTrajectoryError(
                    f"route line {line_number} is not an object"
                )
            rows.append(row)
    if not rows:
        raise LiveControlTrajectoryError("route log is empty")
    return rows


def _reward(result: dict[str, Any]) -> float | None:
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        return None
    return float(reward)


def _terminal_status(value: Any) -> str:
    if value is True:
        return "ready"
    if value is False:
        return "busy"
    return "unknown"


def _observation_text(step: dict[str, Any]) -> str:
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return "No terminal observation was recorded."
    results = observation.get("results")
    if not isinstance(results, list):
        return "No terminal observation was recorded."
    parts: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        content = result.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
    rendered = "\n".join(parts) or "No terminal output was produced."
    return rendered[-MAX_OBSERVATION_CHARS:]


def _wall_time_limit(routes: list[dict[str, Any]]) -> float:
    candidates: list[float] = []
    for route in routes:
        elapsed = route.get("budget_elapsed_s")
        remaining = route.get("budget_remaining_s")
        if (
            not isinstance(elapsed, bool)
            and isinstance(elapsed, (int, float))
            and not isinstance(remaining, bool)
            and isinstance(remaining, (int, float))
        ):
            candidates.append(float(elapsed) + float(remaining))
    return max(candidates, default=1_500.0)


def _budget(
    route: dict[str, Any] | None,
    *,
    paid_calls_used: int,
    paid_call_limit: int,
    wall_time_limit_s: float,
) -> dict[str, Any]:
    elapsed = 0.0 if route is None else route.get("budget_elapsed_s", 0.0)
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        elapsed = 0.0
    return {
        "paid_calls_used": paid_calls_used,
        "paid_call_limit": paid_call_limit,
        "elapsed_s": max(0.0, float(elapsed)),
        "wall_time_limit_s": wall_time_limit_s,
    }


def _initial_state(
    original_task: str,
    *,
    paid_call_limit: int,
    wall_time_limit_s: float,
) -> dict[str, Any]:
    return {
        "original_task": original_task,
        "workflow_id": None,
        "positions": [],
        "active_position_id": None,
        "terminal_status": "ready",
        "terminal_observation": "Initial shared terminal is ready; no worker has acted.",
        "shared_memory": [],
        "budget": _budget(
            None,
            paid_calls_used=0,
            paid_call_limit=paid_call_limit,
            wall_time_limit_s=wall_time_limit_s,
        ),
    }


def _action_after_route(
    routes: list[dict[str, Any]],
    index: int,
) -> dict[str, Any]:
    current_step = int(routes[index]["workflow_step_index"]) - 1
    if index == len(routes) - 1:
        return {
            "action": "complete",
            "reason": (
                "The final active role reports the requested implementation and verification "
                "complete with a stable terminal and no remaining command."
            ),
        }
    next_step = int(routes[index + 1]["workflow_step_index"]) - 1
    if next_step == current_step:
        return {
            "action": "continue",
            "reason": (
                "The active role still owns an unfinished function-call loop and should inspect "
                "the latest terminal result before another workflow decision."
            ),
        }
    if next_step == current_step + 1:
        return {
            "action": "handoff",
            "reason": (
                "The active role completed its assigned subtask at a stable terminal; the next "
                "registered role can now use its permitted evidence and shared workspace."
            ),
            "target_position_id": next_step,
        }
    raise LiveControlTrajectoryError(
        f"route {index} makes an invalid workflow transition {current_step} -> {next_step}"
    )


def convert_successful_harbor_trajectory(
    *,
    collection_id: str,
    task_id: str,
    original_task: str,
    pool_fingerprint: str,
    registered_action: dict[str, Any],
    result_path: Path,
    route_log_path: Path,
    trajectory_path: Path,
    paid_call_limit: int = 120,
) -> list[dict[str, Any]]:
    """Return labels only when reward, topology, models, and tool trace all agree."""
    if not collection_id.strip() or not task_id.strip() or not original_task.strip():
        raise LiveControlTrajectoryError(
            "collection_id, task_id, and original_task are required"
        )
    if not pool_fingerprint.strip():
        raise LiveControlTrajectoryError("pool_fingerprint is required")
    if paid_call_limit <= 0:
        raise LiveControlTrajectoryError("paid_call_limit must be positive")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or _reward(result) != 1.0:
        raise LiveControlTrajectoryError(
            "only audited verifier reward 1.0 trajectories are accepted"
        )

    action = parse_control_action(json.dumps(registered_action, ensure_ascii=True))
    if action.action != "replan" or not action.steps:
        raise LiveControlTrajectoryError(
            "registered collection action must be a non-empty replan"
        )

    routes = _read_jsonl(route_log_path)
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(trajectory, dict) or not isinstance(
        trajectory.get("steps"), list
    ):
        raise LiveControlTrajectoryError("trajectory must contain a steps list")
    all_agent_steps = [
        step for step in trajectory["steps"] if step.get("source") == "agent"
    ]
    route_models = {route.get("worker_model") for route in routes}
    unexpected_models = {
        step.get("model_name")
        for step in all_agent_steps
        if step.get("model_name") not in route_models
        and not str(step.get("model_name") or "").startswith("fugu-runtime-")
        and step.get("model_name") not in {"fugu-conductor-retry", "fugu-live-conductor"}
    }
    if unexpected_models:
        raise LiveControlTrajectoryError(
            f"trajectory contains unknown agent models: {sorted(unexpected_models)}"
        )
    agent_steps = [
        step for step in all_agent_steps if step.get("model_name") in route_models
    ]
    if len(agent_steps) != len(routes):
        raise LiveControlTrajectoryError(
            f"route/agent response count mismatch: {len(routes)} routes vs {len(agent_steps)} responses"
        )
    if len(routes) > paid_call_limit:
        raise LiveControlTrajectoryError(
            "trajectory exceeds the registered paid-call ceiling"
        )

    workflow_ids = {route.get("workflow_id") for route in routes}
    if len(workflow_ids) != 1 or None in workflow_ids:
        raise LiveControlTrajectoryError(
            "collection must contain exactly one concrete workflow"
        )
    workflow_id = next(iter(workflow_ids))
    visited: list[int] = []
    tool_calls_seen = 0
    for index, (route, agent_step) in enumerate(zip(routes, agent_steps, strict=True)):
        step_index = route.get("workflow_step_index")
        if isinstance(step_index, bool) or not isinstance(step_index, int):
            raise LiveControlTrajectoryError(
                f"route {index} has an invalid workflow_step_index"
            )
        position_id = step_index - 1
        if not 0 <= position_id < len(action.steps):
            raise LiveControlTrajectoryError(
                f"route {index} references an unknown workflow position"
            )
        expected = action.steps[position_id]
        if route.get("worker_id") != expected.worker_id:
            raise LiveControlTrajectoryError(
                f"route {index} worker differs from registration"
            )
        if route.get("subtask") != expected.subtask:
            raise LiveControlTrajectoryError(
                f"route {index} subtask differs from registration"
            )
        if tuple(route.get("workflow_access") or ()) != expected.access:
            raise LiveControlTrajectoryError(
                f"route {index} access list differs from registration"
            )
        if agent_step.get("model_name") != route.get("worker_model"):
            raise LiveControlTrajectoryError(
                f"route {index} model differs from trajectory response"
            )
        tool_calls = agent_step.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []
        if not isinstance(tool_calls, list):
            raise LiveControlTrajectoryError(
                f"agent response {index} has invalid tool_calls"
            )
        tool_calls_seen += len(tool_calls)
        if not visited or visited[-1] != position_id:
            visited.append(position_id)

    if visited != list(range(len(action.steps))):
        raise LiveControlTrajectoryError(
            f"trajectory did not execute the complete registered topology: visited {visited}"
        )
    if tool_calls_seen <= 0:
        raise LiveControlTrajectoryError("trajectory contains no tool calls")

    wall_time_limit_s = _wall_time_limit(routes)
    provenance = {
        "collection_id": collection_id,
        "result_sha256": _sha256(result_path),
        "route_log_sha256": _sha256(route_log_path),
        "trajectory_sha256": _sha256(trajectory_path),
    }
    common = {
        "task_id": task_id,
        "pool_fingerprint": pool_fingerprint,
        "terminalbench": False,
        "label_status": "valid_verifier_pass",
        "provenance": provenance,
    }
    rows: list[dict[str, Any]] = [
        {
            **common,
            "record_id": f"{collection_id}__control_000",
            "agentic_evidence": {
                "tool_calls_observed": 0,
                "shared_workspace": True,
                "verifier_audited": True,
            },
            "state": _initial_state(
                original_task,
                paid_call_limit=paid_call_limit,
                wall_time_limit_s=wall_time_limit_s,
            ),
            "action": registered_action,
        }
    ]

    progress: list[Any] = [None for _ in action.steps]
    artifacts: list[list[Any]] = [[] for _ in action.steps]
    cumulative_tool_calls = 0
    for index, (route, agent_step) in enumerate(
        zip(routes, agent_steps, strict=True), start=1
    ):
        position_id = int(route["workflow_step_index"]) - 1
        if route.get("reported_progress") is not None:
            progress[position_id] = route["reported_progress"]
        reported_artifacts = route.get("reported_artifacts")
        if reported_artifacts is not None:
            if not isinstance(reported_artifacts, list):
                raise LiveControlTrajectoryError(
                    f"route {index - 1} reported_artifacts is not a list"
                )
            artifacts[position_id] = reported_artifacts
        tool_calls = agent_step.get("tool_calls") or []
        cumulative_tool_calls += len(tool_calls)
        terminal_status = _terminal_status(route.get("terminal_ready"))
        control_action = _action_after_route(routes, index - 1)
        if (
            control_action["action"] in {"handoff", "complete"}
            and terminal_status != "ready"
        ):
            raise LiveControlTrajectoryError(
                f"route {index - 1} labels {control_action['action']} without a stable terminal"
            )
        positions = []
        for step_id, step in enumerate(action.steps):
            status = (
                "completed"
                if step_id < position_id
                else "active" if step_id == position_id else "pending"
            )
            progress_value = progress[step_id]
            if step_id == position_id and control_action["action"] in {
                "handoff",
                "complete",
            }:
                # The boundary exists because the active worker claimed its
                # subtask complete; the live contract exposes that claim as
                # progress evidence before the conductor may leave the position.
                base_progress = (
                    dict(progress_value) if isinstance(progress_value, dict) else {}
                )
                base_progress["completion_requested"] = True
                progress_value = base_progress
            positions.append(
                {
                    "position_id": step_id,
                    "worker_id": step.worker_id,
                    "subtask": step.subtask,
                    "access": list(step.access),
                    "status": status,
                    "progress": progress_value,
                    "artifacts": artifacts[step_id],
                }
            )
        rows.append(
            {
                **common,
                "record_id": f"{collection_id}__control_{index:03d}",
                "agentic_evidence": {
                    "tool_calls_observed": cumulative_tool_calls,
                    "shared_workspace": True,
                    "verifier_audited": True,
                },
                "state": {
                    "original_task": original_task,
                    "workflow_id": workflow_id,
                    "positions": positions,
                    "active_position_id": position_id,
                    "terminal_status": terminal_status,
                    "terminal_observation": _observation_text(agent_step),
                    "shared_memory": [],
                    "budget": _budget(
                        route,
                        paid_calls_used=index,
                        paid_call_limit=paid_call_limit,
                        wall_time_limit_s=wall_time_limit_s,
                    ),
                },
                "action": control_action,
            }
        )
    return rows


def convert_audited_early_completion_boundary(
    *,
    collection_id: str,
    task_id: str,
    original_task: str,
    pool_fingerprint: str,
    registered_plan: dict[str, Any],
    result_path: Path,
    route_log_path: Path,
    trajectory_path: Path,
    boundary_audit_path: Path,
    paid_call_limit: int = 120,
) -> list[dict[str, Any]]:
    """Convert a verifier-passing prefix whose later work was audited as redundant."""
    if not collection_id.strip() or not task_id.strip() or not original_task.strip():
        raise LiveControlTrajectoryError(
            "collection_id, task_id, and original_task are required"
        )
    if not pool_fingerprint.strip() or paid_call_limit <= 0:
        raise LiveControlTrajectoryError(
            "pool fingerprint and positive paid-call limit are required"
        )

    audit = json.loads(boundary_audit_path.read_text(encoding="utf-8"))
    required_audit_fields = {
        "version",
        "collection_id",
        "task_id",
        "result_sha256",
        "route_log_sha256",
        "trajectory_sha256",
        "boundary_route_index",
        "expected_paid_calls_used",
        "expected_worker_id",
        "expected_worker_model",
        "expected_workflow_step_index",
        "expected_progress_phase",
        "expected_terminal_ready",
        "required_boundary_evidence",
        "excluded_route_count",
        "post_boundary_tool_call_count",
        "post_boundary_tool_calls_sha256",
        "post_boundary_command_scope",
        "post_boundary_production_changes",
        "verifier_reward",
        "training_action",
        "audit_basis",
    }
    if not isinstance(audit, dict) or set(audit) != required_audit_fields:
        raise LiveControlTrajectoryError("early-completion audit has an invalid schema")
    if audit["version"] != EARLY_COMPLETION_AUDIT_VERSION:
        raise LiveControlTrajectoryError("early-completion audit version differs")
    if audit["collection_id"] != collection_id or audit["task_id"] != task_id:
        raise LiveControlTrajectoryError("early-completion audit identity differs")
    artifact_hashes = {
        "result_sha256": _sha256(result_path),
        "route_log_sha256": _sha256(route_log_path),
        "trajectory_sha256": _sha256(trajectory_path),
    }
    if any(audit[key] != value for key, value in artifact_hashes.items()):
        raise LiveControlTrajectoryError("early-completion audit artifact hash differs")
    if audit["verifier_reward"] != 1.0:
        raise LiveControlTrajectoryError(
            "early-completion audit must require reward 1.0"
        )
    if audit["post_boundary_command_scope"] != "read_only_repository_or_tmp_only":
        raise LiveControlTrajectoryError(
            "post-boundary command scope was not explicitly audited"
        )
    if audit["post_boundary_production_changes"] is not False:
        raise LiveControlTrajectoryError(
            "post-boundary production changes prevent an early-completion label"
        )
    audit_basis = audit["audit_basis"]
    if (
        not isinstance(audit_basis, list)
        or not audit_basis
        or any(not isinstance(item, str) or not item.strip() for item in audit_basis)
    ):
        raise LiveControlTrajectoryError("early-completion audit basis is empty")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or _reward(result) != audit["verifier_reward"]:
        raise LiveControlTrajectoryError(
            "early-completion source does not have the audited verifier reward"
        )
    metadata = (result.get("agent_result") or {}).get("metadata") or {}
    if not isinstance(metadata, dict):
        raise LiveControlTrajectoryError("result has no agent metadata")
    if metadata.get("pool_fingerprint") != pool_fingerprint:
        raise LiveControlTrajectoryError(
            "result pool fingerprint differs from registration"
        )
    if metadata.get("collection_registered_workflow") != registered_plan:
        raise LiveControlTrajectoryError(
            "result registered plan attestation differs from registration"
        )
    if (
        metadata.get("provider_owner_retry_limit") != 0
        or metadata.get("provider_owner_retries") != 0
        or metadata.get("provider_request_retries") != 0
    ):
        raise LiveControlTrajectoryError(
            "early-completion source used a provider retry"
        )

    if not isinstance(registered_plan, dict) or set(registered_plan) != {
        "primary",
        "recoveries",
    }:
        raise LiveControlTrajectoryError(
            "registered recovery plan must contain primary and recoveries"
        )
    primary = parse_control_action(
        json.dumps(registered_plan["primary"], ensure_ascii=True)
    )
    training_action_raw = audit["training_action"]
    training_action = parse_control_action(
        json.dumps(training_action_raw, ensure_ascii=True)
    )
    if training_action.action != "replan" or not training_action.steps:
        raise LiveControlTrajectoryError(
            "early-completion training action must be a non-empty replan"
        )
    if len(training_action.steps) >= len(primary.steps):
        raise LiveControlTrajectoryError(
            "early-completion training action must omit a redundant registered role"
        )
    if training_action.steps != primary.steps[: len(training_action.steps)]:
        raise LiveControlTrajectoryError(
            "early-completion training action differs from the registered prefix"
        )

    routes = _read_jsonl(route_log_path)
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(trajectory, dict) or not isinstance(
        trajectory.get("steps"), list
    ):
        raise LiveControlTrajectoryError("trajectory must contain a steps list")
    route_models = {route.get("worker_model") for route in routes}
    all_agent_steps = [
        step for step in trajectory["steps"] if step.get("source") == "agent"
    ]
    unexpected_models = {
        step.get("model_name")
        for step in all_agent_steps
        if step.get("model_name") not in route_models
        and not str(step.get("model_name") or "").startswith("fugu-runtime-")
        and step.get("model_name") not in {"fugu-conductor-retry", "fugu-live-conductor"}
    }
    if unexpected_models:
        raise LiveControlTrajectoryError(
            f"trajectory contains unknown agent models: {sorted(unexpected_models)}"
        )
    agent_steps = [
        step for step in all_agent_steps if step.get("model_name") in route_models
    ]
    if len(agent_steps) != len(routes):
        raise LiveControlTrajectoryError(
            f"route/agent response count mismatch: {len(routes)} routes vs {len(agent_steps)} responses"
        )

    boundary_index = audit["boundary_route_index"]
    if (
        isinstance(boundary_index, bool)
        or not isinstance(boundary_index, int)
        or not 0 <= boundary_index < len(routes) - 1
    ):
        raise LiveControlTrajectoryError("early-completion boundary index is invalid")
    boundary = routes[boundary_index]
    expected_boundary = {
        "paid_call_attempt": audit["expected_paid_calls_used"],
        "worker_id": audit["expected_worker_id"],
        "worker_model": audit["expected_worker_model"],
        "workflow_step_index": audit["expected_workflow_step_index"],
        "terminal_ready": audit["expected_terminal_ready"],
    }
    if any(boundary.get(key) != value for key, value in expected_boundary.items()):
        raise LiveControlTrajectoryError(
            "runtime route differs from the audited completion boundary"
        )
    progress = boundary.get("reported_progress")
    if (
        not isinstance(progress, dict)
        or progress.get("phase") != audit["expected_progress_phase"]
    ):
        raise LiveControlTrajectoryError(
            "completion boundary lacks the audited done progress"
        )
    evidence = str(progress.get("evidence") or "")
    required_evidence = audit["required_boundary_evidence"]
    if (
        not isinstance(required_evidence, list)
        or not required_evidence
        or any(
            not isinstance(fragment, str) or fragment not in evidence
            for fragment in required_evidence
        )
    ):
        raise LiveControlTrajectoryError(
            "completion boundary lacks required verification evidence"
        )
    if audit["expected_terminal_ready"] is not True:
        raise LiveControlTrajectoryError(
            "early-completion boundary must require a stable terminal"
        )
    if boundary["workflow_step_index"] != len(training_action.steps):
        raise LiveControlTrajectoryError(
            "completion boundary is not the final training-action role"
        )
    next_step_index = routes[boundary_index + 1].get("workflow_step_index")
    if next_step_index != boundary["workflow_step_index"] + 1:
        raise LiveControlTrajectoryError(
            "excluded suffix does not begin with the redundant next role"
        )
    excluded_routes = routes[boundary_index + 1 :]
    if len(excluded_routes) != audit["excluded_route_count"]:
        raise LiveControlTrajectoryError("excluded route count differs from audit")
    post_boundary_tool_calls = [
        call
        for step in agent_steps[boundary_index + 1 :]
        for call in (step.get("tool_calls") or [])
    ]
    if len(post_boundary_tool_calls) != audit["post_boundary_tool_call_count"]:
        raise LiveControlTrajectoryError(
            "post-boundary tool-call count differs from audit"
        )
    if (
        _canonical_sha256(post_boundary_tool_calls)
        != audit["post_boundary_tool_calls_sha256"]
    ):
        raise LiveControlTrajectoryError(
            "post-boundary tool-call digest differs from audit"
        )
    failures = metadata.get("provider_failure_events") or []
    if not isinstance(failures, list) or any(
        not isinstance(event, dict) for event in failures
    ):
        raise LiveControlTrajectoryError("provider failure evidence is invalid")
    boundary_calls = audit["expected_paid_calls_used"]
    if any(
        int(event.get("paid_call_attempt", 0)) <= boundary_calls for event in failures
    ):
        raise LiveControlTrajectoryError(
            "provider failure occurred before the audited completion boundary"
        )

    prefix_routes = routes[: boundary_index + 1]
    prefix_agent_steps = agent_steps[: boundary_index + 1]
    if len(prefix_routes) > paid_call_limit:
        raise LiveControlTrajectoryError(
            "early-completion prefix exceeds the paid-call ceiling"
        )
    workflow_ids = {route.get("workflow_id") for route in prefix_routes}
    if len(workflow_ids) != 1 or None in workflow_ids:
        raise LiveControlTrajectoryError(
            "early-completion prefix must contain one concrete workflow"
        )
    workflow_id = next(iter(workflow_ids))
    visited: list[int] = []
    tool_calls_seen = 0
    for index, (route, agent_step) in enumerate(
        zip(prefix_routes, prefix_agent_steps, strict=True)
    ):
        step_index = route.get("workflow_step_index")
        if isinstance(step_index, bool) or not isinstance(step_index, int):
            raise LiveControlTrajectoryError(
                f"route {index} has an invalid workflow_step_index"
            )
        position_id = step_index - 1
        if not 0 <= position_id < len(training_action.steps):
            raise LiveControlTrajectoryError(
                f"route {index} references an excluded workflow position"
            )
        expected = training_action.steps[position_id]
        if (
            route.get("worker_id") != expected.worker_id
            or route.get("subtask") != expected.subtask
            or tuple(route.get("workflow_access") or ()) != expected.access
            or agent_step.get("model_name") != route.get("worker_model")
        ):
            raise LiveControlTrajectoryError(
                f"route {index} differs from the audited training action"
            )
        tool_calls = agent_step.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise LiveControlTrajectoryError(
                f"agent response {index} has invalid tool_calls"
            )
        tool_calls_seen += len(tool_calls)
        if not visited or visited[-1] != position_id:
            visited.append(position_id)
    if visited != list(range(len(training_action.steps))):
        raise LiveControlTrajectoryError(
            f"early-completion prefix did not execute its topology: visited {visited}"
        )
    if tool_calls_seen <= 0:
        raise LiveControlTrajectoryError(
            "early-completion prefix contains no tool calls"
        )

    wall_time_limit_s = _wall_time_limit(routes)
    provenance = {
        "collection_id": collection_id,
        "result_sha256": artifact_hashes["result_sha256"],
        "route_log_sha256": artifact_hashes["route_log_sha256"],
        "trajectory_sha256": artifact_hashes["trajectory_sha256"],
        "boundary_audit_sha256": _sha256(boundary_audit_path),
        "boundary_route_index": boundary_index,
        "excluded_routes": len(excluded_routes),
    }
    common = {
        "task_id": task_id,
        "pool_fingerprint": pool_fingerprint,
        "terminalbench": False,
        "label_status": "audited_early_completion",
        "provenance": provenance,
    }
    rows: list[dict[str, Any]] = [
        {
            **common,
            "record_id": f"{collection_id}__early_control_000",
            "agentic_evidence": {
                "tool_calls_observed": 0,
                "shared_workspace": True,
                "verifier_audited": True,
            },
            "state": _initial_state(
                original_task,
                paid_call_limit=paid_call_limit,
                wall_time_limit_s=wall_time_limit_s,
            ),
            "action": training_action_raw,
        }
    ]
    position_progress: list[Any] = [None for _ in training_action.steps]
    artifacts: list[list[Any]] = [[] for _ in training_action.steps]
    cumulative_tool_calls = 0
    for index, (route, agent_step) in enumerate(
        zip(prefix_routes, prefix_agent_steps, strict=True), start=1
    ):
        position_id = int(route["workflow_step_index"]) - 1
        if route.get("reported_progress") is not None:
            position_progress[position_id] = route["reported_progress"]
        reported_artifacts = route.get("reported_artifacts")
        if reported_artifacts is not None:
            if not isinstance(reported_artifacts, list):
                raise LiveControlTrajectoryError(
                    f"route {index - 1} reported_artifacts is not a list"
                )
            artifacts[position_id] = reported_artifacts
        cumulative_tool_calls += len(agent_step.get("tool_calls") or [])
        terminal_status = _terminal_status(route.get("terminal_ready"))
        control_action = _action_after_route(prefix_routes, index - 1)
        if (
            control_action["action"] in {"handoff", "complete"}
            and terminal_status != "ready"
        ):
            raise LiveControlTrajectoryError(
                f"route {index - 1} labels {control_action['action']} without a stable terminal"
            )
        positions = []
        for step_id, step in enumerate(training_action.steps):
            positions.append(
                {
                    "position_id": step_id,
                    "worker_id": step.worker_id,
                    "subtask": step.subtask,
                    "access": list(step.access),
                    "status": (
                        "completed"
                        if step_id < position_id
                        else "active" if step_id == position_id else "pending"
                    ),
                    "progress": position_progress[step_id],
                    "artifacts": artifacts[step_id],
                }
            )
        rows.append(
            {
                **common,
                "record_id": f"{collection_id}__early_control_{index:03d}",
                "agentic_evidence": {
                    "tool_calls_observed": cumulative_tool_calls,
                    "shared_workspace": True,
                    "verifier_audited": True,
                },
                "state": {
                    "original_task": original_task,
                    "workflow_id": workflow_id,
                    "positions": positions,
                    "active_position_id": position_id,
                    "terminal_status": terminal_status,
                    "terminal_observation": _observation_text(agent_step),
                    "shared_memory": [],
                    "budget": _budget(
                        route,
                        paid_calls_used=index,
                        paid_call_limit=paid_call_limit,
                        wall_time_limit_s=wall_time_limit_s,
                    ),
                },
                "action": control_action,
            }
        )
    return rows


def _registered_recovery_plans(
    registered_plan: dict[str, Any],
) -> tuple[
    dict[str, tuple[Any, dict[str, Any], frozenset[int]]],
    dict[frozenset[int], str],
]:
    if not isinstance(registered_plan, dict) or set(registered_plan) != {
        "primary",
        "recoveries",
    }:
        raise LiveControlTrajectoryError(
            "registered recovery plan must contain primary and recoveries"
        )
    primary_raw = registered_plan["primary"]
    primary = parse_control_action(json.dumps(primary_raw, ensure_ascii=True))
    if primary.action != "replan" or not primary.steps:
        raise LiveControlTrajectoryError(
            "registered primary must be a non-empty replan"
        )
    raw_recoveries = registered_plan["recoveries"]
    if not isinstance(raw_recoveries, list) or not raw_recoveries:
        raise LiveControlTrajectoryError(
            "registered recoveries must be a non-empty list"
        )

    plans: dict[str, tuple[Any, dict[str, Any], frozenset[int]]] = {
        "primary": (primary, primary_raw, frozenset())
    }
    recovery_by_unavailable: dict[frozenset[int], str] = {}
    for index, raw in enumerate(raw_recoveries):
        if not isinstance(raw, dict) or set(raw) != {
            "recovery_id",
            "unavailable_worker_ids",
            "action",
        }:
            raise LiveControlTrajectoryError(
                f"registered recoveries[{index}] has an invalid schema"
            )
        recovery_id = raw["recovery_id"]
        unavailable = raw["unavailable_worker_ids"]
        if not isinstance(recovery_id, str) or not recovery_id.strip():
            raise LiveControlTrajectoryError(
                f"registered recoveries[{index}] has no recovery_id"
            )
        if recovery_id in plans:
            raise LiveControlTrajectoryError("registered recovery IDs must be unique")
        if (
            not isinstance(unavailable, list)
            or not unavailable
            or any(
                isinstance(worker_id, bool) or not isinstance(worker_id, int)
                for worker_id in unavailable
            )
        ):
            raise LiveControlTrajectoryError(
                f"registered recoveries[{index}] has invalid unavailable worker IDs"
            )
        unavailable_set = frozenset(unavailable)
        if len(unavailable_set) != len(unavailable):
            raise LiveControlTrajectoryError(
                "registered unavailable worker IDs must be unique"
            )
        if unavailable_set in recovery_by_unavailable:
            raise LiveControlTrajectoryError(
                "only one recovery may match an unavailable-worker state"
            )
        action_raw = raw["action"]
        action = parse_control_action(json.dumps(action_raw, ensure_ascii=True))
        if action.action != "replan" or not action.steps:
            raise LiveControlTrajectoryError(
                f"registered recovery {recovery_id} must be a non-empty replan"
            )
        if any(step.worker_id in unavailable_set for step in action.steps):
            raise LiveControlTrajectoryError(
                f"registered recovery {recovery_id} selects an unavailable worker"
            )
        plans[recovery_id] = (action, action_raw, unavailable_set)
        recovery_by_unavailable[unavailable_set] = recovery_id
    return plans, recovery_by_unavailable


def _raw_registered_plan_id(
    raw_plan: Any,
    plans: dict[str, tuple[Any, dict[str, Any], frozenset[int]]],
) -> str:
    if not isinstance(raw_plan, str) or not raw_plan:
        raise LiveControlTrajectoryError(
            "runtime route is missing its registered raw plan"
        )
    try:
        raw = json.loads(raw_plan)
    except json.JSONDecodeError as exc:
        raise LiveControlTrajectoryError(
            f"runtime raw plan is invalid JSON: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise LiveControlTrajectoryError("runtime raw plan must be an object")
    plan_id = raw.get("plan_id")
    if plan_id not in plans:
        raise LiveControlTrajectoryError(
            f"runtime selected unregistered plan {plan_id!r}"
        )
    action, _, unavailable = plans[plan_id]
    expected_steps = [
        {
            "worker_id": step.worker_id,
            "subtask": step.subtask,
            "access": list(step.access),
        }
        for step in action.steps
    ]
    if raw.get("reason") != action.reason or raw.get("steps") != expected_steps:
        raise LiveControlTrajectoryError(
            f"runtime plan {plan_id} differs from its registration"
        )
    raw_unavailable = raw.get("unavailable_worker_ids")
    if not isinstance(raw_unavailable, list) or raw_unavailable != sorted(unavailable):
        raise LiveControlTrajectoryError(
            f"runtime plan {plan_id} has the wrong unavailable-worker state"
        )
    return str(plan_id)


def _positions_after_route(
    action: Any,
    *,
    position_id: int,
    progress: list[Any],
    artifacts: list[list[Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "position_id": step_id,
            "worker_id": step.worker_id,
            "subtask": step.subtask,
            "access": list(step.access),
            "status": (
                "completed"
                if step_id < position_id
                else "active" if step_id == position_id else "pending"
            ),
            "progress": progress[step_id],
            "artifacts": artifacts[step_id],
        }
        for step_id, step in enumerate(action.steps)
    ]


def convert_successful_harbor_recovery_trajectory(
    *,
    collection_id: str,
    task_id: str,
    original_task: str,
    pool_fingerprint: str,
    registered_plan: dict[str, Any],
    result_path: Path,
    route_log_path: Path,
    trajectory_path: Path,
    paid_call_limit: int = 120,
) -> list[dict[str, Any]]:
    """Convert a verifier pass containing registered provider-failure replans."""
    if not collection_id.strip() or not task_id.strip() or not original_task.strip():
        raise LiveControlTrajectoryError(
            "collection_id, task_id, and original_task are required"
        )
    if not pool_fingerprint.strip() or paid_call_limit <= 0:
        raise LiveControlTrajectoryError(
            "pool fingerprint and positive paid-call limit are required"
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or _reward(result) != 1.0:
        raise LiveControlTrajectoryError(
            "only audited verifier reward 1.0 trajectories are accepted"
        )
    metadata = (result.get("agent_result") or {}).get("metadata") or {}
    if not isinstance(metadata, dict):
        raise LiveControlTrajectoryError("result has no agent metadata")
    if metadata.get("pool_fingerprint") != pool_fingerprint:
        raise LiveControlTrajectoryError(
            "result pool fingerprint differs from registration"
        )
    if metadata.get("collection_registered_workflow") != registered_plan:
        raise LiveControlTrajectoryError(
            "result registered plan attestation differs from registration"
        )
    if (
        metadata.get("provider_owner_retry_limit") != 0
        or metadata.get("provider_owner_retries") != 0
    ):
        raise LiveControlTrajectoryError("recovery trajectory used a provider retry")
    if metadata.get("provider_request_retries") != 0:
        raise LiveControlTrajectoryError(
            "recovery trajectory lacks zero request-retry attestation"
        )

    plans, recovery_by_unavailable = _registered_recovery_plans(registered_plan)
    routes = _read_jsonl(route_log_path)
    failures = metadata.get("provider_failure_events") or []
    if not isinstance(failures, list) or any(
        not isinstance(event, dict) for event in failures
    ):
        raise LiveControlTrajectoryError(
            "provider failure evidence has an invalid schema"
        )
    if len(routes) + len(failures) > paid_call_limit:
        raise LiveControlTrajectoryError(
            "trajectory exceeds the registered paid-call ceiling"
        )

    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(trajectory, dict) or not isinstance(
        trajectory.get("steps"), list
    ):
        raise LiveControlTrajectoryError("trajectory must contain a steps list")
    route_models = {route.get("worker_model") for route in routes}
    all_agent_steps = [
        step for step in trajectory["steps"] if step.get("source") == "agent"
    ]
    local_steps = [
        step
        for step in all_agent_steps
        if str(step.get("model_name") or "").startswith("fugu-runtime-")
        or step.get("model_name") == "fugu-conductor-retry"
    ]
    unexpected_models = {
        step.get("model_name")
        for step in all_agent_steps
        if step.get("model_name") not in route_models and step not in local_steps
    }
    if unexpected_models:
        raise LiveControlTrajectoryError(
            f"trajectory contains unknown agent models: {sorted(unexpected_models)}"
        )
    retry_steps = [
        step for step in local_steps if step.get("model_name") == "fugu-conductor-retry"
    ]
    if len(retry_steps) != len(failures):
        raise LiveControlTrajectoryError(
            "provider failures do not match local conductor replan boundaries"
        )
    agent_steps = [
        step for step in all_agent_steps if step.get("model_name") in route_models
    ]
    if len(agent_steps) != len(routes):
        raise LiveControlTrajectoryError(
            f"route/agent response count mismatch: {len(routes)} routes vs {len(agent_steps)} responses"
        )

    worker_models = metadata.get("worker_models")
    if not isinstance(worker_models, list) or not all(
        isinstance(model, str) for model in worker_models
    ):
        raise LiveControlTrajectoryError("result has no bound worker-model list")
    timeline: list[dict[str, Any]] = []
    workflow_plans: dict[int, str] = {}
    tool_calls_seen = 0
    for route_index, (route, agent_step) in enumerate(
        zip(routes, agent_steps, strict=True)
    ):
        attempt = route.get("paid_call_attempt")
        workflow_id = route.get("workflow_id")
        step_index = route.get("workflow_step_index")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (attempt, workflow_id, step_index)
        ):
            raise LiveControlTrajectoryError(
                f"route {route_index} has invalid numeric identity"
            )
        plan_id = _raw_registered_plan_id(route.get("raw_plan"), plans)
        previous_plan = workflow_plans.setdefault(workflow_id, plan_id)
        if previous_plan != plan_id:
            raise LiveControlTrajectoryError(
                "one workflow used multiple registered plans"
            )
        action = plans[plan_id][0]
        position_id = step_index - 1
        if not 0 <= position_id < len(action.steps):
            raise LiveControlTrajectoryError(
                f"route {route_index} references an unknown position"
            )
        expected = action.steps[position_id]
        if route.get("workflow_step_count") != len(action.steps):
            raise LiveControlTrajectoryError(
                f"route {route_index} step count differs from registration"
            )
        if (
            route.get("worker_id") != expected.worker_id
            or route.get("subtask") != expected.subtask
            or tuple(route.get("workflow_access") or ()) != expected.access
        ):
            raise LiveControlTrajectoryError(
                f"route {route_index} differs from registered topology"
            )
        if (
            expected.worker_id >= len(worker_models)
            or route.get("worker_model") != worker_models[expected.worker_id]
        ):
            raise LiveControlTrajectoryError(
                f"route {route_index} model differs from pool binding"
            )
        if agent_step.get("model_name") != route.get("worker_model"):
            raise LiveControlTrajectoryError(
                f"route {route_index} model differs from trajectory response"
            )
        tool_calls = agent_step.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise LiveControlTrajectoryError(
                f"agent response {route_index} has invalid tool calls"
            )
        tool_calls_seen += len(tool_calls)
        timeline.append(
            {
                "kind": "route",
                "attempt": attempt,
                "workflow_id": workflow_id,
                "position_id": position_id,
                "plan_id": plan_id,
                "route": route,
                "agent_step": agent_step,
            }
        )

    for failure_index, event in enumerate(failures):
        attempt = event.get("paid_call_attempt")
        workflow_id = event.get("workflow_id")
        step_index = event.get("workflow_step_index")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (attempt, workflow_id, step_index)
        ):
            raise LiveControlTrajectoryError(
                f"provider failure {failure_index} has invalid identity"
            )
        plan_id = _raw_registered_plan_id(event.get("raw_plan"), plans)
        previous_plan = workflow_plans.setdefault(workflow_id, plan_id)
        if previous_plan != plan_id:
            raise LiveControlTrajectoryError(
                "provider failure plan differs from its workflow"
            )
        action = plans[plan_id][0]
        position_id = step_index - 1
        if not 0 <= position_id < len(action.steps):
            raise LiveControlTrajectoryError(
                f"provider failure {failure_index} references an unknown position"
            )
        expected = action.steps[position_id]
        unavailable = event.get("unavailable_worker_ids")
        if not isinstance(unavailable, list) or unavailable != sorted(set(unavailable)):
            raise LiveControlTrajectoryError(
                f"provider failure {failure_index} has invalid unavailable-worker evidence"
            )
        if (
            event.get("worker_id") != expected.worker_id
            or expected.worker_id not in unavailable
        ):
            raise LiveControlTrajectoryError(
                f"provider failure {failure_index} does not identify its failed active worker"
            )
        if (
            expected.worker_id >= len(worker_models)
            or event.get("worker_model") != worker_models[expected.worker_id]
        ):
            raise LiveControlTrajectoryError(
                f"provider failure {failure_index} model differs from pool binding"
            )
        archived = event.get("archived_workflow")
        if not isinstance(archived, dict) or archived.get("workflow_id") != workflow_id:
            raise LiveControlTrajectoryError(
                f"provider failure {failure_index} has no matching archived workflow"
            )
        if event.get("terminal_ready") is not True:
            raise LiveControlTrajectoryError(
                f"provider failure {failure_index} cannot support a stable-terminal replan"
            )
        timeline.append(
            {
                "kind": "failure",
                "attempt": attempt,
                "workflow_id": workflow_id,
                "position_id": position_id,
                "plan_id": plan_id,
                "event": event,
                "unavailable": frozenset(unavailable),
            }
        )

    timeline.sort(key=lambda item: item["attempt"])
    attempts = [item["attempt"] for item in timeline]
    paid_attempts = metadata.get("paid_worker_call_attempts")
    if (
        isinstance(paid_attempts, bool)
        or not isinstance(paid_attempts, int)
        or attempts != list(range(1, paid_attempts + 1))
    ):
        raise LiveControlTrajectoryError(
            "paid-call evidence is not contiguous and complete"
        )
    if tool_calls_seen <= 0:
        raise LiveControlTrajectoryError("trajectory contains no tool calls")

    selected = metadata.get("registered_recovery_selection_history")
    if not isinstance(selected, list) or any(
        not isinstance(row, dict) or row.get("outcome") != "selected"
        for row in selected
    ):
        raise LiveControlTrajectoryError(
            "registered recovery selection history is incomplete"
        )
    first_by_workflow: dict[int, dict[str, Any]] = {}
    for item in timeline:
        first_by_workflow.setdefault(item["workflow_id"], item)
    workflow_order = sorted(
        first_by_workflow,
        key=lambda workflow_id: first_by_workflow[workflow_id]["attempt"],
    )
    selected_ids = [row.get("plan_id") for row in selected]
    if selected_ids != [workflow_plans[workflow_id] for workflow_id in workflow_order]:
        raise LiveControlTrajectoryError(
            "runtime workflow order differs from registered selections"
        )
    if not selected_ids or selected_ids[0] != "primary":
        raise LiveControlTrajectoryError(
            "runtime did not begin with the registered primary"
        )

    routes_by_workflow: dict[int, list[dict[str, Any]]] = {}
    failures_by_workflow: dict[int, list[dict[str, Any]]] = {}
    for item in timeline:
        target = routes_by_workflow if item["kind"] == "route" else failures_by_workflow
        target.setdefault(item["workflow_id"], []).append(item)
    for order_index, workflow_id in enumerate(workflow_order):
        action = plans[workflow_plans[workflow_id]][0]
        workflow_routes = routes_by_workflow.get(workflow_id, [])
        visited: list[int] = []
        for item in workflow_routes:
            if not visited or visited[-1] != item["position_id"]:
                visited.append(item["position_id"])
        workflow_failures = failures_by_workflow.get(workflow_id, [])
        if order_index == len(workflow_order) - 1:
            if workflow_failures or visited != list(range(len(action.steps))):
                raise LiveControlTrajectoryError(
                    "final workflow did not complete its registered topology"
                )
        else:
            if len(workflow_failures) != 1:
                raise LiveControlTrajectoryError(
                    "each interrupted workflow must have exactly one provider failure"
                )
            failure_position = workflow_failures[0]["position_id"]
            if visited != list(range(len(visited))) or failure_position not in {
                len(visited) - 1,
                len(visited),
            }:
                raise LiveControlTrajectoryError(
                    "interrupted workflow is not a valid registered prefix"
                )

    for index, item in enumerate(timeline):
        if item["kind"] != "failure":
            continue
        if index + 1 >= len(timeline) or timeline[index + 1]["kind"] != "route":
            raise LiveControlTrajectoryError(
                "provider failure was not followed by recovery"
            )
        next_item = timeline[index + 1]
        if next_item["workflow_id"] == item["workflow_id"]:
            raise LiveControlTrajectoryError(
                "provider failure recovery did not start a new workflow"
            )
        expected_recovery_id = recovery_by_unavailable.get(item["unavailable"])
        if expected_recovery_id is None or next_item["plan_id"] != expected_recovery_id:
            raise LiveControlTrajectoryError(
                "provider failure did not select its exact registered recovery"
            )

    wall_time_limit_s = _wall_time_limit(routes)
    provenance = {
        "collection_id": collection_id,
        "result_sha256": _sha256(result_path),
        "route_log_sha256": _sha256(route_log_path),
        "trajectory_sha256": _sha256(trajectory_path),
        "provider_failure_events": len(failures),
    }
    common = {
        "task_id": task_id,
        "pool_fingerprint": pool_fingerprint,
        "terminalbench": False,
        "label_status": "valid_verifier_pass",
        "provenance": provenance,
    }
    primary_raw = plans["primary"][1]
    rows: list[dict[str, Any]] = [
        {
            **common,
            "record_id": f"{collection_id}__control_000",
            "agentic_evidence": {
                "tool_calls_observed": 0,
                "shared_workspace": True,
                "verifier_audited": True,
            },
            "state": {
                **_initial_state(
                    original_task,
                    paid_call_limit=paid_call_limit,
                    wall_time_limit_s=wall_time_limit_s,
                ),
                "unavailable_worker_ids": [],
            },
            "action": primary_raw,
        }
    ]

    progress_by_workflow: dict[int, list[Any]] = {}
    artifacts_by_workflow: dict[int, list[list[Any]]] = {}
    shared_memory: list[Any] = []
    cumulative_tool_calls = 0
    for row_index, item in enumerate(timeline, start=1):
        if item["kind"] == "failure":
            event = item["event"]
            shared_memory.append(event["archived_workflow"])
            recovery_id = recovery_by_unavailable[item["unavailable"]]
            action_raw = plans[recovery_id][1]
            state = {
                "original_task": original_task,
                "workflow_id": None,
                "positions": [],
                "active_position_id": None,
                "terminal_status": "ready",
                "terminal_observation": str(event.get("terminal_observation") or "")[
                    -MAX_OBSERVATION_CHARS:
                ],
                "shared_memory": list(shared_memory),
                "budget": _budget(
                    event,
                    paid_calls_used=item["attempt"],
                    paid_call_limit=paid_call_limit,
                    wall_time_limit_s=wall_time_limit_s,
                ),
                "unavailable_worker_ids": sorted(item["unavailable"]),
            }
            control_action = action_raw
        else:
            route = item["route"]
            agent_step = item["agent_step"]
            action = plans[item["plan_id"]][0]
            progress = progress_by_workflow.setdefault(
                item["workflow_id"], [None for _ in action.steps]
            )
            artifacts = artifacts_by_workflow.setdefault(
                item["workflow_id"], [[] for _ in action.steps]
            )
            position_id = item["position_id"]
            if route.get("reported_progress") is not None:
                progress[position_id] = route["reported_progress"]
            reported_artifacts = route.get("reported_artifacts")
            if reported_artifacts is not None:
                if not isinstance(reported_artifacts, list):
                    raise LiveControlTrajectoryError(
                        f"route {row_index - 1} reported_artifacts is not a list"
                    )
                artifacts[position_id] = reported_artifacts
            tool_calls = agent_step.get("tool_calls") or []
            cumulative_tool_calls += len(tool_calls)
            next_item = timeline[row_index] if row_index < len(timeline) else None
            if next_item is None:
                control_action = {
                    "action": "complete",
                    "reason": (
                        "The final registered recovery role reports the requested implementation "
                        "and verification complete at a stable terminal."
                    ),
                }
            else:
                if next_item["workflow_id"] != item["workflow_id"]:
                    raise LiveControlTrajectoryError(
                        "a route changed workflow without a provider-failure boundary"
                    )
                next_position = next_item["position_id"]
                if next_position == position_id:
                    control_action = {
                        "action": "continue",
                        "reason": (
                            "The active role still owns an unfinished function-call loop and "
                            "should process the next terminal boundary."
                        ),
                    }
                elif next_position == position_id + 1:
                    control_action = {
                        "action": "handoff",
                        "reason": (
                            "The active role completed its assigned subtask at a stable terminal; "
                            "the next registered role can now inspect the shared workspace."
                        ),
                        "target_position_id": next_position,
                    }
                else:
                    raise LiveControlTrajectoryError(
                        "route timeline makes an invalid workflow position transition"
                    )
            terminal_status = _terminal_status(route.get("terminal_ready"))
            if (
                control_action["action"] in {"handoff", "complete"}
                and terminal_status != "ready"
            ):
                raise LiveControlTrajectoryError(
                    f"route {row_index - 1} labels {control_action['action']} without a stable terminal"
                )
            unavailable = plans[item["plan_id"]][2]
            state = {
                "original_task": original_task,
                "workflow_id": item["workflow_id"],
                "positions": _positions_after_route(
                    action,
                    position_id=position_id,
                    progress=progress,
                    artifacts=artifacts,
                ),
                "active_position_id": position_id,
                "terminal_status": terminal_status,
                "terminal_observation": _observation_text(agent_step),
                "shared_memory": list(shared_memory),
                "budget": _budget(
                    route,
                    paid_calls_used=item["attempt"],
                    paid_call_limit=paid_call_limit,
                    wall_time_limit_s=wall_time_limit_s,
                ),
                "unavailable_worker_ids": sorted(unavailable),
            }

        rows.append(
            {
                **common,
                "record_id": f"{collection_id}__control_{row_index:03d}",
                "agentic_evidence": {
                    "tool_calls_observed": cumulative_tool_calls,
                    "shared_workspace": True,
                    "verifier_audited": True,
                },
                "state": state,
                "action": control_action,
            }
        )
    return rows
