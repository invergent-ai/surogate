"""Structured control actions for long-horizon Fugu workflows.

The conductor owns workflow decisions. The runtime only validates ownership,
topology, global budget, and terminal safety before applying those decisions.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from openai import AsyncOpenAI

from .anonymous_planner import capability_profile_ref
from .behavior_likelihood import full_vocabulary_behavior_likelihood_contract

ControlActionName = Literal["continue", "handoff", "replan", "complete"]
PositionStatus = Literal["pending", "active", "completed", "interrupted", "failed"]
TerminalStatus = Literal["ready", "busy", "unknown"]

MAX_CONTROL_STEPS = 5
MAX_CONTROL_REASON_CHARS = 2_000
MAX_CONTROL_SUBTASK_CHARS = 8_000
# Decode-time bounds are intentionally tighter than the compatibility parser.
# A control action is an executable routing record, not a worker transcript.
MAX_CONTROL_DECODE_REASON_CHARS = 240
MAX_CONTROL_DECODE_SUBTASK_CHARS = 320
MAX_TERMINAL_OBSERVATION_CHARS = 12_000
MAX_DECISION_TERMINAL_OBSERVATION_CHARS = 8_000
MAX_SHARED_MEMORY_CHARS = 12_000
MAX_WORKER_DESCRIPTOR_ITEMS = 32
MAX_WORKER_DESCRIPTOR_CHARS = 500
MAX_DECISION_INPUT_TOKENS = 8_000
MAX_CONTROL_OUTPUT_TOKENS = 4_096
MAX_DECISION_CORRECTIONS = 2
MAX_DECISION_CORRECTION_ERROR_CHARS = 600
DECISION_PROMPT_COMPACTION_BUDGETS = (
    (6_000, 8_000, 6_000, 2_000, 6_000),
    (4_000, 4_000, 3_500, 1_200, 4_000),
    (2_000, 2_000, 2_000, 800, 2_500),
    (1_000, 1_000, 1_000, 500, 1_500),
    (700, 700, 700, 360, 900),
    (400, 400, 400, 240, 600),
    # Late long-horizon states have a fixed five-position structural cost even
    # after every evidence value is bounded.  Keep two correction-reserve
    # envelopes so a malformed action cannot strand the task merely because
    # the correction message must share the exact optimizer input window.
    (240, 240, 240, 160, 360),
    (96, 96, 120, 80, 160),
)
LIVE_AGENTIC_GRPO_BRIDGE_VERSION = "20260720-live-agentic-grpo-bridge-v4-capability-contract"


class ControlContractError(ValueError):
    """A conductor action cannot be parsed or legally applied."""


@dataclass(frozen=True)
class ControlStep:
    worker_id: int
    subtask: str
    access: tuple[int, ...] = ()


@dataclass(frozen=True)
class WorkerPerformance:
    """Pool-bound evidence about one worker on a task family."""

    task_family: str
    success_rate: float
    sample_size: int


@dataclass(frozen=True)
class WorkerProfile:
    """Runtime description for one stable slot in a versioned pool binding."""

    worker_id: int
    capability_tags: tuple[str, ...] = ()
    tool_tags: tuple[str, ...] = ()
    context_window_tokens: int | None = None
    constraints: tuple[str, ...] = ()
    observed_performance: tuple[WorkerPerformance, ...] = ()


@dataclass(frozen=True)
class ControlPosition:
    position_id: int
    worker_id: int
    subtask: str
    access: tuple[int, ...]
    status: PositionStatus
    progress: Any = None
    artifacts: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ControlBudget:
    paid_calls_used: int
    paid_call_limit: int
    elapsed_s: float
    wall_time_limit_s: float

    @property
    def paid_calls_remaining(self) -> int:
        return max(0, self.paid_call_limit - self.paid_calls_used)

    @property
    def wall_time_remaining_s(self) -> float:
        return max(0.0, self.wall_time_limit_s - self.elapsed_s)


@dataclass(frozen=True)
class LiveControlState:
    original_task: str
    workers: tuple[WorkerProfile, ...]
    workflow_id: int | None
    positions: tuple[ControlPosition, ...]
    active_position_id: int | None
    terminal_status: TerminalStatus
    terminal_observation: str
    shared_memory: tuple[Any, ...]
    budget: ControlBudget
    unavailable_worker_ids: tuple[int, ...] = ()

    @property
    def active_position(self) -> ControlPosition | None:
        return next(
            (position for position in self.positions if position.position_id == self.active_position_id),
            None,
        )

    @property
    def worker_ids(self) -> tuple[int, ...]:
        return tuple(worker.worker_id for worker in self.workers)


@dataclass(frozen=True)
class ControlAction:
    action: ControlActionName
    reason: str
    target_position_id: int | None = None
    steps: tuple[ControlStep, ...] = ()


@dataclass(frozen=True)
class CapabilityReferenceMap:
    """Binding-local translation kept outside the learned control surface."""

    profile_ref_to_worker_id: dict[str, int]
    worker_id_to_profile_ref: dict[int, str]

    def refs_for_workers(self, worker_ids: Sequence[int]) -> tuple[str, ...]:
        try:
            return tuple(self.worker_id_to_profile_ref[item] for item in worker_ids)
        except KeyError as exc:
            raise ControlContractError(f"unknown runtime worker id: {exc.args[0]}") from exc


def _bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"[...{len(value) - limit} earlier characters omitted...]\n{value[-limit:]}"


def _bounded_json_value(value: Any, limit: int) -> Any:
    """Keep ordinary evidence structured and compact only oversized values."""
    rendered = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    if len(rendered) <= limit:
        return value
    low = 0
    high = max(0, min(len(rendered), limit))
    best = ""
    while low <= high:
        midpoint = (low + high) // 2
        candidate = {"truncated_json": _bounded(rendered, midpoint)}
        if len(json.dumps(candidate, ensure_ascii=True, separators=(",", ":"))) <= limit:
            best = candidate["truncated_json"]
            low = midpoint + 1
        else:
            high = midpoint - 1
    return {"truncated_json": best}


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControlContractError(f"{label} must be an integer")
    return value


def _validate_descriptor_items(values: tuple[str, ...], label: str) -> None:
    if len(values) > MAX_WORKER_DESCRIPTOR_ITEMS:
        raise ControlContractError(f"{label} contains too many entries")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ControlContractError(f"{label} entries must be non-empty text")
    if any(len(value) > MAX_WORKER_DESCRIPTOR_CHARS for value in values):
        raise ControlContractError(f"{label} contains an entry that is too long")


def _validate_worker_profile(worker: WorkerProfile) -> None:
    _require_int(worker.worker_id, "worker_id")
    _validate_descriptor_items(worker.capability_tags, "capability_tags")
    _validate_descriptor_items(worker.tool_tags, "tool_tags")
    _validate_descriptor_items(worker.constraints, "constraints")
    if worker.context_window_tokens is not None:
        _require_int(worker.context_window_tokens, "context_window_tokens")
        if worker.context_window_tokens <= 0:
            raise ControlContractError("context_window_tokens must be positive")
    if len(worker.observed_performance) > MAX_WORKER_DESCRIPTOR_ITEMS:
        raise ControlContractError("observed_performance contains too many entries")
    for evidence in worker.observed_performance:
        if not isinstance(evidence.task_family, str) or not evidence.task_family.strip():
            raise ControlContractError("performance task_family must be non-empty")
        if len(evidence.task_family) > MAX_WORKER_DESCRIPTOR_CHARS:
            raise ControlContractError("performance task_family is too long")
        if isinstance(evidence.success_rate, bool) or not isinstance(
            evidence.success_rate,
            (int, float),
        ):
            raise ControlContractError("performance success_rate must be numeric")
        if not 0.0 <= evidence.success_rate <= 1.0:
            raise ControlContractError("performance success_rate must be between 0 and 1")
        _require_int(evidence.sample_size, "performance sample_size")
        if evidence.sample_size < 0:
            raise ControlContractError("performance sample_size cannot be negative")


def validate_worker_profiles(workers: tuple[WorkerProfile, ...]) -> None:
    """Validate the slots exposed by one versioned runtime pool."""
    worker_ids = tuple(worker.worker_id for worker in workers)
    if not workers or len(set(worker_ids)) != len(worker_ids):
        raise ControlContractError("worker_ids must be unique and non-empty")
    for worker in workers:
        _validate_worker_profile(worker)


def capability_reference_map(
    workers: tuple[WorkerProfile, ...],
) -> CapabilityReferenceMap:
    """Derive opaque selectors solely from anonymous capability descriptors."""
    validate_worker_profiles(workers)
    try:
        worker_to_ref = {
            worker.worker_id: capability_profile_ref(worker.capability_tags)
            for worker in workers
        }
    except ValueError as exc:
        raise ControlContractError(str(exc)) from exc
    if len(set(worker_to_ref.values())) != len(worker_to_ref):
        raise ControlContractError(
            "capability profiles must be distinct; enrich anonymous calibration "
            "metadata before using capability references"
        )
    return CapabilityReferenceMap(
        profile_ref_to_worker_id={ref: worker for worker, ref in worker_to_ref.items()},
        worker_id_to_profile_ref=worker_to_ref,
    )


def _parse_access(value: Any, label: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ControlContractError(f"{label} must be a list of earlier step indexes")
    access = tuple(_require_int(item, f"{label} entry") for item in value)
    if len(set(access)) != len(access):
        raise ControlContractError(f"{label} contains duplicate indexes")
    return access


def _parse_steps(value: Any) -> tuple[ControlStep, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ControlContractError("steps must be a list")
    if not 1 <= len(value) <= MAX_CONTROL_STEPS:
        raise ControlContractError(f"steps must contain 1 to {MAX_CONTROL_STEPS} entries")

    steps: list[ControlStep] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ControlContractError(f"steps[{index}] must be an object")
        unexpected = set(raw) - {"worker_id", "subtask", "access"}
        if unexpected:
            raise ControlContractError(f"steps[{index}] has unexpected fields: {sorted(unexpected)}")
        worker_id = _require_int(raw.get("worker_id"), f"steps[{index}].worker_id")
        subtask = raw.get("subtask")
        if not isinstance(subtask, str) or not subtask.strip():
            raise ControlContractError(f"steps[{index}].subtask must be non-empty text")
        subtask = subtask.strip()
        if len(subtask) > MAX_CONTROL_SUBTASK_CHARS:
            raise ControlContractError(f"steps[{index}].subtask is too long")
        access = _parse_access(raw.get("access", []), f"steps[{index}].access")
        if any(dependency < 0 or dependency >= index for dependency in access):
            raise ControlContractError(
                f"steps[{index}].access may reference only earlier steps"
            )
        steps.append(ControlStep(worker_id=worker_id, subtask=subtask, access=access))
    return tuple(steps)


def _parse_control_reason(raw: dict[str, Any], action: str) -> str:
    """Canonicalize only an absent reason; malformed supplied reasons stay invalid."""

    if "reason" not in raw:
        return f"{action} selected from the live task evidence"
    reason = raw["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ControlContractError("reason must be non-empty text")
    reason = reason.strip()
    if len(reason) > MAX_CONTROL_REASON_CHARS:
        raise ControlContractError("reason is too long")
    return reason


def _canonicalize_non_replan_steps(action: str, value: Any) -> Any:
    """Treat only an empty non-replan steps list as an omitted optional field."""

    if action == "replan" or value is None:
        return value
    if isinstance(value, list):
        if not value:
            return None
        raise ControlContractError(f"{action} cannot include replacement steps")
    return value


def parse_control_action(content: str) -> ControlAction:
    """Parse one strict conductor action JSON object."""
    try:
        raw = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ControlContractError(f"control action is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ControlContractError("control action must be an object")
    unexpected = set(raw) - {"action", "reason", "target_position_id", "steps"}
    if unexpected:
        raise ControlContractError(f"control action has unexpected fields: {sorted(unexpected)}")

    action = raw.get("action")
    if action not in {"continue", "handoff", "replan", "complete"}:
        raise ControlContractError("action must be continue, handoff, replan, or complete")
    reason = _parse_control_reason(raw, action)

    target = raw.get("target_position_id")
    if target is not None:
        target = _require_int(target, "target_position_id")
    return ControlAction(
        action=action,
        reason=reason,
        target_position_id=target,
        steps=_parse_steps(_canonicalize_non_replan_steps(action, raw.get("steps"))),
    )


def parse_capability_control_action(
    content: str,
    references: CapabilityReferenceMap,
) -> ControlAction:
    """Parse the typed learned contract and resolve profiles to runtime workers."""
    try:
        raw = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ControlContractError(f"control action is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ControlContractError("control action must be an object")
    unexpected = set(raw) - {"action", "reason", "target_position_id", "steps"}
    if unexpected:
        raise ControlContractError(f"control action has unexpected fields: {sorted(unexpected)}")
    action = raw.get("action")
    if action not in {"continue", "handoff", "replan", "complete"}:
        raise ControlContractError("action must be continue, handoff, replan, or complete")
    reason = _parse_control_reason(raw, action)
    target = raw.get("target_position_id")
    if target is not None:
        target = _require_int(target, "target_position_id")

    raw_steps = _canonicalize_non_replan_steps(action, raw.get("steps"))
    steps: list[ControlStep] = []
    if raw_steps is not None:
        if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= MAX_CONTROL_STEPS:
            raise ControlContractError(
                f"steps must contain 1 to {MAX_CONTROL_STEPS} entries"
            )
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, dict) or set(raw_step) != {
                "profile_ref",
                "subtask",
                "access_positions",
            }:
                raise ControlContractError(
                    f"steps[{index}] must contain only profile_ref, subtask, and access_positions"
                )
            profile_ref = raw_step.get("profile_ref")
            if profile_ref not in references.profile_ref_to_worker_id:
                raise ControlContractError(
                    f"steps[{index}].profile_ref is not in the active capability set"
                )
            subtask = raw_step.get("subtask")
            if not isinstance(subtask, str) or not subtask.strip():
                raise ControlContractError(f"steps[{index}].subtask must be non-empty text")
            subtask = subtask.strip()
            if len(subtask) > MAX_CONTROL_SUBTASK_CHARS:
                raise ControlContractError(f"steps[{index}].subtask is too long")
            access = _parse_access(
                raw_step.get("access_positions", []),
                f"steps[{index}].access_positions",
            )
            if any(dependency < 0 or dependency >= index for dependency in access):
                raise ControlContractError(
                    f"steps[{index}].access_positions may reference only earlier steps"
                )
            steps.append(
                ControlStep(
                    worker_id=references.profile_ref_to_worker_id[profile_ref],
                    subtask=subtask,
                    access=access,
                )
            )
    return ControlAction(
        action=action,
        reason=reason,
        target_position_id=target,
        steps=tuple(steps),
    )


def serialize_capability_control_action(
    action: ControlAction,
    references: CapabilityReferenceMap,
) -> str:
    """Serialize an internal action onto the anonymous learned contract."""
    payload: dict[str, Any] = {
        "action": action.action,
        "reason": action.reason,
    }
    if action.target_position_id is not None:
        payload["target_position_id"] = action.target_position_id
    if action.steps:
        steps: list[dict[str, Any]] = []
        for index, step in enumerate(action.steps):
            profile_ref = references.worker_id_to_profile_ref.get(step.worker_id)
            if profile_ref is None:
                raise ControlContractError(
                    f"steps[{index}].worker_id is not in the active capability set"
                )
            steps.append(
                {
                    "profile_ref": profile_ref,
                    "subtask": step.subtask,
                    "access_positions": list(step.access),
                }
            )
        payload["steps"] = steps
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def parse_control_decision(content: str) -> ControlAction:
    """Parse the compact live decision used before any replacement planning."""
    try:
        raw = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ControlContractError(f"control decision is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ControlContractError("control decision must be an object")
    unexpected = set(raw) - {"action", "target_position_id"}
    if unexpected:
        raise ControlContractError(
            f"control decision has unexpected fields: {sorted(unexpected)}"
        )
    action = raw.get("action")
    if action not in {"continue", "handoff", "replan", "complete"}:
        raise ControlContractError(
            "action must be continue, handoff, replan, or complete"
        )
    target = raw.get("target_position_id")
    if target is not None:
        target = _require_int(target, "target_position_id")
    return ControlAction(
        action=action,
        reason=f"{action} selected from the live task evidence",
        target_position_id=target,
    )


def canonicalize_control_decision(
    action: ControlAction,
    state: LiveControlState,
) -> tuple[ControlAction, dict[str, Any] | None]:
    """Drop a redundant self-target from an otherwise well-formed decision.

    A compact ``continue`` or ``complete`` whose ``target_position_id`` names
    the currently active position can only mean the active position, so the
    redundant target is removed and the normalization is reported for the
    audit record. Every other defect is preserved for validation.
    """
    if (
        action.action in {"continue", "complete"}
        and not action.steps
        and action.target_position_id is not None
        and state.active_position_id is not None
        and action.target_position_id == state.active_position_id
    ):
        return (
            replace(action, target_position_id=None),
            {
                "normalization": "dropped_redundant_self_target",
                "original_target_position_id": action.target_position_id,
            },
        )
    return action, None


def _validate_correction_attempt(correction_attempt: int) -> None:
    if (
        isinstance(correction_attempt, bool)
        or not isinstance(correction_attempt, int)
        or not 1 <= correction_attempt <= MAX_DECISION_CORRECTIONS
    ):
        raise ValueError(
            "correction_attempt must be an integer from 1 through "
            f"{MAX_DECISION_CORRECTIONS}"
        )


def render_decision_correction(
    error: str,
    *,
    correction_attempt: int = 1,
) -> str:
    """Feedback for one rejected decision, re-asked at the same live boundary."""
    _validate_correction_attempt(correction_attempt)
    bounded_error = _bounded(error.strip(), MAX_DECISION_CORRECTION_ERROR_CHARS)
    return (
        f"Correction attempt {correction_attempt} of {MAX_DECISION_CORRECTIONS}.\n"
        "Your previous control decision was rejected before reaching any worker: "
        f"{bounded_error}\n"
        "Decide again for the same live state. Return exactly one compact JSON "
        "control decision and no prose. target_position_id is allowed only for "
        "handoff and must name an eligible pending position."
    )


def render_control_action_correction(
    error: str,
    *,
    correction_attempt: int = 1,
) -> str:
    """Feedback for a rejected unified action at the same control boundary."""
    _validate_correction_attempt(correction_attempt)
    bounded_error = _bounded(error.strip(), MAX_DECISION_CORRECTION_ERROR_CHARS)
    return (
        f"Correction attempt {correction_attempt} of {MAX_DECISION_CORRECTIONS}.\n"
        "Your previous conductor action was rejected before reaching any worker: "
        f"{bounded_error}\n"
        "Decide again for the same state. Return exactly one full JSON action and "
        "no prose. Initial planning and replan require one to five replacement "
        "steps; continue and complete have no target or steps; handoff requires "
        "one eligible target_position_id."
    )


def _completion_requested(position: ControlPosition | None) -> bool:
    if position is None or not isinstance(position.progress, dict):
        return False
    return position.progress.get("completion_requested") is True


def _resolved_position_ids(state: LiveControlState) -> set[int]:
    """Positions whose evidence can satisfy a topology access dependency."""
    return {
        position.position_id
        for position in state.positions
        if position.status in {"completed", "interrupted"}
    }


def _eligible_handoff_targets(state: LiveControlState) -> list[int]:
    active = state.active_position
    if active is None:
        return []
    resolved_ids = _resolved_position_ids(state)
    resolved_ids.add(active.position_id)
    return [
        position.position_id
        for position in state.positions
        if position.status == "pending"
        and position.worker_id not in state.unavailable_worker_ids
        and all(dependency in resolved_ids for dependency in position.access)
    ]


def enabled_control_actions(state: LiveControlState) -> list[ControlActionName]:
    """Return the control actions that are legal at this live state.

    An empty result is a dead-end state: no progress action and no completion
    is legal (for example the paid budget is exhausted while the owner has not
    requested completion, or the terminal is unstable while it has). The
    decode schema cannot be built for such a state, and because the schema is
    derived deterministically from the state, re-asking the conductor cannot
    change the outcome. Callers must handle the dead end directly instead of
    spending correction attempts on it.
    """
    validate_control_state(state)
    available_worker_ids = [
        worker_id
        for worker_id in state.worker_ids
        if worker_id not in state.unavailable_worker_ids
    ]
    has_budget = (
        state.budget.paid_calls_remaining > 0
        and state.budget.wall_time_remaining_s > 0
    )
    handoff_targets = _eligible_handoff_targets(state)
    completion_requested = _completion_requested(state.active_position)
    enabled: list[ControlActionName] = []
    if has_budget and state.active_position is not None and not completion_requested:
        enabled.append("continue")
    if state.active_position is not None:
        if has_budget and state.terminal_status == "ready" and handoff_targets:
            enabled.append("handoff")
        if has_budget and state.terminal_status == "ready" and available_worker_ids:
            enabled.append("replan")
    if (
        state.active_position is not None
        and state.terminal_status == "ready"
        and completion_requested
    ):
        enabled.append("complete")
    elif (
        state.active_position is None
        and has_budget
        and state.terminal_status == "ready"
    ):
        enabled.append("replan")
    return enabled


def control_action_json_schema(state: LiveControlState | None = None) -> dict[str, Any]:
    """Return the mutually exclusive, optionally state-bounded decode schema."""
    reason = {
        "type": "string",
        "minLength": 1,
        "maxLength": MAX_CONTROL_DECODE_REASON_CHARS,
    }
    available_worker_ids = (
        [
            worker_id
            for worker_id in state.worker_ids
            if worker_id not in state.unavailable_worker_ids
        ]
        if state is not None
        else None
    )
    worker_id_schema: dict[str, Any] = {"type": "integer"}
    if available_worker_ids is not None:
        worker_id_schema["enum"] = available_worker_ids
    def step_schema(index: int) -> dict[str, Any]:
        access: dict[str, Any] = {
            "type": "array",
            "maxItems": index,
        }
        if index == 0:
            pass
        else:
            access["items"] = {
                "type": "integer",
                "minimum": 0,
                "maximum": index - 1,
            }
        return {
            "type": "object",
            "properties": {
                "worker_id": worker_id_schema,
                "subtask": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_CONTROL_DECODE_SUBTASK_CHARS,
                },
                "access": access,
            },
            "required": ["worker_id", "subtask", "access"],
            "additionalProperties": False,
        }

    topology_schemas = [
        {
            "type": "array",
            "prefixItems": [step_schema(index) for index in range(length)],
            "items": False,
            "minItems": length,
            "maxItems": length,
        }
        for length in range(1, MAX_CONTROL_STEPS + 1)
    ]

    def action_schema(
        action: ControlActionName,
        *,
        properties: dict[str, Any] | None = None,
        required: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "const": action},
                "reason": reason,
                **(properties or {}),
            },
            "required": ["action", "reason", *(required or [])],
            "additionalProperties": False,
        }

    variants = {
        "continue": action_schema("continue"),
        "handoff": action_schema(
            "handoff",
            properties={"target_position_id": {"type": "integer"}},
            required=["target_position_id"],
        ),
        "replan": action_schema(
            "replan",
            properties={
                "steps": {
                    "oneOf": topology_schemas,
                }
            },
            required=["steps"],
        ),
        "complete": action_schema("complete"),
    }
    if state is None:
        enabled_actions: list[ControlActionName] = [
            "continue",
            "handoff",
            "replan",
            "complete",
        ]
    else:
        validate_control_state(state)
        variants["handoff"]["properties"]["target_position_id"]["enum"] = (
            _eligible_handoff_targets(state)
        )
        enabled_actions = enabled_control_actions(state)
        if not enabled_actions:
            raise ControlContractError("live state has no decodable control action")

    return {
        "name": "fugu_live_control_action",
        "strict": True,
        "schema": {
            "oneOf": [variants[action] for action in enabled_actions]
        },
    }


def capability_control_action_json_schema(
    state: LiveControlState,
    references: CapabilityReferenceMap | None = None,
) -> dict[str, Any]:
    """State-bounded full-action schema with disjoint profile/access types."""
    references = references or capability_reference_map(state.workers)
    schema = control_action_json_schema(state)
    schema["name"] = "fugu_capability_control_action"
    available_refs = [
        references.worker_id_to_profile_ref[worker_id]
        for worker_id in state.worker_ids
        if worker_id not in state.unavailable_worker_ids
    ]
    for variant in schema["schema"]["oneOf"]:
        if variant["properties"]["action"]["const"] != "replan":
            continue
        for topology in variant["properties"]["steps"]["oneOf"]:
            for step in topology["prefixItems"]:
                properties = step["properties"]
                worker_schema = properties.pop("worker_id")
                worker_schema["type"] = "string"
                worker_schema["enum"] = available_refs
                properties["profile_ref"] = worker_schema
                properties["access_positions"] = properties.pop("access")
                step["required"] = [
                    "profile_ref",
                    "subtask",
                    "access_positions",
                ]
    return schema


def control_decision_json_schema(state: LiveControlState | None = None) -> dict[str, Any]:
    """Return the state-bounded schema for action selection without topology text."""
    schema = control_action_json_schema(state)
    schema["name"] = "fugu_live_control_decision"
    for variant in schema["schema"]["oneOf"]:
        action = variant["properties"]["action"]["const"]
        variant["properties"].pop("reason", None)
        variant["required"] = [
            field for field in variant["required"] if field != "reason"
        ]
        if action == "replan":
            variant["properties"].pop("steps", None)
            variant["required"] = [
                field for field in variant["required"] if field != "steps"
            ]
    return schema


def _prompt_action_schema(
    state: LiveControlState,
    *,
    compact_decision: bool,
    references: CapabilityReferenceMap | None = None,
) -> dict[str, Any]:
    """Render the learned schema, bounding it only when the state removes actions."""
    enabled_actions = [
        variant["properties"]["action"]["const"]
        for variant in control_action_json_schema(state)["schema"]["oneOf"]
    ]
    handoff_targets = _eligible_handoff_targets(state)
    if references is None:
        selector_name = "worker_id"
        selector_source = "workers"
        access_name = "access"
    else:
        selector_name = "profile_ref"
        selector_source = "capability_profiles"
        access_name = "access_positions"

    if not _uses_state_bounded_prompt(state):
        schema: dict[str, Any] = {
            "action": "continue | handoff | replan | complete",
            "target_position_id": "required only for handoff",
        }
        if not compact_decision:
            schema["steps"] = [
                {
                    selector_name: (
                        "one supplied capability profile reference"
                        if references is not None
                        else "one supplied worker ID"
                    ),
                    "subtask": "task-specific tool-using work",
                    access_name: [
                        "earlier zero-based replacement-step indexes",
                    ],
                }
            ]
        return schema

    def action_rule() -> dict[str, Any]:
        fields = ["action"]
        if not compact_decision:
            fields.append("reason?")
        return {"fields": fields}

    variants = {
        action: action_rule()
        for action in enabled_actions
    }
    if "handoff" in variants:
        variants["handoff"]["fields"].append("target_position_id")
        variants["handoff"]["target_position_id"] = handoff_targets
    if not compact_decision and "replan" in variants:
        variants["replan"]["fields"].append("steps")
        variants["replan"]["steps"] = {
            "count": f"1..{MAX_CONTROL_STEPS}",
            "item_fields": [selector_name, "subtask", access_name],
            selector_name: f"one value from {selector_source}",
            access_name: "earlier replacement-step indexes only",
        }
    return {
        "valid_actions": enabled_actions,
        "rules": variants,
    }


def _uses_state_bounded_prompt(state: LiveControlState) -> bool:
    return (
        _completion_requested(state.active_position)
        or state.budget.paid_calls_remaining <= 0
        or state.budget.wall_time_remaining_s <= 0
    )


def validate_control_state(state: LiveControlState) -> None:
    """Validate state invariants before exposing the state to the conductor."""
    if not state.original_task.strip():
        raise ControlContractError("original_task must be non-empty")
    validate_worker_profiles(state.workers)
    if len(state.unavailable_worker_ids) != len(set(state.unavailable_worker_ids)):
        raise ControlContractError("unavailable_worker_ids must be unique")
    if any(worker_id not in state.worker_ids for worker_id in state.unavailable_worker_ids):
        raise ControlContractError("an unavailable worker ID is outside the pool")
    if state.budget.paid_call_limit <= 0:
        raise ControlContractError("paid_call_limit must be positive")
    if not 0 <= state.budget.paid_calls_used <= state.budget.paid_call_limit:
        raise ControlContractError("paid call usage is outside the global budget")
    if state.budget.wall_time_limit_s <= 0 or not 0 <= state.budget.elapsed_s:
        raise ControlContractError("wall-time budget is invalid")

    position_ids = [position.position_id for position in state.positions]
    if len(position_ids) != len(set(position_ids)):
        raise ControlContractError("position_id values must be unique")
    if any(position.worker_id not in state.worker_ids for position in state.positions):
        raise ControlContractError("a workflow position references a worker outside the pool")
    if state.active_position_id is None:
        if any(position.status == "active" for position in state.positions):
            raise ControlContractError("state has an active position but no active_position_id")
    else:
        active = state.active_position
        if active is None or active.status != "active":
            raise ControlContractError("active_position_id must reference the active position")
        if sum(position.status == "active" for position in state.positions) != 1:
            raise ControlContractError("exactly one position must be active")


def validate_control_action(action: ControlAction, state: LiveControlState) -> None:
    """Reject control decisions that violate ownership, topology, or safety."""
    validate_control_state(state)
    active = state.active_position
    positions = {position.position_id: position for position in state.positions}

    if action.action == "continue":
        if active is None:
            raise ControlContractError("continue requires an active position")
        if _completion_requested(active):
            raise ControlContractError("continue cannot retain a position that requested completion")
        if action.target_position_id is not None or action.steps:
            raise ControlContractError("continue cannot include a target or replacement steps")
        if state.budget.paid_calls_remaining <= 0 or state.budget.wall_time_remaining_s <= 0:
            raise ControlContractError("continue cannot exceed the global task budget")
        return

    if action.action == "handoff":
        if active is None:
            raise ControlContractError("handoff requires an active position")
        if state.terminal_status != "ready":
            raise ControlContractError("handoff requires a stable terminal")
        if action.steps:
            raise ControlContractError("handoff cannot replace the workflow")
        if action.target_position_id is None:
            raise ControlContractError("handoff requires target_position_id")
        target = positions.get(action.target_position_id)
        if target is None or target.status != "pending":
            raise ControlContractError("handoff target must be a pending workflow position")
        if target.worker_id in state.unavailable_worker_ids:
            raise ControlContractError("handoff target uses an unavailable worker")
        resolved_on_handoff = _resolved_position_ids(state) | {active.position_id}
        if any(dependency not in resolved_on_handoff for dependency in target.access):
            raise ControlContractError("handoff target has incomplete dependencies")
        if state.budget.paid_calls_remaining <= 0 or state.budget.wall_time_remaining_s <= 0:
            raise ControlContractError("handoff cannot exceed the global task budget")
        return

    if action.action == "replan":
        if state.terminal_status != "ready":
            raise ControlContractError("replan requires a stable terminal")
        if action.target_position_id is not None:
            raise ControlContractError("replan cannot include target_position_id")
        if not action.steps:
            raise ControlContractError("replan requires replacement steps")
        if any(step.worker_id not in state.worker_ids for step in action.steps):
            raise ControlContractError("replan selected a worker outside the pool")
        if any(step.worker_id in state.unavailable_worker_ids for step in action.steps):
            raise ControlContractError("replan selected an unavailable worker")
        if state.budget.paid_calls_remaining <= 0 or state.budget.wall_time_remaining_s <= 0:
            raise ControlContractError("replan cannot exceed the global task budget")
        return

    if state.terminal_status != "ready":
        raise ControlContractError("complete requires a stable terminal")
    if not state.positions:
        raise ControlContractError("complete requires workflow evidence")
    if active is None:
        raise ControlContractError("complete requires an active workflow position")
    if not _completion_requested(active):
        raise ControlContractError(
            "complete requires the active position to request completion"
        )
    if action.target_position_id is not None or action.steps:
        raise ControlContractError("complete cannot include a target or replacement steps")


def validate_control_decision(action: ControlAction, state: LiveControlState) -> None:
    """Validate a compact action before a separate planner supplies replan steps."""
    if action.steps:
        raise ControlContractError("a compact control decision cannot include replacement steps")
    if action.action != "replan":
        validate_control_action(action, state)
        return

    validate_control_state(state)
    if state.terminal_status != "ready":
        raise ControlContractError("replan requires a stable terminal")
    if action.target_position_id is not None:
        raise ControlContractError("replan cannot include target_position_id")
    if state.budget.paid_calls_remaining <= 0 or state.budget.wall_time_remaining_s <= 0:
        raise ControlContractError("replan cannot exceed the global task budget")
    if not any(
        worker_id not in state.unavailable_worker_ids for worker_id in state.worker_ids
    ):
        raise ControlContractError("replan requires an available worker")


def _control_state_payload(
    state: LiveControlState,
    *,
    terminal_observation_chars: int = MAX_TERMINAL_OBSERVATION_CHARS,
    shared_memory_chars: int = MAX_SHARED_MEMORY_CHARS,
    position_evidence_chars: int | None = None,
    position_subtask_chars: int | None = None,
    original_task_chars: int | None = None,
) -> dict[str, Any]:
    state_bounded = _uses_state_bounded_prompt(state)
    payload = {
        "original_task": (
            _bounded(state.original_task, original_task_chars)
            if original_task_chars is not None
            else state.original_task
        ),
        "workers": [
            {
                "worker_id": worker.worker_id,
                "capability_tags": list(worker.capability_tags),
                "tool_tags": list(worker.tool_tags),
                "context_window_tokens": worker.context_window_tokens,
                "constraints": list(worker.constraints),
                "observed_performance": [
                    {
                        "task_family": evidence.task_family,
                        "success_rate": evidence.success_rate,
                        "sample_size": evidence.sample_size,
                    }
                    for evidence in worker.observed_performance
                ],
            }
            for worker in state.workers
        ],
        "unavailable_worker_ids": list(state.unavailable_worker_ids),
        "workflow_id": state.workflow_id,
        "positions": [
            {
                "position_id": position.position_id,
                "worker_id": position.worker_id,
                "subtask": (
                    _bounded(position.subtask, position_subtask_chars)
                    if position_subtask_chars is not None
                    else position.subtask
                ),
                "access": list(position.access),
                "status": position.status,
                "progress": (
                    _bounded_json_value(
                        position.progress,
                        max(100, position_evidence_chars * 2 // 3),
                    )
                    if position_evidence_chars is not None
                    else position.progress
                ),
                "artifacts": (
                    _bounded_json_value(
                        list(position.artifacts),
                        max(100, position_evidence_chars // 3),
                    )
                    if position_evidence_chars is not None
                    else list(position.artifacts)
                ),
            }
            for position in state.positions
        ],
        "active_position_id": state.active_position_id,
        "terminal_status": state.terminal_status,
        "terminal_observation": _bounded(
            state.terminal_observation,
            terminal_observation_chars,
        ),
        "shared_memory": _bounded(
            json.dumps(state.shared_memory, ensure_ascii=True),
            shared_memory_chars,
        ),
        "global_budget": {
            "paid_calls_used": state.budget.paid_calls_used,
            "paid_call_limit": state.budget.paid_call_limit,
            "paid_calls_remaining": state.budget.paid_calls_remaining,
            "elapsed_s": state.budget.elapsed_s,
            "wall_time_limit_s": state.budget.wall_time_limit_s,
            "wall_time_remaining_s": state.budget.wall_time_remaining_s,
        },
    }
    if state_bounded:
        enabled_actions = [
            variant["properties"]["action"]["const"]
            for variant in control_action_json_schema(state)["schema"]["oneOf"]
        ]
        payload["valid_actions"] = enabled_actions
        payload["eligible_handoff_target_position_ids"] = _eligible_handoff_targets(
            state
        )
        payload["active_completion_requested"] = _completion_requested(
            state.active_position
        )
        for position_payload, position in zip(
            payload["positions"],
            state.positions,
            strict=True,
        ):
            position_payload["completion_requested"] = _completion_requested(
                position
            )
    return payload


def _replace_runtime_worker_fields(
    value: Any,
    references: CapabilityReferenceMap,
) -> Any:
    if isinstance(value, list):
        return [_replace_runtime_worker_fields(item, references) for item in value]
    if not isinstance(value, dict):
        return value
    mapped: dict[str, Any] = {}
    for key, item in value.items():
        if key == "worker_id":
            if item not in references.worker_id_to_profile_ref:
                raise ControlContractError(f"state contains unknown runtime worker id: {item}")
            mapped["profile_ref"] = references.worker_id_to_profile_ref[item]
        elif key in {"worker_ids", "unavailable_worker_ids"}:
            if not isinstance(item, (list, tuple)):
                raise ControlContractError(f"{key} must be a worker-ID sequence")
            target = "profile_refs" if key == "worker_ids" else "unavailable_profile_refs"
            mapped[target] = list(references.refs_for_workers(item))
        else:
            mapped[key] = _replace_runtime_worker_fields(item, references)
    return mapped


def _capability_control_state_payload(
    state: LiveControlState,
    *,
    references: CapabilityReferenceMap,
    terminal_observation_chars: int,
    shared_memory_chars: int,
    position_evidence_chars: int | None,
    position_subtask_chars: int | None,
    original_task_chars: int | None,
) -> dict[str, Any]:
    payload = _control_state_payload(
        state,
        terminal_observation_chars=terminal_observation_chars,
        shared_memory_chars=shared_memory_chars,
        position_evidence_chars=position_evidence_chars,
        position_subtask_chars=position_subtask_chars,
        original_task_chars=original_task_chars,
    )
    payload["shared_memory"] = _bounded(
        json.dumps(
            _replace_runtime_worker_fields(list(state.shared_memory), references),
            ensure_ascii=True,
        ),
        shared_memory_chars,
    )
    mapped = _replace_runtime_worker_fields(payload, references)
    profiles = mapped.pop("workers")
    mapped["capability_profiles"] = sorted(
        profiles,
        key=lambda row: row["profile_ref"],
    )
    for position in mapped["positions"]:
        position["access_positions"] = position.pop("access")
    return mapped


def render_capability_control_prompt(
    state: LiveControlState,
    *,
    compact_decision: bool = False,
    terminal_observation_chars: int = MAX_TERMINAL_OBSERVATION_CHARS,
    shared_memory_chars: int = MAX_SHARED_MEMORY_CHARS,
    position_evidence_chars: int | None = None,
    position_subtask_chars: int | None = None,
    original_task_chars: int | None = None,
) -> str:
    """Render a permutation-invariant live state with typed selector namespaces."""
    validate_control_state(state)
    references = capability_reference_map(state.workers)
    schema = _prompt_action_schema(
        state,
        compact_decision=compact_decision,
        references=references,
    )
    mode = (
        "Choose only the next workflow action. A separate planning pass creates "
        "replacement steps after replan."
        if compact_decision
        else "Choose the next workflow action and include typed replacement steps for replan."
    )
    bounded_instruction = (
        " ACTION SCHEMA is the exhaustive set of valid actions for this exact "
        "state; do not emit fields outside the selected action rule."
        if _uses_state_bounded_prompt(state)
        else ""
    )
    return (
        f"{mode} Preserve an active worker's private function-call loop with continue "
        "while it is making concrete progress. Handoff only to an eligible pending "
        "position. Replan when no workflow exists or the current topology is exhausted "
        "or wrong. Scale replacement topology to the global paid-call budget. When no "
        "more than 12 paid calls remain and the task names an explicit deliverable, use "
        "a compact plan whose first position owns inspection plus creation of a viable "
        "artifact; add a separate verifier only when the remaining budget can actually "
        "reach it. Complete only from verified overall evidence. Select workers only by "
        "the anonymous capability profiles supplied in this request; no profile is a "
        "default or fallback. profile_ref selects a capability profile. position_id "
        "and target_position_id refer to live workflow positions and are integers. "
        "In replan steps, access_positions contains zero-based indexes of earlier "
        "replacement steps; it never contains live position IDs. Profile references "
        "are never position indexes. Return exactly one "
        "JSON object and no prose. Keep the reason and each subtask concise and "
        f"operational; one or two sentences is sufficient.{bounded_instruction}\n\n"
        f"ACTION SCHEMA:\n{json.dumps(schema, ensure_ascii=True, indent=2)}\n\n"
        "LIVE STATE:\n"
        f"{json.dumps(_capability_control_state_payload(state, references=references, terminal_observation_chars=terminal_observation_chars, shared_memory_chars=shared_memory_chars, position_evidence_chars=position_evidence_chars, position_subtask_chars=position_subtask_chars, original_task_chars=original_task_chars), ensure_ascii=True, indent=2)}"
    )


def render_control_prompt(
    state: LiveControlState,
    *,
    terminal_observation_chars: int = MAX_TERMINAL_OBSERVATION_CHARS,
    shared_memory_chars: int = MAX_SHARED_MEMORY_CHARS,
    position_evidence_chars: int | None = None,
    position_subtask_chars: int | None = None,
    original_task_chars: int | None = None,
) -> str:
    """Render bounded workflow state without leaking private sibling transcripts."""
    validate_control_state(state)
    schema = _prompt_action_schema(
        state,
        compact_decision=False,
    )
    bounded_instruction = (
        " ACTION SCHEMA is the exhaustive set of valid actions for this exact "
        "state; do not emit fields outside the selected action rule."
        if _uses_state_bounded_prompt(state)
        else ""
    )
    return (
        "Choose the next Fugu workflow action from the live task state. Preserve the "
        "current command owner's function-call loop with continue. At a stable terminal, "
        "handoff when a different pending position is now more useful, including when the "
        "current position is stalled or pursuing a bad path. When no workflow exists, "
        "replan creates the initial workflow. Otherwise, replan at a stable terminal when "
        "the existing topology is exhausted or wrong. These are conductor decisions, "
        "not fixed turn or time leases. Scale replacement topology to the remaining "
        "paid calls; under a 12-call-or-smaller budget, an explicit deliverable should "
        "be owned by the first position rather than deferred behind inspection-only "
        "positions. Complete only when the overall task is verified. "
        "A position with progress.completion_requested=true has finished its assigned "
        "subtask and cannot continue; hand it off, replan, or complete from its evidence. "
        "An unfinished handoff or replan preserves that position's partial checkpoint; "
        "share it only through the workflow access list or persistent prior-workflow memory. "
        "Worker IDs are stable only within the current versioned pool binding. Use the "
        "learned pool-specific role priors, runtime descriptors, and live task evidence. "
        "No worker is globally preferred, default, or a fallback; choose by the current "
        "task and role. A replacement pool requires a new binding and continued training. "
        "Return exactly one JSON object and no prose. Keep the reason and each subtask "
        f"concise and operational; one or two sentences is sufficient.{bounded_instruction}\n\n"
        f"ACTION SCHEMA:\n{json.dumps(schema, ensure_ascii=True, indent=2)}\n\n"
        "LIVE STATE:\n"
        f"{json.dumps(_control_state_payload(state, terminal_observation_chars=terminal_observation_chars, shared_memory_chars=shared_memory_chars, position_evidence_chars=position_evidence_chars, position_subtask_chars=position_subtask_chars, original_task_chars=original_task_chars), ensure_ascii=True, indent=2)}"
    )


def render_control_decision_prompt(
    state: LiveControlState,
    *,
    terminal_observation_chars: int = MAX_DECISION_TERMINAL_OBSERVATION_CHARS,
    shared_memory_chars: int = MAX_SHARED_MEMORY_CHARS,
    position_evidence_chars: int | None = None,
    position_subtask_chars: int | None = None,
    original_task_chars: int | None = None,
) -> str:
    """Render the live state for compact action selection only."""
    validate_control_state(state)
    schema = _prompt_action_schema(
        state,
        compact_decision=True,
    )
    return (
        "Choose only the next Fugu workflow action from the live task state. Preserve "
        "the current command owner's private function-call loop with continue while it "
        "is making concrete progress. At a stable terminal, handoff when an existing "
        "pending position is the right successor and can use the preserved checkpoint. "
        "Choose replan when the current topology is exhausted, invalid, or no longer "
        "describes the work required; a separate conductor planning pass will generate "
        "the replacement topology. Complete only when the terminal is stable and the "
        "overall task is verified from workflow evidence. These are evidence-based "
        "conductor decisions, not fixed turn or time leases. No worker is globally preferred, "
        "default, or a fallback. Worker IDs are stable only within the current versioned "
        "pool binding; use the learned pool-specific role priors and live evidence. A "
        "replacement pool requires a new binding and continued training. Return exactly "
        "one compact JSON object containing only the action and optional handoff target; "
        "the runtime records an audit reason from the validated decision. Return no prose; "
        "never include replacement workflow steps in this decision.\n\n"
        f"DECISION SCHEMA:\n{json.dumps(schema, ensure_ascii=True, indent=2)}\n\n"
        "LIVE STATE:\n"
        f"{json.dumps(_control_state_payload(state, terminal_observation_chars=terminal_observation_chars, shared_memory_chars=shared_memory_chars, position_evidence_chars=position_evidence_chars, position_subtask_chars=position_subtask_chars, original_task_chars=original_task_chars), ensure_ascii=True, indent=2)}"
    )


def build_control_decision_messages(
    state: LiveControlState,
    *,
    prompt_token_counter: Callable[[Sequence[dict[str, str]]], int] | None = None,
    max_input_tokens: int = MAX_DECISION_INPUT_TOKENS,
    capability_refs: bool = False,
) -> tuple[list[dict[str, str]], int | None, bool]:
    """Build the exact bounded messages used for one live-control decision."""
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")

    def messages(prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are the live conductor for a multi-step, tool-using agentic task. "
                    "Return one valid compact control-decision JSON object."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    if capability_refs:
        def renderer(value: LiveControlState, **kwargs: Any) -> str:
            return render_capability_control_prompt(
                value,
                compact_decision=True,
                **kwargs,
            )
    else:
        renderer = render_control_decision_prompt
    rendered = messages(renderer(state))
    if prompt_token_counter is None:
        return rendered, None, False
    prompt_tokens = prompt_token_counter(rendered)
    if prompt_tokens <= max_input_tokens:
        return rendered, prompt_tokens, False
    for (
        terminal_chars,
        memory_chars,
        evidence_chars,
        subtask_chars,
        task_chars,
    ) in DECISION_PROMPT_COMPACTION_BUDGETS:
        rendered = messages(
            renderer(
                state,
                terminal_observation_chars=terminal_chars,
                shared_memory_chars=memory_chars,
                position_evidence_chars=evidence_chars,
                position_subtask_chars=subtask_chars,
                original_task_chars=task_chars,
            )
        )
        prompt_tokens = prompt_token_counter(rendered)
        if prompt_tokens <= max_input_tokens:
            return rendered, prompt_tokens, True
    raise ControlContractError(
        "live-control prompt remains above the local input-token ceiling after compaction"
    )


def build_control_action_messages(
    state: LiveControlState,
    *,
    prompt_token_counter: Callable[[Sequence[dict[str, str]]], int] | None = None,
    max_input_tokens: int = MAX_DECISION_INPUT_TOKENS,
    capability_refs: bool = False,
    guidelines: str | None = None,
) -> tuple[list[dict[str, str]], int | None, bool]:
    """Build bounded messages for one unified planning or live-control action.

    ``guidelines`` carries retrieved decision guidance (natural-language rules
    distilled from prior verified outcomes). It rides in the system message so
    the user-prompt compaction budgets below continue to govern live state
    alone; passing ``None`` reproduces the unguided prompt byte for byte.
    """
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")

    system_content = (
        "You are the conductor for a multi-step, tool-using agentic task. "
        "Return one valid full conductor-action JSON object."
    )
    if guidelines:
        system_content = f"{system_content}\n\n{guidelines}"

    def messages(prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

    renderer = render_capability_control_prompt if capability_refs else render_control_prompt
    rendered = messages(renderer(state))
    if prompt_token_counter is None:
        return rendered, None, False
    prompt_tokens = prompt_token_counter(rendered)
    if prompt_tokens <= max_input_tokens:
        return rendered, prompt_tokens, False
    for (
        terminal_chars,
        memory_chars,
        evidence_chars,
        subtask_chars,
        task_chars,
    ) in DECISION_PROMPT_COMPACTION_BUDGETS:
        rendered = messages(
            renderer(
                state,
                terminal_observation_chars=terminal_chars,
                shared_memory_chars=memory_chars,
                position_evidence_chars=evidence_chars,
                position_subtask_chars=subtask_chars,
                original_task_chars=task_chars,
            )
        )
        prompt_tokens = prompt_token_counter(rendered)
        if prompt_tokens <= max_input_tokens:
            return rendered, prompt_tokens, True
    raise ControlContractError(
        "full conductor prompt remains above the local input-token ceiling after compaction"
    )


def _validate_worker_id_map(
    worker_ids: tuple[int, ...],
    worker_id_map: dict[int, int],
) -> None:
    if set(worker_id_map) != set(worker_ids):
        raise ControlContractError("worker_id_map must map every and only current worker ID")
    mapped_ids = tuple(worker_id_map[worker_id] for worker_id in worker_ids)
    if any(isinstance(worker_id, bool) or not isinstance(worker_id, int) for worker_id in mapped_ids):
        raise ControlContractError("mapped worker IDs must be integers")
    if len(set(mapped_ids)) != len(mapped_ids):
        raise ControlContractError("mapped worker IDs must be unique")


def remap_control_state_workers(
    state: LiveControlState,
    worker_id_map: dict[int, int],
    *,
    worker_order: tuple[int, ...] | None = None,
) -> LiveControlState:
    """Relabel and optionally reorder a pool without changing workflow meaning.

    ``worker_order`` contains IDs from the source binding. This mechanical
    helper is for explicit data migration and tests; it must not relabel an
    existing checkpoint's bound slots.
    """
    validate_control_state(state)
    _validate_worker_id_map(state.worker_ids, worker_id_map)
    if worker_order is None:
        worker_order = state.worker_ids
    if len(worker_order) != len(set(worker_order)) or set(worker_order) != set(state.worker_ids):
        raise ControlContractError("worker_order must be a permutation of current worker IDs")

    workers_by_id = {worker.worker_id: worker for worker in state.workers}
    remapped = replace(
        state,
        workers=tuple(
            replace(workers_by_id[worker_id], worker_id=worker_id_map[worker_id])
            for worker_id in worker_order
        ),
        positions=tuple(
            replace(position, worker_id=worker_id_map[position.worker_id])
            for position in state.positions
        ),
        unavailable_worker_ids=tuple(
            worker_id_map[worker_id] for worker_id in state.unavailable_worker_ids
        ),
    )
    validate_control_state(remapped)
    return remapped


def remap_control_action_workers(
    action: ControlAction,
    worker_id_map: dict[int, int],
) -> ControlAction:
    """Apply an explicit source-to-target binding relabeling to an action."""
    unknown = {step.worker_id for step in action.steps} - set(worker_id_map)
    if unknown:
        raise ControlContractError(f"action references unmapped worker IDs: {sorted(unknown)}")
    return replace(
        action,
        steps=tuple(
            replace(step, worker_id=worker_id_map[step.worker_id])
            for step in action.steps
        ),
    )


class OpenAILiveController:
    """Strict single-attempt client for a locally served live conductor."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "x",
        client: AsyncOpenAI | None = None,
        max_tokens: int = 64,
        seed: int = 0,
        temperature: float = 0.0,
        record_token_data: bool = False,
        prompt_token_counter: Callable[[Sequence[dict[str, str]]], int] | None = None,
        max_input_tokens: int = MAX_DECISION_INPUT_TOKENS,
        supplies_topology: bool = False,
        capability_refs: bool = False,
    ) -> None:
        if not model.strip():
            raise ValueError("controller model must be non-empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if not 0.0 <= float(temperature) <= 2.0:
            raise ValueError("temperature must be within [0, 2]")
        if record_token_data and float(temperature) != 1.0:
            raise ValueError(
                "training token collection requires temperature exactly 1.0"
            )
        self._model = model
        self._max_tokens = max_tokens
        self._seed = seed
        self._temperature = float(temperature)
        self._record_token_data = bool(record_token_data)
        self._prompt_token_counter = prompt_token_counter
        self._max_input_tokens = max_input_tokens
        self.supplies_topology = bool(supplies_topology)
        self.capability_refs = bool(capability_refs)
        # Optional callable(state) -> str supplying retrieved decision
        # guidelines. Kept as an injected callback so this module carries no
        # dependency on any particular memory implementation.
        self.guidelines_provider: Callable[[LiveControlState], str | None] | None = None
        self.prompt_compactions = 0
        self.last_prompt_tokens: int | None = None
        self.decision_traces: list[dict[str, Any]] = []
        self._client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=180.0,
            max_retries=0,
        )

    def reset_traces(self) -> None:
        self.decision_traces.clear()

    def _bounded_messages(
        self,
        state: LiveControlState,
        *,
        correction: str | None = None,
    ) -> list[dict[str, str]]:
        builder = (
            build_control_action_messages
            if self.supplies_topology
            else build_control_decision_messages
        )
        prompt_counter = self._prompt_token_counter
        if prompt_counter is not None and correction is not None:
            base_counter = prompt_counter

            def prompt_counter(messages: Sequence[dict[str, str]]) -> int:
                return base_counter(
                    [*messages, {"role": "user", "content": correction}]
                )

        builder_kwargs: dict[str, Any] = {
            "prompt_token_counter": prompt_counter,
            "max_input_tokens": self._max_input_tokens,
            "capability_refs": self.capability_refs,
        }
        if self.supplies_topology and self.guidelines_provider is not None:
            builder_kwargs["guidelines"] = self.guidelines_provider(state)
        messages, prompt_tokens, compacted = builder(state, **builder_kwargs)
        self.last_prompt_tokens = prompt_tokens
        self.prompt_compactions += int(compacted)
        return messages

    async def decide(
        self,
        state: LiveControlState,
        *,
        correction: str | None = None,
    ) -> ControlAction:
        messages = self._bounded_messages(state, correction=correction)
        if correction is not None:
            messages = [*messages, {"role": "user", "content": correction}]
        if self._prompt_token_counter is not None:
            exact_prompt_tokens = self._prompt_token_counter(messages)
            self.last_prompt_tokens = exact_prompt_tokens
            if exact_prompt_tokens > self._max_input_tokens:
                raise ControlContractError(
                    "control prompt exceeds the configured input-token boundary "
                    f"({exact_prompt_tokens} > {self._max_input_tokens})"
                )
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "seed": self._seed,
            "max_tokens": self._max_tokens,
        }
        if not self._record_token_data:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": (
                    capability_control_action_json_schema(state)
                    if self.supplies_topology and self.capability_refs
                    else control_action_json_schema(state)
                    if self.supplies_topology
                    else control_decision_json_schema(state)
                ),
            }
        if self._record_token_data:
            request.update(
                {
                    "logprobs": True,
                    "top_logprobs": 0,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "extra_body": {
                        "return_token_ids": True,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "top_k": 0,
                        "min_p": 0.0,
                        "repetition_penalty": 1.0,
                    },
                }
            )
        response = await self._client.chat.completions.create(**request)
        choice = response.choices[0]
        content = choice.message.content or ""
        token_data: dict[str, Any] = {}
        if self._record_token_data:
            if not hasattr(response, "model_dump"):
                raise ControlContractError(
                    "training conductor response omitted serializable token evidence"
                )
            payload = response.model_dump(mode="json")
            raw_choices = payload.get("choices") if isinstance(payload, dict) else None
            raw_choice = raw_choices[0] if isinstance(raw_choices, list) and raw_choices else {}
            prompt_token_ids = payload.get("prompt_token_ids")
            completion_token_ids = raw_choice.get("token_ids")
            logprob_rows = ((raw_choice.get("logprobs") or {}).get("content") or [])
            completion_logprobs = [
                row.get("logprob") if isinstance(row, dict) else None
                for row in logprob_rows
            ]
            valid_ids = lambda values: (  # noqa: E731 - compact validation predicate
                isinstance(values, list)
                and all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            )
            if (
                not valid_ids(prompt_token_ids)
                or not valid_ids(completion_token_ids)
                or not completion_token_ids
                or len(completion_logprobs) != len(completion_token_ids)
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in completion_logprobs
                )
            ):
                raise ControlContractError(
                    "training conductor response omitted aligned token IDs/log-probabilities"
                )
            token_data = {
                "prompt_token_ids": prompt_token_ids,
                "completion_token_ids": completion_token_ids,
                "completion_logprobs": [float(value) for value in completion_logprobs],
                "temperature": self._temperature,
                "seed": self._seed,
                "behavior_likelihood_contract": (
                    full_vocabulary_behavior_likelihood_contract()
                ),
            }
        self.decision_traces.append(
            {
                "messages": messages,
                "response": content,
                "finish_reason": getattr(choice, "finish_reason", None),
                "correction": correction,
                "prompt_tokens": self.last_prompt_tokens,
                **token_data,
            }
        )
        if getattr(choice, "finish_reason", None) == "length":
            raise ControlContractError(
                "control action reached the local output limit before completion "
                f"({len(content)} characters)"
            )
        if not self.supplies_topology:
            return parse_control_decision(content)
        if self.capability_refs:
            return parse_capability_control_action(
                content,
                capability_reference_map(state.workers),
            )
        return parse_control_action(content)
