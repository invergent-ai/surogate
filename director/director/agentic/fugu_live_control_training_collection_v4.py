"""Recovery-aware, fail-safe current-pool live-control trajectory collection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, override

from harbor.models.agent.context import AgentContext

from director.agentic.fugu_live_control_training_collection_v3 import (
    WORKFLOW_ENV,
    FuguLiveControlTrainingCollectionAgentV3,
)
from director.agentic.fugu_ultra_terminal import (
    DEFAULT_WORKER_MODELS,
    PlannedStep,
    RouteDecision,
)
from ultra.live_control import ControlAction, ControlContractError, parse_control_action


COLLECTION_REVISION = "20260717-live-control-current-pool-v4"


@dataclass(frozen=True)
class RegisteredRecovery:
    recovery_id: str
    unavailable_worker_ids: frozenset[int]
    action: ControlAction


def _parse_replan(raw: Any, label: str) -> ControlAction:
    action = parse_control_action(json.dumps(raw, ensure_ascii=True))
    if action.action != "replan" or not action.steps:
        raise ControlContractError(f"{label} must be a non-empty replan action")
    worker_ids = {step.worker_id for step in action.steps}
    if any(
        worker_id not in range(len(DEFAULT_WORKER_MODELS)) for worker_id in worker_ids
    ):
        raise ControlContractError(f"{label} selected a worker outside the bound pool")
    return action


class RegisteredRecoveryWorkflowPlanner:
    """Select only the registered primary or exact availability-specific recovery."""

    def __init__(self, plan: dict[str, Any]) -> None:
        if set(plan) != {"primary", "recoveries"}:
            raise ControlContractError(
                "registered plan must contain primary and recoveries"
            )
        self.primary = _parse_replan(plan["primary"], "primary")
        raw_recoveries = plan["recoveries"]
        if not isinstance(raw_recoveries, list) or not raw_recoveries:
            raise ControlContractError("registered plan requires recovery entries")
        recoveries: dict[frozenset[int], RegisteredRecovery] = {}
        ids: set[str] = set()
        for index, raw in enumerate(raw_recoveries):
            if not isinstance(raw, dict) or set(raw) != {
                "recovery_id",
                "unavailable_worker_ids",
                "action",
            }:
                raise ControlContractError(f"recoveries[{index}] has an invalid schema")
            recovery_id = raw["recovery_id"]
            unavailable = raw["unavailable_worker_ids"]
            if not isinstance(recovery_id, str) or not recovery_id.strip():
                raise ControlContractError(
                    f"recoveries[{index}].recovery_id is required"
                )
            if recovery_id in ids:
                raise ControlContractError("recovery_id values must be unique")
            if not isinstance(unavailable, list) or not unavailable:
                raise ControlContractError(
                    f"recoveries[{index}].unavailable_worker_ids must be non-empty"
                )
            if any(
                isinstance(worker_id, bool)
                or not isinstance(worker_id, int)
                or worker_id not in range(len(DEFAULT_WORKER_MODELS))
                for worker_id in unavailable
            ):
                raise ControlContractError(
                    f"recoveries[{index}] has an invalid worker ID"
                )
            unavailable_set = frozenset(unavailable)
            if len(unavailable_set) != len(unavailable):
                raise ControlContractError(
                    f"recoveries[{index}] repeats an unavailable worker"
                )
            if unavailable_set in recoveries:
                raise ControlContractError(
                    "only one recovery may match an availability state"
                )
            action = _parse_replan(raw["action"], f"recoveries[{index}].action")
            if any(step.worker_id in unavailable_set for step in action.steps):
                raise ControlContractError(
                    f"recoveries[{index}] selects a worker declared unavailable"
                )
            recoveries[unavailable_set] = RegisteredRecovery(
                recovery_id=recovery_id,
                unavailable_worker_ids=unavailable_set,
                action=action,
            )
            ids.add(recovery_id)
        self.recoveries = recoveries
        self.calls = 0
        self._unavailable_worker_ids: frozenset[int] = frozenset()
        self._selected_availability_states: set[frozenset[int]] = set()
        self.selection_history: list[dict[str, Any]] = []

    @classmethod
    def from_json(cls, content: str) -> "RegisteredRecoveryWorkflowPlanner":
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ControlContractError(
                f"registered recovery plan is invalid JSON: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise ControlContractError("registered recovery plan must be an object")
        return cls(raw)

    def set_task_instruction(self, instruction: str) -> None:
        del instruction

    def set_unavailable_workers(self, worker_ids: frozenset[int]) -> None:
        if any(
            worker_id not in range(len(DEFAULT_WORKER_MODELS))
            for worker_id in worker_ids
        ):
            raise ControlContractError(
                "runtime exposed an unavailable worker outside the bound pool"
            )
        if not self._unavailable_worker_ids.issubset(worker_ids):
            raise ControlContractError(
                "runtime availability cannot restore a failed worker mid-task"
            )
        self._unavailable_worker_ids = worker_ids

    @staticmethod
    def _steps(action: ControlAction) -> tuple[PlannedStep, ...]:
        return tuple(
            PlannedStep(
                worker_id=step.worker_id, subtask=step.subtask, access=step.access
            )
            for step in action.steps
        )

    async def route(
        self,
        prompt: str,
        message_history: list[dict[str, Any] | Any],
    ) -> RouteDecision:
        del prompt, message_history
        self.calls += 1
        unavailable = self._unavailable_worker_ids
        if unavailable in self._selected_availability_states:
            reason = (
                "the registered topology for unavailable worker slots "
                f"{sorted(unavailable)} was already attempted"
            )
            self.selection_history.append(
                {
                    "call": self.calls,
                    "plan_id": None,
                    "unavailable_worker_ids": sorted(unavailable),
                    "outcome": "unrecoverable",
                    "reason": reason,
                }
            )
            return RouteDecision(-1, reason, fallback_reason=reason, unrecoverable=True)

        if not unavailable:
            plan_id = "primary"
            action = self.primary
        else:
            recovery = self.recoveries.get(unavailable)
            if recovery is None:
                reason = (
                    "no preregistered recovery exists for unavailable worker slots "
                    f"{sorted(unavailable)}"
                )
                self.selection_history.append(
                    {
                        "call": self.calls,
                        "plan_id": None,
                        "unavailable_worker_ids": sorted(unavailable),
                        "outcome": "unrecoverable",
                        "reason": reason,
                    }
                )
                return RouteDecision(
                    -1, reason, fallback_reason=reason, unrecoverable=True
                )
            plan_id = recovery.recovery_id
            action = recovery.action

        self._selected_availability_states.add(unavailable)
        steps = self._steps(action)
        raw_plan = json.dumps(
            {
                "collection_revision": COLLECTION_REVISION,
                "plan_id": plan_id,
                "unavailable_worker_ids": sorted(unavailable),
                "reason": action.reason,
                "steps": [
                    {
                        "worker_id": step.worker_id,
                        "subtask": step.subtask,
                        "access": list(step.access),
                    }
                    for step in action.steps
                ],
            },
            ensure_ascii=True,
        )
        self.selection_history.append(
            {
                "call": self.calls,
                "plan_id": plan_id,
                "unavailable_worker_ids": sorted(unavailable),
                "outcome": "selected",
            }
        )
        return RouteDecision(
            worker_id=steps[0].worker_id,
            subtask=steps[0].subtask,
            raw_plan=raw_plan,
            workflow_steps=steps,
        )


class FuguLiveControlTrainingCollectionAgentV4(
    FuguLiveControlTrainingCollectionAgentV3
):
    """Collect registered live recovery traces without provider or task retries."""

    @classmethod
    def _planner_from_json(cls, content: str) -> RegisteredRecoveryWorkflowPlanner:
        return RegisteredRecoveryWorkflowPlanner.from_json(content)

    @staticmethod
    def name() -> str:
        return "fugu-live-control-training-collection-v4"

    def version(self) -> str | None:
        return COLLECTION_REVISION

    @override
    def _record_fugu_metadata(self, context: AgentContext) -> None:
        super()._record_fugu_metadata(context)
        metadata = dict(context.metadata or {})
        planner = self._planner
        metadata.update(
            {
                "collection_revision": COLLECTION_REVISION,
                "collection_id": self._collection_id,
                "collection_registered_workflow": json.loads(os.environ[WORKFLOW_ENV]),
                "collection_training_contract": "live_control_recovery_v2",
                "registered_recovery_selection_history": planner.selection_history,
                "provider_failure_is_replan_signal": True,
                "provider_request_retries": 0,
            }
        )
        context.metadata = metadata
