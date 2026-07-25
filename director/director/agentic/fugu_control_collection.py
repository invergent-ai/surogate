"""Fixed-topology Fugu agent for current-pool live-control data collection.

This is not a product candidate. It collects verifier-backed, shared-workspace
trajectories on train-allowed tasks so the next conductor can learn handoffs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from harbor.models.agent.context import AgentContext

from director.agentic.fugu_ultra_terminal import (
    DEFAULT_WORKER_MODELS,
    FuguUltraTerminalAgent,
    PlannedStep,
    RouteDecision,
)
from ultra.live_control import ControlAction, ControlContractError, parse_control_action


COLLECTION_REVISION = "20260716-live-control-current-pool-v1"
WORKFLOW_ENV = "FUGU_COLLECTION_WORKFLOW_JSON"
COLLECTION_ID_ENV = "FUGU_COLLECTION_ID"


class FixedWorkflowPlanner:
    """Return one preregistered topology without ranking or substituting workers."""

    def __init__(self, action: ControlAction) -> None:
        if action.action != "replan" or not action.steps:
            raise ControlContractError("collection workflow must be a replan action with steps")
        if any(step.worker_id not in range(len(DEFAULT_WORKER_MODELS)) for step in action.steps):
            raise ControlContractError("collection workflow selected a worker outside the frozen pool")
        self.action = action
        self.calls = 0
        self._original_instruction = ""
        self._unavailable_worker_ids: frozenset[int] = frozenset()

    @classmethod
    def from_json(cls, content: str) -> "FixedWorkflowPlanner":
        return cls(parse_control_action(content))

    def set_task_instruction(self, instruction: str) -> None:
        self._original_instruction = instruction

    def set_unavailable_workers(self, worker_ids: frozenset[int]) -> None:
        unknown = worker_ids - set(range(len(DEFAULT_WORKER_MODELS)))
        if unknown:
            raise ControlContractError(
                f"collection runtime exposed unknown unavailable workers: {sorted(unknown)}"
            )
        self._unavailable_worker_ids = worker_ids

    async def route(self, prompt: str, message_history: list[dict[str, Any] | Any]) -> RouteDecision:
        del prompt, message_history
        self.calls += 1
        steps = tuple(
            PlannedStep(
                worker_id=step.worker_id,
                subtask=step.subtask,
                access=step.access,
            )
            for step in self.action.steps
        )
        selected_unavailable = sorted(
            {step.worker_id for step in steps} & self._unavailable_worker_ids
        )
        if selected_unavailable:
            reason = (
                "the preregistered fixed workflow has no legal recovery after "
                f"worker slots became unavailable: {selected_unavailable}"
            )
            return RouteDecision(
                worker_id=steps[0].worker_id,
                subtask=steps[0].subtask,
                raw_plan=json.dumps(
                    {
                        "collection_revision": COLLECTION_REVISION,
                        "unrecoverable": True,
                        "reason": reason,
                    },
                    ensure_ascii=True,
                ),
                fallback_reason=reason,
                unrecoverable=True,
            )
        first = steps[0]
        return RouteDecision(
            worker_id=first.worker_id,
            subtask=first.subtask,
            raw_plan=json.dumps(
                {
                    "collection_revision": COLLECTION_REVISION,
                    "reason": self.action.reason,
                    "steps": [
                        {
                            "worker_id": step.worker_id,
                            "subtask": step.subtask,
                            "access": list(step.access),
                        }
                        for step in self.action.steps
                    ],
                },
                ensure_ascii=True,
            ),
            workflow_steps=steps,
        )


class FuguControlCollectionAgent(FuguUltraTerminalAgent):
    """Run a preregistered current-pool topology in one persistent terminal."""

    def __init__(self, logs_dir: Path, model_name: str | None = None, **kwargs: Any) -> None:
        workflow_json = os.environ.get(WORKFLOW_ENV)
        if not workflow_json:
            raise RuntimeError(f"{WORKFLOW_ENV} is required for control collection")
        planner = FixedWorkflowPlanner.from_json(workflow_json)
        self._collection_id = os.environ.get(COLLECTION_ID_ENV, "unregistered")
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._planner = planner
        self._fugu_llm._planner = planner

    @staticmethod
    def name() -> str:
        return "fugu-control-collection"

    def version(self) -> str | None:
        return COLLECTION_REVISION

    def _record_fugu_metadata(self, context: AgentContext) -> None:
        super()._record_fugu_metadata(context)
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "collection_revision": COLLECTION_REVISION,
                "collection_id": self._collection_id,
                "collection_fixed_workflow": json.loads(os.environ[WORKFLOW_ENV]),
                "collection_is_product_candidate": False,
            }
        )
        context.metadata = metadata
