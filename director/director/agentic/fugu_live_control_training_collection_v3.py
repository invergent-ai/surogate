"""Fail-safe current-pool trajectory collection for agentic coding tasks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, override

from harbor.models.agent.context import AgentContext

from director.agentic.prepared_index_test_protection import (
    PAGER_SETUP,
    PreparedIndexTestProtectionMixin,
    is_benchmark_owned_test_path,
    protected_test_restore_script,
    protected_test_snapshot_script,
)
from director.agentic.fugu_ultra_terminal import (
    DEFAULT_WORKER_MODELS,
    FuguUltraTerminalAgent,
    PlannedStep,
    RouteDecision,
)
from ultra.live_control import ControlAction, ControlContractError, parse_control_action


COLLECTION_REVISION = "20260717-live-control-current-pool-v3"
WORKFLOW_ENV = "FUGU_LIVE_TRAINING_WORKFLOW_JSON"
COLLECTION_ID_ENV = "FUGU_LIVE_TRAINING_COLLECTION_ID"


class RegisteredWorkflowPlanner:
    """Execute one preregistered pool-bound workflow without worker substitution."""

    def __init__(self, action: ControlAction) -> None:
        if action.action != "replan" or not action.steps:
            raise ControlContractError(
                "training workflow must be a replan action with steps"
            )
        if any(
            step.worker_id not in range(len(DEFAULT_WORKER_MODELS))
            for step in action.steps
        ):
            raise ControlContractError(
                "training workflow selected a worker outside the bound pool"
            )
        self.action = action
        self.calls = 0

    @classmethod
    def from_json(cls, content: str) -> "RegisteredWorkflowPlanner":
        return cls(parse_control_action(content))

    def set_task_instruction(self, instruction: str) -> None:
        del instruction

    async def route(
        self, prompt: str, message_history: list[dict[str, Any] | Any]
    ) -> RouteDecision:
        del prompt, message_history
        self.calls += 1
        steps = tuple(
            PlannedStep(
                worker_id=step.worker_id, subtask=step.subtask, access=step.access
            )
            for step in self.action.steps
        )
        return RouteDecision(
            worker_id=steps[0].worker_id,
            subtask=steps[0].subtask,
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


class FuguLiveControlTrainingCollectionAgentV3(
    PreparedIndexTestProtectionMixin, FuguUltraTerminalAgent
):
    """Collect one trace while preserving benchmark-owned test inputs."""

    @classmethod
    def _planner_from_json(cls, content: str) -> RegisteredWorkflowPlanner:
        return RegisteredWorkflowPlanner.from_json(content)

    def __init__(
        self, logs_dir: Path, model_name: str | None = None, **kwargs: Any
    ) -> None:
        workflow_json = os.environ.get(WORKFLOW_ENV)
        if not workflow_json:
            raise RuntimeError(f"{WORKFLOW_ENV} is required")
        planner = self._planner_from_json(workflow_json)
        self._collection_id = os.environ.get(COLLECTION_ID_ENV, "unregistered")
        self._initialize_protected_test_protection()
        kwargs.setdefault("provider_owner_retry_limit", 0)
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._planner = planner
        self._fugu_llm._planner = planner

    @staticmethod
    def name() -> str:
        return "fugu-live-control-training-collection-v3"

    def version(self) -> str | None:
        return COLLECTION_REVISION

    @override
    def _record_fugu_metadata(self, context: AgentContext) -> None:
        FuguUltraTerminalAgent._record_fugu_metadata(self, context)
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "collection_revision": COLLECTION_REVISION,
                "collection_id": self._collection_id,
                "collection_registered_workflow": json.loads(os.environ[WORKFLOW_ENV]),
                "collection_is_product_candidate": False,
                "collection_training_contract": "live_control_v1",
                **self._protected_test_metadata(),
            }
        )
        context.metadata = metadata
