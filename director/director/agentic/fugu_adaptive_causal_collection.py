"""Initial-intervention collection using the exact adaptive Fugu product."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol, override

from harbor.models.agent.context import AgentContext

from director.agentic.fugu_ultra_terminal import (
    MAX_AGENT_TURNS,
    FuguUltraTerminalAgent,
    PlannedStep,
    RouteDecision,
)
from director.agentic.prepared_index_test_protection import (
    PreparedIndexTestProtectionMixin,
)
from ultra.live_control import ControlAction, parse_control_action
from ultra.pool_binding import load_pool_binding


COLLECTION_REVISION = "20260720-adaptive-causal-initial-intervention-v1"
WORKFLOW_ENV = "FUGU_ADAPTIVE_CAUSAL_INITIAL_WORKFLOW_JSON"
COLLECTION_ID_ENV = "FUGU_ADAPTIVE_CAUSAL_COLLECTION_ID"
POOL_BINDING_ENV = "FUGU_ADAPTIVE_CAUSAL_POOL_BINDING"


class RoutePlanner(Protocol):
    def set_task_instruction(self, instruction: str) -> None: ...

    def set_unavailable_workers(self, worker_ids: frozenset[int]) -> None: ...

    async def route(
        self, prompt: str, message_history: list[dict[str, Any] | Any]
    ) -> RouteDecision: ...


class InitialInterventionPlanner:
    """Emit one frozen anonymous seed, then delegate every later replan."""

    def __init__(
        self,
        *,
        action: ControlAction,
        delegate: RoutePlanner,
        worker_ids: frozenset[int],
    ) -> None:
        if action.action != "replan" or not action.steps:
            raise ValueError("initial causal intervention must be a nonempty replan")
        if any(step.worker_id not in worker_ids for step in action.steps):
            raise ValueError("initial causal intervention selects an unknown worker slot")
        self.action = action
        self.delegate = delegate
        self.worker_ids = worker_ids
        self.calls = 0
        self.intervention_calls = 0
        self.delegate_calls = 0
        self._unavailable_worker_ids: frozenset[int] = frozenset()

    @property
    def _max_attempts(self) -> int:
        return int(getattr(self.delegate, "_max_attempts", 1))

    def set_task_instruction(self, instruction: str) -> None:
        self.delegate.set_task_instruction(instruction)

    def set_unavailable_workers(self, worker_ids: frozenset[int]) -> None:
        unknown = worker_ids - self.worker_ids
        if unknown:
            raise ValueError(f"unknown unavailable worker slots: {sorted(unknown)}")
        self._unavailable_worker_ids = worker_ids
        setter = getattr(self.delegate, "set_unavailable_workers", None)
        if callable(setter):
            setter(worker_ids)

    async def route(
        self, prompt: str, message_history: list[dict[str, Any] | Any]
    ) -> RouteDecision:
        self.calls += 1
        if self.intervention_calls:
            self.delegate_calls += 1
            return await self.delegate.route(prompt, message_history)

        self.intervention_calls += 1
        selected_unavailable = sorted(
            {step.worker_id for step in self.action.steps}
            & self._unavailable_worker_ids
        )
        if selected_unavailable:
            reason = (
                "the frozen initial intervention selected unavailable anonymous "
                f"worker slots: {selected_unavailable}"
            )
            first = self.action.steps[0]
            return RouteDecision(
                worker_id=first.worker_id,
                subtask=first.subtask,
                raw_plan=json.dumps(
                    {
                        "collection_revision": COLLECTION_REVISION,
                        "initial_intervention": True,
                        "unrecoverable": True,
                        "reason": reason,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                fallback_reason=reason,
                unrecoverable=True,
            )

        steps = tuple(
            PlannedStep(
                worker_id=step.worker_id,
                subtask=step.subtask,
                access=step.access,
            )
            for step in self.action.steps
        )
        first = steps[0]
        raw_plan = json.dumps(
            {
                "collection_revision": COLLECTION_REVISION,
                "initial_intervention": True,
                "reason": self.action.reason,
                "steps": [
                    {
                        "worker_id": step.worker_id,
                        "subtask": step.subtask,
                        "access": list(step.access),
                    }
                    for step in steps
                ],
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return RouteDecision(
            worker_id=first.worker_id,
            subtask=first.subtask,
            raw_plan=raw_plan,
            workflow_steps=steps,
        )


class FuguAdaptiveCausalCollectionAgent(
    PreparedIndexTestProtectionMixin,
    FuguUltraTerminalAgent,
):
    """Apply one causal seed and preserve the product's downstream adaptivity."""

    _sanitize_prepared_git_history = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        max_turns: int = MAX_AGENT_TURNS,
        **kwargs: Any,
    ) -> None:
        workflow_json = os.environ.get(WORKFLOW_ENV)
        if not workflow_json:
            raise RuntimeError(f"{WORKFLOW_ENV} is required")
        collection_id = os.environ.get(COLLECTION_ID_ENV)
        if not collection_id:
            raise RuntimeError(f"{COLLECTION_ID_ENV} is required")
        raw_binding_path = os.environ.get(POOL_BINDING_ENV)
        if not raw_binding_path:
            raise RuntimeError(f"{POOL_BINDING_ENV} is required")
        binding_path = Path(raw_binding_path)
        binding = load_pool_binding(binding_path)
        action = parse_control_action(workflow_json)

        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name or "fugu-adaptive-causal-collection",
            pool_binding_path=binding_path,
            worker_models=binding.runtime_models,
            reasoning_efforts=binding.reasoning_efforts,
            provider_base_url=binding.provider_base,
            max_turns=max_turns,
            provider_owner_retry_limit=0,
            **kwargs,
        )
        self._initialize_protected_test_protection()
        self._collection_id = collection_id
        self._collection_pool_binding_path = binding_path.resolve()
        self._registered_initial_action = action
        product_planner = self._planner
        planner = InitialInterventionPlanner(
            action=action,
            delegate=product_planner,
            worker_ids=frozenset(slot.worker_id for slot in binding.slots),
        )
        self._planner = planner
        self._fugu_llm._planner = planner
        self._initial_intervention_planner = planner

    @staticmethod
    @override
    def name() -> str:
        return "fugu-adaptive-causal-collection"

    @override
    def version(self) -> str | None:
        return COLLECTION_REVISION

    @override
    def _record_fugu_metadata(self, context: AgentContext) -> None:
        super()._record_fugu_metadata(context)
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "collection_revision": COLLECTION_REVISION,
                "collection_id": self._collection_id,
                "collection_registered_initial_workflow": json.loads(
                    os.environ[WORKFLOW_ENV]
                ),
                "collection_initial_intervention_calls": (
                    self._initial_intervention_planner.intervention_calls
                ),
                "collection_downstream_planner_calls": (
                    self._initial_intervention_planner.delegate_calls
                ),
                "collection_downstream_live_control_preserved": True,
                "collection_dynamic_workflows_allowed": True,
                "collection_is_product_candidate": False,
                "collection_training_eligible": False,
                "collection_training_gate": "independent_causal_pair_admission",
                "initial_workflow_controller": (
                    "frozen_anonymous_intervention_then_product_live_control"
                ),
                "worker_calls_are_paid": True,
                "pool_binding_path": str(self._collection_pool_binding_path),
                **self._protected_test_metadata(),
            }
        )
        context.metadata = metadata
