"""Capture a natural product live-control intervention before it is applied."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol, override

from harbor.models.agent.context import AgentContext

from director.agentic.fugu_live_branchpoint_collection import (
    restore_branchpoint,
    serialize_branchpoint,
    _sha256,
    _write_json_atomic,
)
from director.agentic import fugu_live_branchpoint_collection as base_collection
from director.agentic.fugu_live_branchpoint_collection_v2 import (
    BRANCHPOINT_PATH_ENV,
    BRANCH_ACTION_ENV,
    COLLECTION_ID_ENV,
    ENVIRONMENT_ACK_FILENAME,
    FuguLiveBranchpointCollectionAgentV2,
    INITIAL_WORKFLOW_ENV,
    MODE_ENV,
    POOL_BINDING_ENV,
)
from director.agentic.fugu_ultra_terminal import (
    LOCAL_LIVE_CONTROL_ADAPTER,
    LOCAL_PLANNER_BASE,
    FuguRoutedLLM,
    LocalModelPromptTokenCounter,
)
from ultra.live_control import (
    MAX_CONTROL_OUTPUT_TOKENS,
    ControlAction,
    LiveControlState,
    OpenAILiveController,
)


COLLECTION_REVISION = "20260721-live-natural-intervention-unified-conductor-v6"
CAPTURE_ACTIONS_ENV = "FUGU_NATURAL_CAPTURE_ACTIONS_JSON"


class LiveController(Protocol):
    async def decide(
        self,
        state: LiveControlState,
        *,
        correction: str | None = None,
    ) -> ControlAction: ...


def _action_payload(action: ControlAction) -> dict[str, Any]:
    return {
        "action": action.action,
        "reason": action.reason,
        "target_position_id": action.target_position_id,
        "steps": [
            {
                "worker_id": step.worker_id,
                "subtask": step.subtask,
                "access": list(step.access),
            }
            for step in action.steps
        ],
    }


def serialize_natural_intervention_branchpoint(
    engine: FuguRoutedLLM,
    state: LiveControlState,
    *,
    collection_id: str,
    pool_fingerprint: str,
    proposed_action: ControlAction,
) -> dict[str, Any]:
    workflow = engine._workflow
    if workflow is None or state.active_position_id is None:
        raise RuntimeError("natural intervention requires an active workflow")
    if state.terminal_status != "ready":
        raise RuntimeError("natural intervention requires a stable terminal")
    active = workflow.active
    original_completion = active.completion_requested
    try:
        # Reuse the exact V1 serializer while preserving an unfinished position.
        active.completion_requested = True
        payload = serialize_branchpoint(
            engine,
            state,
            collection_id=collection_id,
            pool_fingerprint=pool_fingerprint,
        )
    finally:
        active.completion_requested = original_completion
    payload["agents"][workflow.active_index]["completion_requested"] = original_completion
    payload["capture_kind"] = "natural_product_live_intervention"
    payload["natural_proposed_action"] = _action_payload(proposed_action)
    payload["natural_intervention_on_unfinished_position"] = not original_completion
    return payload


def restore_natural_intervention_branchpoint(
    engine: FuguRoutedLLM,
    payload: dict[str, Any],
    *,
    instruction: str,
    pool_fingerprint: str,
) -> None:
    if payload.get("capture_kind") != "natural_product_live_intervention":
        raise RuntimeError("branchpoint is not a natural live intervention")
    active_index = payload.get("active_index")
    agents = payload.get("agents")
    if not isinstance(active_index, int) or not isinstance(agents, list):
        raise RuntimeError("natural branchpoint shape is invalid")
    original_completion = bool(agents[active_index].get("completion_requested"))
    compatible = json.loads(json.dumps(payload))
    compatible["agents"][active_index]["completion_requested"] = True
    restore_branchpoint(
        engine,
        compatible,
        instruction=instruction,
        pool_fingerprint=pool_fingerprint,
    )
    if engine._workflow is None:
        raise RuntimeError("natural branchpoint restore lost the workflow")
    engine._workflow.active.completion_requested = original_completion


class NaturalInterventionCaptureController:
    """Freeze the active conductor's first handoff or replan proposal."""

    def __init__(
        self,
        *,
        engine: FuguRoutedLLM,
        delegate: LiveController,
        output: Any,
        collection_id: str,
        pool_fingerprint: str,
        capture_actions: frozenset[str] = frozenset({"handoff", "replan"}),
    ) -> None:
        if not capture_actions or not capture_actions <= {"handoff", "replan"}:
            raise ValueError("capture actions must be a nonempty handoff/replan subset")
        self.engine = engine
        self.delegate = delegate
        self.output = output
        self.collection_id = collection_id
        self.pool_fingerprint = pool_fingerprint
        self.capture_actions = capture_actions
        self.captured = False
        self.workspace_exported = False
        self.proposed_action: ControlAction | None = None

    async def decide(
        self,
        state: LiveControlState,
        *,
        correction: str | None = None,
    ) -> ControlAction:
        decision = await self.delegate.decide(state, correction=correction)
        if (
            not self.captured
            and decision.action in self.capture_actions
            and state.active_position_id is not None
            and state.terminal_status == "ready"
        ):
            payload = serialize_natural_intervention_branchpoint(
                self.engine,
                state,
                collection_id=self.collection_id,
                pool_fingerprint=self.pool_fingerprint,
                proposed_action=decision,
            )
            _write_json_atomic(self.output, payload)
            self.captured = True
            self.proposed_action = decision
            return ControlAction(
                action="complete",
                reason="Captured the product live head's natural intervention before application.",
            )
        return decision


class FuguNaturalInterventionCollectionAgentV3(FuguLiveBranchpointCollectionAgentV2):
    """Collect a natural conductor intervention and exact pre-action environment."""

    @staticmethod
    @override
    def _allow_registered_natural_action() -> bool:
        return True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        original_modes = base_collection.MODES
        if os.environ.get(MODE_ENV) == "handoff":
            base_collection.MODES = frozenset((*original_modes, "handoff"))
        try:
            super().__init__(*args, **kwargs)
        finally:
            base_collection.MODES = original_modes
        if self._collection_mode == "natural" and self._registered_action is not None:
            proposed = (self._branchpoint_payload or {}).get(
                "natural_proposed_action"
            )
            if not isinstance(proposed, dict):
                raise RuntimeError(
                    "captured-action replay requires a natural proposal in the snapshot"
                )
            if _action_payload(self._registered_action) != proposed:
                raise RuntimeError(
                    "registered natural action differs from the captured proposal"
                )
        if self._collection_mode == "prefix":
            raw_capture_actions = os.environ.get(CAPTURE_ACTIONS_ENV)
            capture_actions = frozenset({"handoff", "replan"})
            if raw_capture_actions is not None:
                parsed_capture_actions = json.loads(raw_capture_actions)
                if not isinstance(parsed_capture_actions, list) or any(
                    not isinstance(item, str) for item in parsed_capture_actions
                ):
                    raise RuntimeError("capture actions must be a JSON string list")
                capture_actions = frozenset(parsed_capture_actions)
            typed_model = self._typed_conductor_model
            typed_url = self._typed_conductor_url
            product_controller = OpenAILiveController(
                model=typed_model or LOCAL_LIVE_CONTROL_ADAPTER,
                base_url=typed_url if typed_model else LOCAL_PLANNER_BASE,
                max_tokens=MAX_CONTROL_OUTPUT_TOKENS if typed_model else 64,
                seed=0,
                supplies_topology=bool(typed_model),
                capability_refs=bool(typed_model),
                prompt_token_counter=LocalModelPromptTokenCounter(
                    model=typed_model or LOCAL_LIVE_CONTROL_ADAPTER,
                    models_url=(
                        f"{typed_url.rstrip('/')}/models"
                        if typed_model
                        else f"{LOCAL_PLANNER_BASE.rstrip('/')}/models"
                    ),
                ),
            )
            controller = NaturalInterventionCaptureController(
                engine=self._fugu_llm,
                delegate=product_controller,
                output=self._branchpoint_path,
                collection_id=self._collection_id,
                pool_fingerprint=self._pool_binding.pool_fingerprint,
                capture_actions=capture_actions,
            )
            self._fugu_llm._live_controller = controller
            # Prefix mode has an already frozen initial action installed by the
            # base collector. Do not let a topology-supplying controller bypass
            # that planner and resample the task's initial workflow.
            self._fugu_llm._live_controller_plans_initial_workflow = False
            self._fugu_llm._live_controller_supplies_topology = bool(typed_model)
            self._capture_controller = controller
            self._captured_conductor_source = (
                "unified_capability_action_v2" if typed_model else "accepted_live_control_v16"
            )

    @staticmethod
    @override
    def name() -> str:
        return "fugu-natural-intervention-collection-v3"

    @override
    def version(self) -> str | None:
        return COLLECTION_REVISION

    @override
    def _after_fugu_run_reset(self, instruction: str) -> None:
        if self._branchpoint_payload is None:
            return
        restore_natural_intervention_branchpoint(
            self._fugu_llm,
            self._branchpoint_payload,
            instruction=instruction,
            pool_fingerprint=self._pool_binding.pool_fingerprint,
        )

    @override
    def _record_fugu_metadata(self, context: AgentContext) -> None:
        super()._record_fugu_metadata(context)
        metadata = dict(context.metadata or {})
        proposed = getattr(self._capture_controller, "proposed_action", None)
        metadata.update(
            {
                "collection_revision": COLLECTION_REVISION,
                "natural_live_intervention": (
                    _action_payload(proposed) if proposed is not None else None
                ),
                "natural_live_intervention_captured": proposed is not None,
                "natural_live_intervention_source": (
                    self._captured_conductor_source if proposed is not None else None
                ),
            }
        )
        context.metadata = metadata


__all__ = [
    "BRANCHPOINT_PATH_ENV",
    "CAPTURE_ACTIONS_ENV",
    "BRANCH_ACTION_ENV",
    "COLLECTION_ID_ENV",
    "COLLECTION_REVISION",
    "ENVIRONMENT_ACK_FILENAME",
    "FuguNaturalInterventionCollectionAgentV3",
    "INITIAL_WORKFLOW_ENV",
    "MODE_ENV",
    "POOL_BINDING_ENV",
    "NaturalInterventionCaptureController",
    "restore_natural_intervention_branchpoint",
    "serialize_natural_intervention_branchpoint",
]
