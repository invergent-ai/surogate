"""Same-state live-action collection for the Fugu conductor.

The prefix mode holds one anonymous owner until it requests completion, then
freezes the private owner trajectory and the pre-verifier workspace. Branch
modes restore that exact state and force one registered live action before
returning control to the unchanged product conductor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, override

from harbor.llms.base import LLMResponse
from harbor.models.agent.context import AgentContext

from director.agentic.fugu_adaptive_causal_collection import InitialInterventionPlanner
from director.agentic.fugu_ultra_terminal import (
    MAX_AGENT_TURNS,
    ActiveRoute,
    FuguRoutedLLM,
    FuguUltraTerminalAgent,
    PlannedStep,
    RouteDecision,
    WorkflowAgentState,
    WorkflowExecutionState,
)
from director.agentic.prepared_index_test_protection import PreparedIndexTestProtectionMixin
from ultra.live_control import ControlAction, LiveControlState, parse_control_action
from ultra.pool_binding import load_pool_binding


COLLECTION_REVISION = "20260720-live-branchpoint-causal-v1"
MODE_ENV = "FUGU_LIVE_BRANCHPOINT_MODE"
COLLECTION_ID_ENV = "FUGU_LIVE_BRANCHPOINT_COLLECTION_ID"
POOL_BINDING_ENV = "FUGU_LIVE_BRANCHPOINT_POOL_BINDING"
INITIAL_WORKFLOW_ENV = "FUGU_LIVE_BRANCHPOINT_INITIAL_WORKFLOW_JSON"
BRANCH_ACTION_ENV = "FUGU_LIVE_BRANCHPOINT_ACTION_JSON"
BRANCHPOINT_PATH_ENV = "FUGU_LIVE_BRANCHPOINT_PATH"
SNAPSHOT_VERSION = "fugu_live_branchpoint_snapshot_v1"
MODES = frozenset({"prefix", "continue", "replan", "natural"})


class LiveController(Protocol):
    async def decide(
        self,
        state: LiveControlState,
        *,
        correction: str | None = None,
    ) -> ControlAction: ...


class RoutePlanner(Protocol):
    def set_task_instruction(self, instruction: str) -> None: ...

    def set_unavailable_workers(self, worker_ids: frozenset[int]) -> None: ...

    async def route(
        self, prompt: str, message_history: list[dict[str, Any] | Any]
    ) -> RouteDecision: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _route_payload(route: ActiveRoute) -> dict[str, Any]:
    return {
        "workflow_id": route.workflow_id,
        "step_index": route.step_index,
        "step_count": route.step_count,
        "access": list(route.access),
        "decision": {
            "worker_id": route.decision.worker_id,
            "subtask": route.decision.subtask,
            "raw_plan": route.decision.raw_plan,
        },
        "workflow_steps": [asdict(step) for step in route.workflow_steps],
    }


def serialize_branchpoint(
    engine: FuguRoutedLLM,
    state: LiveControlState,
    *,
    collection_id: str,
    pool_fingerprint: str,
) -> dict[str, Any]:
    workflow = engine._workflow
    if workflow is None or state.active_position_id is None:
        raise RuntimeError("branchpoint capture requires an active workflow")
    if workflow.active_index != state.active_position_id:
        raise RuntimeError("runtime and live-state active positions differ")
    active = workflow.active
    if not active.completion_requested or state.terminal_status != "ready":
        raise RuntimeError("branchpoint must be a stable owner completion request")
    if engine.provider_failure_events:
        raise RuntimeError("invalid provider state cannot be forked")
    # Transient worker protocol slips that the runtime already recovered from
    # do not contaminate the fork; only UNRESOLVED protocol state at the
    # capture boundary does. Two consecutive campaigns were ended by the
    # previous zero-errors-ever precondition while the product runtime itself
    # tolerated the same recovered slips.
    if any(agent.consecutive_protocol_errors for agent in workflow.agents):
        raise RuntimeError("unresolved worker protocol state cannot be forked")

    agents = []
    for agent in workflow.agents:
        agents.append(
            {
                "route": _route_payload(agent.route),
                "messages": agent.messages,
                "turns": agent.turns,
                "status": agent.status,
                "final_response": agent.final_response,
                "completion_requested": agent.completion_requested,
                "consecutive_protocol_errors": agent.consecutive_protocol_errors,
                "handoff_requested": agent.handoff_requested,
                "terminal_ready": agent.terminal_ready,
                "handoff_reason": agent.handoff_reason,
                "checkpoint": agent.checkpoint,
                "progress": agent.progress,
                "artifacts": agent.artifacts,
                "recent_activity": agent.recent_activity,
            }
        )
    return {
        "version": SNAPSHOT_VERSION,
        "collection_id": collection_id,
        "pool_fingerprint": pool_fingerprint,
        "task_instruction_sha256": hashlib.sha256(engine._task_instruction.encode()).hexdigest(),
        "workflow_id": workflow.workflow_id,
        "active_index": workflow.active_index,
        "agents": agents,
        "shared_workflows": engine._without_runtime_identities(engine._shared_workflows),
        "prefix_paid_worker_calls": engine.paid_worker_call_attempts,
        "prefix_live_control_decisions": engine.live_control_decisions,
        "terminal_observation": state.terminal_observation,
        "workspace": None,
    }


def _planned_steps(rows: Any) -> tuple[PlannedStep, ...]:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("branchpoint route has no workflow steps")
    steps = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"worker_id", "subtask", "access"}:
            raise RuntimeError("branchpoint workflow step is malformed")
        access = tuple(row["access"])
        if any(not isinstance(value, int) or value < 0 or value >= index for value in access):
            raise RuntimeError("branchpoint access list is not acyclic")
        steps.append(PlannedStep(int(row["worker_id"]), str(row["subtask"]), access))
    return tuple(steps)


def restore_branchpoint(
    engine: FuguRoutedLLM,
    payload: dict[str, Any],
    *,
    instruction: str,
    pool_fingerprint: str,
) -> None:
    if payload.get("version") != SNAPSHOT_VERSION:
        raise RuntimeError("unsupported live branchpoint snapshot")
    if payload.get("pool_fingerprint") != pool_fingerprint:
        raise RuntimeError("branchpoint pool fingerprint mismatch")
    if payload.get("task_instruction_sha256") != hashlib.sha256(instruction.encode()).hexdigest():
        raise RuntimeError("branchpoint task instruction mismatch")
    raw_agents = payload.get("agents")
    active_index = payload.get("active_index")
    if (
        not isinstance(raw_agents, list)
        or not raw_agents
        or isinstance(active_index, bool)
        or not isinstance(active_index, int)
        or active_index < 0
        or active_index >= len(raw_agents)
    ):
        raise RuntimeError("branchpoint workflow shape is invalid")

    agents: list[WorkflowAgentState] = []
    for raw in raw_agents:
        if not isinstance(raw, dict) or not isinstance(raw.get("route"), dict):
            raise RuntimeError("branchpoint agent is malformed")
        route = raw["route"]
        decision = route.get("decision")
        if not isinstance(decision, dict):
            raise RuntimeError("branchpoint route decision is malformed")
        steps = _planned_steps(route.get("workflow_steps"))
        worker_id = int(decision["worker_id"])
        if worker_id not in engine._workers or route.get("step_count") != len(steps):
            raise RuntimeError("branchpoint route selects an unavailable worker")
        active_route = ActiveRoute(
            decision=RouteDecision(
                worker_id=worker_id,
                subtask=str(decision["subtask"]),
                raw_plan=str(decision.get("raw_plan") or ""),
            ),
            workflow_id=int(route["workflow_id"]),
            step_index=int(route["step_index"]),
            step_count=int(route["step_count"]),
            access=tuple(route["access"]),
            workflow_steps=steps,
        )
        messages = raw.get("messages")
        if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
            raise RuntimeError("branchpoint private trajectory is malformed")
        agents.append(
            WorkflowAgentState(
                route=active_route,
                messages=messages,
                turns=int(raw.get("turns") or 0),
                status=str(raw.get("status") or "running"),
                final_response=raw.get("final_response"),
                completion_requested=bool(raw.get("completion_requested")),
                consecutive_protocol_errors=int(raw.get("consecutive_protocol_errors") or 0),
                handoff_requested=bool(raw.get("handoff_requested")),
                terminal_ready=raw.get("terminal_ready"),
                handoff_reason=raw.get("handoff_reason"),
                checkpoint=raw.get("checkpoint"),
                progress=raw.get("progress"),
                artifacts=list(raw.get("artifacts") or []),
                recent_activity=list(raw.get("recent_activity") or []),
            )
        )
    workflow_id = int(payload["workflow_id"])
    engine._workflow = WorkflowExecutionState(
        workflow_id=workflow_id,
        agents=agents,
        active_index=active_index,
    )
    engine._workflow_id = workflow_id
    engine._shared_workflows = list(payload.get("shared_workflows") or [])
    if not engine._workflow.active.completion_requested:
        raise RuntimeError("restored branchpoint lost its owner completion request")


class CaptureOwnerCompletionController:
    """Hold the owner until its first stable completion request, then freeze it."""

    def __init__(
        self,
        *,
        engine: FuguRoutedLLM,
        output: Path,
        collection_id: str,
        pool_fingerprint: str,
    ) -> None:
        self.engine = engine
        self.output = output
        self.collection_id = collection_id
        self.pool_fingerprint = pool_fingerprint
        self.captured = False
        self.workspace_exported = False

    async def decide(
        self,
        state: LiveControlState,
        *,
        correction: str | None = None,
    ) -> ControlAction:
        if correction is not None:
            raise RuntimeError("registered prefix action unexpectedly required correction")
        active = state.active_position
        completion_requested = bool(
            active is not None
            and isinstance(active.progress, dict)
            and active.progress.get("completion_requested")
        )
        if completion_requested and state.terminal_status == "ready":
            payload = serialize_branchpoint(
                self.engine,
                state,
                collection_id=self.collection_id,
                pool_fingerprint=self.pool_fingerprint,
            )
            _write_json_atomic(self.output, payload)
            self.captured = True
            return ControlAction(
                action="complete",
                reason="Captured the registered stable owner branchpoint before verification.",
            )
        return ControlAction(
            action="continue",
            reason="Preserve the registered common-prefix owner until a stable completion request.",
        )


class ForcedFirstDecisionController:
    """Apply one registered branch action, then restore product live control."""

    def __init__(self, action: ControlAction, delegate: LiveController) -> None:
        self.action = action
        self.delegate = delegate
        self.calls = 0

    async def decide(
        self,
        state: LiveControlState,
        *,
        correction: str | None = None,
    ) -> ControlAction:
        if self.calls == 0:
            if correction is not None:
                raise RuntimeError("registered first branch action required correction")
            self.calls += 1
            return self.action
        self.calls += 1
        return await self.delegate.decide(state, correction=correction)


class FirstReplacementPlanner:
    """Provide one registered replacement topology, then restore product planning."""

    def __init__(self, action: ControlAction, delegate: RoutePlanner) -> None:
        if action.action != "replan" or not action.steps:
            raise ValueError("replacement planner requires a nonempty replan action")
        self.action = action
        self.delegate = delegate
        self.calls = 0

    @property
    def _max_attempts(self) -> int:
        return int(getattr(self.delegate, "_max_attempts", 1))

    def set_task_instruction(self, instruction: str) -> None:
        self.delegate.set_task_instruction(instruction)

    def set_unavailable_workers(self, worker_ids: frozenset[int]) -> None:
        setter = getattr(self.delegate, "set_unavailable_workers", None)
        if callable(setter):
            setter(worker_ids)

    async def route(
        self, prompt: str, message_history: list[dict[str, Any] | Any]
    ) -> RouteDecision:
        if self.calls:
            self.calls += 1
            return await self.delegate.route(prompt, message_history)
        self.calls += 1
        steps = tuple(
            PlannedStep(step.worker_id, step.subtask, step.access)
            for step in self.action.steps
        )
        first = steps[0]
        return RouteDecision(
            worker_id=first.worker_id,
            subtask=first.subtask,
            raw_plan=json.dumps(
                {
                    "collection_revision": COLLECTION_REVISION,
                    "registered_live_replacement": True,
                    "reason": self.action.reason,
                    "steps": [asdict(step) for step in steps],
                },
                sort_keys=True,
            ),
            workflow_steps=steps,
        )


class FuguLiveBranchpointCollectionAgent(
    PreparedIndexTestProtectionMixin,
    FuguUltraTerminalAgent,
):
    """Collect a common prefix or one same-state live-action branch."""

    _sanitize_prepared_git_history = True

    @staticmethod
    def _allow_registered_natural_action() -> bool:
        """Keep ordinary natural arms policy-sampled unless a subclass opts in."""
        return False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        max_turns: int = MAX_AGENT_TURNS,
        **kwargs: Any,
    ) -> None:
        mode = os.environ.get(MODE_ENV)
        collection_id = os.environ.get(COLLECTION_ID_ENV)
        binding_path_raw = os.environ.get(POOL_BINDING_ENV)
        if mode not in MODES or not collection_id or not binding_path_raw:
            raise RuntimeError("live branchpoint collection environment is incomplete")
        binding_path = Path(binding_path_raw)
        binding = load_pool_binding(binding_path)
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name or "fugu-live-branchpoint-collection",
            pool_binding_path=binding_path,
            worker_models=binding.runtime_models,
            reasoning_efforts=binding.reasoning_efforts,
            provider_base_url=binding.provider_base,
            max_turns=max_turns,
            provider_owner_retry_limit=0,
            **kwargs,
        )
        self._initialize_protected_test_protection()
        self._collection_mode = mode
        self._collection_id = collection_id
        self._collection_binding_path = binding_path.resolve()
        self._branchpoint_path = Path(
            os.environ.get(BRANCHPOINT_PATH_ENV) or (logs_dir / "branchpoint.json")
        )
        self._branchpoint_payload: dict[str, Any] | None = None
        self._registered_action: ControlAction | None = None
        self._capture_controller: CaptureOwnerCompletionController | None = None

        if mode == "prefix":
            raw_workflow = os.environ.get(INITIAL_WORKFLOW_ENV)
            if not raw_workflow:
                raise RuntimeError(f"{INITIAL_WORKFLOW_ENV} is required in prefix mode")
            action = parse_control_action(raw_workflow)
            planner = InitialInterventionPlanner(
                action=action,
                delegate=self._planner,
                worker_ids=frozenset(slot.worker_id for slot in binding.slots),
            )
            self._planner = planner
            self._fugu_llm._planner = planner
            controller = CaptureOwnerCompletionController(
                engine=self._fugu_llm,
                output=self._branchpoint_path,
                collection_id=collection_id,
                pool_fingerprint=binding.pool_fingerprint,
            )
            self._fugu_llm._live_controller = controller
            self._capture_controller = controller
        elif mode == "natural":
            # The counterfactual arm: restore the exact branchpoint and let the
            # unmodified product live controller choose every action. A
            # specialized collector may instead replay an action captured
            # before the fork; generic natural arms must remain policy-sampled.
            raw_action = os.environ.get(BRANCH_ACTION_ENV)
            if raw_action and not self._allow_registered_natural_action():
                raise RuntimeError("natural mode must not register a first action")
            if not self._branchpoint_path.is_file():
                raise RuntimeError("natural mode requires a branchpoint snapshot")
            self._branchpoint_payload = json.loads(
                self._branchpoint_path.read_text(encoding="utf-8")
            )
            product_controller = self._fugu_llm._live_controller
            if product_controller is None:
                raise RuntimeError("product live controller is unavailable")
            if raw_action:
                replay_payload = json.loads(raw_action)
                if replay_payload.get("steps") == []:
                    replay_payload.pop("steps")
                action = parse_control_action(
                    json.dumps(replay_payload, ensure_ascii=True)
                )
                if action.action not in {"handoff", "replan"}:
                    raise RuntimeError(
                        "captured natural action must be handoff or replan"
                    )
                self._registered_action = action
                forced_action = action
                if action.action == "replan":
                    planner = FirstReplacementPlanner(action, self._planner)
                    self._planner = planner
                    self._fugu_llm._planner = planner
                    forced_action = ControlAction(
                        action="replan", reason=action.reason
                    )
                self._fugu_llm._live_controller = ForcedFirstDecisionController(
                    forced_action,
                    product_controller,
                )
        else:
            raw_action = os.environ.get(BRANCH_ACTION_ENV)
            if not raw_action or not self._branchpoint_path.is_file():
                raise RuntimeError("branch mode requires an action and branchpoint snapshot")
            action = parse_control_action(raw_action)
            if action.action != mode:
                raise RuntimeError("branch mode and registered action differ")
            self._registered_action = action
            self._branchpoint_payload = json.loads(
                self._branchpoint_path.read_text(encoding="utf-8")
            )
            product_controller = self._fugu_llm._live_controller
            if product_controller is None:
                raise RuntimeError("product live controller is unavailable")
            forced_action = action
            if mode == "replan":
                planner = FirstReplacementPlanner(action, self._planner)
                self._planner = planner
                self._fugu_llm._planner = planner
                forced_action = ControlAction(action="replan", reason=action.reason)
            self._fugu_llm._live_controller = ForcedFirstDecisionController(
                forced_action,
                product_controller,
            )

    @staticmethod
    @override
    def name() -> str:
        return "fugu-live-branchpoint-collection"

    @override
    def version(self) -> str | None:
        return COLLECTION_REVISION

    @override
    def _after_fugu_run_reset(self, instruction: str) -> None:
        if self._branchpoint_payload is None:
            return
        restore_branchpoint(
            self._fugu_llm,
            self._branchpoint_payload,
            instruction=instruction,
            pool_fingerprint=self._pool_binding.pool_fingerprint,
        )

    @staticmethod
    def _captured_completion_response() -> LLMResponse:
        return LLMResponse(
            content=json.dumps(
                {
                    "analysis": "The registered live branchpoint is frozen.",
                    "plan": "Stop before any post-branch worker action.",
                    "commands": [],
                    "task_complete": True,
                }
            ),
            model_name="fugu-live-branchpoint-capture",
        )

    @override
    async def _query_llm(
        self,
        chat: Any,
        prompt: str,
        original_instruction: str = "",
        session: Any | None = None,
    ) -> LLMResponse:
        if self._capture_controller is not None and self._capture_controller.captured:
            return self._captured_completion_response()
        return await super()._query_llm(chat, prompt, original_instruction, session)

    async def _export_workspace(self) -> None:
        controller = self._capture_controller
        environment = self._active_environment
        if controller is None or environment is None or controller.workspace_exported:
            return
        artifact = "/logs/artifacts/live-branchpoint/testbed.tar.gz"
        command = f"""set -eu
mkdir -p /logs/artifacts/live-branchpoint
tar --exclude={shlex.quote('.fugu-runtime-workspace-root')} -C {shlex.quote(self._workspace_root)} -czf {shlex.quote(artifact)} .
sha256sum {shlex.quote(artifact)}
stat -c '%s' {shlex.quote(artifact)}
"""
        result = await environment.exec(command, cwd="/", timeout_sec=600, user="root")
        if result.return_code != 0:
            raise RuntimeError(f"failed to export branchpoint workspace: {result.stderr}")
        match = re.search(r"^([0-9a-f]{64})\s+", result.stdout or "", re.MULTILINE)
        sizes = re.findall(r"^([0-9]+)$", result.stdout or "", re.MULTILINE)
        if match is None or not sizes:
            raise RuntimeError("workspace export did not return a hash and size")
        payload = json.loads(self._branchpoint_path.read_text(encoding="utf-8"))
        payload["workspace"] = {
            "artifact_path": artifact,
            "sha256": match.group(1),
            "size_bytes": int(sizes[-1]),
        }
        _write_json_atomic(self._branchpoint_path, payload)
        controller.workspace_exported = True

    @override
    async def _handle_llm_interaction(
        self,
        chat: Any,
        prompt: str,
        original_instruction: str = "",
        session: Any | None = None,
    ) -> tuple[list[Any], bool, str, str, str, LLMResponse]:
        result = await super()._handle_llm_interaction(
            chat,
            prompt,
            original_instruction,
            session,
        )
        if self._capture_controller is not None and self._capture_controller.captured:
            await self._export_workspace()
        return result

    @override
    def _record_fugu_metadata(self, context: AgentContext) -> None:
        super()._record_fugu_metadata(context)
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "collection_revision": COLLECTION_REVISION,
                "collection_id": self._collection_id,
                "live_branchpoint_mode": self._collection_mode,
                "live_branchpoint_snapshot_path": str(self._branchpoint_path),
                "live_branchpoint_snapshot_sha256": (
                    _sha256(self._branchpoint_path)
                    if self._branchpoint_path.is_file()
                    else None
                ),
                "live_branchpoint_workspace_exported": bool(
                    self._capture_controller is not None
                    and self._capture_controller.workspace_exported
                ),
                "registered_first_live_action": (
                    json.loads(os.environ[BRANCH_ACTION_ENV])
                    if self._registered_action is not None
                    else None
                ),
                "collection_training_eligible": False,
                "collection_training_gate": "same_state_clean_outcome_pair",
                "worker_calls_are_paid": True,
                "worker_provider_base": self._pool_binding.provider_base,
                "pool_binding_path": str(self._collection_binding_path),
                **self._protected_test_metadata(),
            }
        )
        context.metadata = metadata
