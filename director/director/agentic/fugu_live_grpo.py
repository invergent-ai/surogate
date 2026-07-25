"""External decision bridge for on-policy Fugu live-control GRPO."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, override

from harbor.models.agent.context import AgentContext

from director.agentic.prepared_index_test_protection import (
    PreparedIndexTestProtectionMixin,
)
from director.agentic.fugu_ultra_terminal import (
    LOCAL_PRODUCT_PLANNER_ADAPTER,
    FuguUltraTerminalAgent,
    LocalModelPromptTokenCounter,
)
from ultra.live_control import (
    LIVE_AGENTIC_GRPO_BRIDGE_VERSION,
    MAX_DECISION_INPUT_TOKENS,
    ControlAction,
    LiveControlState,
    build_control_action_messages,
    build_control_decision_messages,
    capability_reference_map,
    parse_capability_control_action,
    parse_control_action,
    parse_control_decision,
)


BRIDGE_VERSION = LIVE_AGENTIC_GRPO_BRIDGE_VERSION
CONTROL_DIR_ENV = "FUGU_GRPO_CONTROL_DIR"
ROLLOUT_ID_ENV = "FUGU_GRPO_ROLLOUT_ID"
DECISION_TIMEOUT_ENV = "FUGU_GRPO_DECISION_TIMEOUT_S"
UNIFIED_CONTROL_ENV = "FUGU_GRPO_UNIFIED_CONTROL"
CAPABILITY_REFS_ENV = "FUGU_GRPO_CAPABILITY_REFS"
DEFAULT_DECISION_TIMEOUT_S = 180.0
MIN_WORKER_TIMEOUT_S = 600.0


def control_protocol(*, supplies_topology: bool, capability_refs: bool) -> str:
    if capability_refs:
        if not supplies_topology:
            raise ValueError("capability references require unified topology control")
        return "unified_capability_action_v2"
    return "unified_full_action_v1" if supplies_topology else "compact_live_decision_v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class ExternalDecisionLiveController:
    """Pause the product runtime until the GRPO environment supplies a decision."""

    def __init__(
        self,
        *,
        control_dir: Path,
        rollout_id: str,
        decision_timeout_s: float = DEFAULT_DECISION_TIMEOUT_S,
        prompt_token_counter: Any | None = None,
        max_input_tokens: int = MAX_DECISION_INPUT_TOKENS,
        supplies_topology: bool = False,
        capability_refs: bool = False,
    ) -> None:
        if not rollout_id.strip():
            raise ValueError("rollout_id must be non-empty")
        if decision_timeout_s <= 0:
            raise ValueError("decision_timeout_s must be positive")
        self.control_dir = control_dir.resolve()
        self.rollout_id = rollout_id
        self.decision_timeout_s = float(decision_timeout_s)
        self.prompt_token_counter = prompt_token_counter
        self.max_input_tokens = max_input_tokens
        self.supplies_topology = bool(supplies_topology)
        self.capability_refs = bool(capability_refs)
        control_protocol(
            supplies_topology=self.supplies_topology,
            capability_refs=self.capability_refs,
        )
        self.requests = 0
        self.responses = 0
        self.prompt_compactions = 0
        self.last_prompt_tokens: int | None = None
        self.control_dir.mkdir(parents=True, exist_ok=True)

    def _request_path(self, request_id: int) -> Path:
        return self.control_dir / f"request_{request_id:04d}.json"

    def _response_path(self, request_id: int) -> Path:
        return self.control_dir / f"response_{request_id:04d}.json"

    async def decide(
        self,
        state: LiveControlState,
        *,
        correction: str | None = None,
    ) -> ControlAction:
        self.requests += 1
        request_id = self.requests
        builder = (
            build_control_action_messages
            if self.supplies_topology
            else build_control_decision_messages
        )
        messages, prompt_tokens, compacted = builder(
            state,
            prompt_token_counter=self.prompt_token_counter,
            max_input_tokens=self.max_input_tokens,
            capability_refs=self.capability_refs,
        )
        if correction is not None:
            messages = [*messages, {"role": "user", "content": correction}]
        self.last_prompt_tokens = prompt_tokens
        self.prompt_compactions += int(compacted)
        _atomic_json(
            self._request_path(request_id),
            {
                "version": BRIDGE_VERSION,
                "rollout_id": self.rollout_id,
                "request_id": request_id,
                "created_at_unix_s": time.time(),
                "messages": messages,
                "prompt_tokens": prompt_tokens,
                "compacted": compacted,
                "correction": correction,
                "control_protocol": control_protocol(
                    supplies_topology=self.supplies_topology,
                    capability_refs=self.capability_refs,
                ),
                "state": asdict(state),
            },
        )

        response_path = self._response_path(request_id)
        cancel_path = self.control_dir / "cancel.json"
        deadline = time.monotonic() + self.decision_timeout_s
        while True:
            if cancel_path.exists():
                raise asyncio.CancelledError("GRPO rollout cancelled")
            if response_path.exists():
                payload = json.loads(response_path.read_text(encoding="utf-8"))
                if payload.get("version") != BRIDGE_VERSION:
                    raise RuntimeError("GRPO response bridge version mismatch")
                if payload.get("rollout_id") != self.rollout_id:
                    raise RuntimeError("GRPO response rollout ID mismatch")
                if payload.get("request_id") != request_id:
                    raise RuntimeError("GRPO response request ID mismatch")
                completion = payload.get("completion")
                if not isinstance(completion, str):
                    raise RuntimeError("GRPO response completion must be text")
                self.responses += 1
                if not self.supplies_topology:
                    return parse_control_decision(completion)
                if self.capability_refs:
                    return parse_capability_control_action(
                        completion,
                        capability_reference_map(state.workers),
                    )
                return parse_control_action(completion)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"GRPO conductor decision {request_id} exceeded "
                    f"{self.decision_timeout_s:g}s"
                )
            await asyncio.sleep(0.05)


class FuguLiveGRPOAgent(PreparedIndexTestProtectionMixin, FuguUltraTerminalAgent):
    """Product agent whose live-control actions come from a GRPO rollout."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        raw_control_dir = os.environ.get(CONTROL_DIR_ENV)
        rollout_id = os.environ.get(ROLLOUT_ID_ENV)
        if not raw_control_dir or not rollout_id:
            raise RuntimeError(
                f"{CONTROL_DIR_ENV} and {ROLLOUT_ID_ENV} are required"
            )
        self._initialize_protected_test_protection()
        kwargs.setdefault("provider_owner_retry_limit", 0)
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        decision_timeout_s = float(
            os.environ.get(DECISION_TIMEOUT_ENV, str(DEFAULT_DECISION_TIMEOUT_S))
        )
        self._grpo_worker_timeout_s = float(
            os.environ.get("FUGU_TB_TOTAL_S", str(MIN_WORKER_TIMEOUT_S))
        )
        if self._grpo_worker_timeout_s < MIN_WORKER_TIMEOUT_S:
            raise RuntimeError("live GRPO worker timeout must be at least 600 seconds")
        unified_control = os.environ.get(UNIFIED_CONTROL_ENV) == "1"
        capability_refs = os.environ.get(CAPABILITY_REFS_ENV) == "1"
        controller = ExternalDecisionLiveController(
            control_dir=Path(raw_control_dir),
            rollout_id=rollout_id,
            decision_timeout_s=decision_timeout_s,
            prompt_token_counter=LocalModelPromptTokenCounter(
                model=LOCAL_PRODUCT_PLANNER_ADAPTER,
            ),
            supplies_topology=unified_control,
            capability_refs=capability_refs,
        )
        self._external_live_controller = controller
        self._fugu_llm._live_controller = controller
        if unified_control:
            self._fugu_llm._live_controller_plans_initial_workflow = True
            self._fugu_llm._live_controller_supplies_topology = True

    @staticmethod
    @override
    def name() -> str:
        return "fugu-live-agentic-grpo"

    @override
    def version(self) -> str | None:
        return BRIDGE_VERSION

    @override
    def _record_fugu_metadata(self, context: AgentContext) -> None:
        super()._record_fugu_metadata(context)
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "grpo_bridge_version": BRIDGE_VERSION,
                "grpo_rollout_id": self._external_live_controller.rollout_id,
                "grpo_control_requests": self._external_live_controller.requests,
                "grpo_control_responses": self._external_live_controller.responses,
                "grpo_prompt_compactions": (
                    self._external_live_controller.prompt_compactions
                ),
                "grpo_last_prompt_tokens": (
                    self._external_live_controller.last_prompt_tokens
                ),
                "grpo_external_live_controller": True,
                "grpo_control_protocol": control_protocol(
                    supplies_topology=self._external_live_controller.supplies_topology,
                    capability_refs=self._external_live_controller.capability_refs,
                ),
                "grpo_worker_timeout_s": self._grpo_worker_timeout_s,
                **self._protected_test_metadata(),
            }
        )
        context.metadata = metadata
