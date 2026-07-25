"""External-decision GRPO agent over the local mechanics pool.

The trainer's sampled policy supplies compact live-control decisions through
the same file bridge as the paid GRPO agent, while workers stay local and
free. Zero paid calls by construction: no Yunwu key, localhost-pinned pool.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, override

from harbor.models.agent.context import AgentContext

from director.agentic.fugu_live_grpo import (
    CAPABILITY_REFS_ENV,
    CONTROL_DIR_ENV,
    DECISION_TIMEOUT_ENV,
    DEFAULT_DECISION_TIMEOUT_S,
    ROLLOUT_ID_ENV,
    UNIFIED_CONTROL_ENV,
    ExternalDecisionLiveController,
    control_protocol,
)
from director.agentic.fugu_mechanics_live import FuguMechanicsLiveAgent
from director.agentic.fugu_ultra_terminal import LocalModelPromptTokenCounter
from ultra.live_control import LIVE_AGENTIC_GRPO_BRIDGE_VERSION

MECHANICS_GRPO_REVISION = "20260719-mechanics-grpo-v1"


class FuguMechanicsGRPOAgent(FuguMechanicsLiveAgent):
    """Local mechanics runtime whose decisions come from a GRPO rollout."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        raw_control_dir = os.environ.get(CONTROL_DIR_ENV)
        rollout_id = os.environ.get(ROLLOUT_ID_ENV)
        if not raw_control_dir or not rollout_id:
            raise RuntimeError(f"{CONTROL_DIR_ENV} and {ROLLOUT_ID_ENV} are required")
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        decision_timeout_s = float(
            os.environ.get(DECISION_TIMEOUT_ENV, str(DEFAULT_DECISION_TIMEOUT_S))
        )
        unified_control = os.environ.get(UNIFIED_CONTROL_ENV) == "1"
        capability_refs = os.environ.get(CAPABILITY_REFS_ENV) == "1"
        controller = ExternalDecisionLiveController(
            control_dir=Path(raw_control_dir),
            rollout_id=rollout_id,
            decision_timeout_s=decision_timeout_s,
            prompt_token_counter=LocalModelPromptTokenCounter(
                model=os.environ["FUGU_MECHANICS_LIVE_CONTROLLER"],
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
        return "fugu-mechanics-grpo"

    @override
    def version(self) -> str | None:
        return MECHANICS_GRPO_REVISION

    @override
    def _record_fugu_metadata(self, context: AgentContext) -> None:
        super()._record_fugu_metadata(context)
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "mechanics_grpo_revision": MECHANICS_GRPO_REVISION,
                "grpo_bridge_version": LIVE_AGENTIC_GRPO_BRIDGE_VERSION,
                "grpo_rollout_id": self._external_live_controller.rollout_id,
                "grpo_control_requests": self._external_live_controller.requests,
                "grpo_control_responses": self._external_live_controller.responses,
                "grpo_external_live_controller": True,
                "grpo_control_protocol": control_protocol(
                    supplies_topology=self._external_live_controller.supplies_topology,
                    capability_refs=self._external_live_controller.capability_refs,
                ),
                "grpo_worker_timeout_s": float(
                    os.environ.get("FUGU_TB_TOTAL_S", "600")
                ),
            }
        )
        context.metadata = metadata
