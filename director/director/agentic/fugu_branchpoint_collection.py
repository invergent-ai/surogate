"""Pool-bound fixed-workflow collection for targeted recovery evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, override

from harbor.agents.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM
from harbor.models.agent.context import AgentContext

from director.agentic.fugu_control_collection import FixedWorkflowPlanner
from director.agentic.fugu_ultra_terminal import (
    MAX_AGENT_TURNS,
    PRODUCT_POOL_BINDING,
    PRODUCT_RUNTIME_REVISION,
    TERMINAL_TASK_BUDGET_S,
    WORKER_CALL_TIMEOUT_S,
    FuguRoutedLLM,
    FuguUltraTerminalAgent,
    YunwuLiteLLM,
)
from director.agentic.prepared_index_test_protection import (
    PreparedIndexTestProtectionMixin,
)
from ultra.live_control import WorkerProfile, validate_worker_profiles
from ultra.pool_binding import load_pool_binding, verify_runtime_pool


COLLECTION_REVISION = "20260719-branchpoint-recovery-v1"
WORKFLOW_ENV = "FUGU_BRANCHPOINT_WORKFLOW_JSON"
COLLECTION_ID_ENV = "FUGU_BRANCHPOINT_COLLECTION_ID"
POOL_BINDING_ENV = "FUGU_BRANCHPOINT_POOL_BINDING"


class FuguBranchpointCollectionAgent(
    PreparedIndexTestProtectionMixin,
    FuguUltraTerminalAgent,
):
    """Run a preregistered recovery topology over a manifest-bound paid pool."""

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
        provider_key = os.environ.get("YUNWU_API_KEY")
        if not provider_key:
            raise RuntimeError("YUNWU_API_KEY is required for paid recovery workers")
        binding_path = Path(os.environ.get(POOL_BINDING_ENV, PRODUCT_POOL_BINDING))
        binding = load_pool_binding(binding_path)
        verify_runtime_pool(
            binding,
            runtime_models=binding.runtime_models,
            reasoning_efforts=binding.reasoning_efforts,
            provider_base=binding.provider_base,
        )
        planner = FixedWorkflowPlanner.from_json(workflow_json)
        self._collection_id = os.environ.get(COLLECTION_ID_ENV, "unregistered")
        self._collection_pool_binding_path = binding_path.resolve()
        self._initialize_protected_test_protection()

        Terminus2.__init__(
            self,
            logs_dir=logs_dir,
            model_name=model_name or "fugu-branchpoint-recovery",
            max_turns=max_turns,
            suppress_max_turns_warning=True,
            **kwargs,
        )
        worker_profiles = tuple(
            WorkerProfile(
                worker_id=slot.worker_id,
                capability_tags=slot.role_prior,
                tool_tags=("terminal", "filesystem", "test_runner"),
            )
            for slot in binding.slots
        )
        validate_worker_profiles(worker_profiles)
        worker_llms: dict[int, BaseLLM] = {}
        worker_names: dict[int, str] = {}
        for slot in binding.slots:
            worker_names[slot.worker_id] = slot.runtime_model
            worker_llms[slot.worker_id] = YunwuLiteLLM(
                model_name=f"openai/{slot.runtime_model}",
                api_base=binding.provider_base,
                api_key=provider_key,
                timeout=WORKER_CALL_TIMEOUT_S,
                reasoning_effort=slot.reasoning_effort,
                session_id=f"fugu-branchpoint-worker-{slot.worker_id}",
            )

        self._planner = planner
        self._pool_binding = binding
        self._configured_max_turns = max_turns
        self._run_started_monotonic: float | None = None
        self._terminal_task_budget_s = TERMINAL_TASK_BUDGET_S
        self._fugu_llm = FuguRoutedLLM(
            planner=planner,
            workers=worker_llms,
            worker_names=worker_names,
            route_log=logs_dir / "fugu_routes.jsonl",
            budget_status=self._terminal_budget_status,
            max_agent_turns=max_turns,
            fair_position_call_budget=None,
            provider_owner_retry_limit=0,
            fail_closed_conductor_errors=True,
        )
        self._llm = self._fugu_llm
        self._rejected_commandful_confirmations = 0
        self._blocked_workspace_commands = 0
        self._isolated_errexit_batches = 0
        self._collapsed_empty_wait_commands = 0
        self._converted_shell_wait_commands = 0
        self._rejected_overtime_commands = 0
        self._initialize_workspace_state()

    @staticmethod
    @override
    def name() -> str:
        return "fugu-branchpoint-collection"

    @override
    def version(self) -> str | None:
        return COLLECTION_REVISION

    @override
    def _verify_serving_dependencies(self) -> None:
        """Fixed recovery workflows do not query a conductor checkpoint."""

    @override
    def _record_fugu_metadata(self, context: AgentContext) -> None:
        super()._record_fugu_metadata(context)
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "runtime_revision": PRODUCT_RUNTIME_REVISION,
                "collection_revision": COLLECTION_REVISION,
                "collection_id": self._collection_id,
                "collection_registered_workflow": json.loads(
                    os.environ[WORKFLOW_ENV]
                ),
                "collection_is_product_candidate": False,
                "collection_training_eligible": False,
                "collection_training_gate": "two_verified_recovery_wins",
                "worker_calls_are_paid": True,
                "worker_provider_base": self._pool_binding.provider_base,
                "worker_provider_bases": [
                    self._pool_binding.provider_base for _ in self._pool_binding.slots
                ],
                "worker_models": list(self._pool_binding.runtime_models),
                "worker_reasoning_efforts": list(
                    self._pool_binding.reasoning_efforts
                ),
                "pool_id": self._pool_binding.pool_id,
                "pool_fingerprint": self._pool_binding.pool_fingerprint,
                "pool_binding_path": str(self._collection_pool_binding_path),
                "frozen_adapter": None,
                "planner_adapter": None,
                "live_control_adapter": None,
                **self._protected_test_metadata(),
            }
        )
        context.metadata = metadata
