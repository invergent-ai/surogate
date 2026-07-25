"""Live-conductor agent over the local mechanics pool for zero-cost evals.

Mirror of the pool-bound live agent, bound to a mechanics pool binding:
the conductor checkpoint makes real continue/handoff/replan/complete
decisions at terminal boundaries while local workers execute. Used for
paired held-out evaluation of conductor candidates. Not a product
candidate and never a paid path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, override
from urllib.parse import urlparse

from harbor.agents.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM
from harbor.models.agent.context import AgentContext

from director.agentic.fugu_mechanics_terminal import (
    LOCAL_WORKER_TIMEOUT_S,
    LocalWorkerLiteLLM,
    _require_local_base,
)
from director.agentic.fugu_ultra_terminal import (
    MAX_AGENT_TURNS,
    TERMINAL_TASK_BUDGET_S,
    FrozenFuguPlanner,
    FuguRoutedLLM,
    FuguUltraTerminalAgent,
    LocalModelPromptTokenCounter,
)
from director.agentic.prepared_index_test_protection import (
    PreparedIndexTestProtectionMixin,
)
from ultra.live_control import OpenAILiveController
from ultra.pool_binding import (
    load_pool_binding,
    verify_checkpoint_artifacts,
    verify_checkpoint_sidecar,
)

MECHANICS_LIVE_REVISION = "20260718-mechanics-live-v1"
BINDING_ENV = "FUGU_MECHANICS_LIVE_BINDING"
CONTROLLER_MODEL_ENV = "FUGU_MECHANICS_LIVE_CONTROLLER"
PLANNER_MODEL_ENV = "FUGU_MECHANICS_LIVE_PLANNER"
COLLECTION_ID_ENV = "FUGU_MECHANICS_LIVE_COLLECTION_ID"
CONTROLLER_URL = "http://localhost:8007/v1"
DEFAULT_PLANNER_ADAPTER = "planner-v11-s20"
DEFAULT_WORKER_CONTEXT_TOKENS = 57344
REPO_ROOT = Path(__file__).resolve().parents[3]


class FuguMechanicsLiveAgent(PreparedIndexTestProtectionMixin, FuguUltraTerminalAgent):
    """Harbor agent serving a conductor checkpoint over the local mechanics pool."""

    _sanitize_prepared_git_history = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        max_turns: int = MAX_AGENT_TURNS,
        **kwargs: Any,
    ) -> None:
        binding_path = os.environ.get(BINDING_ENV)
        controller_model = os.environ.get(CONTROLLER_MODEL_ENV)
        if not binding_path or not controller_model:
            raise RuntimeError(
                f"{BINDING_ENV} and {CONTROLLER_MODEL_ENV} are required"
            )
        binding = load_pool_binding(Path(binding_path))
        _require_local_base(binding.provider_base)
        verify_checkpoint_artifacts(binding, repo_root=REPO_ROOT)
        verify_checkpoint_sidecar(
            REPO_ROOT / binding.checkpoint.adapter_path / "pool_binding.json",
            binding,
        )
        if urlparse(CONTROLLER_URL).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise RuntimeError("the mechanics conductor must be served locally")
        self._collection_id = os.environ.get(COLLECTION_ID_ENV, "unregistered")
        self._mechanics_binding = binding
        self._initialize_protected_test_protection()

        Terminus2.__init__(
            self,
            logs_dir=logs_dir,
            model_name=model_name or f"fugu-mechanics-live/{binding.pool_id}",
            max_turns=max_turns,
            suppress_max_turns_warning=True,
            **kwargs,
        )
        worker_llms: dict[int, BaseLLM] = {}
        worker_names: dict[int, str] = {}
        for slot in binding.slots:
            worker_names[slot.worker_id] = slot.runtime_model
            worker_llms[slot.worker_id] = LocalWorkerLiteLLM(
                model_name=f"openai/{slot.runtime_model}",
                api_base=binding.provider_base,
                api_key="local",
                timeout=LOCAL_WORKER_TIMEOUT_S,
                reasoning_effort=None,
                session_id=(
                    f"fugu-mechanics-live-{binding.pool_id}-worker-{slot.worker_id}"
                ),
                context_window_tokens=DEFAULT_WORKER_CONTEXT_TOKENS,
            )
        controller = OpenAILiveController(
            model=controller_model.strip(),
            base_url=CONTROLLER_URL,
            max_tokens=64,
            seed=int(os.environ.get("FUGU_MECHANICS_LIVE_SEED", "0")),
            temperature=float(
                os.environ.get("FUGU_MECHANICS_LIVE_TEMPERATURE", "0.0")
            ),
            prompt_token_counter=LocalModelPromptTokenCounter(
                model=controller_model.strip(),
            ),
        )
        from ultra.live_control import WorkerProfile, validate_worker_profiles

        profiles = tuple(
            WorkerProfile(
                worker_id=slot.worker_id,
                capability_tags=slot.role_prior,
                tool_tags=("terminal", "filesystem", "test_runner"),
            )
            for slot in binding.slots
        )
        validate_worker_profiles(profiles)

        self._pool_binding = binding
        self._planner = FrozenFuguPlanner(
            base_url=CONTROLLER_URL,
            adapter=os.environ.get(PLANNER_MODEL_ENV, DEFAULT_PLANNER_ADAPTER),
            max_attempts=1,
        )
        self._configured_max_turns = max_turns
        self._run_started_monotonic: float | None = None
        self._terminal_task_budget_s = TERMINAL_TASK_BUDGET_S
        self._fugu_llm = FuguRoutedLLM(
            planner=self._planner,
            workers=worker_llms,
            worker_names=worker_names,
            worker_profiles=profiles,
            live_controller=controller,
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
        return "fugu-mechanics-live"

    @override
    def version(self) -> str | None:
        return MECHANICS_LIVE_REVISION

    @override
    def _verify_serving_dependencies(self) -> None:
        """The conductor adapter is verified by binding hash, not by name list."""

    @override
    def _record_fugu_metadata(self, context: AgentContext) -> None:
        super()._record_fugu_metadata(context)
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "mechanics_live_revision": MECHANICS_LIVE_REVISION,
                "pool_id": self._mechanics_binding.pool_id,
                "pool_fingerprint": self._mechanics_binding.pool_fingerprint,
                "worker_provider_base": self._mechanics_binding.provider_base,
                "worker_models": list(self._mechanics_binding.runtime_models),
                "worker_reasoning_efforts": None,
                "worker_calls_are_paid": False,
                "collection_id": self._collection_id,
                "collection_is_product_candidate": False,
                "conductor_adapter": os.environ.get(CONTROLLER_MODEL_ENV),
                "bound_adapter": self._mechanics_binding.checkpoint.adapter_path,
                "frozen_adapter": None,
                "planner_adapter": None,
                "live_control_adapter": os.environ.get(CONTROLLER_MODEL_ENV),
                **self._protected_test_metadata(),
            }
        )
        context.metadata = metadata
