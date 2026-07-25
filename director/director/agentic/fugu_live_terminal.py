"""Pool-bound live-control agent for multi-step terminal tasks."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, override
from urllib.parse import urlparse

from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import BaseLLM
from harbor.models.agent.context import AgentContext
from ultra.live_control import OpenAILiveController, WorkerProfile, validate_worker_profiles
from ultra.pool_binding import (
    PoolBinding,
    PoolBindingError,
    load_pool_binding,
    verify_checkpoint_artifacts,
)

from .fugu_ultra_terminal import (
    CURRENT_POOL_BINDING,
    LOCAL_HANDOFF_INTERRUPT_WAIT_S,
    LOCAL_PLANNER_BASE,
    LOCAL_POLL_EPISODE_GUARD,
    MAX_AGENT_TURNS,
    LocalModelPromptTokenCounter,
    REPO_ROOT,
    TERMINAL_TASK_BUDGET_S,
    WORKER_CALL_TIMEOUT_S,
    YUNWU_API_BASE,
    FuguRoutedLLM,
    FuguUltraTerminalAgent,
    RouteDecision,
    YunwuLiteLLM,
    task_agent_timeout_s,
)


LIVE_RUNTIME_REVISION = "20260717-pool-bound-live-control-v2-provider-recovery"
LIVE_CONTROL_CONTRACT = "live_control_v1"


@dataclass(frozen=True)
class RuntimeWorkerSpec:
    """One stable checkpoint-bound worker slot."""

    model: str
    reasoning_effort: str
    profile: WorkerProfile


class _LiveOnlyPlanner:
    async def route(self, prompt: str, message_history: list[Any]) -> RouteDecision:
        raise RuntimeError("the pool-bound runtime must plan through its live controller")


def worker_specs_from_binding(binding: PoolBinding) -> tuple[RuntimeWorkerSpec, ...]:
    """Build live runtime workers without changing the checkpoint's ordinals."""
    specs = tuple(
        RuntimeWorkerSpec(
            model=slot.runtime_model,
            reasoning_effort=slot.reasoning_effort,
            profile=WorkerProfile(
                worker_id=slot.worker_id,
                capability_tags=slot.role_prior,
                tool_tags=("terminal", "filesystem", "test_runner"),
            ),
        )
        for slot in binding.slots
    )
    validate_worker_profiles(tuple(spec.profile for spec in specs))
    return specs


def load_live_pool_binding(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    require_live_contract: bool = True,
) -> PoolBinding:
    binding = load_pool_binding(path)
    verify_checkpoint_artifacts(binding, repo_root=repo_root)
    if (
        require_live_contract
        and binding.checkpoint.trained_control_contract != LIVE_CONTROL_CONTRACT
    ):
        raise PoolBindingError(
            f"checkpoint {binding.checkpoint.adapter_path} is trained for "
            f"{binding.checkpoint.trained_control_contract}, not {LIVE_CONTROL_CONTRACT}"
        )
    return binding


class FuguLiveTerminalAgent(FuguUltraTerminalAgent):
    """Harbor agent serving a live-control checkpoint and its exact pool."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        pool_binding_path: str | Path = CURRENT_POOL_BINDING,
        controller_model: str | None = None,
        controller_url: str = LOCAL_PLANNER_BASE,
        provider_base_url: str = YUNWU_API_BASE,
        max_turns: int = MAX_AGENT_TURNS,
        **kwargs: Any,
    ) -> None:
        if not controller_model or not controller_model.strip():
            raise ValueError("controller_model is required")
        controller_host = urlparse(controller_url).hostname
        if controller_host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("the live conductor must be served locally")

        binding = load_live_pool_binding(Path(pool_binding_path))
        if provider_base_url.rstrip("/") != binding.provider_base:
            raise ValueError(
                f"all external worker requests must use the bound gateway at {binding.provider_base}"
            )
        provider_key = os.environ.get("YUNWU_API_KEY")
        if not provider_key:
            raise RuntimeError("YUNWU_API_KEY is required for all external worker requests")

        specs = worker_specs_from_binding(binding)
        Terminus2.__init__(
            self,
            logs_dir=logs_dir,
            model_name=model_name or f"fugu-live/{binding.pool_id}",
            max_turns=max_turns,
            suppress_max_turns_warning=True,
            **kwargs,
        )
        worker_llms: dict[int, BaseLLM] = {}
        worker_names: dict[int, str] = {}
        for spec in specs:
            worker_id = spec.profile.worker_id
            worker_names[worker_id] = spec.model
            worker_llms[worker_id] = YunwuLiteLLM(
                model_name=f"openai/{spec.model}",
                api_base=binding.provider_base,
                api_key=provider_key,
                timeout=WORKER_CALL_TIMEOUT_S,
                reasoning_effort=spec.reasoning_effort,
                session_id=f"fugu-live-{binding.pool_id}-worker-{worker_id}",
            )

        controller = OpenAILiveController(
            model=controller_model.strip(),
            base_url=controller_url,
            max_tokens=64,
            prompt_token_counter=LocalModelPromptTokenCounter(
                model=controller_model.strip(),
            ),
        )
        self._planner = _LiveOnlyPlanner()
        self._pool_binding = binding
        self._worker_specs = specs
        self._configured_max_turns = max_turns
        self._run_started_monotonic: float | None = None
        self._terminal_task_budget_s = TERMINAL_TASK_BUDGET_S
        self._fugu_llm = FuguRoutedLLM(
            planner=self._planner,
            workers=worker_llms,
            worker_names=worker_names,
            worker_profiles=tuple(spec.profile for spec in specs),
            live_controller=controller,
            route_log=logs_dir / "fugu_routes.jsonl",
            budget_status=self._terminal_budget_status,
            max_agent_turns=max_turns,
            fair_position_call_budget=None,
            provider_owner_retry_limit=0,
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
        return "fugu-live-terminal"

    @override
    def version(self) -> str | None:
        return LIVE_RUNTIME_REVISION

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._fugu_llm.reset_for_run()
        self._rejected_commandful_confirmations = 0
        self._blocked_workspace_commands = 0
        self._isolated_errexit_batches = 0
        self._collapsed_empty_wait_commands = 0
        self._converted_shell_wait_commands = 0
        self._rejected_overtime_commands = 0
        self._terminal_task_budget_s = task_agent_timeout_s(environment)
        self._max_episodes = (
            self._configured_max_turns
            + int(self._terminal_task_budget_s // LOCAL_HANDOFF_INTERRUPT_WAIT_S)
            + LOCAL_POLL_EPISODE_GUARD
        )
        self._run_started_monotonic = time.monotonic()
        self._fugu_llm.set_task_instruction(instruction)
        try:
            await Terminus2.run(self, instruction, environment, context)
        finally:
            self._record_fugu_metadata(context)

    @override
    def _record_fugu_metadata(self, context: AgentContext) -> None:
        super()._record_fugu_metadata(context)
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "runtime_revision": LIVE_RUNTIME_REVISION,
                "pool_id": self._pool_binding.pool_id,
                "pool_binding_revision": self._pool_binding.binding_revision,
                "worker_models": [spec.model for spec in self._worker_specs],
                "worker_reasoning_efforts": [
                    spec.reasoning_effort for spec in self._worker_specs
                ],
                "stable_worker_profiles": [
                    asdict(spec.profile) for spec in self._worker_specs
                ],
                "live_control_decisions": self._fugu_llm.live_control_decisions,
                "live_control_failures": self._fugu_llm.live_control_failures,
                "bound_adapter": self._pool_binding.checkpoint.adapter_path,
                "trained_control_contract": (
                    self._pool_binding.checkpoint.trained_control_contract
                ),
            }
        )
        context.metadata = metadata
