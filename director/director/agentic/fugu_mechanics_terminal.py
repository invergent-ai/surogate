"""Local-worker mechanics collection agent for live-control training data.

This rig runs the exact product runtime and control contract against a
local, zero-cost worker pool so control-mechanics trajectories can be
collected and graded at scale. It is not a product candidate, its worker
calls are not paid external requests, and its trajectories can never
attest as current-pool paid evidence because their provider base is not
the bound Yunwu gateway.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, override
from urllib.parse import urlparse

from harbor.agents.terminus_2 import Terminus2
from harbor.llms.base import BaseLLM, LLMResponse
from harbor.models.agent.context import AgentContext

from director.agentic.fugu_control_collection import FixedWorkflowPlanner
from director.agentic.fugu_ultra_terminal import (
    MAX_AGENT_TURNS,
    TERMINAL_TASK_BUDGET_S,
    FuguRoutedLLM,
    FuguUltraTerminalAgent,
    YunwuLiteLLM,
)
from director.agentic.prepared_index_test_protection import (
    PreparedIndexTestProtectionMixin,
)
from ultra.live_control import WorkerProfile, validate_worker_profiles


MECHANICS_REVISION = "20260719-mechanics-local-pool-v2"
MECHANICS_POOL_SCHEMA = "fugu_mechanics_pool_v1"
MECHANICS_POOL_SCHEMA_V2 = "fugu_mechanics_pool_v2"
POOL_ENV = "FUGU_MECHANICS_POOL_PATH"
WORKFLOW_ENV = "FUGU_MECHANICS_WORKFLOW_JSON"
COLLECTION_ID_ENV = "FUGU_MECHANICS_COLLECTION_ID"
LOCAL_WORKER_TIMEOUT_S = 600.0
LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1"}


class MechanicsPoolError(ValueError):
    """A mechanics pool manifest is invalid or is not local."""


@dataclass(frozen=True)
class MechanicsWorkerSlot:
    worker_id: int
    served_model: str
    api_base: str
    role_prior: tuple[str, ...]
    context_window_tokens: int


@dataclass(frozen=True)
class MechanicsPool:
    schema_version: str
    pool_id: str
    slots: tuple[MechanicsWorkerSlot, ...]
    temperature: float | None = None
    top_p: float | None = None

    @property
    def worker_ids(self) -> tuple[int, ...]:
        return tuple(slot.worker_id for slot in self.slots)


def _require_local_base(api_base: str) -> str:
    base = api_base.strip().rstrip("/")
    host = urlparse(base).hostname
    if host not in LOCAL_HOSTNAMES:
        raise MechanicsPoolError(
            f"mechanics workers must be served locally, got {api_base!r}"
        )
    return base


def mechanics_pool_fingerprint(pool: MechanicsPool) -> str:
    payload: dict[str, Any] = {
        "schema_version": pool.schema_version,
        "pool_id": pool.pool_id,
        "slots": [
            {
                "worker_id": slot.worker_id,
                "served_model": slot.served_model,
                "api_base": slot.api_base,
                "role_prior": list(slot.role_prior),
                "context_window_tokens": slot.context_window_tokens,
            }
            for slot in pool.slots
        ],
    }
    if pool.schema_version == MECHANICS_POOL_SCHEMA_V2:
        payload["sampling"] = {
            "temperature": pool.temperature,
            "top_p": pool.top_p,
        }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def load_mechanics_pool(path: Path) -> MechanicsPool:
    raw = json.loads(path.read_text(encoding="utf-8"))
    schema_version = raw.get("schema_version") if isinstance(raw, dict) else None
    if schema_version not in {MECHANICS_POOL_SCHEMA, MECHANICS_POOL_SCHEMA_V2}:
        raise MechanicsPoolError(f"{path} has an unsupported mechanics pool schema")
    allowed = {"schema_version", "pool_id", "slots"}
    if schema_version == MECHANICS_POOL_SCHEMA_V2:
        allowed.add("sampling")
    unexpected = set(raw) - allowed
    if unexpected:
        raise MechanicsPoolError(
            f"mechanics pool has unexpected fields: {sorted(unexpected)}"
        )
    pool_id = raw.get("pool_id")
    if not isinstance(pool_id, str) or not pool_id.strip():
        raise MechanicsPoolError("pool_id must be non-empty text")
    raw_slots = raw.get("slots")
    if not isinstance(raw_slots, list) or not raw_slots:
        raise MechanicsPoolError("slots must be a non-empty list")
    slots: list[MechanicsWorkerSlot] = []
    for index, row in enumerate(raw_slots):
        if not isinstance(row, dict) or set(row) != {
            "worker_id",
            "served_model",
            "api_base",
            "role_prior",
            "context_window_tokens",
        }:
            raise MechanicsPoolError(f"slots[{index}] has an invalid schema")
        worker_id = row.get("worker_id")
        if isinstance(worker_id, bool) or not isinstance(worker_id, int):
            raise MechanicsPoolError(f"slots[{index}].worker_id must be an integer")
        served_model = row.get("served_model")
        if not isinstance(served_model, str) or not served_model.strip():
            raise MechanicsPoolError(f"slots[{index}].served_model must be text")
        context_window = row.get("context_window_tokens")
        if (
            isinstance(context_window, bool)
            or not isinstance(context_window, int)
            or context_window < 8192
        ):
            raise MechanicsPoolError(
                f"slots[{index}].context_window_tokens must be an integer >= 8192"
            )
        roles = row.get("role_prior")
        if not isinstance(roles, list) or any(
            not isinstance(role, str) or not role.strip() for role in roles
        ):
            raise MechanicsPoolError(f"slots[{index}].role_prior must be a list of text")
        slots.append(
            MechanicsWorkerSlot(
                worker_id=worker_id,
                served_model=served_model.strip(),
                api_base=_require_local_base(str(row.get("api_base", ""))),
                role_prior=tuple(role.strip() for role in roles),
                context_window_tokens=context_window,
            )
        )
    temperature: float | None = None
    top_p: float | None = None
    if schema_version == MECHANICS_POOL_SCHEMA_V2:
        sampling = raw.get("sampling")
        if not isinstance(sampling, dict) or set(sampling) != {
            "temperature",
            "top_p",
        }:
            raise MechanicsPoolError(
                "v2 mechanics pool sampling must contain temperature and top_p"
            )
        raw_temperature = sampling.get("temperature")
        raw_top_p = sampling.get("top_p")
        if (
            isinstance(raw_temperature, bool)
            or not isinstance(raw_temperature, (int, float))
            or not 0.0 <= float(raw_temperature) <= 2.0
        ):
            raise MechanicsPoolError("sampling.temperature must be between 0 and 2")
        if (
            isinstance(raw_top_p, bool)
            or not isinstance(raw_top_p, (int, float))
            or not 0.0 < float(raw_top_p) <= 1.0
        ):
            raise MechanicsPoolError("sampling.top_p must be greater than 0 and at most 1")
        temperature = float(raw_temperature)
        top_p = float(raw_top_p)
    slot_tuple = tuple(slots)
    if tuple(slot.worker_id for slot in slot_tuple) != tuple(range(len(slot_tuple))):
        raise MechanicsPoolError(
            "worker_id values must be consecutive stable ordinals starting at zero"
        )
    return MechanicsPool(
        schema_version=schema_version,
        pool_id=pool_id.strip(),
        slots=slot_tuple,
        temperature=temperature,
        top_p=top_p,
    )


LOCAL_WORKER_OUTPUT_TOKENS = 4096


CONTEXT_CHARS_PER_TOKEN = 3.0


class LocalWorkerLiteLLM(YunwuLiteLLM):
    """The Yunwu streaming worker client pinned to a local server instead.

    Local served models are absent from LiteLLM's registry, whose fallback
    context limit disables Terminus context summarization. The true serving
    window is declared explicitly so summarization keeps prompts inside it.

    The runtime's private per-position histories bypass Terminus context
    management entirely, so this client additionally enforces a sliding
    window over the composed request: when the conservative character
    estimate exceeds the declared window, the oldest non-system history
    turns are dropped (with an elision marker) until the request fits.
    """

    @override
    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any] | Any] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        budget_chars = int(
            (self._local_context_window_tokens - LOCAL_WORKER_OUTPUT_TOKENS)
            * CONTEXT_CHARS_PER_TOKEN
        )

        def content_of(message: dict[str, Any] | Any) -> str:
            if isinstance(message, dict):
                return str(message.get("content") or "")
            return str(getattr(message, "content", "") or "")

        def role_of(message: dict[str, Any] | Any) -> str:
            if isinstance(message, dict):
                return str(message.get("role") or "")
            return str(getattr(message, "role", "") or "")

        history = list(message_history or [])
        total = len(prompt) + sum(len(content_of(m)) for m in history)
        if total > budget_chars:
            kept: list[Any] = []
            head: list[Any] = []
            for message in history:
                (head if role_of(message) == "system" else kept).append(message)
            while kept and total > budget_chars:
                dropped = kept.pop(0)
                total -= len(content_of(dropped))
            if kept or head:
                kept.insert(
                    0,
                    {
                        "role": "user",
                        "content": (
                            "[Earlier turns of this private position history were "
                            "elided to fit the local worker context window. Continue "
                            "from the retained recent turns and the terminal state.]"
                        ),
                    },
                )
            history = head + kept
        return await super().call(
            prompt,
            message_history=history,
            response_format=response_format,
            **kwargs,
        )

    def __init__(
        self,
        *args: Any,
        context_window_tokens: int,
        **kwargs: Any,
    ) -> None:
        if context_window_tokens < 8192:
            raise MechanicsPoolError("context_window_tokens must be >= 8192")
        self._local_context_window_tokens = context_window_tokens
        super().__init__(*args, **kwargs)

    @override
    def _require_worker_api_base(self) -> None:
        _require_local_base(self._api_base)

    @override
    def get_model_context_limit(self) -> int:
        return self._local_context_window_tokens - LOCAL_WORKER_OUTPUT_TOKENS

    @override
    def get_model_output_limit(self) -> int:
        return LOCAL_WORKER_OUTPUT_TOKENS


class FuguMechanicsCollectionAgent(
    PreparedIndexTestProtectionMixin,
    FuguUltraTerminalAgent,
):
    """Run one preregistered topology over the local mechanics pool."""

    _sanitize_prepared_git_history = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        max_turns: int = MAX_AGENT_TURNS,
        **kwargs: Any,
    ) -> None:
        pool_path = os.environ.get(POOL_ENV)
        workflow_json = os.environ.get(WORKFLOW_ENV)
        if not pool_path or not workflow_json:
            raise RuntimeError(f"{POOL_ENV} and {WORKFLOW_ENV} are required")
        pool = load_mechanics_pool(Path(pool_path))
        planner = FixedWorkflowPlanner.from_json(workflow_json)
        planner._max_attempts = 1
        self._collection_id = os.environ.get(COLLECTION_ID_ENV, "unregistered")
        self._mechanics_pool = pool
        self._initialize_protected_test_protection()

        Terminus2.__init__(
            self,
            logs_dir=logs_dir,
            model_name=model_name or f"fugu-mechanics/{pool.pool_id}",
            max_turns=max_turns,
            suppress_max_turns_warning=True,
            **kwargs,
        )
        validate_worker_profiles(
            tuple(
                WorkerProfile(
                    worker_id=slot.worker_id,
                    capability_tags=slot.role_prior,
                    tool_tags=("terminal", "filesystem", "test_runner"),
                )
                for slot in pool.slots
            )
        )
        worker_llms: dict[int, BaseLLM] = {}
        worker_names: dict[int, str] = {}
        sampling_kwargs = {
            key: value
            for key, value in {
                "temperature": pool.temperature,
                "top_p": pool.top_p,
            }.items()
            if value is not None
        }
        for slot in pool.slots:
            worker_names[slot.worker_id] = slot.served_model
            worker_llms[slot.worker_id] = LocalWorkerLiteLLM(
                model_name=f"openai/{slot.served_model}",
                api_base=slot.api_base,
                api_key="local",
                timeout=LOCAL_WORKER_TIMEOUT_S,
                reasoning_effort=None,
                session_id=f"fugu-mechanics-{pool.pool_id}-worker-{slot.worker_id}",
                context_window_tokens=slot.context_window_tokens,
                **sampling_kwargs,
            )

        self._planner = planner
        self._pool_binding = SimpleNamespace(
            pool_id=f"mechanics:{pool.pool_id}",
            pool_fingerprint=mechanics_pool_fingerprint(pool),
        )
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
        return "fugu-mechanics-collection"

    @override
    def version(self) -> str | None:
        return MECHANICS_REVISION

    @override
    def _verify_serving_dependencies(self) -> None:
        """Mechanics collection has no conductor checkpoint dependency."""

    @override
    def _record_fugu_metadata(self, context: AgentContext) -> None:
        super()._record_fugu_metadata(context)
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "mechanics_revision": MECHANICS_REVISION,
                "mechanics_pool_id": self._mechanics_pool.pool_id,
                "pool_id": self._pool_binding.pool_id,
                "pool_fingerprint": self._pool_binding.pool_fingerprint,
                "worker_provider_base": self._mechanics_pool.slots[0].api_base,
                "worker_provider_bases": [
                    slot.api_base for slot in self._mechanics_pool.slots
                ],
                "worker_models": [
                    slot.served_model for slot in self._mechanics_pool.slots
                ],
                "worker_reasoning_efforts": None,
                "worker_sampling": {
                    "temperature": self._mechanics_pool.temperature,
                    "top_p": self._mechanics_pool.top_p,
                },
                "worker_calls_are_paid": False,
                "collection_id": self._collection_id,
                "collection_fixed_workflow": json.loads(os.environ[WORKFLOW_ENV]),
                "collection_is_product_candidate": False,
                "frozen_adapter": None,
                "planner_adapter": None,
                "live_control_adapter": None,
                **self._protected_test_metadata(),
            }
        )
        context.metadata = metadata
