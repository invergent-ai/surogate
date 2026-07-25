"""Fugu-Ultra adapter for Harbor's TerminalBench 2.1 harness.

Terminus 2 owns the terminal interaction loop. The conductor emits a complete
workflow whose steps execute across successive terminal turns against the same
workspace. A new workflow is planned after the prior one finishes or is reset.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, override
from urllib.request import urlopen

from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_2 import Command
from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import BaseLLM, LLMResponse
from harbor.llms.lite_llm import LiteLLM
from harbor.models.agent.context import AgentContext
from harbor.models.metric import UsageInfo
from openai import AsyncOpenAI
from ultra.anonymous_planner import (
    anonymous_planner_config,
    capability_set_planner_view,
    verify_anonymous_planner_config,
    verify_capability_set_planner_config,
)
from ultra.conductor_prompt import extract_workflow_payload, prompt_for_task
from ultra.live_control import (
    MAX_CONTROL_OUTPUT_TOKENS,
    MAX_DECISION_CORRECTIONS,
    enabled_control_actions,
    MAX_DECISION_INPUT_TOKENS,
    ControlAction,
    ControlBudget,
    ControlContractError,
    ControlPosition,
    ControlStep,
    LiveControlState,
    OpenAILiveController,
    WorkerProfile,
    canonicalize_control_decision,
    render_control_action_correction,
    render_decision_correction,
    validate_control_action,
    validate_control_decision,
)
from ultra.pool_binding import (
    PoolBinding,
    PoolBindingError,
    load_pool_binding,
    verify_checkpoint_artifacts,
    verify_runtime_pool,
)

from director.agentic.prepared_index_test_protection import (
    repository_discovery_python,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CURRENT_POOL_BINDING = (
    REPO_ROOT
    / "director/manifests/fugu_clean_v1/grpo_pilot_train/current_pool_binding.json"
)
PRODUCT_POOL_BINDING = (
    REPO_ROOT
    / "director/manifests/fugu_clean_v1/grpo_pilot_train/current_pool_binding_v11.json"
)
_PRODUCT_BINDING_DEFAULTS = load_pool_binding(PRODUCT_POOL_BINDING)
# Backward-compatible exports for historical collectors. The values come only
# from the versioned binding; worker identities are not encoded in runtime source.
DEFAULT_WORKER_MODELS = _PRODUCT_BINDING_DEFAULTS.runtime_models
DEFAULT_REASONING_EFFORTS = _PRODUCT_BINDING_DEFAULTS.reasoning_efforts
YUNWU_API_BASE = "https://yunwu.ai/v1"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
# Worker requests may use either the Yunwu gateway or OpenRouter. OpenRouter
# serves the open-weight pool with price-priority routing; Yunwu remains valid
# for measuring the proprietary frontier comparator. Each provider resolves its
# own API key: OpenRouter -> OPENROUTER_KEY, Yunwu -> YUNWU_API_KEY.
ALLOWED_WORKER_PROVIDER_BASES = (YUNWU_API_BASE, OPENROUTER_API_BASE)
WORKER_PROVIDER_KEY_ENV = {
    YUNWU_API_BASE: "YUNWU_API_KEY",
    OPENROUTER_API_BASE: "OPENROUTER_KEY",
}
LOCAL_PLANNER_BASE = "http://localhost:8007/v1"
LOCAL_PLANNER_ADAPTER = "default"
LOCAL_PRODUCT_PLANNER_ADAPTER = "planner-v11-s20"
LOCAL_LIVE_CONTROL_ADAPTER = "live-control-v16-grpo"
PRODUCT_TYPED_CONDUCTOR_MODEL = "fugu-27b-conductor"
PRODUCT_TYPED_CONDUCTOR_BASE = "http://localhost:8010/v1"
PRODUCT_POLICY_REVISION = "fugu-ale-r2-continue-balanced-20260722"
PRODUCT_TYPED_CONDUCTOR_MAX_INPUT_TOKENS = 7_680
PRODUCT_TYPED_CONDUCTOR_MAX_OUTPUT_TOKENS = 512
# Historical collectors and replay auditors bind this base runtime identifier.
RUNTIME_REVISION = "20260717-r36-task-budget-stop"
PRODUCT_RUNTIME_REVISION = "20260724-r88-context-8192"
TERMINUS_RESPONSE_FORMAT = {"type": "json_object"}
PRODUCT_PLANNER_TEMPERATURE = 0.0
INVALID_JSON_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')
TERMINAL_TASK_BUDGET_S = 900.0
WORKER_CALL_TIMEOUT_S = 600.0
MAX_AGENT_TURNS = 120
FAIR_POSITION_CALL_BUDGET = 60
# Retained replay auditors use these frozen r26/r28 policy values when proving
# historical score provenance. r30 does not use either value for live scheduling.
MIN_TURNS_PER_PENDING_POSITION = 4
MIN_NEW_POSITION_BUDGET_S = 120.0
MIN_COMMAND_START_BUDGET_S = 30.0
MIN_WORKER_CALL_START_BUDGET_S = 30.0
COMMAND_COMPLETION_GUARD_S = 10.0
POSITION_CALL_TIME_GUARD_S = 5.0
LOCAL_POLL_INTERVAL_S = 60.0
LOCAL_POLL_OWNER_RECHECK_S = 180.0
LOCAL_HANDOFF_GRACE_S = 60.0
LOCAL_HANDOFF_INTERRUPT_WAIT_S = 2.0
LOCAL_POLL_EPISODE_GUARD = 4
REPEATED_ENVIRONMENT_ACTION_LIMIT = 2
WORKSPACE_ROOT = "/app"
WORKSPACE_SNAPSHOT_ROOT = "/tmp/fugu-runtime-workspace"
WORKSPACE_GUARD_TOKEN = "ROOT_GUARD_TOKEN"
WORKSPACE_SNAPSHOT_TIMEOUT_S = 600
FROZEN_ADAPTER = REPO_ROOT / "output/fugu_ultra_stage2/final_adapter"
FROZEN_ADAPTER_CONFIG = FROZEN_ADAPTER / "adapter_config.json"
FROZEN_ADAPTER_WEIGHTS = FROZEN_ADAPTER / "adapter_model.safetensors"
PRODUCT_PLANNER_ADAPTER = (
    REPO_ROOT / "output/fugu_ultra_planner_composite_v11_s20"
)
PRODUCT_PLANNER_ADAPTER_CONFIG = PRODUCT_PLANNER_ADAPTER / "adapter_config.json"
PRODUCT_PLANNER_ADAPTER_WEIGHTS = PRODUCT_PLANNER_ADAPTER / "adapter_model.safetensors"
LIVE_CONTROL_ADAPTER = (
    REPO_ROOT / "output/fugu_ultra_live_control_grpo_v16/final_adapter"
)
LIVE_CONTROL_ADAPTER_CONFIG = LIVE_CONTROL_ADAPTER / "adapter_config.json"
LIVE_CONTROL_ADAPTER_WEIGHTS = LIVE_CONTROL_ADAPTER / "adapter_model.safetensors"
LOCAL_MODELS_URL = f"{LOCAL_PLANNER_BASE}/models"
MAX_DEPENDENCY_CONTEXT_CHARS = 40_000
MAX_SHARED_MEMORY_CHARS = 80_000
MAX_PLANNER_SHARED_MEMORY_CHARS = 1_500
PLANNER_MAX_TASK_CHARS = 9_000
PLANNING_QUALITY_CONTRACT = (
    "WORKFLOW QUALITY CONTRACT: Create a tool-using workflow for the persistent "
    "shared workspace, not a prose-only consultation. Assign at least one position "
    "that explicitly implements, modifies, edits, applies, repairs, or builds the "
    "concrete result in the environment; proposing a fix or returning a code snippet "
    "is not implementation. The final/root position must explicitly and independently "
    "inspect the resulting artifacts or run task-relevant checks, repair residual "
    "defects when necessary, and claim overall completion only from evidence. Every "
    "assigned strategy must be feasible within the task budget and start from evidence "
    "actually available in the environment. For unknown, deleted, opaque, or externally "
    "sourced artifacts, assign inspection, recovery, measurement, or source validation "
    "before synthesis. Never guess or invent missing data, and never enumerate a large "
    "combinatorial space when direct evidence can be recovered, inspected, or measured. "
    "access_list refers only to earlier positions in the new current workflow: the "
    "first entry must be empty, and completed prior workflows are already available "
    "through persistent shared memory rather than current-workflow indices. "
    "Choose the workers, depth, topology, and aggregator from the current task and bound "
    "pool; no worker is a default or fallback."
)


def render_initial_planning_prompt(
    terminal_state: str,
    *,
    shared_memory: str = "None.",
    unavailable_worker_ids: Sequence[int | str] = (),
    selector_field: str = "worker_id",
) -> str:
    """Render the exact pool-neutral prompt for an initial or replacement workflow."""
    terminal_observation = latest_terminal_observation(terminal_state)
    if selector_field not in {"worker_id", "profile_ref"}:
        raise ValueError(f"unsupported planner selector field: {selector_field}")
    unavailable = json.dumps(
        [{selector_field: worker_id} for worker_id in sorted(unavailable_worker_ids)],
        ensure_ascii=True,
    )
    return (
        "Current terminal state:\n"
        "Persistent shared memory from completed Fugu workflows:\n"
        f"{shared_memory}\n\n"
        f"Latest terminal observation:\n{terminal_observation}\n\n"
        f"{PLANNING_QUALITY_CONTRACT}\n\n"
        "ACTIVE POOL CONSTRAINT: The following workers are unavailable for the "
        "remainder of this task after exhausted owner repair. Do not include "
        f"these {selector_field} values anywhere in {selector_field}. Choose any suitable worker "
        "from the remaining pool; the runtime will not substitute one:\n"
        f"{unavailable}"
    )


class LocalModelPromptTokenCounter:
    """Count chat-template tokens with the exact parent of a served local adapter."""

    def __init__(self, *, model: str, models_url: str = LOCAL_MODELS_URL) -> None:
        self._model = model
        self._models_url = models_url
        self._tokenizer: Any | None = None

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        with urlopen(self._models_url, timeout=10) as response:  # noqa: S310 - local only
            payload = json.load(response)
        models = payload.get("data") if isinstance(payload, dict) else None
        served = next(
            (row for row in models or [] if row.get("id") == self._model),
            None,
        )
        # Current vLLM reports a LoRA's *served base model id* as ``parent`` and
        # the adapter directory as ``root``. Older versions reported the base
        # snapshot path directly as ``parent``. Resolve both shapes before
        # falling back to the directly served model root.
        parent = served.get("parent") if isinstance(served, dict) else None
        if parent and not Path(parent).is_dir():
            parent_row = next(
                (row for row in models or [] if row.get("id") == parent),
                None,
            )
            parent = parent_row.get("root") if isinstance(parent_row, dict) else None
        if not parent or not Path(parent).is_dir():
            parent = served.get("root") if isinstance(served, dict) else None
        if not parent or not Path(parent).is_dir():
            raise RuntimeError(
                f"cannot resolve local tokenizer parent for served model {self._model!r} "
                f"at {self._models_url!r} (served={served!r})"
            )
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            parent,
            local_files_only=True,
        )
        return self._tokenizer

    def __call__(self, messages: Sequence[dict[str, str]]) -> int:
        encoded = self._load_tokenizer().apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            # The vLLM request below explicitly disables Qwen thinking.  This
            # changes the final chat-template sentinel and adds two tokens for
            # the current base tokenizer, so the preflight counter must use the
            # identical setting rather than estimating a shorter prompt.
            enable_thinking=False,
        )
        try:
            token_ids = encoded["input_ids"]
        except (KeyError, TypeError):
            token_ids = encoded
        if token_ids and isinstance(token_ids[0], (list, tuple)):
            token_ids = token_ids[0]
        return len(token_ids)


FINAL_ROOT_QUALITY_CONTRACT = (
    "Regardless of the assigned subtask wording, independently inspect the current "
    "workspace artifacts, run task-relevant checks, and repair any residual defect "
    "before claiming overall completion."
)


@dataclass(frozen=True)
class PlannedStep:
    worker_id: int
    subtask: str
    access: tuple[int, ...] = ()


@dataclass(frozen=True)
class RouteDecision:
    worker_id: int
    subtask: str
    raw_plan: str = ""
    fallback_reason: str | None = None
    workflow_steps: tuple[PlannedStep, ...] = ()
    unrecoverable: bool = False


@dataclass(frozen=True)
class ActiveRoute:
    decision: RouteDecision
    workflow_id: int
    step_index: int
    step_count: int
    access: tuple[int, ...]
    workflow_steps: tuple[PlannedStep, ...]


@dataclass
class WorkflowAgentState:
    """Private function-call trajectory for one position in a conductor workflow."""

    route: ActiveRoute
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    status: str = "pending"
    final_response: str | None = None
    completion_requested: bool = False
    consecutive_protocol_errors: int = 0
    handoff_requested: bool = False
    terminal_ready: bool | None = None
    local_poll_enabled: bool = False
    local_poll_elapsed_s: float = 0.0
    paid_call_start: int | None = None
    paid_call_limit: int | None = None
    lease_started_elapsed_s: float | None = None
    lease_deadline_elapsed_s: float | None = None
    handoff_reason: str | None = None
    checkpoint: dict[str, Any] | None = None
    progress: Any = None
    artifacts: list[Any] = field(default_factory=list)
    recent_activity: list[dict[str, Any]] = field(default_factory=list)
    latest_material_change: bool | None = None
    last_material_progress_turn: int | None = None
    turns_without_material_progress: int = 0
    last_environment_action_signature: str | None = None
    repeated_environment_action_batches: int = 0
    local_handoff_interrupts: int = 0


@dataclass
class WorkflowExecutionState:
    """The active conductor topology and its isolated per-position trajectories."""

    workflow_id: int
    agents: list[WorkflowAgentState]
    active_index: int = 0

    @property
    def active(self) -> WorkflowAgentState:
        return self.agents[self.active_index]


class Planner(Protocol):
    async def route(
        self, prompt: str, message_history: list[dict[str, Any] | Any]
    ) -> RouteDecision: ...


class LiveController(Protocol):
    async def decide(
        self,
        state: LiveControlState,
        *,
        correction: str | None = None,
    ) -> ControlAction: ...


class YunwuLiteLLM(LiteLLM):
    """Single-attempt streaming client for Yunwu's OpenAI-compatible endpoint."""

    def __init__(
        self, *args: Any, client: AsyncOpenAI | None = None, **kwargs: Any
    ) -> None:
        self._sampling_temperature = kwargs.get("temperature")
        self._sampling_top_p = kwargs.get("top_p")
        self.reasoning_json_promotions = 0
        super().__init__(*args, **kwargs)
        self._require_worker_api_base()
        # These model slugs are not in LiteLLM's registry, so capability discovery
        # returns unknown even though the gateway implements OpenAI response_format.
        self._supports_response_format = True
        timeout = float(self._llm_kwargs.get("timeout", WORKER_CALL_TIMEOUT_S))
        self._yunwu_client = client or AsyncOpenAI(
            base_url=self._api_base,
            api_key=self._llm_kwargs.get("api_key"),
            timeout=timeout,
            max_retries=0,
            default_headers={"X-Session-ID": self._session_id or "fugu-ultra"},
        )

    def _require_worker_api_base(self) -> None:
        """External paid workers use an allowed gateway (Yunwu or OpenRouter)."""
        if self._api_base.rstrip("/") not in ALLOWED_WORKER_PROVIDER_BASES:
            raise ValueError(
                f"workers must use one of {ALLOWED_WORKER_PROVIDER_BASES}, "
                f"got {self._api_base!r}"
            )
        self._is_openrouter = self._api_base.rstrip("/") == OPENROUTER_API_BASE

    def _total_timeout_s(self) -> float:
        configured_timeout_s = float(
            self._llm_kwargs.get("timeout", WORKER_CALL_TIMEOUT_S)
        )
        requested_timeout_s = float(
            os.environ.get("FUGU_TB_TOTAL_S", str(configured_timeout_s))
        )
        return min(configured_timeout_s, requested_timeout_s)

    @override
    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any] | Any] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # Streaming keeps Yunwu's upstream connection active during reasoning and
        # permits a hard wall-clock disconnect. There is deliberately no retry.
        def message_dict(message: dict[str, Any] | Any) -> dict[str, Any]:
            if isinstance(message, dict):
                return message
            if hasattr(message, "model_dump"):
                return message.model_dump(exclude_none=True)
            return {"role": message.role, "content": message.content}

        position_timeout_s = kwargs.pop("fugu_call_timeout_s", None)
        messages = [message_dict(message) for message in (message_history or [])]
        messages.append({"role": "user", "content": prompt})
        first_content_s = float(
            os.environ.get("FUGU_TB_FIRST_CONTENT_S", str(WORKER_CALL_TIMEOUT_S))
        )
        total_s = self._total_timeout_s()
        if position_timeout_s is not None:
            total_s = min(total_s, max(1.0, float(position_timeout_s)))
        started = time.monotonic()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        prompt_tokens = completion_tokens = cache_tokens = 0
        cost_usd = 0.0
        finish_reason: str | None = None
        stream = None
        try:
            async with asyncio.timeout(min(first_content_s, total_s)):
                sampling: dict[str, float] = {}
                for name, value in (
                    ("temperature", self._sampling_temperature),
                    ("top_p", self._sampling_top_p),
                ):
                    if value is not None:
                        sampling[name] = float(value)
                extra_body: dict[str, Any] = {"usage": {"include": True}}
                if getattr(self, "_is_openrouter", False):
                    # Route each worker model to its cheapest upstream provider.
                    extra_body["provider"] = {"sort": "price"}
                stream = await self._yunwu_client.chat.completions.create(
                    model=self._canonical_model_name,
                    messages=messages,
                    reasoning_effort=self._reasoning_effort,
                    response_format=response_format,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body=extra_body,
                    **sampling,
                )
            iterator = stream.__aiter__()
            while True:
                elapsed = time.monotonic() - started
                deadline = total_s if content_parts else min(first_content_s, total_s)
                remaining = deadline - elapsed
                if remaining <= 0:
                    finish_reason = "abort_budget"
                    break
                try:
                    async with asyncio.timeout(remaining):
                        chunk = await anext(iterator)
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    finish_reason = "abort_budget"
                    break
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    content = getattr(delta, "content", None)
                    reasoning = getattr(delta, "reasoning_content", None) or getattr(
                        delta, "reasoning", None
                    )
                    if content:
                        content_parts.append(content)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                    details = getattr(usage, "prompt_tokens_details", None)
                    cache_tokens = getattr(details, "cached_tokens", 0) or 0
                    cost_usd = float(getattr(usage, "cost", 0.0) or 0.0)
                    if cost_usd <= 0.0:
                        cost_details = getattr(usage, "cost_details", None)
                        if isinstance(cost_details, dict):
                            cost_usd = float(
                                cost_details.get("upstream_inference_cost", 0.0) or 0.0
                            )
                        elif cost_details is not None:
                            cost_usd = float(
                                getattr(
                                    cost_details,
                                    "upstream_inference_cost",
                                    0.0,
                                )
                                or 0.0
                            )
        except TimeoutError:
            finish_reason = "abort_budget"
        finally:
            if stream is not None:
                try:
                    async with asyncio.timeout(5):
                        await stream.close()
                except Exception:
                    pass

        content = "".join(content_parts)
        reasoning_content = "".join(reasoning_parts)
        if not content and reasoning_content and response_format is not None:
            parser = TerminusJSONPlainParser()
            json_content, _ = parser._extract_json_content(reasoning_content)
            try:
                data = json.loads(json_content, strict=False) if json_content else None
            except (json.JSONDecodeError, TypeError, ValueError):
                data = None
            if (
                isinstance(data, dict)
                and any(key in data for key in ("commands", "command", "keystrokes"))
                and any(key in data for key in ("task_complete", "done"))
            ):
                content = json.dumps(data, ensure_ascii=True)
                self.reasoning_json_promotions += 1
        if not content:
            raise RuntimeError(
                f"Yunwu worker {self._canonical_model_name} produced no executable content "
                f"(finish_reason={finish_reason or 'stream_error'}, "
                f"reasoning_chars={len(reasoning_content)})"
            )
        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content or None,
            model_name=self._canonical_model_name,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_tokens=cache_tokens,
                cost_usd=cost_usd,
            ),
        )


def latest_terminal_observation(prompt: str) -> str:
    """Remove Terminus' response protocol and retain the latest shell observation."""
    matches = list(
        re.finditer(r"(?:Current terminal state|New Terminal Output):\s*", prompt)
    )
    if not matches:
        return prompt.strip()
    observation = prompt[matches[-1].end() :]
    return observation.split(
        "Are you sure you want to mark the task as complete?", 1
    )[0].strip()


def render_route_state(
    prompt: str,
    message_history: list[dict[str, Any] | Any],
    *,
    original_instruction: str | None = None,
    max_chars: int = 12_000,
) -> str:
    """Keep the original task and the latest terminal exchanges in the route prompt."""

    def content(message: dict[str, Any] | Any) -> str:
        if isinstance(message, dict):
            value = message.get("content", "")
        else:
            value = getattr(message, "content", "")
        return value if isinstance(value, str) else str(value)

    if original_instruction is not None:
        first = original_instruction
        if message_history:
            recent_messages = [
                *message_history[-6:],
                {"role": "user", "content": prompt},
            ]
            current = "\n\n".join(
                f"{(m.get('role', '') if isinstance(m, dict) else getattr(m, 'role', ''))}: {content(m)}"
                for m in recent_messages
            )
        else:
            if re.search(
                r"(?:Current terminal state|New Terminal Output):\s*", prompt
            ):
                current = latest_terminal_observation(prompt)
            else:
                current = "No commands have run yet. Begin by inspecting the terminal environment."
        original_header = "ORIGINAL TASK:\n"
        transcript_header = "\n\nCURRENT TRANSCRIPT:\n"
        content_budget = max(
            0,
            max_chars - len(original_header) - len(transcript_header),
        )
        recent_reserve = min(len(current), min(3_000, content_budget // 4))
        first_budget = min(len(first), content_budget - recent_reserve)
        current_budget = content_budget - first_budget
        current_tail = current[-current_budget:] if current_budget else ""
        return (
            f"{original_header}{first[:first_budget]}{transcript_header}{current_tail}"
        )

    first = content(message_history[0]) if message_history else prompt
    recent_messages = [*message_history[-6:], {"role": "user", "content": prompt}]
    recent = "\n\n".join(
        f"{(m.get('role', '') if isinstance(m, dict) else getattr(m, 'role', ''))}: {content(m)}"
        for m in recent_messages
    )
    first_budget = min(len(first), max_chars // 2)
    recent_budget = max_chars - first_budget
    return f"ORIGINAL TASK AND INITIAL STATE:\n{first[:first_budget]}\n\nCURRENT TRANSCRIPT:\n{recent[-recent_budget:]}"


def _protocol_text(value: Any, default: str) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    if isinstance(value, list):
        return "\n".join(
            f"{index}. {item}" for index, item in enumerate(value, start=1)
        )
    try:
        return json.dumps(value, ensure_ascii=True)
    except (TypeError, ValueError):
        return str(value)


def normalize_worker_command_payload(data: Any) -> tuple[Any, bool]:
    """Normalize recognized worker command aliases without inventing commands."""
    if not isinstance(data, dict):
        return data, False
    payload = dict(data)
    changed = False

    analysis = _protocol_text(
        payload.get("analysis", payload.get("reasoning")),
        "Execute the assigned terminal subtask.",
    )
    plan = _protocol_text(
        payload.get("plan", payload.get("next_steps")),
        "Run the requested terminal command and inspect its result.",
    )
    if payload.get("analysis") != analysis:
        payload["analysis"] = analysis
        changed = True
    if payload.get("plan") != plan:
        payload["plan"] = plan
        changed = True

    if "commands" in payload:
        raw_commands = payload["commands"]
    elif "command" in payload:
        raw_commands = payload.pop("command")
        changed = True
    elif "keystrokes" in payload:
        raw_commands = {
            "keystrokes": payload.pop("keystrokes"),
            "duration": payload.pop("duration", 1.0),
        }
        changed = True
    else:
        raw_commands = []
        payload["commands"] = []
        changed = True

    if isinstance(raw_commands, (str, dict)):
        raw_commands = [raw_commands]
        changed = True
    if raw_commands is None:
        raw_commands = []
        changed = True
    if not isinstance(raw_commands, list):
        return data, False

    normalized_commands: list[Any] = []
    aliases = ("keystrokes", "command", "cmd", "shell_command", "bash")
    for raw_command in raw_commands:
        if isinstance(raw_command, str):
            keystrokes = raw_command
            duration: Any = 1.0
            changed = True
        elif isinstance(raw_command, dict):
            selected_alias = next(
                (alias for alias in aliases if alias in raw_command),
                None,
            )
            if selected_alias is None:
                normalized_commands.append(raw_command)
                continue
            keystrokes = raw_command[selected_alias]
            duration = raw_command.get(
                "duration",
                raw_command.get("duration_sec", raw_command.get("timeout", 1.0)),
            )
            if selected_alias != "keystrokes" or set(raw_command) - {
                "keystrokes",
                "duration",
            }:
                changed = True
        else:
            normalized_commands.append(raw_command)
            continue

        if not isinstance(keystrokes, str):
            normalized_commands.append(raw_command)
            continue
        symbolic_key = re.fullmatch(r"\s*((?:C|M)-[^\s]+)\s*", keystrokes)
        if symbolic_key is not None:
            normalized_key = symbolic_key.group(1)
            if normalized_key != keystrokes:
                changed = True
            keystrokes = normalized_key
        elif keystrokes and not keystrokes.endswith(("\n", "\r")):
            keystrokes += "\n"
            changed = True
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = 1.0
            changed = True
        normalized_commands.append({"keystrokes": keystrokes, "duration": duration})

    if payload.get("commands") != normalized_commands:
        payload["commands"] = normalized_commands
        changed = True

    if "task_complete" not in payload and "done" in payload:
        payload["task_complete"] = payload.pop("done")
        changed = True
    task_complete = payload.get("task_complete", False)
    if isinstance(task_complete, str):
        normalized_complete = task_complete.strip().lower() in {
            "true",
            "1",
            "yes",
        }
        payload["task_complete"] = normalized_complete
        changed = True
    elif "task_complete" not in payload:
        payload["task_complete"] = False
        changed = True

    return payload, changed


def repair_terminus_json(
    content: str,
    parser: TerminusJSONPlainParser,
) -> tuple[str, Any, bool]:
    """Canonicalize recoverable JSON syntax and recognized command wire shapes."""
    parsed = parser.parse_response(content)

    json_content, _ = parser._extract_json_content(content)
    if not json_content:
        return content, parsed, False
    candidates = (json_content, INVALID_JSON_ESCAPE_RE.sub(r"\\\\", json_content))
    for candidate in dict.fromkeys(candidates):
        try:
            data = json.loads(candidate, strict=False)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        normalized_data, semantically_repaired = normalize_worker_command_payload(data)
        normalized = json.dumps(normalized_data, ensure_ascii=True)
        repaired = parser.parse_response(normalized)
        if not repaired.error and (parsed.error or semantically_repaired):
            return normalized, repaired, True
    if not parsed.error:
        return content, parsed, False
    return content, parsed, False


def destroys_workspace_root(
    keystrokes: str,
    workspace_root: str = WORKSPACE_ROOT,
) -> bool:
    """Reject a catastrophic move or removal of the active workspace root."""
    normalized_root = workspace_root.rstrip("/")
    for line in keystrokes.splitlines():
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if not tokens:
            continue
        if tokens[0] == "sudo":
            tokens = tokens[1:]
        if not tokens or tokens[0] not in {"mv", "rm"}:
            continue
        operands = [token for token in tokens[1:] if not token.startswith("-")]
        if (
            tokens[0] == "mv"
            and operands
            and operands[0].rstrip("/") == normalized_root
        ):
            return True
        if tokens[0] == "rm" and any(
            operand.rstrip("/") in {normalized_root, f"{normalized_root}/*"}
            for operand in operands
        ):
            return True
    return False


def task_agent_timeout_s(
    environment: BaseEnvironment | Any,
    default: float = TERMINAL_TASK_BUDGET_S,
) -> float:
    """Read the official task agent timeout that Harbor applies to this trial."""
    declared_timeout = getattr(environment, "agent_timeout_s", None)
    try:
        declared_timeout = float(declared_timeout)
    except (TypeError, ValueError):
        declared_timeout = None
    if declared_timeout is not None and declared_timeout > 0:
        return declared_timeout
    environment_dir = getattr(environment, "environment_dir", None)
    if environment_dir is None:
        return default
    task_path = Path(environment_dir).parent / "task.toml"
    try:
        value = tomllib.loads(task_path.read_text())["agent"]["timeout_sec"]
        timeout = float(value)
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError):
        return default
    return timeout if timeout > 0 else default


def enables_interactive_errexit(keystrokes: str) -> bool:
    """Detect direct or sourced shell state that could kill the persistent shell."""
    for line in keystrokes.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        command_boundary = r"(?:^|(?:&&|\|\||;)\s*)"
        if re.search(
            rf"{command_boundary}set\s+-[^\s;]*e[^\s;]*(?:\s|;|$)",
            stripped,
        ):
            return True
        if re.search(
            rf"{command_boundary}set\s+-o\s+errexit(?:\s|;|$)",
            stripped,
        ):
            return True
        if re.search(
            rf"{command_boundary}(?:source|\.)\s+[^;&|\s]+",
            stripped,
        ):
            return True
    return False


def normalize_terminal_commands(
    commands: list[Command],
) -> tuple[list[Command], int, int, int]:
    """Isolate errexit, convert shell sleeps, and collapse adjacent terminal waits."""
    normalized: list[Command] = []
    isolated_errexit = 0
    collapsed_empty = 0
    converted_shell_waits = 0
    for command in commands:
        current = command
        if enables_interactive_errexit(current.keystrokes):
            isolated_errexit += 1
            current = Command(
                keystrokes=f"bash -lc {shlex.quote(current.keystrokes)}\n",
                duration_sec=current.duration_sec,
            )
        sleep_match = re.fullmatch(
            r"\s*sleep\s+([0-9]+(?:\.[0-9]+)?)\s*;?\s*",
            current.keystrokes,
        )
        if sleep_match:
            converted_shell_waits += 1
            current = Command(
                keystrokes="",
                duration_sec=min(
                    LOCAL_POLL_INTERVAL_S,
                    max(float(sleep_match.group(1)), current.duration_sec),
                ),
            )
        if (
            not current.keystrokes.strip()
            and normalized
            and not normalized[-1].keystrokes.strip()
        ):
            previous = normalized[-1]
            normalized[-1] = Command(
                keystrokes="",
                duration_sec=min(
                    LOCAL_POLL_INTERVAL_S,
                    previous.duration_sec + current.duration_sec,
                ),
            )
            collapsed_empty += 1
        else:
            normalized.append(current)
    return normalized, isolated_errexit, collapsed_empty, converted_shell_waits


def validate_frozen_planner_config(
    config: dict[str, Any],
    binding: PoolBinding | None = None,
) -> None:
    """Reject anonymous prompt metadata that differs from its pool binding."""
    try:
        verify_anonymous_planner_config(
            config,
            binding or _PRODUCT_BINDING_DEFAULTS,
            max_workflow_steps=5,
        )
    except PoolBindingError as exc:
        raise ValueError(str(exc)) from exc


def normalize_conductor_workflow_payload(
    raw: str,
    paper_payload_parser: Callable[[str], str],
    *,
    profile_ref_to_worker_id: dict[str, int] | None = None,
) -> str:
    """Normalize the JSON spelling of the trained three-list conductor contract."""
    payload = paper_payload_parser(raw)
    json_payload = payload.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        json_payload,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        json_payload = fenced.group(1)
    try:
        data = json.loads(json_payload)
    except (json.JSONDecodeError, TypeError):
        return payload
    if not isinstance(data, dict):
        return payload

    if "steps" in data:
        if profile_ref_to_worker_id is None:
            return payload
        translated = []
        for step in data.get("steps") or []:
            if not isinstance(step, dict) or set(step) != {
                "profile_ref",
                "subtask",
                "access_positions",
            }:
                return payload
            profile_ref = step.get("profile_ref")
            if profile_ref not in profile_ref_to_worker_id:
                raise ValueError(f"unknown capability profile reference: {profile_ref!r}")
            translated.append(
                {
                    "worker_id": profile_ref_to_worker_id[profile_ref],
                    "subtask": step["subtask"],
                    "access": step["access_positions"],
                }
            )
        return json.dumps({"steps": translated}, ensure_ascii=True)

    if profile_ref_to_worker_id is not None:
        refs = data.get("profile_ref")
        if not isinstance(refs, list):
            return payload
        try:
            ids = [profile_ref_to_worker_id[ref] for ref in refs]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unknown capability profile reference: {exc}") from exc
    else:
        ids = data.get("model_id")
    subtasks = data.get("subtasks")
    access_list = data.get("access_list")
    if not all(isinstance(value, list) for value in (ids, subtasks, access_list)):
        return payload
    if len({len(ids), len(subtasks), len(access_list)}) != 1:
        return payload

    steps: list[dict[str, Any]] = []
    for index, (worker_id, subtask, access_entry) in enumerate(
        zip(ids, subtasks, access_list, strict=True)
    ):
        if isinstance(worker_id, bool) or not isinstance(worker_id, int):
            return payload
        if not isinstance(subtask, str):
            return payload
        if isinstance(access_entry, str):
            access_entry = [access_entry]
        if not isinstance(access_entry, list):
            return payload
        if len(access_entry) == 1 and (
            isinstance(access_entry[0], str)
            and access_entry[0].strip().lower() == "all"
        ):
            access = list(range(index))
        elif all(
            isinstance(entry, int) and not isinstance(entry, bool)
            for entry in access_entry
        ):
            access = access_entry
        else:
            return payload
        steps.append({"worker_id": worker_id, "subtask": subtask, "access": access})
    return json.dumps({"steps": steps})


def verify_frozen_adapter_served() -> None:
    """Check the required adapter files and local serving root."""
    for label, path in (
        ("config", FROZEN_ADAPTER_CONFIG),
        ("weights", FROZEN_ADAPTER_WEIGHTS),
    ):
        if not path.is_file():
            raise RuntimeError(f"adapter {label} is missing: {path}")
    with urlopen(LOCAL_MODELS_URL, timeout=10) as response:  # noqa: S310 - local only
        payload = json.load(response)
    models = payload.get("data") if isinstance(payload, dict) else None
    default = next(
        (model for model in models or [] if model.get("id") == LOCAL_PLANNER_ADAPTER),
        None,
    )
    served_root = default.get("root") if isinstance(default, dict) else None
    if not served_root or Path(served_root).resolve() != FROZEN_ADAPTER.resolve():
        raise RuntimeError(
            f"{LOCAL_MODELS_URL} default adapter is {served_root!r}, expected {str(FROZEN_ADAPTER)!r}"
        )


def verify_product_adapters_served() -> None:
    """Check both trained conductor heads and their local serving roots."""
    for label, path in (
        ("planner config", PRODUCT_PLANNER_ADAPTER_CONFIG),
        ("planner weights", PRODUCT_PLANNER_ADAPTER_WEIGHTS),
        ("live-control config", LIVE_CONTROL_ADAPTER_CONFIG),
        ("live-control weights", LIVE_CONTROL_ADAPTER_WEIGHTS),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} is missing: {path}")
    with urlopen(LOCAL_MODELS_URL, timeout=10) as response:  # noqa: S310 - local only
        payload = json.load(response)
    models = payload.get("data") if isinstance(payload, dict) else None
    planner = next(
        (
            model
            for model in models or []
            if model.get("id") == LOCAL_PRODUCT_PLANNER_ADAPTER
        ),
        None,
    )
    planner_root = planner.get("root") if isinstance(planner, dict) else None
    if not planner_root or Path(planner_root).resolve() != PRODUCT_PLANNER_ADAPTER.resolve():
        raise RuntimeError(
            f"{LOCAL_MODELS_URL} product planner is {planner_root!r}, "
            f"expected {str(PRODUCT_PLANNER_ADAPTER)!r}"
        )
    live = next(
        (
            model
            for model in models or []
            if model.get("id") == LOCAL_LIVE_CONTROL_ADAPTER
        ),
        None,
    )
    served_root = live.get("root") if isinstance(live, dict) else None
    if not served_root or Path(served_root).resolve() != LIVE_CONTROL_ADAPTER.resolve():
        raise RuntimeError(
            f"{LOCAL_MODELS_URL} live-control adapter is {served_root!r}, "
            f"expected {str(LIVE_CONTROL_ADAPTER)!r}"
        )


class FrozenFuguPlanner:
    """Generate and validate a complete workflow with the frozen Stage-2 adapter."""

    def __init__(
        self,
        *,
        base_url: str = LOCAL_PLANNER_BASE,
        adapter: str = LOCAL_PLANNER_ADAPTER,
        max_attempts: int = 3,
        client: AsyncOpenAI | None = None,
        binding: PoolBinding | None = None,
        capability_set_interface: bool = False,
    ) -> None:
        from ultra.sources.hf import make_taskspec
        from ultra.workflow import parse_workflow, validate_workflow

        self._make_taskspec = make_taskspec
        self._parse_workflow = parse_workflow
        self._validate_workflow = validate_workflow
        self._adapter = adapter
        self._max_attempts = max_attempts
        self._binding = binding or _PRODUCT_BINDING_DEFAULTS
        self._capability_set_interface = capability_set_interface
        self._client = client or AsyncOpenAI(
            base_url=base_url,
            api_key="x",
            timeout=180.0,
            max_retries=0,
        )
        self._original_instruction: str | None = None

        template_path = (
            REPO_ROOT
            / "director/manifests/fugu_clean_v1/grpo_pilot_train/pilot_config_singleturn.json"
        )
        template = json.loads(template_path.read_text())
        max_workflow_steps = int(template["workflow_policy"]["max_workflow_steps"])
        self._capability_view = capability_set_planner_view(
            self._binding,
            max_workflow_steps=max_workflow_steps,
        )
        if capability_set_interface:
            self._config = self._capability_view.config
            verify_capability_set_planner_config(
                self._config,
                self._binding,
                max_workflow_steps=max_workflow_steps,
            )
        else:
            self._config = anonymous_planner_config(
                self._binding,
                max_workflow_steps=max_workflow_steps,
            )
            validate_frozen_planner_config(self._config, self._binding)

    def planner_availability_refs(
        self, worker_ids: tuple[int, ...]
    ) -> tuple[int | str, ...]:
        if not self._capability_set_interface:
            return tuple(sorted(worker_ids))
        return self._capability_view.profile_refs_for_workers(worker_ids)

    def planner_selector_field(self) -> str:
        return "profile_ref" if self._capability_set_interface else "worker_id"

    def set_task_instruction(self, instruction: str) -> None:
        self._original_instruction = instruction

    def build_messages(
        self, prompt: str, message_history: list[dict[str, Any] | Any]
    ) -> list[dict[str, Any]]:
        """Build the exact planner request used by the product runtime."""
        state = render_route_state(
            prompt,
            message_history,
            original_instruction=self._original_instruction,
            max_chars=PLANNER_MAX_TASK_CHARS,
        )
        task = self._make_taskspec(
            task_id="terminalbench-live-turn",
            capability="agentic_coding",
            source_name="terminal_bench_official",
            source_version="2.1",
            policy="final_eval_only",
            harness="terminal_sandbox",
            grader_type="harbor_verifier",
            expected_answer={},
            prompt=state,
            system="Direct tool-using workers to complete the terminal task.",
            group_id="terminalbench-2.1",
            domain="code",
            tags=["agentic", "terminal"],
            url_or_ref="terminal-bench/terminal-bench-2-1",
        )
        return [
            dict(message)
            for message in prompt_for_task(
                task,
                self._config,
                "single_turn",
                max_task_chars=PLANNER_MAX_TASK_CHARS,
            )
        ]

    async def route(
        self, prompt: str, message_history: list[dict[str, Any] | Any]
    ) -> RouteDecision:
        messages = self.build_messages(prompt, message_history)

        last_error = "empty conductor response"
        last_raw = ""
        for _ in range(self._max_attempts):
            try:
                response = await self._client.chat.completions.create(
                    model=self._adapter,
                    messages=messages,
                    temperature=PRODUCT_PLANNER_TEMPERATURE,
                    top_p=1.0,
                    max_tokens=1024,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                last_raw = response.choices[0].message.content or ""
                workflow = self._parse_workflow(
                    normalize_conductor_workflow_payload(
                        last_raw,
                        extract_workflow_payload,
                        profile_ref_to_worker_id=(
                            self._capability_view.profile_ref_to_worker_id
                            if self._capability_set_interface
                            else None
                        ),
                    )
                )
                self._validate_workflow(workflow, len(self._binding.slots))
                workflow_steps = tuple(
                    PlannedStep(
                        worker_id=step.worker_id,
                        subtask=step.subtask.strip(),
                        access=tuple(step.access),
                    )
                    for step in workflow.steps
                )
                step = workflow_steps[0]
                return RouteDecision(
                    worker_id=step.worker_id,
                    subtask=step.subtask,
                    raw_plan=last_raw,
                    workflow_steps=workflow_steps,
                )
            except Exception as exc:  # retry parse and transient local-serving failures
                last_error = f"{type(exc).__name__}: {exc}"

        return RouteDecision(
            worker_id=-1,
            subtask="No worker selected because conductor planning failed.",
            raw_plan=last_raw,
            fallback_reason=last_error,
        )


class FixedSoloPlanner:
    """Seed one full-task owner without invoking a conductor model."""

    def __init__(self, worker_id: int) -> None:
        self.worker_id = worker_id
        self._max_attempts = 1
        self._task_instruction = ""
        self._unavailable_worker_ids: frozenset[int] = frozenset()

    def set_task_instruction(self, instruction: str) -> None:
        self._task_instruction = instruction

    def set_unavailable_workers(self, worker_ids: frozenset[int]) -> None:
        self._unavailable_worker_ids = worker_ids

    async def route(
        self, prompt: str, message_history: list[dict[str, Any] | Any]
    ) -> RouteDecision:
        del prompt, message_history
        if self.worker_id in self._unavailable_worker_ids:
            return RouteDecision(
                worker_id=-1,
                subtask="The sole worker is unavailable.",
                raw_plan=json.dumps(
                    {"mode": "solo", "worker_id": self.worker_id, "unavailable": True},
                    sort_keys=True,
                ),
                fallback_reason="the sole worker is unavailable",
                unrecoverable=True,
            )
        subtask = (
            "Own the complete task from initial investigation through implementation "
            "and final verification. Use the terminal iteratively, preserve task-owned "
            "tests, and report completion only after the requested behavior is checked."
        )
        step = PlannedStep(worker_id=self.worker_id, subtask=subtask)
        return RouteDecision(
            worker_id=self.worker_id,
            subtask=subtask,
            raw_plan=json.dumps(
                {"mode": "solo", "worker_id": self.worker_id}, sort_keys=True
            ),
            workflow_steps=(step,),
        )


class FuguRoutedLLM(BaseLLM):
    """Stateful Fugu-Ultra workflow engine behind Terminus' command executor.

    Harbor owns the shared terminal. This class owns the conductor topology and one
    private conversation per workflow position, so tool results return to the agent
    that emitted the call and sibling trajectories are visible only through access.
    """

    def __init__(
        self,
        *,
        planner: Planner,
        workers: dict[int, BaseLLM],
        worker_names: dict[int, str],
        route_log: Path,
        budget_status: Callable[[], tuple[float, float]] | None = None,
        max_agent_turns: int = MAX_AGENT_TURNS,
        fair_position_call_budget: int | None = FAIR_POSITION_CALL_BUDGET,
        provider_owner_retry_limit: int = 1,
        live_controller: LiveController | None = None,
        worker_profiles: tuple[WorkerProfile, ...] | None = None,
        live_controller_plans_initial_workflow: bool = True,
        live_controller_supplies_topology: bool = False,
        fail_closed_conductor_errors: bool = False,
        fail_closed_provider_errors: bool = False,
        local_live_completion_confirmation: bool = False,
        worker_tool_contract: str | None = None,
    ) -> None:
        super().__init__()
        if not workers or set(workers) != set(worker_names):
            raise ValueError(
                "workers and worker_names must have identical non-empty keys"
            )
        self._planner = planner
        self._workers = workers
        self._worker_names = worker_names
        self._live_controller = live_controller
        self._worker_profiles = worker_profiles
        self._live_controller_plans_initial_workflow = (
            live_controller_plans_initial_workflow
        )
        self._live_controller_supplies_topology = live_controller_supplies_topology
        self._fail_closed_conductor_errors = fail_closed_conductor_errors
        self._fail_closed_provider_errors = bool(fail_closed_provider_errors)
        self._local_live_completion_confirmation = bool(
            local_live_completion_confirmation
        )
        self._worker_tool_contract = (worker_tool_contract or "").strip()
        if self._local_live_completion_confirmation and live_controller is None:
            raise ValueError(
                "local live completion confirmation requires a live controller"
            )
        if (
            live_controller_supplies_topology
            and not live_controller_plans_initial_workflow
        ):
            raise ValueError(
                "a topology-supplying live controller must plan the initial workflow"
            )
        if live_controller is not None:
            if worker_profiles is None:
                raise ValueError("worker_profiles are required with a live controller")
            profile_ids = tuple(profile.worker_id for profile in worker_profiles)
            if len(profile_ids) != len(set(profile_ids)) or set(profile_ids) != set(
                workers
            ):
                raise ValueError(
                    "worker_profiles must describe every and only runtime worker"
                )
        elif worker_profiles is not None:
            raise ValueError("worker_profiles require a live controller")
        self._route_log = route_log
        self._paid_call_log = route_log.with_name("fugu_paid_call_attempts.jsonl")
        self._provider_failure_log = route_log.with_name("fugu_provider_failures.jsonl")
        self._budget_status = budget_status
        self._max_agent_turns = max_agent_turns
        if fair_position_call_budget is not None and fair_position_call_budget <= 0:
            raise ValueError("fair_position_call_budget must be positive or None")
        self._fair_position_call_budget = fair_position_call_budget
        if provider_owner_retry_limit < 0:
            raise ValueError("provider_owner_retry_limit cannot be negative")
        self._provider_owner_retry_limit = provider_owner_retry_limit
        self._task_instruction = ""
        self.routes: list[dict[str, Any]] = []
        self._workflow: WorkflowExecutionState | None = None
        self._shared_workflows: list[dict[str, Any]] = []
        self._workflow_id = 0
        self._unavailable_workers: dict[int, str] = {}
        self._consecutive_planner_failures = 0
        self.worker_protocol_errors = 0
        self.worker_protocol_repairs = 0
        self.protocol_replans = 0
        self.provider_owner_retries = 0
        self.provider_replans = 0
        self.provider_failure_events: list[dict[str, Any]] = []
        self.unrecoverable_planning_failures = 0
        self.planner_failures = 0
        self.conductor_workflows = 0
        self.workflow_agent_continuations = 0
        self.completed_workflow_steps = 0
        self.completed_workflows = 0
        self.discarded_workflow_steps = 0
        self.local_completion_confirmations = 0
        self.forced_workflow_handoffs = 0
        self.call_lease_handoffs = 0
        self.time_lease_handoffs = 0
        self.late_root_promotions = 0
        self.unstable_completion_rejections = 0
        self.local_terminal_polls = 0
        self.local_terminal_poll_seconds = 0.0
        self.local_handoff_interrupts = 0
        self.paid_worker_call_attempts = 0
        self.paid_worker_call_limit_responses = 0
        self.task_budget_stop_responses = 0
        self.runtime_turns = 0
        self.live_control_decisions: list[dict[str, Any]] = []
        self.live_control_failures = 0
        self.live_control_corrections = 0
        self.live_control_dead_end_completions = 0
        self.live_control_dead_end_polls = 0
        self.live_control_normalizations = 0
        self.live_control_replacement_plans = 0
        self.live_control_replacement_plan_failures = 0
        self.conductor_interrupted_positions = 0
        self._live_completion_pending: str | None = None
        self._terminus_parser = TerminusJSONPlainParser()

    def set_task_instruction(self, instruction: str) -> None:
        self._task_instruction = instruction

    def note_workspace_recovery(self, detail: dict[str, Any]) -> None:
        """Expose a runtime recovery as persistent memory to every later workflow."""
        self._shared_workflows.append(
            {
                "workflow_id": (
                    self._workflow.workflow_id if self._workflow is not None else None
                ),
                "outcome": "runtime_workspace_recovery",
                "detail": detail,
                "steps": [],
            }
        )

    def reset_for_run(self) -> None:
        """Clear all task-scoped workflow and private-agent state."""
        self.routes.clear()
        self._workflow = None
        self._shared_workflows.clear()
        self._workflow_id = 0
        self._unavailable_workers.clear()
        self._consecutive_planner_failures = 0
        self.worker_protocol_errors = 0
        self.worker_protocol_repairs = 0
        self.protocol_replans = 0
        self.provider_owner_retries = 0
        self.provider_replans = 0
        self.provider_failure_events.clear()
        self.unrecoverable_planning_failures = 0
        self.planner_failures = 0
        self.conductor_workflows = 0
        self.workflow_agent_continuations = 0
        self.completed_workflow_steps = 0
        self.completed_workflows = 0
        self.discarded_workflow_steps = 0
        self.local_completion_confirmations = 0
        self.forced_workflow_handoffs = 0
        self.call_lease_handoffs = 0
        self.time_lease_handoffs = 0
        self.late_root_promotions = 0
        self.unstable_completion_rejections = 0
        self.local_terminal_polls = 0
        self.local_terminal_poll_seconds = 0.0
        self.local_handoff_interrupts = 0
        self.paid_worker_call_attempts = 0
        self.paid_worker_call_limit_responses = 0
        self.task_budget_stop_responses = 0
        self.runtime_turns = 0
        self.live_control_decisions.clear()
        reset_controller_traces = getattr(self._live_controller, "reset_traces", None)
        if callable(reset_controller_traces):
            reset_controller_traces()
        self.live_control_failures = 0
        self.live_control_corrections = 0
        self.live_control_dead_end_completions = 0
        self.live_control_dead_end_polls = 0
        self.live_control_normalizations = 0
        self.live_control_replacement_plans = 0
        self.live_control_replacement_plan_failures = 0
        self.conductor_interrupted_positions = 0
        self._live_completion_pending = None
        self._route_log.unlink(missing_ok=True)
        self._paid_call_log.unlink(missing_ok=True)
        self._provider_failure_log.unlink(missing_ok=True)

    @staticmethod
    def _bounded(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return (
            f"[...{len(text) - limit} earlier characters omitted...]\n{text[-limit:]}"
        )

    def _render_shared_memory(self, limit: int = MAX_SHARED_MEMORY_CHARS) -> str:
        if not self._shared_workflows:
            return "None."
        rendered = json.dumps(self._shared_workflows, ensure_ascii=True, indent=2)
        return self._bounded(rendered, limit)

    @staticmethod
    def _without_runtime_identities(value: Any) -> Any:
        """Remove private model/provider provenance before conductor inference."""
        if isinstance(value, dict):
            return {
                key: FuguRoutedLLM._without_runtime_identities(item)
                for key, item in value.items()
                if key
                not in {"worker_model", "worker_models", "provider", "provider_base"}
            }
        if isinstance(value, list):
            return [FuguRoutedLLM._without_runtime_identities(item) for item in value]
        return value

    def _live_control_state(
        self,
        terminal_observation: str,
        *,
        terminal_ready: bool | None,
        elapsed_s: float | None,
        remaining_s: float | None,
    ) -> LiveControlState:
        if self._worker_profiles is None:
            raise RuntimeError("live control state requested without worker profiles")
        workflow = self._workflow
        positions: tuple[ControlPosition, ...] = ()
        active_position_id: int | None = None
        workflow_id: int | None = None
        if workflow is not None:
            workflow_id = workflow.workflow_id
            active_position_id = workflow.active_index
            position_rows: list[ControlPosition] = []
            for index, agent in enumerate(workflow.agents):
                if index == workflow.active_index:
                    status = "active"
                elif agent.status == "pending":
                    status = "pending"
                elif agent.status == "completed":
                    status = "completed"
                elif agent.status == "interrupted_by_conductor":
                    status = "interrupted"
                else:
                    status = "failed"
                position_rows.append(
                    ControlPosition(
                        position_id=index,
                        worker_id=agent.route.decision.worker_id,
                        subtask=agent.route.decision.subtask,
                        access=agent.route.access,
                        status=status,
                        progress={
                            "worker_report": agent.progress,
                            "completion_requested": agent.completion_requested,
                            "turns": agent.turns,
                            "checkpoint": agent.checkpoint,
                            "recent_activity": agent.recent_activity,
                            "material_progress": (
                                {
                                    "latest_turn_changed_material_state": (
                                        agent.latest_material_change
                                    ),
                                    "last_material_progress_turn": (
                                        agent.last_material_progress_turn
                                    ),
                                    "turns_without_material_progress": (
                                        agent.turns_without_material_progress
                                    ),
                                    "repeated_environment_action_batches": (
                                        agent.repeated_environment_action_batches
                                    ),
                                }
                                if agent.latest_material_change is not None
                                else None
                            ),
                        },
                        artifacts=tuple(agent.artifacts),
                    )
                )
            positions = tuple(position_rows)

        effective_elapsed_s = max(0.0, elapsed_s or 0.0)
        wall_time_limit_s = (
            effective_elapsed_s + max(0.0, remaining_s)
            if remaining_s is not None
            else TERMINAL_TASK_BUDGET_S
        )
        effective_terminal_ready = terminal_ready
        if workflow is None and effective_terminal_ready is None:
            effective_terminal_ready = True
        return LiveControlState(
            original_task=self._task_instruction,
            workers=self._worker_profiles,
            workflow_id=workflow_id,
            positions=positions,
            active_position_id=active_position_id,
            terminal_status=(
                "ready"
                if effective_terminal_ready is True
                else "busy" if effective_terminal_ready is False else "unknown"
            ),
            terminal_observation=terminal_observation,
            shared_memory=tuple(
                self._without_runtime_identities(self._shared_workflows)
            ),
            budget=ControlBudget(
                paid_calls_used=self.paid_worker_call_attempts,
                paid_call_limit=self._max_agent_turns,
                elapsed_s=effective_elapsed_s,
                wall_time_limit_s=max(1.0, wall_time_limit_s),
            ),
            unavailable_worker_ids=tuple(sorted(self._unavailable_workers)),
        )

    @staticmethod
    def _live_completion_response(reason: str) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(
                {
                    "analysis": reason,
                    "plan": "Return the conductor's verified completion decision.",
                    "commands": [],
                    "task_complete": True,
                }
            ),
            model_name="fugu-live-conductor",
        )

    def _apply_live_control_action(
        self,
        action: ControlAction,
        *,
        replacement_raw_plan: str | None = None,
        normalization: dict[str, Any] | None = None,
        rejected_attempts: tuple[dict[str, Any], ...] = (),
        controller_trace_index: int | None = None,
    ) -> LLMResponse | None:
        active = self._workflow.active if self._workflow is not None else None
        interrupts_unfinished = bool(
            active is not None
            and not active.completion_requested
            and action.action in {"handoff", "replan"}
        )
        decision_record = {
            "decision": len(self.live_control_decisions) + 1,
            "workflow_id": (
                self._workflow.workflow_id if self._workflow is not None else None
            ),
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
            "interrupts_unfinished_position": interrupts_unfinished,
            "controller_trace_index": controller_trace_index,
        }
        if normalization is not None:
            decision_record["normalization"] = normalization
        if rejected_attempts:
            decision_record["rejected_attempts"] = list(rejected_attempts)
        if replacement_raw_plan is not None:
            decision_record["replacement_raw_plan"] = self._bounded(
                replacement_raw_plan,
                8_000,
            )
        self.live_control_decisions.append(decision_record)
        if action.action == "continue":
            return None
        if action.action == "handoff":
            if self._workflow is None or action.target_position_id is None:
                raise RuntimeError("validated handoff lost its workflow target")
            active = self._workflow.active
            if active.completion_requested:
                active.status = "completed"
                active.final_response = self._latest_agent_response(active)
                self.completed_workflow_steps += 1
            else:
                self._interrupt_active_position(action.reason, action="handoff")
            self._workflow.active_index = action.target_position_id
            return None
        if action.action == "replan":
            if self._workflow is not None:
                if not self._workflow.active.completion_requested:
                    self._interrupt_active_position(action.reason, action="replan")
                self._archive_workflow("replanned", action.reason)
            steps = tuple(
                PlannedStep(
                    worker_id=step.worker_id,
                    subtask=step.subtask,
                    access=step.access,
                )
                for step in action.steps
            )
            decision = RouteDecision(
                worker_id=steps[0].worker_id,
                subtask=steps[0].subtask,
                raw_plan=(
                    replacement_raw_plan
                    or json.dumps(self.live_control_decisions[-1], ensure_ascii=True)
                ),
                workflow_steps=steps,
            )
            self._start_workflow(decision)
            return None

        if self._workflow is not None:
            self._live_completion_pending = action.reason
        return self._live_completion_response(action.reason)

    def _interrupt_active_position(self, reason: str, *, action: str) -> None:
        """Preserve an unfinished position without claiming that it completed."""
        if self._workflow is None:
            raise RuntimeError("cannot interrupt a position without an active workflow")
        agent = self._workflow.active
        agent.status = "interrupted_by_conductor"
        agent.final_response = self._latest_agent_response(agent)
        agent.handoff_reason = reason
        agent.checkpoint = {
            "kind": "conductor_interruption",
            "action": action,
            "reason": reason,
            "agent_turns": agent.turns,
            "paid_calls_used": self.paid_worker_call_attempts,
            "terminal_ready": agent.terminal_ready,
            "progress": agent.progress,
            "artifacts": agent.artifacts,
            "recent_activity": agent.recent_activity,
            "material_progress": {
                "latest_turn_changed_material_state": agent.latest_material_change,
                "last_material_progress_turn": agent.last_material_progress_turn,
                "turns_without_material_progress": (
                    agent.turns_without_material_progress
                ),
                "repeated_environment_action_batches": (
                    agent.repeated_environment_action_batches
                ),
            },
        }
        self.conductor_interrupted_positions += 1

    async def _plan_live_replacement(
        self,
        decision: ControlAction,
        state: LiveControlState,
    ) -> tuple[ControlAction, str]:
        availability_setter = getattr(self._planner, "set_unavailable_workers", None)
        if callable(availability_setter):
            availability_setter(frozenset(self._unavailable_workers))
        planned = await self._planner.route(
            self._replacement_planning_prompt(state, decision.reason),
            [],
        )
        if planned.fallback_reason:
            self.planner_failures += 1
            self.live_control_replacement_plan_failures += 1
            self._consecutive_planner_failures += 1
            raise RuntimeError(
                f"replacement workflow planning failed: {planned.fallback_reason}"
            )
        steps = planned.workflow_steps or (
            PlannedStep(planned.worker_id, planned.subtask),
        )
        if planned.worker_id not in self._workers or any(
            step.worker_id not in self._workers for step in steps
        ):
            self.planner_failures += 1
            self.live_control_replacement_plan_failures += 1
            self._consecutive_planner_failures += 1
            raise RuntimeError("replacement workflow selected an unknown worker")
        selected_unavailable = sorted(
            {step.worker_id for step in steps if step.worker_id in self._unavailable_workers}
        )
        if selected_unavailable:
            self.planner_failures += 1
            self.live_control_replacement_plan_failures += 1
            self._consecutive_planner_failures += 1
            raise RuntimeError(
                "replacement workflow selected unavailable worker slots "
                f"{selected_unavailable}"
            )
        action = replace(
            decision,
            steps=tuple(
                ControlStep(
                    worker_id=step.worker_id,
                    subtask=step.subtask,
                    access=step.access,
                )
                for step in steps
            ),
        )
        validate_control_action(action, state)
        self._consecutive_planner_failures = 0
        self.live_control_replacement_plans += 1
        return action, planned.raw_plan

    async def _run_live_controller(
        self,
        terminal_observation: str,
        *,
        terminal_ready: bool | None,
        elapsed_s: float | None,
        remaining_s: float | None,
    ) -> LLMResponse | None:
        if self._live_controller is None:
            return None
        terminal_observation = latest_terminal_observation(terminal_observation)
        state = self._live_control_state(
            terminal_observation,
            terminal_ready=terminal_ready,
            elapsed_s=elapsed_s,
            remaining_s=remaining_s,
        )
        if not enabled_control_actions(state):
            # Dead-end state: no progress action and no completion is legal.
            # The decode schema is derived deterministically from this state,
            # so asking the conductor — or re-asking it under a correction —
            # cannot produce a legal action. Two very different causes reach
            # here and they need opposite responses.
            if state.budget.paid_calls_remaining > 0 and (
                state.budget.wall_time_remaining_s > 0
            ):
                # TRANSIENT: budget remains, but the terminal is unstable while
                # the owner has already requested completion (e.g. a long
                # install still running). Finalizing here would abandon a task
                # that merely needs the foreground process to settle, so poll
                # locally and re-decide once the terminal is stable.
                self.live_control_dead_end_polls += 1
                return self._local_poll_response(LOCAL_POLL_INTERVAL_S)
            # TERMINAL: the paid or wall-clock budget is exhausted, so no
            # further progress is possible. Finish on the work already in the
            # workspace instead of failing closed, which would discard every
            # artifact produced so far.
            self.live_control_dead_end_completions += 1
            return self._live_completion_response(
                "no legal control action remains at this state; finalizing on "
                "the work already completed"
            )
        correction: str | None = None
        rejected_attempts: list[dict[str, Any]] = []
        last_contract_error = "no decision attempt was made"
        for attempt in range(1 + MAX_DECISION_CORRECTIONS):
            try:
                if correction is None:
                    decision = await self._live_controller.decide(state)
                else:
                    decision = await self._live_controller.decide(
                        state,
                        correction=correction,
                    )
            except ControlContractError as exc:
                last_contract_error = str(exc)
                rejected_attempts.append(
                    {
                        "action": None,
                        "target_position_id": None,
                        "error": self._bounded(last_contract_error, 600),
                    }
                )
                if attempt < MAX_DECISION_CORRECTIONS:
                    correction = (
                        render_control_action_correction(
                            last_contract_error,
                            correction_attempt=attempt + 1,
                        )
                        if self._live_controller_supplies_topology
                        else render_decision_correction(
                            last_contract_error,
                            correction_attempt=attempt + 1,
                        )
                    )
                    self.live_control_corrections += 1
                continue
            except Exception as exc:
                self.live_control_failures += 1
                return self._conductor_error_response(
                    f"live conductor action failed validation: {type(exc).__name__}: {exc}"
                )
            decision, normalization = canonicalize_control_decision(decision, state)
            if normalization is not None:
                self.live_control_normalizations += 1
            try:
                active = self._workflow.active if self._workflow is not None else None
                if (
                    decision.action == "continue"
                    and active is not None
                    and active.repeated_environment_action_batches
                    >= REPEATED_ENVIRONMENT_ACTION_LIMIT
                ):
                    raise ControlContractError(
                        "continue is invalid after the active position repeated the "
                        "same environment-action batch without objective material "
                        "progress; choose handoff or replan"
                    )
                replacement_raw_plan = None
                action = decision
                if self._live_controller_supplies_topology:
                    validate_control_action(decision, state)
                    action = decision
                    replacement_raw_plan = (
                        json.dumps(
                            {
                                "action": decision.action,
                                "reason": decision.reason,
                                "target_position_id": decision.target_position_id,
                                "steps": [
                                    {
                                        "worker_id": step.worker_id,
                                        "subtask": step.subtask,
                                        "access": list(step.access),
                                    }
                                    for step in decision.steps
                                ],
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                        )
                        if decision.action == "replan"
                        else None
                    )
                    if decision.action == "replan":
                        self.live_control_replacement_plans += 1
                else:
                    validate_control_decision(decision, state)
                    if decision.action == "replan":
                        action, replacement_raw_plan = await self._plan_live_replacement(
                            decision,
                            state,
                        )
                    else:
                        validate_control_action(action, state)
            except ControlContractError as exc:
                last_contract_error = str(exc)
                rejected_attempts.append(
                    {
                        "action": decision.action,
                        "target_position_id": decision.target_position_id,
                        "error": self._bounded(last_contract_error, 600),
                    }
                )
                if attempt < MAX_DECISION_CORRECTIONS:
                    correction = (
                        render_control_action_correction(
                            last_contract_error,
                            correction_attempt=attempt + 1,
                        )
                        if self._live_controller_supplies_topology
                        else render_decision_correction(
                            last_contract_error,
                            correction_attempt=attempt + 1,
                        )
                    )
                    self.live_control_corrections += 1
                continue
            except Exception as exc:
                self.live_control_failures += 1
                return self._conductor_error_response(
                    f"live conductor action failed validation: {type(exc).__name__}: {exc}"
                )
            return self._apply_live_control_action(
                action,
                replacement_raw_plan=replacement_raw_plan,
                normalization=normalization,
                rejected_attempts=tuple(rejected_attempts),
                controller_trace_index=(
                    len(self._live_controller.decision_traces) - 1
                    if isinstance(
                        getattr(self._live_controller, "decision_traces", None),
                        list,
                    )
                    and self._live_controller.decision_traces
                    else None
                ),
            )
        self.live_control_failures += 1
        return self._conductor_error_response(
            "live conductor decision rejected after "
            f"{MAX_DECISION_CORRECTIONS} corrections: {last_contract_error}"
        )

    def _render_dependencies(self, agent: WorkflowAgentState) -> str:
        if self._workflow is None or not agent.route.access:
            return "None."
        dependencies = []
        for index in agent.route.access:
            source = self._workflow.agents[index]
            dependencies.append(
                {
                    "step": index + 1,
                    (
                        "worker_id"
                        if self._live_controller is not None
                        else "worker_model"
                    ): (
                        source.route.decision.worker_id
                        if self._live_controller is not None
                        else self._worker_names[source.route.decision.worker_id]
                    ),
                    "subtask": source.route.decision.subtask,
                    "final_response": source.final_response,
                    "checkpoint": source.checkpoint,
                    "progress": source.progress,
                    "artifacts": source.artifacts,
                    "recent_activity": source.recent_activity,
                    "private_trajectory": source.messages,
                }
            )
        rendered = json.dumps(dependencies, ensure_ascii=True, indent=2)
        return self._bounded(rendered, MAX_DEPENDENCY_CONTEXT_CHARS)

    def _archive_workflow(self, outcome: str, detail: str | None = None) -> None:
        workflow = self._workflow
        if workflow is None:
            return
        self._shared_workflows.append(
            {
                "workflow_id": workflow.workflow_id,
                "outcome": outcome,
                "detail": detail,
                "steps": [
                    {
                        "step": state.route.step_index + 1,
                        "worker_id": state.route.decision.worker_id,
                        "worker_model": self._worker_names.get(
                            state.route.decision.worker_id,
                            f"worker-{state.route.decision.worker_id}",
                        ),
                        "subtask": state.route.decision.subtask,
                        "access": list(state.route.access),
                        "status": state.status,
                        "trajectory": state.messages,
                        "final_response": state.final_response,
                        "checkpoint": state.checkpoint,
                        "progress": state.progress,
                        "artifacts": state.artifacts,
                        "recent_activity": state.recent_activity,
                    }
                    for state in workflow.agents
                ],
            }
        )
        if outcome == "completed":
            self.completed_workflows += 1
        self._workflow = None
        self._live_completion_pending = None

    def discard_pending_workflow(self, reason: str = "runtime_discard") -> None:
        if self._workflow is None:
            return
        self.discarded_workflow_steps += sum(
            state.status != "completed" for state in self._workflow.agents
        )
        self._archive_workflow("aborted", reason)

    def _start_workflow(self, decision: RouteDecision) -> WorkflowAgentState:
        steps = decision.workflow_steps or (
            PlannedStep(decision.worker_id, decision.subtask),
        )
        self._workflow_id += 1
        self.conductor_workflows += 1
        routes = [
            ActiveRoute(
                decision=RouteDecision(
                    worker_id=step.worker_id,
                    subtask=step.subtask,
                    raw_plan=decision.raw_plan,
                ),
                workflow_id=self._workflow_id,
                step_index=index,
                step_count=len(steps),
                access=step.access,
                workflow_steps=steps,
            )
            for index, step in enumerate(steps)
        ]
        self._workflow = WorkflowExecutionState(
            workflow_id=self._workflow_id,
            agents=[WorkflowAgentState(route=route) for route in routes],
        )
        return self._workflow.active

    def _complete_active_step(self, response_content: str) -> None:
        if self._workflow is None:
            return
        agent = self._workflow.active
        agent.status = "completed"
        agent.final_response = response_content
        self.completed_workflow_steps += 1
        if self._workflow.active_index < len(self._workflow.agents) - 1:
            self._workflow.active_index += 1

    def _exhaust_active_position(
        self,
        response_content: str,
        *,
        reason: str = "fair_call_lease_exhausted",
        terminal_observation: str | None = None,
    ) -> None:
        """Release a position that consumed its turn allocation without completing."""
        if self._workflow is None:
            return
        agent = self._workflow.active
        agent.status = reason
        agent.handoff_reason = reason
        agent.final_response = response_content
        agent.checkpoint = {
            "reason": reason,
            "agent_turns": agent.turns,
            "paid_calls_used": (
                self.paid_worker_call_attempts - agent.paid_call_start
                if agent.paid_call_start is not None
                else None
            ),
            "paid_call_limit": agent.paid_call_limit,
            "terminal_ready": agent.terminal_ready,
            "latest_terminal_observation": self._bounded(
                terminal_observation or "Unavailable.",
                6_000,
            ),
            "latest_worker_response": self._bounded(response_content, 6_000),
            "progress": agent.progress,
            "artifacts": agent.artifacts,
            "local_handoff_interrupts": agent.local_handoff_interrupts,
        }
        self.forced_workflow_handoffs += 1
        if reason == "fair_call_lease_exhausted":
            self.call_lease_handoffs += 1
        elif reason == "fair_time_lease_exhausted":
            self.time_lease_handoffs += 1
        if self._workflow.active_index < len(self._workflow.agents) - 1:
            self._workflow.active_index += 1
            return
        self._archive_workflow(
            reason,
            "root position exhausted its fair lease; conductor must replan",
        )

    def _initialize_position_lease(
        self,
        agent: WorkflowAgentState,
        *,
        elapsed_s: float | None,
        remaining_s: float | None,
    ) -> None:
        """Give the active position a fair share while forwarding unused capacity."""
        if agent.paid_call_start is not None:
            return
        if self._workflow is None:
            return
        positions_remaining = len(self._workflow.agents) - self._workflow.active_index
        agent.paid_call_start = self.paid_worker_call_attempts
        if self._fair_position_call_budget is None:
            agent.paid_call_limit = self._max_agent_turns
            agent.lease_started_elapsed_s = elapsed_s
            return
        if agent.route.step_index >= agent.route.step_count - 1:
            agent.paid_call_limit = self._max_agent_turns
            if elapsed_s is not None and remaining_s is not None:
                agent.lease_started_elapsed_s = elapsed_s
                agent.lease_deadline_elapsed_s = elapsed_s + max(0.0, remaining_s)
            return
        fair_allocation_ceiling = min(
            self._max_agent_turns,
            self._fair_position_call_budget,
        )
        paid_calls_remaining = max(
            0,
            fair_allocation_ceiling - self.paid_worker_call_attempts,
        )
        fair_calls = max(1, paid_calls_remaining // max(1, positions_remaining))
        agent.paid_call_limit = min(
            self._max_agent_turns,
            self.paid_worker_call_attempts + fair_calls,
        )
        if elapsed_s is not None and remaining_s is not None:
            agent.lease_started_elapsed_s = elapsed_s
            agent.lease_deadline_elapsed_s = elapsed_s + max(0.0, remaining_s) / max(
                1, positions_remaining
            )

    def _position_allocation_reason(
        self,
        agent: WorkflowAgentState,
        *,
        elapsed_s: float | None,
    ) -> str | None:
        if self._fair_position_call_budget is None:
            return None
        route = agent.route
        if route.step_index >= route.step_count - 1:
            return None
        if (
            agent.paid_call_limit is not None
            and self.paid_worker_call_attempts >= agent.paid_call_limit
        ):
            return "fair_call_lease_exhausted"
        if (
            elapsed_s is not None
            and agent.lease_deadline_elapsed_s is not None
            and elapsed_s + POSITION_CALL_TIME_GUARD_S >= agent.lease_deadline_elapsed_s
        ):
            return "fair_time_lease_exhausted"
        return None

    @staticmethod
    def _position_call_timeout_s(
        agent: WorkflowAgentState,
        *,
        elapsed_s: float | None,
        remaining_s: float | None,
    ) -> float | None:
        limits = [WORKER_CALL_TIMEOUT_S]
        if remaining_s is not None:
            limits.append(max(1.0, remaining_s - COMMAND_COMPLETION_GUARD_S))
        if elapsed_s is not None and agent.lease_deadline_elapsed_s is not None:
            limits.append(
                max(
                    1.0,
                    agent.lease_deadline_elapsed_s
                    - elapsed_s
                    - POSITION_CALL_TIME_GUARD_S,
                )
            )
        return min(limits) if limits else None

    def note_terminal_wait(
        self,
        commands: list[Command],
        *,
        is_task_complete: bool,
    ) -> None:
        """Allow local polling only after the command owner explicitly chose to wait."""
        if self._workflow is None:
            return
        agent = self._workflow.active
        wait_only = (
            bool(commands)
            and not is_task_complete
            and all(not command.keystrokes.strip() for command in commands)
        )
        if wait_only and agent.terminal_ready is False:
            wait_duration_s = sum(
                max(0.0, float(command.duration_sec)) for command in commands
            )
            if agent.local_poll_enabled:
                # A runtime-generated poll is fed back through this hook. Keep the
                # elapsed time already accumulated in call() so the owner is
                # actually rechecked at LOCAL_POLL_OWNER_RECHECK_S.
                agent.local_poll_elapsed_s = max(
                    agent.local_poll_elapsed_s,
                    wait_duration_s,
                )
            else:
                agent.local_poll_enabled = True
                agent.local_poll_elapsed_s = wait_duration_s
            return
        agent.local_poll_enabled = False
        agent.local_poll_elapsed_s = 0.0

    @staticmethod
    def _local_poll_response(duration_s: float) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(
                {
                    "analysis": (
                        "The worker-owned foreground process is still running; the runtime is polling it locally."
                    ),
                    "plan": "Wait for a stable terminal before returning control.",
                    "commands": [{"keystrokes": "", "duration": round(duration_s, 3)}],
                    "task_complete": False,
                }
            ),
            model_name="fugu-runtime-poll",
        )

    @staticmethod
    def _local_handoff_interrupt_response() -> LLMResponse:
        return LLMResponse(
            content=json.dumps(
                {
                    "analysis": (
                        "The command owner's fair lease is exhausted and its "
                        "foreground process remained busy after the local grace poll."
                    ),
                    "plan": (
                        "Interrupt the owner process locally, then keep ownership until the terminal is stable."
                    ),
                    "commands": [
                        {
                            "keystrokes": "\u0003",
                            "duration": LOCAL_HANDOFF_INTERRUPT_WAIT_S,
                        }
                    ],
                    "task_complete": False,
                }
            ),
            model_name="fugu-runtime-handoff-interrupt",
        )

    @staticmethod
    def _position_retry_limit_response(reason: str) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(
                {
                    "analysis": (
                        "The worker's transient provider attempt consumed the "
                        f"remainder of this position's fair lease ({reason})."
                    ),
                    "plan": (
                        "Preserve the environment and release the position only after the terminal is stable."
                    ),
                    "commands": [],
                    "task_complete": False,
                }
            ),
            model_name="fugu-runtime-position-lease",
        )

    @staticmethod
    def _latest_agent_response(agent: WorkflowAgentState) -> str:
        for message in reversed(agent.messages):
            if message.get("role") == "assistant" and isinstance(
                message.get("content"), str
            ):
                return message["content"]
        return "The position reached its turn allocation at a stable terminal."

    @classmethod
    def _summarize_worker_activity(
        cls,
        payload: dict[str, Any],
        *,
        turn: int,
    ) -> dict[str, Any]:
        """Expose bounded action evidence to the conductor, not a sibling transcript."""
        commands = payload.get("commands")
        command_summaries: list[str] = []
        if isinstance(commands, list):
            for command in commands[:4]:
                if not isinstance(command, dict):
                    continue
                keystrokes = command.get("keystrokes")
                if isinstance(keystrokes, str) and keystrokes.strip():
                    command_summaries.append(cls._bounded(keystrokes.strip(), 500))

        activity: dict[str, Any] = {
            "turn": turn,
            "task_complete": (
                payload.get("task_complete") is True
                and not isinstance(payload.get("computer_action"), dict)
            ),
            "command_count": len(commands) if isinstance(commands, list) else 0,
            "command_summaries": command_summaries,
        }
        computer_action = payload.get("computer_action")
        if isinstance(computer_action, dict) and isinstance(
            computer_action.get("name"), str
        ):
            activity["computer_action"] = computer_action["name"]
        for key in ("analysis", "plan"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                activity[key] = cls._bounded(value.strip(), 1_000)
        if "progress" in payload:
            activity["progress"] = payload["progress"]
        if "artifacts" in payload:
            activity["artifacts"] = payload["artifacts"]
        return activity

    @staticmethod
    def _environment_action_signature(activity: dict[str, Any]) -> str | None:
        """Return a stable signature only for worker actions that touch the environment."""
        commands = activity.get("command_summaries")
        computer_action = activity.get("computer_action")
        if not commands and not computer_action:
            return None
        encoded = json.dumps(
            {
                "commands": commands or [],
                "computer_action": computer_action,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return encoded.decode("utf-8")

    def record_active_tool_observation(
        self,
        *,
        tool_name: str,
        text: str,
        is_error: bool = False,
        image_base64: str | None = None,
        media_type: str = "image/png",
    ) -> None:
        """Return a tool result only to the worker position that emitted it."""
        if self._workflow is None:
            raise RuntimeError("cannot record a tool result without an active workflow")
        active = self._workflow.active
        label = f"computer.{tool_name} {'failed' if is_error else 'result'}: {text}"
        if image_base64 is None:
            content: Any = label
        else:
            content = [
                {"type": "text", "text": label},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_base64}",
                        "detail": "auto",
                    },
                },
            ]
        active.messages.append({"role": "user", "content": content})
        active.recent_activity.append(
            {
                "turn": active.turns,
                "tool_observation": f"computer.{tool_name}",
                "is_error": is_error,
                "evidence": self._bounded(text, 1_000),
                "has_image": image_base64 is not None,
            }
        )
        del active.recent_activity[:-6]

    @staticmethod
    def _json_flag(content: str, name: str) -> bool:
        try:
            value = json.loads(content).get(name, False)
        except (AttributeError, json.JSONDecodeError, TypeError):
            return False
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes"}
        return bool(value)

    @staticmethod
    def _set_task_complete(content: str, value: bool) -> str:
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content
        payload["task_complete"] = value
        return json.dumps(payload, ensure_ascii=True)

    @staticmethod
    def _set_commands_empty(content: str) -> str:
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            payload = {
                "analysis": "The workflow position budget was exhausted.",
                "plan": "Release control to the next conductor-selected position.",
            }
        payload["commands"] = []
        payload["task_complete"] = False
        return json.dumps(payload, ensure_ascii=True)

    @staticmethod
    def _terminal_ready(prompt: str) -> bool | None:
        """Return whether a recognized Terminus observation ends at a shell prompt."""
        if not re.search(
            r"(?:Current terminal state|New Terminal Output):\s*", prompt
        ):
            return None
        observation = latest_terminal_observation(prompt)
        observation = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", observation).strip()
        if not observation:
            return None
        lines = [line.rstrip() for line in observation.splitlines() if line.strip()]
        if not lines:
            return None
        last_line = lines[-1]
        shell_prompt = re.search(
            r"(?:^|\s)(?:\([^\n]*\)\s*)?(?:[\w.-]+@[^\s:]+:)?[^\n]*[#$>]\s*$",
            last_line,
        )
        return shell_prompt is not None

    def _planning_prompt(self, terminal_state: str) -> str:
        unavailable: tuple[int | str, ...] = tuple(self._unavailable_workers)
        availability_mapper = getattr(self._planner, "planner_availability_refs", None)
        if callable(availability_mapper):
            unavailable = availability_mapper(tuple(self._unavailable_workers))
        selector_field_getter = getattr(self._planner, "planner_selector_field", None)
        selector_field = (
            selector_field_getter() if callable(selector_field_getter) else "worker_id"
        )
        return render_initial_planning_prompt(
            terminal_state,
            shared_memory=self._render_shared_memory(
                MAX_PLANNER_SHARED_MEMORY_CHARS
            ),
            unavailable_worker_ids=unavailable,
            selector_field=selector_field,
        )

    def _replacement_planning_prompt(
        self,
        state: LiveControlState,
        reason: str,
    ) -> str:
        positions = [
            {
                "position_id": position.position_id,
                "worker_id": position.worker_id,
                "subtask": self._bounded(position.subtask, 700),
                "access": list(position.access),
                "status": position.status,
                "progress": self._without_runtime_identities(position.progress),
                "artifacts": self._without_runtime_identities(list(position.artifacts)),
            }
            for position in state.positions
        ]
        current_context = self._bounded(
            json.dumps(
                {
                    "workflow_id": state.workflow_id,
                    "active_position_id": state.active_position_id,
                    "replan_reason": reason,
                    "positions": positions,
                },
                ensure_ascii=True,
            ),
            2_400,
        )
        return (
            f"{self._planning_prompt(state.terminal_observation)}\n\n"
            "LIVE REPLAN REQUEST: The live decision head selected replan. Generate a "
            "complete replacement workflow for the original task using the current "
            "workspace and the bounded partial evidence below. New access_list indexes "
            "start at zero and may reference only earlier positions in the replacement "
            "workflow. Do not assume a globally preferred or fallback worker. The old "
            "workflow is archived only after this replacement validates.\n"
            f"CURRENT WORKFLOW EVIDENCE:\n{current_context}"
        )

    def _agent_prompt(
        self,
        agent: WorkflowAgentState,
        terminal_observation: str,
        *,
        confirmation_turn: bool,
        elapsed_s: float | None,
        remaining_s: float | None,
    ) -> str:
        route = agent.route
        final_step = route.step_index == route.step_count - 1
        lease_calls_remaining = (
            max(0, agent.paid_call_limit - self.paid_worker_call_attempts)
            if agent.paid_call_limit is not None
            else None
        )
        lease_seconds_remaining = (
            max(0.0, agent.lease_deadline_elapsed_s - elapsed_s)
            if elapsed_s is not None and agent.lease_deadline_elapsed_s is not None
            else None
        )
        lease_status = (
            (
                "There is no per-position lease. Retain ownership of your private tool loop "
                "until the assigned subtask is complete. The workflow changes position only "
                "after an explicit completion; the global task budget is shared by all positions."
            )
            if self._fair_position_call_budget is None
            else (
                "Position lease: "
                f"{lease_calls_remaining} paid calls and "
                f"{lease_seconds_remaining:.0f}s remain before a downstream handoff."
                if lease_calls_remaining is not None
                and lease_seconds_remaining is not None
                and not final_step
                else (
                    f"Position lease: {lease_calls_remaining} paid calls remain before a downstream handoff."
                    if lease_calls_remaining is not None and not final_step
                    else "This is the final/root position and owns the remaining task budget."
                )
            )
        )
        global_calls_remaining = max(
            0,
            self._max_agent_turns - self.paid_worker_call_attempts,
        )
        deliverable_budget_contract = (
            f"Global worker-call budget: {global_calls_remaining} paid calls remain, "
            "including this response. For a task with an explicit output artifact, "
            "the position that owns implementation must create a minimally viable "
            "artifact at the exact required path by the end of its second owned call "
            "(or its first call if only one remains), then refine it in place. "
            "Inspection-only positions must hand off as soon as the facts needed to "
            "start implementation are known. Reserve the final available call for "
            "targeted artifact presence, syntax/schema, and task-level verification. "
            "Do not spend scarce calls probing optional packages when static inspection "
            "or a lightweight local check can advance the deliverable."
        )
        progress_contract = (
            "Treat the terminal as the source of truth. On the first turn, inspect the "
            "relevant actual files, interfaces, inputs, and tests before modifying the "
            "workspace; a workflow subtask never authorizes guessing unavailable facts. "
            "The workspace opened by the terminal is the authoritative challenge tree. "
            "If it already contains a repository, inspect and use that exact checkout; "
            "do not clone, install, or redirect later work to a replacement checkout "
            "elsewhere. Environment-setup prose never overrides an already prepared "
            "workspace. Keep task edits and verification in the authoritative tree. "
            "Move from inspection to concrete "
            "implementation as soon as the blocker is understood. In every response, add "
            "a `progress` object with `phase` (inspect, implement, verify, or done), "
            "`evidence`, and `material_change` (a boolean). Set `material_change=true` "
            "only when this turn established a new concrete fact, changed an artifact, "
            "or completed a new verification result; repeated inspection, plans, and a "
            "failed command are false. If the terminal reveals that a prior claimed edit "
            "did not occur, report false. Also add an `artifacts` array listing concrete "
            "paths and their current state. Before overwriting, moving, truncating, or deleting a pre-existing "
            "task input, first make a recoverable copy under "
            f"/tmp/fugu-input-snapshots/workflow-{route.workflow_id}-step-"
            f"{route.step_index + 1}/. Do not modify that snapshot."
        )
        if self._worker_tool_contract:
            progress_contract = (
                f"{progress_contract}\n{self._worker_tool_contract}"
            )
        if agent.handoff_requested:
            return (
                "Return exactly one valid json object matching the terminal response "
                "contract. This workflow position has consumed its turn allocation. "
                "Do not begin new work. If the shell prompt is available, return a concise "
                "status summary with commands=[] so the runtime can release the next "
                "conductor-selected position. If a command you started is still running, "
                "only poll it or interrupt it; you retain ownership until the terminal is "
                "stable. Include progress.phase, progress.evidence, and every artifact path "
                "created or changed in this position.\n\n"
                f"{lease_status}\n{deliverable_budget_contract}\n{progress_contract}\n\n"
                f"Terminal observation:\n{terminal_observation}"
            )
        if agent.turns == 0:
            initial_environment = (
                terminal_observation
                if route.workflow_id == 1 and route.step_index == 0
                else (
                    "You have access to the same persistent terminal environment. Inspect "
                    "it yourself when your subtask requires current filesystem state; no "
                    "unlisted sibling transcript is visible."
                )
            )
            completion_contract = (
                f"You are the final/root position. {FINAL_ROOT_QUALITY_CONTRACT} "
                "Set task_complete=true only when the "
                "overall user task is finished and no command is needed in that response. "
                "If this workflow is exhausted but the overall task needs a new conductor "
                "workflow, return task_complete=false, commands=[], and add "
                '"workflow_complete":true to the JSON object.'
                if final_step
                else (
                    "Set task_complete=true when your assigned subtask is finished and no "
                    "command is needed in that response. The runtime will release dependent "
                    "workflow positions without ending the overall user task."
                )
            )
            budget = (
                f"\nRuntime budget: {elapsed_s:.0f}s elapsed, {remaining_s:.0f}s remaining."
                if elapsed_s is not None and remaining_s is not None
                else ""
            )
            return (
                "You are an isolated agent position in a Fugu-Ultra workflow.\n"
                "Return exactly one valid json object matching the terminal response "
                "contract; do not return prose outside that json object.\n"
                f"Original user task:\n{self._task_instruction}\n\n"
                f"Workflow position: {route.step_index + 1}/{route.step_count}\n"
                f"Assigned subtask:\n{route.decision.subtask}\n\n"
                "Permitted current-workflow dependency trajectories from access_list:\n"
                f"{self._render_dependencies(agent)}\n\n"
                "Persistent memory from completed prior workflows:\n"
                f"{self._render_shared_memory()}\n\n"
                f"Initial environment information:\n{initial_environment}\n\n"
                f"{completion_contract}{budget}\n{lease_status}\n"
                f"{deliverable_budget_contract}\n{progress_contract}"
            )

        verification = (
            "This is the completion confirmation for your own prior claim. Recheck the "
            "overall task and return task_complete=true with no commands only if it is done."
            if confirmation_turn
            else (
                "This observation belongs to the terminal call you emitted. Continue your "
                "same assigned subtask from your private trajectory."
            )
        )
        return (
            "Return exactly one valid json object matching the terminal response "
            "contract.\n"
            f"{verification}\n{lease_status}\n{deliverable_budget_contract}\n"
            f"{progress_contract}\n\n"
            f"Terminal observation:\n{terminal_observation}"
        )

    @staticmethod
    def _planning_retry_response(error: str) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(
                {
                    "analysis": f"The conductor workflow was invalid: {error}",
                    "plan": "Retry conductor planning while preserving the environment.",
                    "commands": [],
                    "task_complete": False,
                }
            ),
            model_name="fugu-conductor-retry",
        )

    def _conductor_error_response(self, error: str) -> LLMResponse:
        if self._fail_closed_conductor_errors:
            raise RuntimeError(f"conductor failed closed: {error}")
        return self._planning_retry_response(error)

    @staticmethod
    def _planning_unrecoverable_response(error: str) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(
                {
                    "analysis": f"The registered conductor has no legal recovery: {error}",
                    "plan": "Stop immediately and preserve the current workspace.",
                    "commands": [],
                    "task_complete": True,
                }
            ),
            model_name="fugu-runtime-unrecoverable",
        )

    @staticmethod
    def _paid_worker_call_limit_response() -> LLMResponse:
        return LLMResponse(
            content=json.dumps(
                {
                    "analysis": (
                        "The registered paid worker-call limit has been reached; "
                        "the runtime will not issue another external request."
                    ),
                    "plan": "Stop worker execution and preserve the current workspace.",
                    "commands": [],
                    "task_complete": True,
                }
            ),
            model_name="fugu-runtime-paid-worker-cap",
        )

    @staticmethod
    def _task_budget_stop_response() -> LLMResponse:
        return LLMResponse(
            content=json.dumps(
                {
                    "analysis": (
                        "The global task budget is too small for another external "
                        "worker request; the runtime will preserve the workspace for grading."
                    ),
                    "plan": "Stop worker execution and submit the current workspace.",
                    "commands": [],
                    "task_complete": True,
                }
            ),
            model_name="fugu-runtime-task-budget",
        )

    def _paid_worker_call_available(self) -> bool:
        return self.paid_worker_call_attempts < self._max_agent_turns

    async def _call_worker(
        self,
        worker_id: int,
        prompt: str,
        message_history: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        if not self._paid_worker_call_available():
            raise RuntimeError("paid worker-call limit reached before dispatch")
        self.paid_worker_call_attempts += 1
        self._paid_call_log.parent.mkdir(parents=True, exist_ok=True)
        with self._paid_call_log.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"paid_call_attempt": self.paid_worker_call_attempts}
                )
                + "\n"
            )
        return await self._workers[worker_id].call(
            prompt,
            message_history=message_history,
            **kwargs,
        )

    @staticmethod
    def _provider_error_is_transient(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(
            needle in text
            for needle in (
                "timeout",
                "timed out",
                "connection",
                "rate limit",
                "ratelimit",
                "429",
                "502",
                "503",
                "504",
                "temporarily unavailable",
                "produced no executable content",
                "upstream request failed",
            )
        )

    def _mark_worker_unavailable(self, worker_id: int, reason: str) -> None:
        self._unavailable_workers[worker_id] = self._bounded(reason, 2_000)

    def _record_provider_failure(
        self,
        *,
        worker_id: int,
        exc: Exception,
        transient: bool,
        elapsed_s: float | None,
        failure_kind: str,
        terminal_observation: str,
    ) -> None:
        workflow = self._workflow
        agent = workflow.active if workflow is not None else None
        event: dict[str, Any] = {
            "event": len(self.provider_failure_events) + 1,
            "paid_call_attempt": self.paid_worker_call_attempts,
            "worker_id": worker_id,
            "worker_model": self._worker_names[worker_id],
            "error_type": type(exc).__name__,
            "error": self._bounded(str(exc), 2_000),
            "transient": transient,
            "failure_kind": failure_kind,
            "budget_elapsed_s": round(elapsed_s, 3) if elapsed_s is not None else None,
            "workflow_id": workflow.workflow_id if workflow is not None else None,
            "workflow_step_index": (
                agent.route.step_index + 1 if agent is not None else None
            ),
            "workflow_step_count": (
                agent.route.step_count if agent is not None else None
            ),
            "workflow_access": list(agent.route.access) if agent is not None else [],
            "subtask": agent.route.decision.subtask if agent is not None else None,
            "raw_plan": (
                agent.route.decision.raw_plan[:4_000] if agent is not None else ""
            ),
            "terminal_ready": agent.terminal_ready if agent is not None else None,
            "terminal_observation": self._bounded(terminal_observation, 12_000),
            "unavailable_worker_ids": tuple(sorted(self._unavailable_workers)),
        }
        self.provider_failure_events.append(event)

    def _finish_provider_failure_event(self) -> None:
        if not self.provider_failure_events:
            return
        event = self.provider_failure_events[-1]
        event["unavailable_worker_ids"] = tuple(sorted(self._unavailable_workers))
        event["archived_workflow"] = (
            self._without_runtime_identities(self._shared_workflows[-1])
            if self._shared_workflows
            else None
        )
        self._provider_failure_log.parent.mkdir(parents=True, exist_ok=True)
        with self._provider_failure_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any] | Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        del message_history  # Harbor's shared chat must never leak into private agents.
        elapsed_s = remaining_s = None
        if self._budget_status is not None:
            elapsed_s, remaining_s = self._budget_status()
        confirmation_turn = (
            "Are you sure you want to mark the task as complete?" in prompt
        )
        observed_terminal_ready = self._terminal_ready(prompt)

        if self._workflow is not None:
            active = self._workflow.active
            if observed_terminal_ready is not None:
                active.terminal_ready = observed_terminal_ready
            if observed_terminal_ready is True:
                active.local_poll_enabled = False
                active.local_poll_elapsed_s = 0.0
            elif observed_terminal_ready is False:
                allocation_reason = self._position_allocation_reason(
                    active,
                    elapsed_s=elapsed_s,
                )
                if allocation_reason is not None:
                    active.handoff_requested = True
                    active.handoff_reason = allocation_reason
                poll_budget_s = (
                    LOCAL_POLL_INTERVAL_S
                    if remaining_s is None
                    else min(
                        LOCAL_POLL_INTERVAL_S,
                        max(0.0, remaining_s - COMMAND_COMPLETION_GUARD_S),
                    )
                )
                if active.handoff_requested:
                    grace_remaining_s = max(
                        0.0,
                        LOCAL_HANDOFF_GRACE_S - active.local_poll_elapsed_s,
                    )
                    if (
                        active.local_handoff_interrupts == 0
                        and grace_remaining_s >= 1.0
                        and poll_budget_s >= 1.0
                    ):
                        duration_s = min(poll_budget_s, grace_remaining_s)
                        active.local_poll_enabled = True
                        active.local_poll_elapsed_s += duration_s
                        self.local_terminal_polls += 1
                        self.local_terminal_poll_seconds += duration_s
                        return self._local_poll_response(duration_s)
                    active.local_handoff_interrupts += 1
                    self.local_handoff_interrupts += 1
                    return self._local_handoff_interrupt_response()
                if active.local_poll_enabled:
                    should_recheck_owner = (
                        active.local_poll_elapsed_s >= LOCAL_POLL_OWNER_RECHECK_S
                    )
                    if not should_recheck_owner and poll_budget_s >= 1.0:
                        active.local_poll_elapsed_s += poll_budget_s
                        self.local_terminal_polls += 1
                        self.local_terminal_poll_seconds += poll_budget_s
                        return self._local_poll_response(poll_budget_s)
                    active.local_poll_enabled = False
                    active.local_poll_elapsed_s = 0.0

            if active.handoff_requested and observed_terminal_ready is None:
                poll_budget_s = (
                    LOCAL_POLL_INTERVAL_S
                    if remaining_s is None
                    else min(
                        LOCAL_POLL_INTERVAL_S,
                        max(0.0, remaining_s - COMMAND_COMPLETION_GUARD_S),
                    )
                )
                grace_remaining_s = max(
                    0.0,
                    LOCAL_HANDOFF_GRACE_S - active.local_poll_elapsed_s,
                )
                if (
                    active.local_handoff_interrupts == 0
                    and grace_remaining_s >= 1.0
                    and poll_budget_s >= 1.0
                ):
                    duration_s = min(poll_budget_s, grace_remaining_s)
                    active.local_poll_enabled = True
                    active.local_poll_elapsed_s += duration_s
                    self.local_terminal_polls += 1
                    self.local_terminal_poll_seconds += duration_s
                    return self._local_poll_response(duration_s)
                active.local_handoff_interrupts += 1
                self.local_handoff_interrupts += 1
                return self._local_handoff_interrupt_response()

        if (
            confirmation_turn
            and self._local_live_completion_confirmation
            and self._live_completion_pending is not None
            and observed_terminal_ready is True
        ):
            reason = self._live_completion_pending
            self.local_completion_confirmations += 1
            if self._workflow is not None:
                self._complete_active_step(
                    self._latest_agent_response(self._workflow.active)
                )
                self._archive_workflow("completed", "conductor completion confirmed")
            return self._live_completion_response(reason)

        active_completion_requested = bool(
            self._workflow is not None
            and self._workflow.active.completion_requested
        )
        if (
            not self._paid_worker_call_available()
            and not active_completion_requested
        ):
            self.paid_worker_call_limit_responses += 1
            return self._paid_worker_call_limit_response()

        live_control_terminal_ready = observed_terminal_ready
        if live_control_terminal_ready is None and self._workflow is not None:
            live_control_terminal_ready = self._workflow.active.terminal_ready

        # The preserved planning head was trained to create task-specific workflows;
        # the continuation head was trained on decisions inside an active workflow.
        # Keep those responsibilities aligned with their respective supervision.
        live_control_response = (
            await self._run_live_controller(
                prompt,
                terminal_ready=live_control_terminal_ready,
                elapsed_s=elapsed_s,
                remaining_s=remaining_s,
            )
            if (
                self._workflow is not None
                or self._live_controller_plans_initial_workflow
            )
            and not (confirmation_turn and self._live_completion_pending is not None)
            else None
        )
        if live_control_response is not None:
            return live_control_response

        if remaining_s is not None and remaining_s < MIN_WORKER_CALL_START_BUDGET_S:
            self.task_budget_stop_responses += 1
            self.discard_pending_workflow(
                "global task budget cannot support another worker request"
            )
            return self._task_budget_stop_response()

        if not self._paid_worker_call_available():
            self.paid_worker_call_limit_responses += 1
            return self._paid_worker_call_limit_response()
        self.runtime_turns += 1

        if self._workflow is not None:
            active = self._workflow.active
            allocation_reason = self._position_allocation_reason(
                active,
                elapsed_s=elapsed_s,
            )
            if active.terminal_ready is True and (
                active.handoff_requested or allocation_reason is not None
            ):
                self._exhaust_active_position(
                    self._latest_agent_response(active),
                    reason=(
                        active.handoff_reason
                        or allocation_reason
                        or "position_handoff_requested"
                    ),
                    terminal_observation=prompt,
                )

        if self._workflow is None:
            availability_setter = getattr(
                self._planner, "set_unavailable_workers", None
            )
            if callable(availability_setter):
                availability_setter(frozenset(self._unavailable_workers))
            planned = await self._planner.route(self._planning_prompt(prompt), [])
            if planned.fallback_reason:
                self.planner_failures += 1
                self._consecutive_planner_failures += 1
                if planned.unrecoverable:
                    self.unrecoverable_planning_failures += 1
                    return self._planning_unrecoverable_response(
                        planned.fallback_reason
                    )
                return self._conductor_error_response(planned.fallback_reason)
            if planned.worker_id not in self._workers or any(
                step.worker_id not in self._workers for step in planned.workflow_steps
            ):
                self.planner_failures += 1
                self._consecutive_planner_failures += 1
                return self._conductor_error_response(
                    "conductor selected an unknown worker"
                )
            selected_unavailable = sorted(
                {
                    step.worker_id
                    for step in (
                        planned.workflow_steps
                        or (PlannedStep(planned.worker_id, planned.subtask),)
                    )
                    if step.worker_id in self._unavailable_workers
                }
            )
            if selected_unavailable:
                self.planner_failures += 1
                self._consecutive_planner_failures += 1
                return self._conductor_error_response(
                    "conductor selected unavailable worker slots "
                    f"{selected_unavailable}; choose from the remaining pool"
                )
            self._consecutive_planner_failures = 0
            agent = self._start_workflow(planned)
            route_source = "conductor_workflow"
        else:
            agent = self._workflow.active
            if agent.turns:
                self.workflow_agent_continuations += 1
                route_source = "workflow_agent_continuation"
            else:
                route_source = "workflow_step_start"

        if observed_terminal_ready is not None:
            agent.terminal_ready = observed_terminal_ready
        self._initialize_position_lease(
            agent,
            elapsed_s=elapsed_s,
            remaining_s=remaining_s,
        )

        route = agent.route
        worker_id = route.decision.worker_id
        worker_name = self._worker_names[worker_id]
        routed_prompt = self._agent_prompt(
            agent,
            prompt,
            confirmation_turn=confirmation_turn,
            elapsed_s=elapsed_s,
            remaining_s=remaining_s,
        )
        worker_kwargs = dict(kwargs)
        worker_kwargs.pop("previous_response_id", None)
        worker_kwargs.setdefault("response_format", TERMINUS_RESPONSE_FORMAT)
        position_timeout_s = self._position_call_timeout_s(
            agent,
            elapsed_s=elapsed_s,
            remaining_s=remaining_s,
        )
        if position_timeout_s is not None:
            worker_kwargs["fugu_call_timeout_s"] = position_timeout_s

        try:
            response = await self._call_worker(
                worker_id,
                routed_prompt,
                list(agent.messages),
                **worker_kwargs,
            )
        except Exception as exc:
            if self._fail_closed_provider_errors:
                post_attempt_elapsed_s = elapsed_s
                if self._budget_status is not None:
                    post_attempt_elapsed_s, _ = self._budget_status()
                transient = self._provider_error_is_transient(exc)
                reason = (
                    "provider call failed in fail-closed collection mode: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._mark_worker_unavailable(worker_id, reason)
                self._record_provider_failure(
                    worker_id=worker_id,
                    exc=exc,
                    transient=transient,
                    elapsed_s=post_attempt_elapsed_s,
                    failure_kind="fail_closed_collection_provider_failure",
                    terminal_observation=prompt,
                )
                self.discard_pending_workflow(reason)
                self._finish_provider_failure_event()
                raise RuntimeError(reason) from exc
            if self._provider_error_is_transient(exc):
                post_attempt_elapsed_s = elapsed_s
                if self._budget_status is not None:
                    post_attempt_elapsed_s, _ = self._budget_status()
                retry_allocation_reason = self._position_allocation_reason(
                    agent,
                    elapsed_s=post_attempt_elapsed_s,
                )
                if retry_allocation_reason is not None:
                    agent.handoff_requested = True
                    agent.handoff_reason = retry_allocation_reason
                    return self._position_retry_limit_response(retry_allocation_reason)
                if not self._paid_worker_call_available():
                    self.paid_worker_call_limit_responses += 1
                    return self._paid_worker_call_limit_response()
                if self.provider_owner_retries >= self._provider_owner_retry_limit:
                    self.provider_replans += 1
                    reason = f"owner provider call failed without retry: {type(exc).__name__}: {exc}"
                    self._mark_worker_unavailable(worker_id, reason)
                    self._record_provider_failure(
                        worker_id=worker_id,
                        exc=exc,
                        transient=True,
                        elapsed_s=post_attempt_elapsed_s,
                        failure_kind="owner_call_failed_without_retry",
                        terminal_observation=prompt,
                    )
                    self.discard_pending_workflow(reason)
                    self._finish_provider_failure_event()
                    return self._planning_retry_response(str(exc))
                self.provider_owner_retries += 1
                try:
                    response = await self._call_worker(
                        worker_id,
                        routed_prompt,
                        list(agent.messages),
                        **worker_kwargs,
                    )
                except Exception as retry_exc:
                    self.provider_replans += 1
                    reason = f"owner provider retry failed: {type(retry_exc).__name__}: {retry_exc}"
                    self._mark_worker_unavailable(worker_id, reason)
                    self._record_provider_failure(
                        worker_id=worker_id,
                        exc=retry_exc,
                        transient=self._provider_error_is_transient(retry_exc),
                        elapsed_s=post_attempt_elapsed_s,
                        failure_kind="owner_retry_failed",
                        terminal_observation=prompt,
                    )
                    self.discard_pending_workflow(reason)
                    self._finish_provider_failure_event()
                    return self._planning_retry_response(str(retry_exc))
            else:
                self.provider_replans += 1
                reason = f"owner provider call failed: {type(exc).__name__}: {exc}"
                self._mark_worker_unavailable(worker_id, reason)
                self._record_provider_failure(
                    worker_id=worker_id,
                    exc=exc,
                    transient=False,
                    elapsed_s=elapsed_s,
                    failure_kind="owner_call_failed",
                    terminal_observation=prompt,
                )
                self.discard_pending_workflow(reason)
                self._finish_provider_failure_event()
                return self._planning_retry_response(str(exc))

        repaired_content, parse_result, was_repaired = repair_terminus_json(
            response.content,
            self._terminus_parser,
        )
        if was_repaired:
            self.worker_protocol_repairs += 1
            response = replace(response, content=repaired_content)

        try:
            response_payload = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            response_payload = {}
        if isinstance(response_payload, dict):
            activity = self._summarize_worker_activity(
                response_payload,
                turn=agent.turns + 1,
            )
            action_signature = self._environment_action_signature(activity)
            if action_signature is None:
                agent.last_environment_action_signature = None
                agent.repeated_environment_action_batches = 0
            elif action_signature == agent.last_environment_action_signature:
                agent.repeated_environment_action_batches += 1
            else:
                agent.last_environment_action_signature = action_signature
                agent.repeated_environment_action_batches = 1

            reported_material_change: bool | None = None
            if "progress" in response_payload:
                agent.progress = response_payload["progress"]
                reported_material_change = (
                    agent.progress.get("material_change")
                    if isinstance(agent.progress, dict)
                    else None
                )
            if agent.repeated_environment_action_batches >= 2:
                reported_material_change = False
            if isinstance(reported_material_change, bool):
                agent.latest_material_change = reported_material_change
                if reported_material_change:
                    agent.last_material_progress_turn = agent.turns + 1
                    agent.turns_without_material_progress = 0
                else:
                    agent.turns_without_material_progress += 1
            artifacts = response_payload.get("artifacts")
            if isinstance(artifacts, list):
                agent.artifacts = artifacts
            agent.recent_activity.append(activity)
            del agent.recent_activity[:-6]

        agent.messages.extend(
            [
                {"role": "user", "content": routed_prompt},
                {"role": "assistant", "content": response.content},
            ]
        )
        agent.turns += 1
        agent.status = "running"
        post_elapsed_s = elapsed_s
        if self._budget_status is not None:
            post_elapsed_s, _ = self._budget_status()
        allocation_reason = self._position_allocation_reason(
            agent,
            elapsed_s=post_elapsed_s,
        )

        record = {
            "turn": len(self.routes) + 1,
            "runtime_turn": self.runtime_turns,
            "worker_id": worker_id,
            "worker_model": worker_name,
            "route_source": route_source,
            "workflow_id": route.workflow_id,
            "workflow_step_index": route.step_index + 1,
            "workflow_step_count": route.step_count,
            "workflow_access": list(route.access),
            "agent_private_turn": agent.turns,
            "budget_elapsed_s": round(elapsed_s, 3) if elapsed_s is not None else None,
            "budget_remaining_s": (
                round(remaining_s, 3) if remaining_s is not None else None
            ),
            "subtask": route.decision.subtask,
            "fallback_reason": None,
            "raw_plan": route.decision.raw_plan[:4000],
            "terminal_ready": agent.terminal_ready,
            "position_paid_call_start": agent.paid_call_start,
            "position_paid_call_limit": agent.paid_call_limit,
            "paid_call_attempt": self.paid_worker_call_attempts,
            "position_lease_started_elapsed_s": agent.lease_started_elapsed_s,
            "position_lease_deadline_elapsed_s": agent.lease_deadline_elapsed_s,
            "reported_progress": agent.progress,
            "reported_artifacts": agent.artifacts,
            "material_progress": {
                "latest_turn_changed_material_state": agent.latest_material_change,
                "last_material_progress_turn": agent.last_material_progress_turn,
                "turns_without_material_progress": (
                    agent.turns_without_material_progress
                ),
                "repeated_environment_action_batches": (
                    agent.repeated_environment_action_batches
                ),
            },
        }
        self.routes.append(record)
        self._route_log.parent.mkdir(parents=True, exist_ok=True)
        with self._route_log.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

        if parse_result.error:
            self.worker_protocol_errors += 1
            if agent.handoff_requested and agent.terminal_ready is not False:
                self._exhaust_active_position(
                    response.content,
                    reason=agent.handoff_reason or "position_handoff_requested",
                    terminal_observation=prompt,
                )
                return replace(
                    response,
                    content=self._set_commands_empty(response.content),
                )
            agent.consecutive_protocol_errors += 1
            if agent.consecutive_protocol_errors >= 2:
                self.protocol_replans += 1
                error = parse_result.error
                self.discard_pending_workflow(
                    f"owner protocol repair exhausted: {error}"
                )
                return self._planning_retry_response(error)
            return response

        agent.consecutive_protocol_errors = 0
        has_commands = bool(parse_result.commands)
        has_external_action = bool(
            isinstance(response_payload, dict)
            and "computer_action" in response_payload
        )
        has_environment_actions = has_commands or has_external_action
        is_final = route.step_index == route.step_count - 1
        workflow_complete = self._json_flag(response.content, "workflow_complete")
        pending_live_confirmation = bool(
            confirmation_turn and self._live_completion_pending is not None
        )
        if pending_live_confirmation and (
            not parse_result.is_task_complete
            or has_environment_actions
            or agent.terminal_ready is False
        ):
            self._live_completion_pending = None
            agent.completion_requested = False

        if agent.handoff_requested and agent.terminal_ready is not False:
            self._exhaust_active_position(
                response.content,
                reason=agent.handoff_reason or "position_handoff_requested",
                terminal_observation=prompt,
            )
            return replace(
                response,
                content=self._set_commands_empty(response.content),
            )

        if parse_result.is_task_complete and has_environment_actions:
            agent.completion_requested = has_commands and not has_external_action
            if allocation_reason is not None:
                agent.handoff_requested = True
                agent.handoff_reason = allocation_reason
            return replace(
                response,
                content=self._set_task_complete(response.content, False),
            )

        if parse_result.is_task_complete and agent.terminal_ready is False:
            self.unstable_completion_rejections += 1
            if allocation_reason is not None:
                agent.handoff_requested = True
                agent.handoff_reason = allocation_reason
            return replace(
                response,
                content=self._set_task_complete(response.content, False),
            )

        if (
            self._live_controller is not None
            and (parse_result.is_task_complete or workflow_complete)
            and not (
                pending_live_confirmation
                and parse_result.is_task_complete
            )
        ):
            agent.completion_requested = True
            return replace(
                response,
                content=self._set_task_complete(response.content, False),
            )

        if (
            parse_result.is_task_complete
            and not is_final
            and not pending_live_confirmation
        ):
            self._complete_active_step(response.content)
            return replace(
                response,
                content=self._set_task_complete(response.content, False),
            )

        if workflow_complete and is_final and not has_environment_actions:
            self._complete_active_step(response.content)
            self._archive_workflow("completed", "root requested another workflow")
            return replace(
                response,
                content=self._set_task_complete(response.content, False),
            )

        if parse_result.is_task_complete and confirmation_turn and (
            is_final or pending_live_confirmation
        ):
            self.local_completion_confirmations += 1
            self._complete_active_step(response.content)
            self._archive_workflow("completed", "overall task confirmed")

        if (
            self._workflow is not None
            and allocation_reason is not None
            and not parse_result.is_task_complete
            and not workflow_complete
        ):
            if has_environment_actions or agent.terminal_ready is False:
                agent.handoff_requested = True
                agent.handoff_reason = allocation_reason
            else:
                self._exhaust_active_position(
                    response.content,
                    reason=allocation_reason,
                    terminal_observation=prompt,
                )
                return replace(
                    response,
                    content=self._set_commands_empty(response.content),
                )

        return response

    @override
    def get_model_context_limit(self) -> int:
        return min(
            worker.get_model_context_limit() for worker in self._workers.values()
        )

    @override
    def get_model_output_limit(self) -> int | None:
        limits = [
            limit
            for worker in self._workers.values()
            if (limit := worker.get_model_output_limit()) is not None
        ]
        return min(limits) if limits else None


DEFAULT_CONDUCTOR_GUIDANCE = (
    Path(__file__).resolve().parents[3]
    / "director/manifests/fugu_clean_v1/lesson_memory/conductor_guidance_v1.md"
)


def _attach_lesson_memory(live_controller: Any) -> None:
    """Attach the conductor's universal strategy guidance.

    One static text, versioned with the manifests, byte-identical for every
    task and every decision — the conductor itself judges which principles the
    current state warrants. ``FUGU_LESSON_MEMORY`` overrides the file path;
    the value ``off`` disables guidance. Any failure to load leaves the
    conductor unguided rather than failing the task.
    """
    path = os.environ.get("FUGU_LESSON_MEMORY", str(DEFAULT_CONDUCTOR_GUIDANCE))
    if not path or path.lower() in {"off", "none", "0", "disabled"}:
        return
    try:
        guidance = Path(path).read_text().strip()
    except OSError:
        return
    if not guidance:
        return

    live_controller.guidelines_provider = lambda state: guidance


class FuguUltraTerminalAgent(Terminus2):
    """Harbor custom agent for the accepted Fugu-Ultra product configuration."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        vllm_url: str = LOCAL_PLANNER_BASE,
        adapter: str = LOCAL_PRODUCT_PLANNER_ADAPTER,
        live_control_adapter: str = LOCAL_LIVE_CONTROL_ADAPTER,
        provider_base_url: str = YUNWU_API_BASE,
        worker_models: list[str] | tuple[str, ...] | None = None,
        reasoning_efforts: list[str] | tuple[str, ...] | None = None,
        pool_binding_path: str | Path = PRODUCT_POOL_BINDING,
        max_turns: int = MAX_AGENT_TURNS,
        provider_owner_retry_limit: int = 0,
        fail_closed_provider_errors: bool = False,
        solo_worker_id: int | None = None,
        typed_conductor_model: str | None = None,
        typed_conductor_url: str | None = None,
        typed_conductor_temperature: float = 0.0,
        typed_conductor_seed: int = 0,
        typed_conductor_record_token_data: bool = False,
        typed_conductor_policy_revision: str | None = None,
        typed_conductor_max_input_tokens: int | None = None,
        typed_conductor_max_output_tokens: int | None = None,
        worker_tool_tags: dict[int, tuple[str, ...]] | None = None,
        worker_tool_contract: str | None = None,
        worker_session_namespace: str = "fugu-ultra",
        **kwargs: Any,
    ) -> None:
        # Env overrides let harness entry points that construct this agent by
        # import path (e.g. harbor for Terminal-Bench) select the pool binding
        # and worker gateway without constructor access. Unset -> defaults.
        pool_binding_path = os.environ.get(
            "FUGU_POOL_BINDING", str(pool_binding_path)
        )
        env_solo = os.environ.get("FUGU_SOLO_WORKER_ID")
        if env_solo and solo_worker_id is None:
            # Lets an import-path harness measure one bound worker alone, as a
            # control for how much of a gap is orchestration versus the worker
            # harness itself.
            solo_worker_id = int(env_solo)
        provider_base_url = os.environ.get(
            "FUGU_PROVIDER_BASE_URL", provider_base_url
        )
        binding = load_pool_binding(Path(pool_binding_path))
        worker_models = (
            tuple(worker_models) if worker_models is not None else binding.runtime_models
        )
        reasoning_efforts = (
            tuple(reasoning_efforts)
            if reasoning_efforts is not None
            else binding.reasoning_efforts
        )
        verify_checkpoint_artifacts(binding, repo_root=REPO_ROOT)
        if tuple(worker_models) != binding.runtime_models:
            raise ValueError(
                f"the frozen conductor worker pool must remain {binding.runtime_models!r}"
            )
        if tuple(reasoning_efforts) != binding.reasoning_efforts:
            raise ValueError(
                f"the frozen conductor reasoning efforts must remain {binding.reasoning_efforts!r}"
            )
        if provider_base_url.rstrip("/") != binding.provider_base:
            raise ValueError(
                "all external worker requests must use the Yunwu gateway at "
                f"{binding.provider_base}"
            )
        try:
            verify_runtime_pool(
                binding,
                runtime_models=tuple(worker_models),
                reasoning_efforts=tuple(reasoning_efforts),
                provider_base=provider_base_url,
            )
        except PoolBindingError as exc:
            raise ValueError(
                f"the frozen conductor worker pool must remain bound to {binding.pool_id}: {exc}"
            ) from exc
        if vllm_url.rstrip("/") != LOCAL_PLANNER_BASE:
            raise ValueError(
                f"the frozen conductor planner must use {LOCAL_PLANNER_BASE}"
            )
        if adapter != LOCAL_PRODUCT_PLANNER_ADAPTER:
            raise ValueError(
                "the product conductor adapter must remain "
                f"{LOCAL_PRODUCT_PLANNER_ADAPTER!r}"
            )
        if live_control_adapter != LOCAL_LIVE_CONTROL_ADAPTER:
            raise ValueError(
                "the live-control adapter must remain "
                f"{LOCAL_LIVE_CONTROL_ADAPTER!r}"
            )
        if provider_base_url.rstrip("/") not in ALLOWED_WORKER_PROVIDER_BASES:
            raise ValueError(
                "external worker requests must use one of "
                f"{ALLOWED_WORKER_PROVIDER_BASES}, got {provider_base_url!r}"
            )
        if isinstance(solo_worker_id, bool) or (
            solo_worker_id is not None
            and solo_worker_id not in {slot.worker_id for slot in binding.slots}
        ):
            raise ValueError("solo_worker_id must identify one worker in the bound pool")
        bound_worker_ids = {slot.worker_id for slot in binding.slots}
        if worker_tool_tags is None:
            worker_tool_tags = {
                worker_id: ("terminal", "filesystem", "test_runner")
                for worker_id in bound_worker_ids
            }
        if set(worker_tool_tags) != bound_worker_ids:
            raise ValueError(
                "worker_tool_tags must describe every and only bound worker slot"
            )
        provider_key_env = WORKER_PROVIDER_KEY_ENV[provider_base_url.rstrip("/")]
        provider_key = os.environ.get(provider_key_env)
        if not provider_key:
            raise RuntimeError(
                f"{provider_key_env} is required for external worker requests "
                f"to {provider_base_url}"
            )
        worker_session_namespace = worker_session_namespace.strip()
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", worker_session_namespace):
            raise ValueError(
                "worker_session_namespace must be a non-empty, header-safe runtime identity"
            )

        configured_typed_model = (
            typed_conductor_model
            if typed_conductor_model is not None
            else os.environ.get("FUGU_TYPED_CONDUCTOR_MODEL")
        )
        using_product_default = (
            configured_typed_model is None and solo_worker_id is None
        )
        if using_product_default:
            configured_typed_model = PRODUCT_TYPED_CONDUCTOR_MODEL
        typed_conductor_model = configured_typed_model
        typed_conductor = bool(typed_conductor_model)
        typed_conductor_url = (
            typed_conductor_url
            or os.environ.get("FUGU_TYPED_CONDUCTOR_URL")
            or (PRODUCT_TYPED_CONDUCTOR_BASE if typed_conductor else vllm_url)
        )
        # Harness entry points that construct this agent by import path select
        # the conductor via env, which turns off `using_product_default` above.
        # Allow the product token budgets to be supplied the same way so the
        # decision request still fits the served context window.
        env_in = os.environ.get("FUGU_TYPED_CONDUCTOR_MAX_INPUT_TOKENS")
        if env_in and typed_conductor_max_input_tokens is None:
            typed_conductor_max_input_tokens = int(env_in)
        env_out = os.environ.get("FUGU_TYPED_CONDUCTOR_MAX_OUTPUT_TOKENS")
        if env_out and typed_conductor_max_output_tokens is None:
            typed_conductor_max_output_tokens = int(env_out)
        if typed_conductor_policy_revision is None and using_product_default:
            typed_conductor_policy_revision = PRODUCT_POLICY_REVISION
        if using_product_default:
            if typed_conductor_max_input_tokens is None:
                typed_conductor_max_input_tokens = (
                    PRODUCT_TYPED_CONDUCTOR_MAX_INPUT_TOKENS
                )
            if typed_conductor_max_output_tokens is None:
                typed_conductor_max_output_tokens = (
                    PRODUCT_TYPED_CONDUCTOR_MAX_OUTPUT_TOKENS
                )

        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name
            or (
                f"fugu-ultra/{typed_conductor_policy_revision}"
                if typed_conductor_policy_revision
                else "fugu-ultra/typed-conductor"
                if typed_conductor
                else "fugu-ultra/legacy-split"
            ),
            max_turns=max_turns,
            suppress_max_turns_warning=True,
            **kwargs,
        )
        worker_llms: dict[int, BaseLLM] = {}
        worker_names: dict[int, str] = {}
        for worker_id, (worker_model, effort) in enumerate(
            zip(worker_models, reasoning_efforts, strict=True)
        ):
            if solo_worker_id is not None and worker_id != solo_worker_id:
                continue
            worker_names[worker_id] = worker_model
            worker_llms[worker_id] = YunwuLiteLLM(
                # LiteLLM's openai/ prefix selects its OpenAI-compatible wire
                # protocol; api_base below fixes the actual provider (Yunwu or
                # OpenRouter, both OpenAI-compatible).
                model_name=f"openai/{worker_model}",
                api_base=provider_base_url,
                api_key=provider_key,
                timeout=WORKER_CALL_TIMEOUT_S,
                reasoning_effort=effort,
                session_id=f"{worker_session_namespace}-worker-{worker_id}",
            )

        if solo_worker_id is not None:
            if typed_conductor:
                raise ValueError(
                    "solo_worker_id cannot be combined with a typed conductor"
                )
            planner: Planner = FixedSoloPlanner(solo_worker_id)
            live_controller: LiveController | None = None
            worker_profiles: tuple[WorkerProfile, ...] | None = None
        else:
            planner = FrozenFuguPlanner(
                base_url=typed_conductor_url if typed_conductor else vllm_url,
                adapter=typed_conductor_model if typed_conductor else adapter,
                max_attempts=1,
                binding=binding,
                capability_set_interface=typed_conductor,
            )
            live_controller = OpenAILiveController(
                model=typed_conductor_model if typed_conductor else live_control_adapter,
                base_url=typed_conductor_url if typed_conductor else vllm_url,
                # Full typed replans may contain five independently scoped steps.
                # Keep this local structured action budget separate from paid
                # worker generation, which has no output-token cap here.
                max_tokens=(
                    typed_conductor_max_output_tokens
                    if typed_conductor and typed_conductor_max_output_tokens is not None
                    else MAX_CONTROL_OUTPUT_TOKENS
                    if typed_conductor
                    else 64
                ),
                seed=typed_conductor_seed,
                temperature=typed_conductor_temperature,
                record_token_data=typed_conductor_record_token_data,
                max_input_tokens=(
                    typed_conductor_max_input_tokens
                    if typed_conductor and typed_conductor_max_input_tokens is not None
                    else MAX_DECISION_INPUT_TOKENS
                ),
                supplies_topology=typed_conductor,
                capability_refs=typed_conductor,
                prompt_token_counter=LocalModelPromptTokenCounter(
                    model=(
                        typed_conductor_model
                        if typed_conductor
                        else live_control_adapter
                    ),
                    models_url=(
                        f"{typed_conductor_url}/models"
                        if typed_conductor
                        else LOCAL_MODELS_URL
                    ),
                ),
            )
            _attach_lesson_memory(live_controller)
            worker_profiles = tuple(
                WorkerProfile(
                    worker_id=slot.worker_id,
                    capability_tags=slot.role_prior,
                    tool_tags=tuple(worker_tool_tags[slot.worker_id]),
                )
                for slot in binding.slots
            )
        self._pool_binding = binding
        self._solo_worker_id = solo_worker_id
        self._typed_conductor_model = typed_conductor_model
        self._typed_conductor_url = typed_conductor_url
        self._typed_conductor_policy_revision = typed_conductor_policy_revision
        self._planner = planner
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
            provider_owner_retry_limit=provider_owner_retry_limit,
            fail_closed_provider_errors=fail_closed_provider_errors,
            live_controller=live_controller,
            worker_profiles=worker_profiles,
            live_controller_plans_initial_workflow=(
                typed_conductor and solo_worker_id is None
            ),
            live_controller_supplies_topology=(
                typed_conductor and solo_worker_id is None
            ),
            fail_closed_conductor_errors=True,
            local_live_completion_confirmation=(
                typed_conductor and solo_worker_id is None
            ),
            worker_tool_contract=worker_tool_contract,
        )
        self._llm = self._fugu_llm
        self._rejected_commandful_confirmations = 0
        self._blocked_workspace_commands = 0
        self._isolated_errexit_batches = 0
        self._collapsed_empty_wait_commands = 0
        self._converted_shell_wait_commands = 0
        self._rejected_overtime_commands = 0
        self._initialize_workspace_state()

    def _initialize_workspace_state(self) -> None:
        """Reset per-run workspace snapshot and recovery accounting."""
        self._active_environment: BaseEnvironment | None = None
        self._workspace_snapshot_ready = False
        self._workspace_root = WORKSPACE_ROOT
        self._workspace_snapshot_token = ""
        self._workspace_root_identity = ""
        self._workspace_snapshot_summary: dict[str, Any] = {}
        self._workspace_integrity_checks = 0
        self._workspace_recoveries = 0
        self._workspace_recovery_failures = 0
        self._workspace_cleanup_failures = 0

    def _after_fugu_run_reset(self, instruction: str) -> None:
        """Allow collection-only subclasses to restore a preregistered live state."""
        del instruction

    @staticmethod
    @override
    def name() -> str:
        return "fugu-ultra-terminal"

    @override
    def version(self) -> str | None:
        return f"stage2-final-{PRODUCT_RUNTIME_REVISION}"

    @override
    async def _query_llm(
        self,
        chat: Any,
        prompt: str,
        logging_paths: tuple[Any | None, Any | None, Any | None] = (
            None,
            None,
            None,
        ),
        original_instruction: str = "",
        session: Any | None = None,
    ) -> LLMResponse:
        # Terminus2 retries every exception three times. Bypass only that decorator;
        # the underlying context/summarization handling remains intact.
        undecorated_query = Terminus2._query_llm.__wrapped__
        return await undecorated_query(
            self,
            chat=chat,
            prompt=prompt,
            logging_paths=logging_paths,
            original_instruction=original_instruction,
            session=session,
        )

    @staticmethod
    def _workspace_exec_error(result: Any) -> str:
        output = getattr(result, "stderr", None) or getattr(result, "stdout", None)
        return str(output or "no command output").strip()

    async def _prepare_workspace_snapshot(self, environment: BaseEnvironment) -> None:
        """Capture the pristine task workspace before any paid worker can mutate it."""
        token = f"{time.time_ns()}-{id(self)}"
        discovery = repository_discovery_python(("/app", "/testbed"))
        command = f"""set -eu
snapshot_root={shlex.quote(WORKSPACE_SNAPSHOT_ROOT)}
workspace=$(python3 - <<'PY'
import pathlib
import subprocess

def run(*args):
    try:
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(args, 127, b'', b'git binary missing')

{discovery}
repository = discover_repository()
if repository is not None:
    print(repository)
else:
    for candidate in roots:
        if pathlib.Path(candidate).is_dir():
            print(candidate)
            break
PY
)
if [ -z "$workspace" ]; then
  printf 'no supported workspace root found (checked /app and /testbed)\n' >&2
  exit 1
fi
rm -rf -- "$snapshot_root"
mkdir -p "$snapshot_root/original"
cp -a --reflink=auto "$workspace"/. "$snapshot_root/original"/
file_count=$(find "$snapshot_root/original" -xdev -type f -print | wc -l)
size_kib=$(du -sk "$snapshot_root/original" | cut -f1)
workspace_identity=$(stat -Lc '%d:%i' "$workspace")
printf '%s\n' {shlex.quote(token)} > "$snapshot_root/{WORKSPACE_GUARD_TOKEN}"
printf 'ready\n' > "$snapshot_root/READY"
printf 'workspace_root=%s workspace_identity=%s file_count=%s size_kib=%s\n' "$workspace" "$workspace_identity" "$file_count" "$size_kib"
"""
        result = await environment.exec(
            command,
            cwd="/",
            timeout_sec=WORKSPACE_SNAPSHOT_TIMEOUT_S,
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(
                "failed to create the enforced workspace snapshot before paid work: "
                f"{self._workspace_exec_error(result)}"
            )
        root_match = re.search(
            r"(?:^|\s)workspace_root=((?:/app|/testbed)(?:/[^\s]*)?)(?:\s|$)",
            result.stdout or "",
        )
        if root_match is None:
            raise RuntimeError(
                "workspace snapshot did not attest a supported workspace root"
            )
        identity_match = re.search(
            r"(?:^|\s)workspace_identity=([0-9]+:[0-9]+)(?:\s|$)",
            result.stdout or "",
        )
        if identity_match is None:
            raise RuntimeError("workspace snapshot did not attest its root identity")
        self._workspace_root = root_match.group(1)
        self._workspace_root_identity = identity_match.group(1)
        summary: dict[str, Any] = {
            "raw": (result.stdout or "").strip(),
            "workspace_root": self._workspace_root,
            "workspace_identity": self._workspace_root_identity,
        }
        for key, value in re.findall(r"(file_count|size_kib)=([0-9]+)", result.stdout or ""):
            summary[key] = int(value)
        self._workspace_snapshot_token = token
        self._workspace_snapshot_summary = summary
        self._workspace_snapshot_ready = True

    async def _ensure_workspace_integrity(self) -> bool:
        """Restore the pristine root when its out-of-band identity guard changes."""
        environment = self._active_environment
        if environment is None or not self._workspace_snapshot_ready:
            return False
        self._workspace_integrity_checks += 1
        workspace_root = self._workspace_root
        guard_path = f"{WORKSPACE_SNAPSHOT_ROOT}/{WORKSPACE_GUARD_TOKEN}"
        probe = await environment.exec(
            (
                f"test -d {shlex.quote(workspace_root)} && "
                f"test ! -L {shlex.quote(workspace_root)} && "
                f"test \"$(stat -Lc '%d:%i' {shlex.quote(workspace_root)})\" = "
                f"{shlex.quote(self._workspace_root_identity)} && "
                f"test -f {shlex.quote(guard_path)} && "
                f"test \"$(cat {shlex.quote(guard_path)})\" = "
                f"{shlex.quote(self._workspace_snapshot_token)}"
            ),
            cwd="/",
            timeout_sec=30,
            user="root",
        )
        if probe.return_code == 0:
            return False

        recovery_number = self._workspace_recoveries + 1
        damaged = f"{WORKSPACE_SNAPSHOT_ROOT}/damaged-{recovery_number}"
        command = f"""set -eu
workspace={shlex.quote(workspace_root)}
snapshot_root={shlex.quote(WORKSPACE_SNAPSHOT_ROOT)}
snapshot={shlex.quote(WORKSPACE_SNAPSHOT_ROOT + '/original')}
damaged={shlex.quote(damaged)}
rm -rf -- "$damaged"
mkdir -p "$damaged"
if [ -d "$workspace" ] && [ ! -L "$workspace" ]; then
  find "$workspace" -mindepth 1 -maxdepth 1 -exec mv -t "$damaged" -- {{}} +
else
  if [ -e "$workspace" ] || [ -L "$workspace" ]; then mv -- "$workspace" "$damaged/root-object"; fi
  mkdir -p "$workspace"
fi
cp -a --reflink=auto "$snapshot"/. "$workspace"/
workspace_identity=$(stat -Lc '%d:%i' "$workspace")
printf '%s\n' {shlex.quote(self._workspace_snapshot_token)} > "$snapshot_root/{WORKSPACE_GUARD_TOKEN}"
printf 'workspace_identity=%s\n' "$workspace_identity"
"""
        result = await environment.exec(
            command,
            cwd="/",
            timeout_sec=WORKSPACE_SNAPSHOT_TIMEOUT_S,
            user="root",
        )
        if result.return_code != 0:
            self._workspace_recovery_failures += 1
            raise RuntimeError(
                "workspace corruption was detected but recovery failed: "
                f"{self._workspace_exec_error(result)}"
            )
        identity_match = re.search(
            r"(?:^|\s)workspace_identity=([0-9]+:[0-9]+)(?:\s|$)",
            result.stdout or "",
        )
        if identity_match is None:
            self._workspace_recovery_failures += 1
            raise RuntimeError("workspace recovery did not attest its root identity")
        self._workspace_root_identity = identity_match.group(1)
        self._workspace_recoveries = recovery_number
        self._fugu_llm.note_workspace_recovery(
            {
                "recovery_number": recovery_number,
                "reason": "workspace root missing, replaced, or guard changed",
                "restored_from": "enforced pristine task snapshot",
                "damaged_workspace_preserved": damaged,
            }
        )
        return True

    async def _remove_workspace_sentinel(self) -> None:
        environment = self._active_environment
        if environment is None or not self._workspace_snapshot_ready:
            return
        result = await environment.exec(
            f"rm -f -- {shlex.quote(WORKSPACE_SNAPSHOT_ROOT + '/' + WORKSPACE_GUARD_TOKEN)}",
            cwd="/",
            timeout_sec=30,
            user="root",
        )
        if result.return_code != 0:
            self._workspace_cleanup_failures += 1

    @override
    async def _handle_llm_interaction(
        self,
        chat: Any,
        prompt: str,
        logging_paths: tuple[Any | None, Any | None, Any | None] = (
            None,
            None,
            None,
        ),
        original_instruction: str = "",
        session: Any | None = None,
    ) -> tuple[list[Any], bool, str, str, str, LLMResponse]:
        workspace_recovered = await self._ensure_workspace_integrity()
        interaction = await super()._handle_llm_interaction(
            chat,
            prompt,
            logging_paths,
            original_instruction,
            session,
        )
        commands, is_task_complete, feedback, analysis, plan, response = interaction
        (
            commands,
            isolated_errexit,
            collapsed_empty,
            converted_shell_waits,
        ) = normalize_terminal_commands(commands)
        self._isolated_errexit_batches += isolated_errexit
        self._collapsed_empty_wait_commands += collapsed_empty
        self._converted_shell_wait_commands += converted_shell_waits
        if workspace_recovered:
            commands.insert(
                0,
                Command(
                    keystrokes=f"cd {shlex.quote(self._workspace_root)}\n",
                    duration_sec=1.0,
                ),
            )
            is_task_complete = False
            warning = (
                "WARNINGS: The runtime detected workspace-root corruption, preserved "
                "the damaged tree under /tmp, and restored the pristine task snapshot. "
                f"Re-evaluate the task from {self._workspace_root} before "
                "claiming completion."
            )
            feedback = f"{feedback}\n{warning}" if feedback else warning
        if commands and self._run_started_monotonic is not None:
            _, remaining_s = self._terminal_budget_status()
            declared_duration_s = sum(
                max(0.0, float(command.duration_sec)) for command in commands
            )
            if (
                remaining_s <= MIN_COMMAND_START_BUDGET_S
                or declared_duration_s + COMMAND_COMPLETION_GUARD_S > remaining_s
            ):
                self._rejected_overtime_commands += 1
                commands = []
                is_task_complete = False
                warning = (
                    "WARNINGS: The command batch was not started because its declared "
                    "duration does not fit safely inside the remaining task budget. "
                    "Return a command-free completion decision or concise status now."
                )
                feedback = f"{feedback}\n{warning}" if feedback else warning
        if any(
            destroys_workspace_root(command.keystrokes, self._workspace_root)
            for command in commands
        ):
            self._blocked_workspace_commands += 1
            self._fugu_llm.discard_pending_workflow(
                "blocked workspace-root destruction"
            )
            commands = []
            is_task_complete = False
            warning = (
                "WARNINGS: The command batch was blocked because it would move or "
                f"remove Harbor's {self._workspace_root} workspace root. Operate "
                f"on files beneath {self._workspace_root} without destroying the "
                "workspace itself."
            )
            feedback = f"{feedback}\n{warning}" if feedback else warning
        if self._pending_completion and is_task_complete and commands:
            self._rejected_commandful_confirmations += 1
            commands = []
            is_task_complete = False
            warning = (
                "WARNINGS: Completion confirmation was rejected because the same "
                "response issued terminal commands. Those commands were not executed, "
                "so the verified workspace was preserved. If repairs are needed, issue "
                "them with task_complete=false; otherwise confirm again without commands."
            )
            feedback = f"{feedback}\n{warning}" if feedback else warning
        self._fugu_llm.note_terminal_wait(
            commands,
            is_task_complete=is_task_complete,
        )
        return commands, is_task_complete, feedback, analysis, plan, response

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
        self._active_environment = environment
        self._workspace_snapshot_ready = False
        self._workspace_root = WORKSPACE_ROOT
        self._workspace_snapshot_token = ""
        self._workspace_root_identity = ""
        self._workspace_snapshot_summary = {}
        self._workspace_integrity_checks = 0
        self._workspace_recoveries = 0
        self._workspace_recovery_failures = 0
        self._workspace_cleanup_failures = 0
        self._terminal_task_budget_s = task_agent_timeout_s(environment)
        self._max_episodes = (
            self._configured_max_turns
            + int(self._terminal_task_budget_s // LOCAL_HANDOFF_INTERRUPT_WAIT_S)
            + LOCAL_POLL_EPISODE_GUARD
        )
        self._run_started_monotonic = time.monotonic()
        self._planner.set_task_instruction(instruction)
        self._fugu_llm.set_task_instruction(instruction)
        self._after_fugu_run_reset(instruction)
        try:
            self._verify_serving_dependencies()
            await self._prepare_workspace_snapshot(environment)
            await super().run(instruction, environment, context)
        finally:
            try:
                await self._remove_workspace_sentinel()
            finally:
                self._record_fugu_metadata(context)
                self._active_environment = None

    def _verify_serving_dependencies(self) -> None:
        """The product agent requires its exact conductor adapters to be served."""
        if self._solo_worker_id is not None:
            return
        # The opt-in typed single-model conductor is served directly; the legacy
        # two-head path still checks that its required files and serving roots exist.
        if self._typed_conductor_model:
            return
        verify_product_adapters_served()

    def _terminal_budget_status(self) -> tuple[float, float]:
        if self._run_started_monotonic is None:
            return 0.0, self._terminal_task_budget_s
        elapsed = max(0.0, time.monotonic() - self._run_started_monotonic)
        return elapsed, max(0.0, self._terminal_task_budget_s - elapsed)

    def _record_fugu_metadata(self, context: AgentContext) -> None:
        """Persist audit metadata even when Harbor cancels the agent at its deadline."""
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "fugu_routes": self._fugu_llm.routes,
                "worker_turn_counts": dict(
                    Counter(route["worker_model"] for route in self._fugu_llm.routes)
                ),
                "worker_protocol_errors": self._fugu_llm.worker_protocol_errors,
                "worker_protocol_repairs": self._fugu_llm.worker_protocol_repairs,
                "worker_reasoning_json_promotions": {
                    str(worker_id): getattr(worker, "reasoning_json_promotions", 0)
                    for worker_id, worker in self._fugu_llm._workers.items()
                },
                "protocol_replans": self._fugu_llm.protocol_replans,
                "provider_owner_retries": self._fugu_llm.provider_owner_retries,
                "provider_owner_retry_limit": self._fugu_llm._provider_owner_retry_limit,
                "provider_replans": self._fugu_llm.provider_replans,
                "provider_failure_events": self._fugu_llm.provider_failure_events,
                "fail_closed_provider_errors": (
                    self._fugu_llm._fail_closed_provider_errors
                ),
                "unrecoverable_planning_failures": (
                    self._fugu_llm.unrecoverable_planning_failures
                ),
                "planner_failures": self._fugu_llm.planner_failures,
                "conductor_workflows": self._fugu_llm.conductor_workflows,
                "workflow_agent_continuations": (
                    self._fugu_llm.workflow_agent_continuations
                ),
                "runtime_turns": self._fugu_llm.runtime_turns,
                "completed_workflow_steps": self._fugu_llm.completed_workflow_steps,
                "completed_workflows": self._fugu_llm.completed_workflows,
                "shared_workflow_memories": len(self._fugu_llm._shared_workflows),
                "unavailable_worker_models": {
                    self._fugu_llm._worker_names[worker_id]: reason
                    for worker_id, reason in self._fugu_llm._unavailable_workers.items()
                },
                "discarded_workflow_steps": self._fugu_llm.discarded_workflow_steps,
                "local_completion_confirmations": (
                    self._fugu_llm.local_completion_confirmations
                ),
                "forced_workflow_handoffs": (self._fugu_llm.forced_workflow_handoffs),
                "call_lease_handoffs": self._fugu_llm.call_lease_handoffs,
                "time_lease_handoffs": self._fugu_llm.time_lease_handoffs,
                "late_root_promotions": self._fugu_llm.late_root_promotions,
                "unstable_completion_rejections": (
                    self._fugu_llm.unstable_completion_rejections
                ),
                "local_terminal_polls": self._fugu_llm.local_terminal_polls,
                "local_terminal_poll_seconds": round(
                    self._fugu_llm.local_terminal_poll_seconds,
                    3,
                ),
                "local_handoff_interrupts": self._fugu_llm.local_handoff_interrupts,
                "paid_worker_call_attempts": (self._fugu_llm.paid_worker_call_attempts),
                "paid_worker_call_limit_responses": (
                    self._fugu_llm.paid_worker_call_limit_responses
                ),
                "task_budget_stop_responses": self._fugu_llm.task_budget_stop_responses,
                "live_control_decisions": self._fugu_llm.live_control_decisions,
                "live_control_model_traces": list(
                    getattr(
                        self._fugu_llm._live_controller,
                        "decision_traces",
                        (),
                    )
                ),
                "live_control_failures": self._fugu_llm.live_control_failures,
                "live_control_corrections": self._fugu_llm.live_control_corrections,
                "live_control_normalizations": (
                    self._fugu_llm.live_control_normalizations
                ),
                "live_control_prompt_compactions": getattr(
                    self._fugu_llm._live_controller,
                    "prompt_compactions",
                    0,
                ),
                "live_control_last_prompt_tokens": getattr(
                    self._fugu_llm._live_controller,
                    "last_prompt_tokens",
                    None,
                ),
                "live_control_seed": getattr(
                    self._fugu_llm._live_controller,
                    "_seed",
                    None,
                ),
                "live_control_temperature": getattr(
                    self._fugu_llm._live_controller,
                    "_temperature",
                    None,
                ),
                "live_control_records_token_data": getattr(
                    self._fugu_llm._live_controller,
                    "_record_token_data",
                    False,
                ),
                "live_control_replacement_plans": (
                    self._fugu_llm.live_control_replacement_plans
                ),
                "live_control_replacement_plan_failures": (
                    self._fugu_llm.live_control_replacement_plan_failures
                ),
                "conductor_interrupted_positions": (
                    self._fugu_llm.conductor_interrupted_positions
                ),
                "live_control_architecture": (
                    "none_fixed_solo_worker"
                    if self._solo_worker_id is not None
                    else (
                        "unified_full_action_controller"
                        if self._fugu_llm._live_controller_supplies_topology
                        else "compact_decision_then_planner_generated_replacement_topology"
                    )
                ),
                "initial_workflow_controller": (
                    "fixed_solo_worker_no_conductor"
                    if self._solo_worker_id is not None
                    else (
                        "unified_full_action_controller"
                        if self._fugu_llm._live_controller_supplies_topology
                        else "live_control_head"
                        if self._fugu_llm._live_controller_plans_initial_workflow
                        else "environment_first_planning_head_v11"
                    )
                ),
                "planner_max_attempts": getattr(self._planner, "_max_attempts", 1),
                "planner_temperature": PRODUCT_PLANNER_TEMPERATURE,
                "fail_closed_conductor_errors": (
                    self._fugu_llm._fail_closed_conductor_errors
                ),
                "rejected_commandful_confirmations": (
                    self._rejected_commandful_confirmations
                ),
                "blocked_workspace_commands": self._blocked_workspace_commands,
                "workspace_snapshot_ready": self._workspace_snapshot_ready,
                "workspace_root": self._workspace_root,
                "workspace_snapshot_summary": self._workspace_snapshot_summary,
                "workspace_integrity_checks": self._workspace_integrity_checks,
                "workspace_recoveries": self._workspace_recoveries,
                "workspace_recovery_failures": self._workspace_recovery_failures,
                "workspace_cleanup_failures": self._workspace_cleanup_failures,
                "isolated_errexit_batches": self._isolated_errexit_batches,
                "collapsed_empty_wait_commands": self._collapsed_empty_wait_commands,
                "converted_shell_wait_commands": (self._converted_shell_wait_commands),
                "rejected_overtime_commands": self._rejected_overtime_commands,
                "terminal_task_budget_s": self._terminal_task_budget_s,
                "max_agent_turns": self._configured_max_turns,
                "fair_position_call_budget": self._fugu_llm._fair_position_call_budget,
                "completion_confirmation_policy": (
                    "local_conductor_confirmation"
                    if self._fugu_llm._local_live_completion_confirmation
                    else "same_live_selected_agent_clean_confirmation"
                ),
                "position_lease_policy": (
                    "conductor_explicit_completion_no_per_position_lease"
                    if self._fugu_llm._fair_position_call_budget is None
                    else "dynamic_equal_share_of_remaining_paid_calls_and_wall_time"
                ),
                "position_busy_handoff_policy": (
                    "owner_continues_local_polling_until_terminal_stable"
                    if self._fugu_llm._fair_position_call_budget is None
                    else "local_60s_grace_then_ctrl_c_until_terminal_stable"
                ),
                "input_snapshot_policy": (
                    "runtime_enforced_pristine_snapshot_with_out_of_band_root_identity"
                ),
                "runtime_revision": PRODUCT_RUNTIME_REVISION,
                "worker_provider_base": self._pool_binding.provider_base,
                "worker_models": list(self._pool_binding.runtime_models),
                "active_worker_ids": sorted(self._fugu_llm._workers),
                "solo_worker_id": self._solo_worker_id,
                "worker_reasoning_efforts": list(DEFAULT_REASONING_EFFORTS),
                "pool_id": self._pool_binding.pool_id,
                "pool_binding_revision": self._pool_binding.binding_revision,
                "typed_conductor_model": self._typed_conductor_model,
                "typed_conductor_url": self._typed_conductor_url,
                "typed_conductor_policy_revision": (
                    self._typed_conductor_policy_revision
                ),
                "planner_adapter": (
                    self._typed_conductor_model
                    or "output/fugu_ultra_planner_composite_v11_s20"
                ),
                "live_control_adapter": (
                    self._typed_conductor_model
                    or str(LIVE_CONTROL_ADAPTER.relative_to(REPO_ROOT))
                ),
            }
        )
        context.metadata = metadata
