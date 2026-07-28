"""The trained conductor as workflow planner for the heavy orchestrator.

Replaces the hardcoded planner: r2 (Qwen3.6-27B + LoRA, served locally) emits a
typed capability workflow — a sequence of steps, each an anonymous capability
role with a subtask and an access list — which the orchestrator executes with
one isolated worker loop per step (Fugu report §3.2.1).

Plans are SAMPLED, not computed: the same task can yield different topologies,
and a multi-worker plan can beat every individual worker, which is why the
result must be measured rather than derived from published solo numbers.

Model-agnostic by construction: the conductor sees only capability profile
refs, never model names; the binding resolves role -> model afterwards.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any

from .fugu_heavy import WorkflowStep
from .live_control import (
    ControlBudget,
    LiveControlState,
    WorkerProfile,
    build_control_action_messages,
    capability_control_action_json_schema,
    capability_reference_map,
    parse_capability_control_action,
)

CONDUCTOR_URL = "http://localhost:8011/v1/chat/completions"
# r4-merged: first conductor whose TRAINED behavior reaches serving (gates
# 2026-07-26: shape 40/40 holdout trees; execution 87.2% vs solo 82.5%).
# Served as MERGED weights — never as a LoRA on this stack.
CONDUCTOR_MODEL = "fugu-27b-r4-merged"


@dataclass
class TrainedConductor:
    """Plans workflows with the trained typed conductor."""

    roles: list[tuple[str, ...]]
    model: str = CONDUCTOR_MODEL
    url: str = CONDUCTOR_URL
    temperature: float = 0.0
    wall_time_limit_s: float = 900.0
    max_tokens: int = 2048
    # Optional planning guidance carried in the system message (the sanctioned
    # extension slot in build_control_action_messages). None reproduces the
    # r4 SFT serving prompt byte for byte — the default MUST stay None unless
    # a measured comparison says otherwise, and whatever value serves during
    # GRPO collection must be frozen into training's tokenize_prompt too
    # (exact-token contract).
    guidelines: str | None = None

    def _messages(self, state: LiveControlState) -> list[dict[str, str]]:
        messages, _, _ = build_control_action_messages(
            state, capability_refs=True, guidelines=self.guidelines
        )
        return messages

    def _state(self, instruction: str) -> LiveControlState:
        workers = tuple(
            WorkerProfile(worker_id=index, capability_tags=tuple(tags))
            for index, tags in enumerate(self.roles)
        )
        return LiveControlState(
            original_task=instruction,
            workers=workers,
            workflow_id=None,
            positions=(),
            active_position_id=None,
            terminal_status="ready",
            terminal_observation="",
            shared_memory=(),
            budget=ControlBudget(
                paid_calls_used=0,
                paid_call_limit=120,
                elapsed_s=0.0,
                wall_time_limit_s=self.wall_time_limit_s,
            ),
        )

    def sample_plan(
        self, instruction: str, roles: list[str]
    ) -> tuple[list[WorkflowStep], dict]:
        """Sample a plan AND return the raw completion + token logprobs.

        The GRPO trainer's exact-token contract needs the behavior policy's
        sampled tokens and their logprobs, so rollout collection must keep the
        raw completion — `plan()` (production) discards it. `meta` carries:
        raw, parse_ok, logprobs (as served), finish_reason.
        """
        state = self._state(instruction)
        messages = self._messages(state)
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "logprobs": True,
            # The server's own ids are the ONLY safe exact-token source:
            # grammar-constrained decoding may sample non-canonical splits,
            # so re-encoding the text can differ from what the policy
            # actually emitted (measured: 0/480 clean round-trips).
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": capability_control_action_json_schema(state),
            },
        }
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.load(response)
        choice = payload["choices"][0]
        raw = (choice["message"].get("content") or "").strip()
        meta = {
            "raw": raw,
            "finish_reason": choice.get("finish_reason"),
            "logprobs": choice.get("logprobs"),
            "token_ids": choice.get("token_ids"),
            "prompt_token_ids": payload.get("prompt_token_ids"),
            "parse_ok": False,
        }
        try:
            action = parse_capability_control_action(
                raw, capability_reference_map(state.workers)
            )
        except Exception:
            return [], meta
        if action.action != "replan" or not action.steps:
            return [], meta
        meta["parse_ok"] = True
        steps = [
            WorkflowStep(
                role=roles[step.worker_id % len(roles)],
                subtask=step.subtask,
                access=tuple(step.access),
            )
            for step in action.steps
        ]
        return steps, meta

    def plan(self, instruction: str, roles: list[str]) -> list[WorkflowStep]:
        """Sample a typed workflow and convert it to executable steps.

        Falls back to a single builder step if the conductor is unreachable or
        emits an unusable action: a degraded plan is better than a failed task,
        and the fallback is the same shape as a one-worker plan.
        """
        state = self._state(instruction)
        messages = self._messages(state)
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # Serving must match the training template (qwen3_nothinking).
            # The r3 v3 gate failure was traced to this mismatch: trained
            # non-thinking, served with the model's default thinking template.
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": capability_control_action_json_schema(state),
            },
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = json.load(response)
            content = (payload["choices"][0]["message"].get("content") or "").strip()
            action = parse_capability_control_action(
                content, capability_reference_map(state.workers)
            )
        except Exception:
            return [WorkflowStep(role=roles[0], subtask=instruction)]

        if action.action != "replan" or not action.steps:
            return [WorkflowStep(role=roles[0], subtask=instruction)]
        return [
            WorkflowStep(
                role=roles[step.worker_id % len(roles)],
                subtask=step.subtask,
                access=tuple(step.access),
            )
            for step in action.steps
        ]
