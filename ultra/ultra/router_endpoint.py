"""OpenAI-compatible endpoint that routes each agent turn to a pool model.

This is the product's serving surface. A mature agent loop (Terminus 2,
mini-swe-agent, Claude Code, …) drives the terminal and calls this endpoint as
if it were a single model; the endpoint decides which open-weight worker
answers the current turn and forwards the request unchanged.

Why this shape: measured on Terminal-Bench 2.1, one pool model inside a mature
agent loop scores ~84% while the same pool inside our own worker loop scores
61%. The loop, not the models and not the routing, held performance down. So
the loop comes from upstream and the product keeps only what it uniquely adds
— choosing which model answers, and recovering when one stalls.

Routing is deliberately conservative at v1: a single strong default backed by
the pool binding, with the seams for conductor-driven selection already in
place (`select_worker`). Anything cleverer must first beat the solo baseline
measured under the same loop.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fugu_heavy import FuguHeavyOrchestrator
from .trained_conductor import TrainedConductor
from .worker_loop import FunctionCallingWorkerLoop, openrouter_client
from .pool_binding import load_pool_binding

DEFAULT_BINDING = Path(
    "director/manifests/fugu_clean_v1/grpo_pilot_train/current_pool_binding_ow2.json"
)


@dataclass(frozen=True)
class WorkerSlot:
    """One bound pool member the router may select."""

    worker_id: int
    model: str
    reasoning_effort: str
    role: tuple[str, ...]


class PoolRouter:
    """Selects a pool worker per turn and forwards the request to it."""

    def __init__(
        self,
        binding_path: Path | str = DEFAULT_BINDING,
        *,
        provider_base: str | None = None,
        api_key_env: str = "OPENROUTER_KEY",
        default_worker_id: int | None = None,
        heavy: bool = False,
    ) -> None:
        binding = load_pool_binding(Path(binding_path))
        self.slots = {
            slot.worker_id: WorkerSlot(
                worker_id=slot.worker_id,
                model=slot.runtime_model,
                reasoning_effort=slot.reasoning_effort,
                role=tuple(slot.role_prior),
            )
            for slot in binding.slots
        }
        self.provider_base = (provider_base or binding.provider_base).rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise RuntimeError(f"{api_key_env} is not set")
        # Default to the implementer slot: on terminal work the agent loop
        # spends nearly every turn producing and running code.
        self.default_worker_id = (
            default_worker_id
            if default_worker_id is not None
            else self._slot_for_role("implementer")
        )
        # message index -> worker that produced it, for per-worker isolation
        self._emitted_by: dict[int, int] = {}
        # Light-path selection: the conductor assigns a role per TASK (cached);
        # the binding resolves role -> model. No content string-matching, ever.
        self._role_names_all = ["/".join(s.role) for s in self.slots.values()]
        self._selection_conductor = TrainedConductor(
            roles=[slot.role for slot in self.slots.values()]
        )
        self._selection_cache: dict[int, int] = {}
        # Heavy path (Fugu 3.2.1-3.2.2): the trained conductor plans a workflow
        # over capability roles and each step runs its own isolated worker loop.
        self.heavy: FuguHeavyOrchestrator | None = None
        if heavy:
            role_names = ["/".join(s.role) for s in self.slots.values()]
            self.heavy = FuguHeavyOrchestrator(
                binding={
                    "/".join(slot.role): slot.model for slot in self.slots.values()
                },
                worker_loop=FunctionCallingWorkerLoop(
                    client=openrouter_client(
                        self.api_key,
                        self.provider_base,
                        efforts={
                            slot.model: slot.reasoning_effort
                            for slot in self.slots.values()
                            if slot.reasoning_effort
                        },
                    )
                ),
                conductor=TrainedConductor(
                    roles=[slot.role for slot in self.slots.values()]
                ),
            )
            self._role_names = role_names

    def _slot_for_role(self, tag: str) -> int:
        for worker_id, slot in sorted(self.slots.items()):
            if tag in slot.role:
                return worker_id
        return min(self.slots)

    def select_worker(self, messages: list[dict[str, Any]]) -> WorkerSlot:
        """Pick the worker for this request — the CONDUCTOR decides.

        No word/string classification anywhere (hard rule, 2026-07-27: prompts
        can be in any language; this is an orchestrator, not a word-based
        router). The trained conductor reads the task and assigns a capability
        role; the binding resolves role -> model. The decision is cached per
        task text, so a multi-turn agent session pays for one conductor call,
        not one per turn. Unreachable conductor -> binding default slot.
        """
        text = self._first_user_text(messages)
        if text:
            cache_key = hash(text)
            if cache_key in self._selection_cache:
                return self.slots[self._selection_cache[cache_key]]
            try:
                steps = self._selection_conductor.plan(text, self._role_names_all)
                worker_id = self._slot_for_role_name(steps[0].role)
                self._selection_cache[cache_key] = worker_id
                return self.slots[worker_id]
            except Exception:  # noqa: BLE001 - degraded routing beats failure
                pass
        return self.slots[self.default_worker_id]

    def _slot_for_role_name(self, role_name: str) -> int:
        for worker_id, slot in sorted(self.slots.items()):
            if "/".join(slot.role) == role_name:
                return worker_id
        return self.default_worker_id

    @staticmethod
    def _first_user_text(messages: list[dict[str, Any]]) -> str:
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
        return ""

    @staticmethod
    def _latest_user_text(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
        return ""

    @staticmethod
    def _latest_environment_text(messages: list[dict[str, Any]]) -> str:
        """The most recent terminal/tool observation the agent received."""
        for message in reversed(messages):
            if message.get("role") in ("user", "tool"):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    )
        return ""

    @staticmethod
    def isolate_for_worker(
        messages: list[dict[str, Any]],
        worker_id: int,
        emitted_by: dict[int, int],
    ) -> list[dict[str, Any]]:
        """Rewrite the shared transcript into one worker's isolated view.

        Sakana's Fugu report (§3.2.2) documents why a shared transcript is
        unsafe once more than one model answers: handing a fresh agent the
        previous agent's trajectory causes *orchestration collapse* — the new
        agent follows the path already laid down instead of finding its own,
        so the second opinion is worth nothing.

        The fix keeps two kinds of history apart:

        * agent trajectory (assistant turns) — ISOLATED. A worker sees only
          its own prior replies; another model's reasoning is dropped.
        * environment interaction (system/user/tool observations) — SHARED.
          Terminal output is a fact about the world, and re-discovering it
          wastes turns, which is exactly what the short-budget regime cannot
          afford.

        ``emitted_by`` maps message index -> worker_id for assistant turns.
        Foreign assistant turns are replaced by a compact factual note so the
        transcript stays coherent without transplanting another agent's
        solution path.
        """
        view: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                view.append(message)
                continue
            author = emitted_by.get(index)
            if author is None or author == worker_id:
                view.append(message)
                continue
            # Foreign turn: mask the REASONING (agent trajectory, isolated) but
            # preserve tool_calls verbatim. A tool call and its result are
            # ENVIRONMENT interaction — shared memory under 3.2.2 — and dropping
            # the call would orphan the paired `tool` message, violating the
            # OpenAI contract and breaking any tool-using agent loop.
            masked: dict[str, Any] = {
                "role": "assistant",
                "content": (
                    "[earlier work by another position; its actions and their "
                    "results are retained]"
                ),
            }
            if message.get("tool_calls"):
                masked["tool_calls"] = message["tool_calls"]
                masked["content"] = None
            view.append(masked)
        return view

    def forward(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Serve one request.

        Heavy path: the trained conductor plans a capability workflow and each
        step runs its own isolated worker loop; the final step's output is the
        response (Fugu 3.2.1). Light path: route the request to one worker.
        """
        messages = payload.get("messages") or []
        if self.heavy is not None:
            instruction = self._latest_user_text(messages)
            if instruction:
                result = self.heavy.execute(instruction, env=None)
                return {
                    "id": "fugu-heavy",
                    "object": "chat.completion",
                    "model": payload.get("model", "fugu-open"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": result["output"],
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "_routing": {"plan": result["steps"]},
                }
        worker = self.select_worker(messages)
        body = dict(payload)
        # Isolation is a no-op while one worker answers every turn (v1), and
        # becomes load-bearing the moment the conductor switches mid-session.
        body["messages"] = self.isolate_for_worker(
            messages, worker.worker_id, self._emitted_by
        )
        body["model"] = worker.model
        body.setdefault("reasoning_effort", worker.reasoning_effort)
        # price-sort removed (user, 2026-07-26): hung cheapest-provider stalls
        # outweigh the savings; OpenRouter default routing balances uptime.
        request = urllib.request.Request(
            f"{self.provider_base}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
        # Report the served identity, not the backing model: callers integrate
        # against the endpoint, and the pool must stay swappable behind it.
        result["model"] = payload.get("model", "fugu-open")
        result.setdefault("_routing", {})["worker_id"] = worker.worker_id
        # Record authorship so the next turn can isolate correctly: the agent
        # loop appends this reply to the transcript it will send back.
        self._emitted_by[len(messages)] = worker.worker_id
        return result

    def models_payload(self, served_name: str = "fugu-open") -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": served_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "fugu",
                }
            ],
        }
