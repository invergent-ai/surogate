"""Per-worker function-calling loop (Fugu report §3.2.2).

Each worker in a planned workflow runs its OWN complete function-calling loop:
it calls a model, receives tool calls, executes them against the SHARED
environment, feeds results back into ITS OWN transcript, and iterates until it
answers or hits its budget. Workers never share a transcript — that is the
isolation guarantee that prevents orchestration collapse.

What the report requires and this implements:

* "any agent may make a function call at any time" — every worker owns a full
  tool loop, not a single turn.
* "retain which agent emitted every call ... such that each agent's function
  call loop is correctly routed to the corresponding agent" — each loop keeps
  its own message list, so a tool result is appended to the transcript of the
  worker that requested it and to no other.
* "intra-workflow isolation" — a worker's transcript contains only its own
  assistant turns; prior workers reach it solely through the access list, as
  pre-formatted context.
* "persistent shared memory" — tools act on one shared environment, so what a
  worker does to the world persists for later workers without re-discovery.

Model-agnostic: the loop is handed a model id resolved from the binding. It
contains no model names and no provider logic beyond the injected client.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

MAX_WORKER_TURNS = 30


class ToolExecutor(Protocol):
    """Executes one tool call against the shared environment."""

    def __call__(self, name: str, arguments: dict[str, Any]) -> str: ...


class ChatClient(Protocol):
    """Model transport: one chat-completions call."""

    def __call__(
        self, *, model: str, messages: list[dict[str, Any]], tools: list[dict] | None
    ) -> dict[str, Any]: ...


@dataclass
class FunctionCallingWorkerLoop:
    """One worker's own tool loop, isolated from every other worker."""

    client: ChatClient
    tools: list[dict] = field(default_factory=list)
    execute_tool: ToolExecutor | None = None
    max_turns: int = MAX_WORKER_TURNS
    # Deliberately minimal. Workflow framing ("you are one position in a
    # multi-position workflow...") competes with the task for attention and
    # measurably diluted long prompts; the context itself already carries the
    # role. Keep this to the format guarantee that must never be lost.
    system_prompt: str = (
        "Answer the request exactly as asked, including any answer format it "
        "specifies (e.g. ending with 'ANSWER: [LETTER]'). Use any tools "
        "available when they help."
    )
    # audit trail: which worker emitted each call, per the report's requirement
    call_log: list[dict[str, Any]] = field(default_factory=list)

    def run(
        self,
        *,
        model: str,
        subtask: str,
        context: str,
        env: Any = None,
    ) -> str:
        """Run this worker's loop to completion and return its final message.

        `context` is the worker's ISOLATED view (its subtask plus only the
        access-listed prior results). The transcript built here is private to
        this worker; no other worker ever sees it.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context or subtask},
        ]
        for turn in range(self.max_turns):
            response = self.client(
                model=model, messages=messages, tools=self.tools or None
            )
            message = (response.get("choices") or [{}])[0].get("message") or {}
            tool_calls = message.get("tool_calls") or []
            # The assistant turn joins THIS worker's transcript only.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                }
            )
            if not tool_calls:
                return self._message_text(message)
            for call in tool_calls:
                name, arguments = self._decode(call)
                self.call_log.append(
                    {"model": model, "turn": turn, "tool": name}
                )
                result = (
                    self.execute_tool(name, arguments)
                    if self.execute_tool is not None
                    else "no tool executor configured"
                )
                # The result is routed back into the emitting worker's loop.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": result,
                    }
                )
        return self._last_text(messages) or "worker turn budget exhausted"

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        """Extract a worker's answer, tolerating reasoning-only responses.

        Reasoning models return `content: null` with the text in `reasoning`
        when they hit the token limit mid-trace. Reading only `content` turns
        a usable (if truncated) answer into an empty string, which then scores
        as wrong — 2/70 MMLU-Pro samples were lost this way on 2026-07-26.
        """
        content = (message.get("content") or "").strip()
        if content:
            return content
        for field in ("reasoning", "reasoning_content"):
            fallback = message.get(field)
            if isinstance(fallback, str) and fallback.strip():
                return fallback.strip()
        return ""

    @staticmethod
    def _decode(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        function = call.get("function") or {}
        name = function.get("name", "")
        raw = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        return name, arguments

    @staticmethod
    def _last_text(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("content"):
                return str(message["content"])
        return ""


def openrouter_client(
    api_key: str,
    base_url: str,
    efforts: dict[str, str] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> Callable[..., dict[str, Any]]:
    """Minimal chat client for the pool provider.

    `efforts` maps model id -> reasoning_effort from the pool binding, so the
    heavy path honours the binding's declared per-worker effort (it silently
    ran at provider defaults until 2026-07-26).

    `max_tokens`/`temperature` are for TRAINING ROLLOUTS ONLY (the Conductor
    recipe constrains workers to ~4096 tokens at temp 0.2 so rollouts stay
    cheap). Production serving leaves both unset: capping reasoning workers
    mid-answer burned money and yielded unusable results (2026-07-26).
    """
    import urllib.request

    def call(
        *, model: str, messages: list[dict[str, Any]], tools: list[dict] | None
    ) -> dict[str, Any]:
        # No provider price-sort: cheapest-first routed huge reasoning prompts
        # to a provider that accepted and never answered (2026-07-26, GPQA
        # stall). OpenRouter's default routing balances price with uptime.
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        effort = (efforts or {}).get(model)
        if effort:
            body["reasoning_effort"] = effort
        if tools:
            body["tools"] = tools
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.load(response)

    return call
