"""Disk cache for worker completions, keyed on (model, messages, sampling).

The dominant cost in single-step label generation is calling every worker n times
over thousands of questions. Caching at this grain makes re-runs free and lets the
SFT/CMA-ES stages iterate without re-paying the API.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from .types import Completion, Message, Sampling, ToolCall, ToolCompletion

if TYPE_CHECKING:
    pass


def completion_key(model: str, messages: list[Message], sampling: Sampling) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "sampling": sampling.as_dict(),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def tool_completion_key(model: str, messages: list, sampling: Sampling, tools: list) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "sampling": sampling.as_dict(),
        "tools": tools,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CompletionCache:
    """Thin wrapper over ``diskcache.Cache``.

    Falls back to an in-memory dict if ``diskcache`` is unavailable (e.g. in a
    minimal test environment), so the rest of the system never has to care.
    """

    def __init__(self, path: str | None):
        self._mem: dict[str, dict] | None = None
        self._dc = None
        if path is None:
            self._mem = {}
            return
        try:
            import diskcache

            self._dc = diskcache.Cache(path)
        except Exception:
            self._mem = {}

    def get(self, key: str) -> Completion | None:
        raw = self._mem.get(key) if self._mem is not None else self._dc.get(key)
        if raw is None:
            return None
        return Completion(**raw)

    def set(self, key: str, value: Completion) -> None:
        raw = asdict(value)
        if self._mem is not None:
            self._mem[key] = raw
        else:
            self._dc.set(key, raw)

    def get_tool(self, key: str) -> ToolCompletion | None:
        raw = self._mem.get(key) if self._mem is not None else self._dc.get(key)
        if raw is None:
            return None
        raw = dict(raw)
        raw["tool_calls"] = [ToolCall(**tc) for tc in raw.get("tool_calls", [])]
        return ToolCompletion(**raw)

    def set_tool(self, key: str, value: ToolCompletion) -> None:
        raw = asdict(value)
        if self._mem is not None:
            self._mem[key] = raw
        else:
            self._dc.set(key, raw)
