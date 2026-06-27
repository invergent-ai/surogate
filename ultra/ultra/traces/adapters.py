"""Normalize coding-assistant session exports into ``AgentTrace`` records.

These adapters intentionally define the ingestion contract before binding to a
specific vendor export format. Each concrete adapter accepts a plain dictionary
so offline tests and future exporters can feed the same normalization path:

raw session export -> redact/validate -> extract events/artifacts/grade -> AgentTrace
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC
from typing import Any

from ..schemas import (
    AgentTrace,
    Grade,
    RepoStateRef,
    TraceArtifacts,
    TraceEvent,
    TraceOrigin,
    TracePrivacy,
    TracePromptRef,
    TraceUsage,
)


def _stable_hash(data: Any) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


class BaseTraceAdapter(ABC):
    origin_harness: TraceOrigin

    def normalize(self, raw: dict[str, Any]) -> AgentTrace:
        """Return a canonical trace.

        Expected raw keys are deliberately close to ``AgentTrace`` so vendor
        exporters can stay thin. Secret/PII redaction should happen before this
        method or inside a provider-specific override.
        """

        prompt = raw.get("prompt", {})
        grade = raw.get("grade")
        return AgentTrace(
            trace_id=raw.get("trace_id") or _stable_hash(raw),
            origin_harness=self.origin_harness,
            harness_version=raw.get("harness_version"),
            worker_model=raw["worker_model"],
            worker_config_hash=raw.get("worker_config_hash") or _stable_hash(
                {
                    "origin_harness": self.origin_harness,
                    "harness_version": raw.get("harness_version"),
                    "worker_model": raw["worker_model"],
                    "settings": raw.get("settings", {}),
                }
            ),
            task_id=raw["task_id"],
            repo=RepoStateRef(**raw.get("repo", {})),
            prompt=TracePromptRef(
                user_task=prompt.get("user_task", raw.get("user_task", "")),
                system_prompt_ref=prompt.get("system_prompt_ref"),
                developer_prompt_ref=prompt.get("developer_prompt_ref"),
            ),
            events=[TraceEvent(**event) for event in raw.get("events", [])],
            artifacts=TraceArtifacts(**raw.get("artifacts", {})),
            grade=Grade(**grade) if isinstance(grade, dict) else grade,
            usage=TraceUsage(**raw.get("usage", {})),
            privacy=TracePrivacy(**raw.get("privacy", {})),
        )


class OpenCodeTraceAdapter(BaseTraceAdapter):
    origin_harness: TraceOrigin = "opencode"


class ClaudeCodeTraceAdapter(BaseTraceAdapter):
    origin_harness: TraceOrigin = "claude_code"


class CodexTraceAdapter(BaseTraceAdapter):
    origin_harness: TraceOrigin = "codex"


TRACE_ADAPTERS: dict[str, type[BaseTraceAdapter]] = {
    "opencode": OpenCodeTraceAdapter,
    "claude_code": ClaudeCodeTraceAdapter,
    "codex": CodexTraceAdapter,
}
