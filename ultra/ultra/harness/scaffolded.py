"""Scaffold-aware coding harness placeholders.

OpenCode, Claude Code, and Codex manage repository context, edits, commands, and
execution feedback differently. Ultra's Conductor should route by worker ID and
backend, while concrete harness adapters own those scaffold-specific details.

These classes register the backend names now and fail closed until full executors
are wired in.
"""

from __future__ import annotations

from ..schemas import Grade, TaskSpec
from ..workers import Sampling, WorkerPool
from .base import StepInput, StepResult, register_harness


class _PendingScaffoldHarness:
    name: str

    async def run_step(
        self, step: StepInput, pool: WorkerPool, sampling: Sampling
    ) -> StepResult:
        return StepResult(
            text="",
            error=f"{self.name} harness adapter is registered but not implemented",
            termination="not_implemented",
        )

    def grade(self, task: TaskSpec, final: StepResult) -> Grade:
        return Grade(score=0.0, success=False, details={"error": final.error})


@register_harness
class OpenCodeHarness(_PendingScaffoldHarness):
    name = "opencode"


@register_harness
class ClaudeCodeHarness(_PendingScaffoldHarness):
    name = "claude_code"


@register_harness
class CodexHarness(_PendingScaffoldHarness):
    name = "codex"
