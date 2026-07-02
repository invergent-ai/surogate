"""tau-bench tool-use harness with deterministic user exposure.

This wraps the installed ``tau_bench`` package but avoids a second LLM provider path
for the user simulator: the user gives the full task instruction on reset and emits
STOP after the agent responds. The environment tools and programmatic reward remain
tau-bench's own implementation.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path
from typing import Any

from ..schemas import Grade, TaskSpec
from ..workers import Sampling, ToolCall, WorkerPool
from .base import StepInput, StepResult, register_harness
from .repo_artifacts import write_json

TAU_RESPOND_ACTION = "respond"

_MAX_TURNS_BY_BUDGET = {
    "short": 4,
    "medium": 8,
    "long": 20,
    "max": None,
}


class _InstructionUser:
    def reset(self, instruction: str | None = None) -> str:
        return instruction or ""

    def step(self, content: str) -> str:
        return "###STOP###"

    def get_total_cost(self) -> float:
        return 0.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_tau_bench_path() -> None:
    try:
        import tau_bench  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    candidates = []
    for root in (_repo_root() / "director" / ".venv", _repo_root() / ".venv"):
        candidates.extend(sorted(root.glob("lib/python*/site-packages")))
    for path in candidates:
        if (path / "tau_bench").exists():
            sys.path.insert(0, str(path))
            return


def _payload(task: TaskSpec) -> dict[str, Any]:
    payload = task.grader.expected_answer
    return payload if isinstance(payload, dict) else {}


def _max_turns_for_step(task: TaskSpec, budget: str) -> int:
    payload = _payload(task)
    configured = int(payload.get("max_turns") or task.metadata.estimated_worker_calls or 30)
    cap = _MAX_TURNS_BY_BUDGET.get(budget, _MAX_TURNS_BY_BUDGET["medium"])
    return min(configured, cap) if cap is not None else configured


def _tool_message(call: ToolCall, result: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "content": str(result),
    }


def _assistant_message(comp) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": comp.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in comp.tool_calls
        ],
    }


def _assemble_messages(step: StepInput, transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are solving a tau-bench tool-use task. Use tools to inspect state "
                "and make only the changes required by the user. When finished, respond "
                "to the user instead of calling another tool."
            ),
        }
    ]
    messages.extend(transcript)
    if step.prior_artifacts:
        blocks = [
            f"[Worker {a.get('worker_id')} result]\n{a.get('response', '')}"
            for a in step.prior_artifacts
        ]
        messages.append({"role": "user", "content": "Authorized prior-step results:\n\n" + "\n\n".join(blocks)})
    if step.subtask.strip():
        messages.append({"role": "user", "content": f"Your subtask: {step.subtask}"})
    return messages


@register_harness
class TauBenchHarness:
    name = "tau_bench"

    def __init__(self) -> None:
        self.env = None
        self.tools: list[dict[str, Any]] = []
        self.transcript: list[dict[str, Any]] = []
        self.reward: float = 0.0
        self.done: bool = False
        self.last_info: dict[str, Any] = {}

    def _init_env(self, task: TaskSpec) -> None:
        if self.env is not None:
            return
        _ensure_tau_bench_path()
        from tau_bench.envs import get_env

        payload = _payload(task)
        env_name = str(payload.get("env_name") or "")
        task_split = str(payload.get("task_split") or "train")
        task_index = int(payload.get("task_index"))
        self.env = get_env(
            env_name,
            user_strategy="human",
            user_model="",
            user_provider=None,
            task_split=task_split,
            task_index=task_index,
        )
        self.env.user = _InstructionUser()
        reset = self.env.reset(task_index=task_index)
        self.transcript = [{"role": "user", "content": reset.observation}]
        self.tools = list(self.env.tools_info)

    async def run_step(self, step: StepInput, pool: WorkerPool, sampling: Sampling) -> StepResult:
        self._init_env(step.task)
        assert self.env is not None
        if not self.tools:
            return StepResult(text="", error="tau_bench task has no tools", termination="missing_tools")

        max_turns = _max_turns_for_step(step.task, step.budget)
        prompt_tokens = 0
        completion_tokens = 0
        cost = 0.0
        final_content = ""
        termination = "max_turns"

        for _ in range(max_turns):
            comp = await pool.call_tools(step.worker_id, _assemble_messages(step, self.transcript), self.tools, sampling)
            prompt_tokens += comp.prompt_tokens
            completion_tokens += comp.completion_tokens
            cost += comp.cost_usd
            final_content = comp.content or ""
            if comp.tool_calls:
                self.transcript.append(_assistant_message(comp))
                for call in comp.tool_calls:
                    result = await asyncio.to_thread(self._step_tool, call)
                    self.transcript.append(_tool_message(call, result.get("observation", "")))
                    if result.get("done"):
                        termination = "completed"
                        break
                if self.done:
                    break
                continue

            response = await asyncio.to_thread(self._step_respond, final_content)
            self.transcript.append({"role": "assistant", "content": final_content})
            self.transcript.append({"role": "user", "content": response.get("observation", "")})
            termination = "completed" if response.get("done") else "responded_without_stop"
            if self.done:
                break

        artifact_dir = Path(step.artifact_dir) if step.artifact_dir else None
        messages_ref = None
        if artifact_dir is not None:
            messages_ref = write_json(
                artifact_dir / "tau_bench_transcript.json",
                {
                    "done": self.done,
                    "reward": self.reward,
                    "transcript": self.transcript,
                    "info": self.last_info,
                },
            )
        return StepResult(
            text=json.dumps(
                {
                    "done": self.done,
                    "reward": self.reward,
                    "content": final_content,
                    "turns": len([m for m in self.transcript if m.get("role") == "assistant"]),
                },
                sort_keys=True,
            ),
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost_usd=cost,
            termination=termination,
            messages_ref=messages_ref,
        )

    def _step_tool(self, call: ToolCall) -> dict[str, Any]:
        from tau_bench.types import Action

        assert self.env is not None
        resp = self.env.step(Action(name=call.name, kwargs=copy.deepcopy(call.arguments)))
        self.reward = float(resp.reward)
        self.done = bool(resp.done)
        self.last_info = resp.info.model_dump(mode="json") if hasattr(resp.info, "model_dump") else {}
        return {"observation": resp.observation, "done": resp.done, "reward": resp.reward}

    def _step_respond(self, content: str) -> dict[str, Any]:
        from tau_bench.types import Action

        assert self.env is not None
        resp = self.env.step(Action(name=TAU_RESPOND_ACTION, kwargs={"content": content}))
        self.reward = float(resp.reward)
        self.done = bool(resp.done)
        self.last_info = resp.info.model_dump(mode="json") if hasattr(resp.info, "model_dump") else {}
        return {"observation": resp.observation, "done": resp.done, "reward": resp.reward}

    def grade(self, task: TaskSpec, final: StepResult) -> Grade:
        return Grade(score=self.reward, success=self.reward >= task.grader.success_threshold, details={"done": self.done})

    def close(self) -> None:
        self.env = None
