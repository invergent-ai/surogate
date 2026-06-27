"""Tool-use (function-calling) agentic environment contract + a scripted env.

This is the second agentic modality alongside the shell ``AgentEnv``: instead of bash
commands, the worker emits **tool calls** against a domain (e.g. tau-bench retail/airline).
Reward is judge-free (programmatic state/output checks). The per-step router picks which
worker handles each assistant turn.

    reset()           -> (first user message, OpenAI-style tools schema)
    step(action)      -> ToolStep(observation, done)   (action.name == "respond" -> user turn)
    reward()          -> terminal reward in [0, 1]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

RESPOND = "respond"


@dataclass
class ToolAction:
    name: str
    arguments: dict


@dataclass
class ToolStep:
    observation: str
    done: bool = False


@runtime_checkable
class ToolEnv(Protocol):
    def reset(self) -> tuple[str, list[dict]]: ...
    def step(self, action: ToolAction) -> ToolStep: ...
    def reward(self) -> float: ...
    def close(self) -> None: ...


@dataclass
class ScriptedToolEnv:
    """Deterministic tool-use env for tests/demos.

    Models a tiny task: the agent must call ``success_tool`` with ``success_args`` (a
    subset match), then respond. Reward 1.0 iff that happened. Tool calls return a canned
    observation; ``respond`` ends the episode.
    """

    task: str
    tools: list[dict]
    success_tool: str
    success_args: dict
    tool_result: str = "ok"
    calls: list[ToolAction] = field(default_factory=list)
    succeeded: bool = False

    def reset(self) -> tuple[str, list[dict]]:
        self.calls.clear()
        self.succeeded = False
        return self.task, self.tools

    def step(self, action: ToolAction) -> ToolStep:
        if action.name == RESPOND:
            return ToolStep(observation="(user) thanks, goodbye", done=True)
        self.calls.append(action)
        if action.name == self.success_tool and all(
            action.arguments.get(k) == v for k, v in self.success_args.items()
        ):
            self.succeeded = True
        return ToolStep(observation=self.tool_result, done=False)

    def reward(self) -> float:
        return 1.0 if self.succeeded else 0.0

    def close(self) -> None:
        return None


ToolEnvFactory = Callable[[], ToolEnv]
