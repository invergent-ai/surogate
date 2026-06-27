"""Agentic environment protocol + a scripted in-memory env for offline tests/demos.

An ``AgentEnv`` exposes a minimal shell-style contract:

    reset()            -> task description (str)
    step(command)      -> StepResult(observation, done)
    evaluate()         -> terminal reward in [0, 1]
    close()            -> release resources

Real harnesses (SWE-Bench Docker, Terminal-Bench) implement this same interface; the
rollout driver and CMA-ES fitness are harness-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class StepResult:
    observation: str
    done: bool = False


@runtime_checkable
class AgentEnv(Protocol):
    def reset(self) -> str: ...
    def step(self, command: str) -> StepResult: ...
    def evaluate(self) -> float: ...
    def close(self) -> None: ...


@dataclass
class ScriptedEnv:
    """A deterministic, dependency-free env for testing the rollout/routing loop.

    Models a tiny "fix the bug" task: a file contains a wrong expression; the agent
    must issue a command that writes the correct fix, then the tests pass. ``solved``
    flips when a command contains ``fix_token``; ``test`` commands report PASS/FAIL.
    """

    task: str
    fix_token: str
    test_token: str = "pytest"
    solved: bool = False
    steps_taken: int = 0
    log: list[str] = field(default_factory=list)

    def reset(self) -> str:
        self.solved = False
        self.steps_taken = 0
        self.log.clear()
        return self.task

    def step(self, command: str) -> StepResult:
        self.steps_taken += 1
        self.log.append(command)
        if self.fix_token in command:
            self.solved = True
            return StepResult(observation="edit applied", done=False)
        if self.test_token in command:
            return StepResult(observation="PASSED" if self.solved else "FAILED")
        return StepResult(observation=f"$ {command}\n(no output)")

    def evaluate(self) -> float:
        return 1.0 if self.solved else 0.0

    def close(self) -> None:  # nothing to release
        return None
