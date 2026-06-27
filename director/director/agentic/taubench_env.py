"""tau-bench tool-use environment, wrapped as a Director ToolEnv.

tau-bench (sierra-research/tau-bench) is the canonical judge-free tool-use benchmark:
the agent makes function calls against a simulated domain (retail/airline), an LLM
*user simulator* plays the customer, and reward is **programmatic** (final DB-state hash
+ required-output checks) — no judge. This is the tool-use breadth axis (the report's
τ³-Banking), and it slots into our tool-call rollout + sep-CMA-ES unchanged.

Requires the tau-bench package (install from github.com/sierra-research/tau-bench) and an
LLM for the user simulator (we route it through OpenRouter via litellm:
``user_provider="openrouter"``, ``user_model="openrouter/<slug>"``, ``OPENROUTER_API_KEY``).
"""

from __future__ import annotations

from collections.abc import Callable

from .toolenv import ToolAction, ToolStep


class TauBenchEnv:
    def __init__(
        self,
        env_name: str,           # "retail" | "airline"
        task_index: int,
        user_model: str = "openrouter/openai/gpt-5-mini",
        user_provider: str = "openrouter",
        user_strategy: str = "llm",
        task_split: str = "test",
    ):
        from tau_bench.envs import get_env

        self._env = get_env(
            env_name, user_strategy=user_strategy, user_model=user_model,
            task_split=task_split, user_provider=user_provider, task_index=task_index,
        )
        self._task_index = task_index
        self._reward = 0.0

    def reset(self) -> tuple[str, list[dict]]:
        r = self._env.reset(task_index=self._task_index)
        return r.observation, self._env.tools_info

    def step(self, action: ToolAction) -> ToolStep:
        from tau_bench.types import Action

        resp = self._env.step(Action(name=action.name, kwargs=action.arguments))
        self._reward = resp.reward
        return ToolStep(observation=resp.observation, done=resp.done)

    def reward(self) -> float:
        return self._reward

    def close(self) -> None:
        return None


def load_taubench_tasks(env_name: str = "retail", task_split: str = "test", limit: int | None = None) -> list[int]:
    """Return the task indices for a tau-bench domain/split (for factory construction)."""
    from tau_bench.envs import get_env

    env = get_env(env_name, user_strategy="llm", user_model="gpt-4o-mini",
                  task_split=task_split, user_provider="openai")
    n = len(env.tasks)
    idxs = list(range(n))
    return idxs[:limit] if limit else idxs


def build_taubench_factories(
    env_name: str, task_indices: list[int], *, user_model: str = "openrouter/openai/gpt-5-mini",
    user_provider: str = "openrouter", task_split: str = "test",
) -> list[Callable[[], TauBenchEnv]]:
    return [
        (lambda i=i: TauBenchEnv(env_name, i, user_model=user_model,
                                 user_provider=user_provider, task_split=task_split))
        for i in task_indices
    ]
