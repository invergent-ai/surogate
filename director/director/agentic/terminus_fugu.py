"""Terminal-Bench via the shipped harness (Terminus 2) with per-step routing.

Terminus 2 is the report's Terminal-Bench EVALUATION harness (Appendix A). It owns the
task container, tmux session, and native grading; it calls an LLM each episode. We inject
Fugu by subclassing terminal-bench's LiteLLM so every call() runs our SelectionRouter,
picks a worker, sets the model_name to that worker's OpenRouter slug, and delegates — the
same per-step routing as the SWE-Bench path. ``Terminus2Fugu`` swaps the routed LLM into
the real Terminus 2 agent.

Solo baselines: ``allowed={worker_id}`` forces one worker (proper per-worker resolve rate).
Run via terminal-bench's harness with a Terminus2Fugu instance (terminal-bench grades).
"""

from __future__ import annotations

import threading

from terminal_bench.agents.terminus_2.terminus_2 import Terminus2
from terminal_bench.llms.lite_llm import LiteLLM

from ..fugu.inference import select_worker


def _render_tb(prompt: str, history: list) -> str:
    """Raw role:content surface form for the router (matches our featurizer)."""
    parts = []
    for m in history:
        if isinstance(m, dict):
            role, content = m.get("role", ""), m.get("content", "")
        else:
            role, content = getattr(m, "role", ""), getattr(m, "content", "")
        parts.append(f"{role}: {content}")
    parts.append(f"user: {prompt}")
    return "\n".join(parts)


class FuguLLM(LiteLLM):
    """terminal-bench LLM that routes each call to a worker via the SelectionRouter."""

    _gpu_lock = threading.Lock()  # serialize the 0.6B featurize across parallel rollouts

    def __init__(self, router, worker_slugs: dict[str, str], allowed=None,
                 temperature: float = 0.7, max_tokens: int = 32768, **kwargs):
        first = "openrouter/" + next(iter(worker_slugs.values()))
        super().__init__(model_name=first, temperature=temperature, **kwargs)
        self.router = router
        self.worker_slugs = worker_slugs
        self.allowed = allowed
        self.worker_sequence: list[str] = []
        self._max_tokens = max_tokens

    def call(self, prompt: str, message_history: list = [], **kwargs) -> str:
        with FuguLLM._gpu_lock:
            wid = select_worker(self.router, _render_tb(prompt, message_history), allowed=self.allowed)
        self.worker_sequence.append(wid)
        self._model_name = "openrouter/" + self.worker_slugs[wid]  # delegate to chosen worker
        kwargs.setdefault("max_tokens", self._max_tokens)
        eb = dict(kwargs.get("extra_body") or {})
        eb.setdefault("provider", {"sort": "price"})       # cheapest provider first
        eb.setdefault("reasoning", {"effort": "high"})      # max reasoning (Fugu setting)
        kwargs["extra_body"] = eb
        return super().call(prompt, message_history=message_history, **kwargs)


class Terminus2Fugu(Terminus2):
    """The real Terminus 2 agent with a routed (FuguLLM) backend."""

    def __init__(self, router, worker_slugs: dict[str, str], allowed=None,
                 temperature: float = 0.7, **kwargs):
        first = "openrouter/" + next(iter(worker_slugs.values()))
        super().__init__(model_name=first, temperature=temperature, **kwargs)
        self._llm = FuguLLM(router, worker_slugs, allowed=allowed, temperature=temperature)

    @property
    def worker_sequence(self) -> list[str]:
        return self._llm.worker_sequence
