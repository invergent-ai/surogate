"""Configuration for the Director orchestrator.

Pydantic models loaded from YAML/JSON or env. Kept deliberately small: paths,
worker pool membership, sampling, and spend caps. Training hyperparameters live
next to their stage (see ``fugu.sft`` / ``fugu.cmaes``).
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class WorkerSpec(BaseModel):
    """One worker in the pool. ``worker_id`` is the stable, ordered class index the
    router learns to predict; ``model`` is the OpenRouter slug actually called."""

    worker_id: str
    model: str
    # Optional static USD-per-million-token costs, used only if OpenRouter does not
    # return a per-call cost. None => trust the provider's reported cost.
    cost_in_per_mtok: float | None = None
    cost_out_per_mtok: float | None = None
    # OpenRouter provider routing: "price" = cheapest-provider-first; None = OpenRouter's
    # default (reliability/uptime-weighted) routing. Use None for models whose cheap
    # providers are flaky/slow (e.g. they hang or ignore the reasoning-effort hint).
    provider_sort: str | None = "price"


class SamplingConfig(BaseModel):
    # Fugu generation setting (Fugu report §4.1.1): workers at MAXIMUM reasoning effort,
    # with a high token cap so max-reasoning outputs are not truncated.
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 32768
    seed: int | None = None
    reasoning_effort: str | None = "high"


class PoolConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    max_concurrency: int = 8
    requests_per_minute: float | None = None
    max_retries: int = 4
    timeout_s: float = 120.0
    cache_dir: str = "./.director_cache/completions"
    budget_usd: float | None = None  # None => unlimited
    prompt_caching: bool = True  # inject Anthropic cache_control breakpoints (others auto-cache)

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


class FeaturizerConfig(BaseModel):
    """How the frozen backbone turns a (possibly long) transcript into the routing
    feature. context_window is a cap (auto-capped to the model's max); short transcripts
    forward cheaply regardless. head_tail keeps goal + recent state when over the cap."""

    hidden_position: str = "penultimate"          # decision token (-2)
    context_window: int = 4096                      # cap; bounded so router forwards don't OOM the
    #                                                 GPU on long agentic transcripts (terminal/swe)
    context_strategy: str = "head_tail"            # head_tail | tail | full
    head_tokens: int = 512                          # goal/system prefix kept in head_tail
    svf_targets: list[str] | None = None            # None => default layer-26 projections


class DirectorConfig(BaseModel):
    workers: list[WorkerSpec] = Field(default_factory=list)
    pool: PoolConfig = Field(default_factory=PoolConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    featurizer: FeaturizerConfig = Field(default_factory=FeaturizerConfig)
    backbone: str = "Qwen/Qwen3-0.6B"
    work_dir: str = "./.director"

    @property
    def worker_ids(self) -> list[str]:
        return [w.worker_id for w in self.workers]


def default_frontier_pool() -> list[WorkerSpec]:
    """The Fugu worker pool: a cost-escalation ladder anchored on GLM-5.2.

    The product is cost-aware escalation — route to cheap GLM by default, escalate to a frontier
    model only when the query needs it. The pool is a 3-tier cost ladder, each tier earning its slot
    on measured complementarity (n=8 screen + Fugu report):
      - GLM-5.2  ($4)  default/hero: matched the per-item oracle on ~69% of queries.
      - Gemini-3.1-Pro ($12) cheap escalation: best on math/general (screen), GPQA leader (report).
      - Opus-4.8 ($25) premium escalation: best on code/science (screen), SWE-bench-Pro + debug (report).
    GPT-5.5 was DROPPED: best at nothing on the single-step screen and the most expensive ($30); its
    only edge is the agentic-coding builder role (Terminal-Bench), out of current scope.
    Ordering defines the router's class index ``j ↔ M_j`` and must stay stable.
    """
    return [
        WorkerSpec(worker_id="glm", model="z-ai/glm-5.2", provider_sort=None),  # cheap providers hang/ignore effort
        WorkerSpec(worker_id="gemini", model="google/gemini-3.1-pro-preview"),
        WorkerSpec(worker_id="opus", model="anthropic/claude-opus-4.8"),
    ]
