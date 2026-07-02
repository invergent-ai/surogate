"""Worker-pool configuration (vendored from the router's shared infra, trimmed).

Only the pieces ``ultra.workers`` needs: a worker spec, sampling, and pool settings.
Router-specific config (backbone/featurizer/SVF) is intentionally absent — Ultra has
no frozen-backbone router.
"""

from __future__ import annotations

import os

from pydantic import BaseModel


class WorkerSpec(BaseModel):
    """One worker in the pool. ``worker_id`` is the stable, ordered class index the
    Conductor addresses; ``model`` is a logical name or provider slug resolved by
    the live provider router."""

    worker_id: str
    model: str
    # Static USD-per-million-token costs; used only if OpenRouter does not report a
    # per-call cost. None => trust the provider's reported cost.
    cost_in_per_mtok: float | None = None
    cost_out_per_mtok: float | None = None
    # Provider routing: "price" = cheapest-first where supported; None = provider default.
    provider_sort: str | None = "price"


class SamplingConfig(BaseModel):
    # Ultra worker regime (ultra-intro §1): workers sampled at temperature 0.2 with a
    # 4,096-token cap in the constrained training regime.
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 4096
    seed: int | None = None
    reasoning_effort: str | None = "high"


class PoolConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    split_provider_routing: bool = True
    max_concurrency: int = 8
    requests_per_minute: float | None = None
    max_retries: int = 4
    timeout_s: float = 120.0
    cache_dir: str = "./.ultra_cache/completions"
    budget_usd: float | None = None  # None => unlimited
    prompt_caching: bool = True

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)
