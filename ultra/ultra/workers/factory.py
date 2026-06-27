"""Live worker-pool wiring. Offline tests construct ``WorkerPool`` directly with a
``FakeProvider``; this factory builds the OpenRouter-backed pool for real runs."""

from __future__ import annotations

from ..config import PoolConfig, WorkerSpec
from .budget import BudgetTracker
from .cache import CompletionCache
from .pool import RateGate
from .providers import OpenRouterProvider, Provider, WorkerPool


def build_pool(
    workers: list[WorkerSpec],
    config: PoolConfig | None = None,
    provider: Provider | None = None,
) -> WorkerPool:
    config = config or PoolConfig()
    if provider is None:
        provider = OpenRouterProvider(
            base_url=config.base_url,
            api_key=config.api_key(),
            timeout_s=config.timeout_s,
            sort_by_model={w.model: w.provider_sort for w in workers},
        )
    return WorkerPool(
        workers,
        provider,
        cache=CompletionCache(config.cache_dir),
        budget=BudgetTracker(config.budget_usd),
        gate=RateGate(config.max_concurrency, config.requests_per_minute),
        max_retries=config.max_retries,
        prompt_caching=config.prompt_caching,
    )
