"""Concurrency, rate-limiting and retry helpers for outbound worker calls.

Adapts the pattern used by surogate's JudgeClientPool (asyncio.Semaphore for a
global concurrency cap + tenacity-style retry), with an optional aiolimiter
token bucket for RPM. Kept dependency-light: aiolimiter is optional.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator


class RateGate:
    """Bounds in-flight requests (semaphore) and, optionally, requests-per-minute.

    Uses ``aiolimiter`` for the RPM bucket when available; otherwise the RPM bound
    is skipped (the semaphore still applies).
    """

    def __init__(self, max_concurrency: int, requests_per_minute: float | None = None):
        self._sem = asyncio.Semaphore(max_concurrency)
        self._limiter = None
        if requests_per_minute:
            try:
                from aiolimiter import AsyncLimiter

                self._limiter = AsyncLimiter(requests_per_minute, time_period=60.0)
            except Exception:
                self._limiter = None

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._sem:
            if self._limiter is not None:
                async with self._limiter:
                    yield
            else:
                yield


async def call_with_retry(fn, *, max_retries: int, base_delay: float = 1.0):
    """Retry ``fn`` (a zero-arg coroutine factory) with exponential backoff.

    Retries on any exception; re-raises the last one if all attempts fail.
    """
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - provider errors are heterogeneous
            last = exc
            if attempt == max_retries:
                break
            await asyncio.sleep(base_delay * (2**attempt))
    assert last is not None
    raise last
