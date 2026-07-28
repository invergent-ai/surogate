"""Worker providers and the WorkerPool.

Ultra workers use OpenAI-compatible provider endpoints. ``RoutedOpenAIProvider``
lets one ``WorkerPool`` route commercial frontier workers through Yunwu and
open/specialist workers through OpenRouter while preserving the same completion
interface. ``WorkerPool`` adds caching, budgeting, concurrency control and an
n-sample helper on top.

A ``FakeProvider`` lets the whole pipeline run offline and for free in tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from ..config import PoolConfig, WorkerSpec
from ..providers import provider as registry_provider
from ..providers import assert_live_provider_allowed
from ..providers import routed_provider_name, routed_slug
from .budget import BudgetTracker
from .cache import CompletionCache, completion_key, tool_completion_key
from .pool import RateGate, call_with_retry
from .prompt_cache import with_cache_control
from .types import Completion, Message, Sampling, ToolCall, ToolCompletion

# OpenRouter app-attribution (shown in OpenRouter's logs/rankings). Sent on every call.
OPENROUTER_REFERER = "https://surogate.ai"
OPENROUTER_TITLE = "Surogate"
OPENROUTER_CATEGORIES = "cloud-agent,personal-agent"


def openrouter_attribution_headers() -> dict[str, str]:
    return {
        "HTTP-Referer": OPENROUTER_REFERER,
        "X-OpenRouter-Title": OPENROUTER_TITLE,
        "X-OpenRouter-Categories": OPENROUTER_CATEGORIES,
    }


@runtime_checkable
class Provider(Protocol):
    async def complete(
        self, model: str, messages: list[Message], sampling: Sampling
    ) -> Completion: ...

    async def complete_tools(
        self, model: str, messages: list, tools: list, sampling: Sampling
    ) -> ToolCompletion: ...


class OpenRouterProvider:
    """OpenAI-compatible client pointed at OpenRouter."""

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        timeout_s: float = 120.0,
        sort_by_model: dict[str, str | None] | None = None,
    ):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout_s,
            default_headers=openrouter_attribution_headers(),
        )
        # per-model provider routing: model slug -> "price" | None (default). Missing => "price".
        self._sort_by_model = sort_by_model or {}

    def _provider_routing(self, model: str) -> dict | None:
        sort = self._sort_by_model.get(model, "price")
        return {"sort": sort} if sort else None  # None => omit, use OpenRouter default routing

    async def complete(
        self, model: str, messages: list[Message], sampling: Sampling
    ) -> Completion:
        from openai import NOT_GIVEN

        extra = {"usage": {"include": True}}
        routing = self._provider_routing(model)
        if routing is not None:
            extra["provider"] = routing
        # OpenAI-native reasoning models (gpt-5.x, reached via Yunwu -- GPT never routes
        # through OpenRouter) ignore max_tokens AND the OpenRouter-style reasoning object,
        # reasoning unbounded to a ~20k internal ceiling (observed 2026-07-06: 6.7M excess
        # tokens billed invisibly). They require max_completion_tokens + top-level
        # reasoning_effort instead.
        # gpt-*/grok-*/gemini-* BARE slugs (all yunwu; OpenRouter slugs are prefixed "x-ai/"/"google/" and
        # honors caps, so it self-selects out): unenforced caps + unbounded server-side
        # reasoning (grok-4.5 probe 2026-07-09: max_tokens=400 ground >2 min) -> the
        # streaming wall-clock budget is the only real cap.
        openai_native_reasoning = model.startswith(("gpt-", "grok-", "gemini-"))
        if openai_native_reasoning:
            # effort flag ONLY -- no token-cap param: Yunwu doesn't enforce caps for these
            # models anyway (probe 2026-07-06: mct=1024 -> 4559 tok), and sending
            # max_completion_tokens ALONGSIDE reasoning_effort can make the API ignore the
            # effort flag entirely (documented gpt-5.x interaction bug).
            if sampling.reasoning_effort is not None:
                extra["reasoning_effort"] = sampling.reasoning_effort
            # Streaming with a client-side wall-clock budget is the ONLY enforceable cap:
            # these models reason ADAPTIVELY (per Yunwu docs) BEFORE the first visible token,
            # to ~20k tokens on brutal prompts regardless of parameters. Disconnecting stops
            # server-side generation and billing at that point.
            return await self._complete_streaming_budget(model, messages, sampling, extra)
        elif sampling.reasoning_effort is not None:
            extra["reasoning"] = {"effort": sampling.reasoning_effort}
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=NOT_GIVEN if openai_native_reasoning else sampling.max_tokens,
            seed=sampling.seed,
            extra_body=extra,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = float(getattr(usage, "cost", 0.0) or 0.0)
        if cost <= 0.0 and usage is not None:
            # cheapest-provider routing often reports top-level cost=0 with the real cost in
            # cost_details.upstream_inference_cost — use it so budget + cost-tiebreak have signal.
            cd = getattr(usage, "cost_details", None)
            if isinstance(cd, dict):
                cost = float(cd.get("upstream_inference_cost", 0.0) or 0.0)
            elif cd is not None:
                cost = float(getattr(cd, "upstream_inference_cost", 0.0) or 0.0)
        return Completion(
            text=text,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            finish_reason=getattr(choice, "finish_reason", None),
        )

    async def _complete_streaming_budget(
        self, model: str, messages: list[Message], sampling: Sampling, extra: dict
    ) -> Completion:
        """Streamed completion with wall-clock budgets (adaptive-reasoning models only).

        first-token deadline bounds the silent reasoning phase (probe: healthy minimal
        calls reach first token in ~25-30 s; blowouts sit silent for minutes); the total
        ceiling bounds pathological streams. An abort returns a normal truncated
        Completion (finish_reason='abort_budget') so the retry ladder does NOT re-burn it.
        Budgets are env-tunable: ULTRA_GPT_FIRST_TOKEN_S / ULTRA_GPT_TOTAL_S."""
        import os
        import time

        import asyncio

        first_token_s = float(os.environ.get("ULTRA_GPT_FIRST_TOKEN_S", "120"))
        total_s = float(os.environ.get("ULTRA_GPT_TOTAL_S", "240"))
        t0 = time.monotonic()
        text_parts: list[str] = []
        finish_reason = None
        prompt_tokens = completion_tokens = 0
        got_first = False
        stream = None
        try:
            # HARD outer deadline: the per-chunk budget checks below only run when chunks
            # ARRIVE — a completely silent stream (observed 2026-07-06: zombie gpt calls
            # froze the whole 12-wide lane for ~1 h) would otherwise hang forever, dodging
            # both the chunk checks and the client read-timeout.
            async with asyncio.timeout(total_s + 30):
                stream = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    seed=sampling.seed,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body=extra,
                )
                async for chunk in stream:
                    now = time.monotonic() - t0
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta is not None and delta.content:
                            got_first = True
                            text_parts.append(delta.content)
                        fr = getattr(chunk.choices[0], "finish_reason", None)
                        if fr:
                            finish_reason = fr
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                    if (not got_first and now > first_token_s) or now > total_s:
                        finish_reason = "abort_budget"
                        break
        except (TimeoutError, asyncio.TimeoutError):
            finish_reason = "abort_budget"
        except Exception:
            # a broken stream with partial text is still a valid (truncated) completion;
            # with none it degrades to an empty wrong answer rather than a retry-burn
            if finish_reason is None:
                finish_reason = "stream_error"
        finally:
            if stream is not None:
                try:
                    async with asyncio.timeout(5):
                        await stream.close()
                except Exception:
                    pass
        return Completion(
            text="".join(text_parts),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=0.0,
            finish_reason=finish_reason,
        )

    async def complete_tools(
        self, model: str, messages: list, tools: list, sampling: Sampling
    ) -> ToolCompletion:
        from openai import NOT_GIVEN

        extra = {"usage": {"include": True}}
        routing = self._provider_routing(model)
        if routing is not None:
            extra["provider"] = routing
        # same OpenAI-native reasoning-model handling as complete() (see comment there)
        # gpt-*/grok-*/gemini-* BARE slugs (all yunwu; OpenRouter slugs are prefixed "x-ai/"/"google/" and
        # honors caps, so it self-selects out): unenforced caps + unbounded server-side
        # reasoning (grok-4.5 probe 2026-07-09: max_tokens=400 ground >2 min) -> the
        # streaming wall-clock budget is the only real cap.
        openai_native_reasoning = model.startswith(("gpt-", "grok-", "gemini-"))
        if openai_native_reasoning:
            # effort flag ONLY -- no token-cap param: Yunwu doesn't enforce caps for these
            # models anyway (probe 2026-07-06: mct=1024 -> 4559 tok), and sending
            # max_completion_tokens ALONGSIDE reasoning_effort can make the API ignore the
            # effort flag entirely (documented gpt-5.x interaction bug).
            if sampling.reasoning_effort is not None:
                extra["reasoning_effort"] = sampling.reasoning_effort
        elif sampling.reasoning_effort is not None:
            extra["reasoning"] = {"effort": sampling.reasoning_effort}
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=sampling.temperature,
            max_tokens=NOT_GIVEN if openai_native_reasoning else sampling.max_tokens,
            extra_body=extra,
        )
        msg = resp.choices[0].message
        calls = []
        for tc in getattr(msg, "tool_calls", None) or []:
            import json as _json

            try:
                args = _json.loads(tc.function.arguments or "{}")
            except (ValueError, TypeError):
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = getattr(resp, "usage", None)
        return ToolCompletion(
            content=msg.content,
            tool_calls=calls,
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=float(getattr(usage, "cost", 0.0) or 0.0),
        )


class RoutedOpenAIProvider:
    """OpenAI-compatible router keyed by logical model/known provider slug."""

    def __init__(
        self,
        timeout_s: float = 120.0,
        sort_by_model: dict[str, str | None] | None = None,
    ):
        self._timeout_s = timeout_s
        self._clients: dict[str, OpenRouterProvider] = {}
        self._sort_by_provider_model: dict[str, dict[str, str | None]] = {}
        for model, sort in (sort_by_model or {}).items():
            provider_name = routed_provider_name(model)
            provider_model = routed_slug(model, provider_name)
            self._sort_by_provider_model.setdefault(provider_name, {})[provider_model] = sort

    def _client_for(self, model: str) -> tuple[OpenRouterProvider, str]:
        provider_name = routed_provider_name(model)
        provider_model = routed_slug(model, provider_name)
        assert_live_provider_allowed(provider_name, model=model, context="direct worker call")
        client = self._clients.get(provider_name)
        if client is None:
            cfg = registry_provider(provider_name)
            key_env = str(cfg.get("key_env") or "")
            api_key = None
            if key_env:
                from ultra.providers import resolve_api_key

                api_key = resolve_api_key(key_env)
                if not api_key:
                    raise RuntimeError(
                        f"{key_env} is not set; model {model!r} routes to provider {provider_name!r}"
                    )
            client = OpenRouterProvider(
                base_url=str(cfg["base_url"]),
                api_key=api_key,
                timeout_s=self._timeout_s,
                sort_by_model=self._sort_by_provider_model.get(provider_name, {}),
            )
            self._clients[provider_name] = client
        return client, provider_model

    async def complete(
        self, model: str, messages: list[Message], sampling: Sampling
    ) -> Completion:
        client, provider_model = self._client_for(model)
        return await client.complete(provider_model, messages, sampling)

    async def complete_tools(
        self, model: str, messages: list, tools: list, sampling: Sampling
    ) -> ToolCompletion:
        client, provider_model = self._client_for(model)
        return await client.complete_tools(provider_model, messages, tools, sampling)


class FakeProvider:
    """Deterministic, free provider for tests and offline development.

    ``answer_fn(model, messages, sampling) -> str`` controls the response. The
    default echoes the model + last user message.
    """

    def __init__(
        self,
        answer_fn: Callable[[str, list[Message], Sampling], str] | None = None,
        tool_fn: Callable[[str, list, list, Sampling], ToolCompletion] | None = None,
    ):
        self._fn = answer_fn or self._default
        self._tool_fn = tool_fn
        self.calls: int = 0

    @staticmethod
    def _default(model: str, messages: list[Message], sampling: Sampling) -> str:
        last = messages[-1]["content"] if messages else ""
        return f"[{model}] {last}"

    async def complete(
        self, model: str, messages: list[Message], sampling: Sampling
    ) -> Completion:
        self.calls += 1
        return Completion(text=self._fn(model, messages, sampling), model=model)

    async def complete_tools(
        self, model: str, messages: list, tools: list, sampling: Sampling
    ) -> ToolCompletion:
        self.calls += 1
        if self._tool_fn is not None:
            return self._tool_fn(model, messages, tools, sampling)
        return ToolCompletion(content=self._fn(model, messages, sampling), tool_calls=[], model=model)


class WorkerPool:
    """Routes ``worker_id -> model slug`` and calls the provider with caching,
    budgeting and concurrency control."""

    def __init__(
        self,
        workers: list[WorkerSpec],
        provider: Provider,
        cache: CompletionCache | None = None,
        budget: BudgetTracker | None = None,
        gate: RateGate | None = None,
        max_retries: int = 4,
        prompt_caching: bool = True,
        call_timeout: float = 300.0,
    ):
        if not workers:
            raise ValueError("WorkerPool requires at least one worker")
        self._workers = list(workers)
        self._by_id = {w.worker_id: w for w in self._workers}
        self._provider = provider
        self._cache = cache or CompletionCache(None)
        self._budget = budget or BudgetTracker(None)
        self._gate = gate or RateGate(max_concurrency=8)
        self._max_retries = max_retries
        self._prompt_caching = prompt_caching
        self._call_timeout = call_timeout  # hard backstop over the client's own timeout

    @property
    def worker_ids(self) -> list[str]:
        """Ordered worker ids; index j is the router's class for worker M_j."""
        return [w.worker_id for w in self._workers]

    @property
    def budget(self) -> BudgetTracker:
        return self._budget

    def model_for(self, worker_id: str) -> str:
        return self._by_id[worker_id].model

    async def call(
        self, worker_id: str, messages: list[Message], sampling: Sampling
    ) -> Completion:
        model = self.model_for(worker_id)
        key = completion_key(model, messages, sampling)
        hit = self._cache.get(key)
        if hit is not None:
            hit.cached = True
            return hit
        self._budget.check()
        spec = self._by_id[worker_id]
        send_msgs = (
            with_cache_control(messages, model=model)[0] if self._prompt_caching else messages
        )

        async def _do() -> Completion:
            async with self._gate.slot():
                return await asyncio.wait_for(
                    self._provider.complete(model, send_msgs, sampling), self._call_timeout
                )

        comp = await call_with_retry(_do, max_retries=self._max_retries)
        if comp.cost_usd == 0.0:
            comp.cost_usd = _estimate_cost(spec, comp)
        self._budget.add(comp.cost_usd)
        self._cache.set(key, comp)
        return comp

    async def call_tools(
        self, worker_id: str, messages: list, tools: list, sampling: Sampling
    ) -> ToolCompletion:
        """Function-calling worker call (for tool-use agentic tasks). Cached like call()."""
        model = self.model_for(worker_id)
        key = tool_completion_key(model, messages, sampling, tools)
        hit = self._cache.get_tool(key)
        if hit is not None:
            hit.cached = True
            return hit
        self._budget.check()
        spec = self._by_id[worker_id]
        send_msgs, send_tools = (
            with_cache_control(messages, tools, model=model) if self._prompt_caching else (messages, tools)
        )

        async def _do() -> ToolCompletion:
            async with self._gate.slot():
                return await asyncio.wait_for(
                    self._provider.complete_tools(model, send_msgs, send_tools, sampling), self._call_timeout
                )

        comp = await call_with_retry(_do, max_retries=self._max_retries)
        if comp.cost_usd == 0.0:
            comp.cost_usd = _estimate_cost_tool(spec, comp)
        self._budget.add(comp.cost_usd)
        self._cache.set_tool(key, comp)
        return comp

    async def sample(
        self, worker_id: str, messages: list[Message], n: int, sampling: Sampling
    ) -> list[Completion]:
        """n independent samples. Seeds are varied per sample so each is cached
        distinctly and reproducibly."""
        import asyncio

        base_seed = sampling.seed or 0
        tasks = [
            self.call(
                worker_id,
                messages,
                Sampling(
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    max_tokens=sampling.max_tokens,
                    seed=base_seed + i,
                ),
            )
            for i in range(n)
        ]
        return await asyncio.gather(*tasks)


def _estimate_cost(spec: WorkerSpec, comp: Completion) -> float:
    if spec.cost_in_per_mtok is None or spec.cost_out_per_mtok is None:
        return 0.0
    return (
        comp.prompt_tokens * spec.cost_in_per_mtok
        + comp.completion_tokens * spec.cost_out_per_mtok
    ) / 1_000_000.0


def _estimate_cost_tool(spec: WorkerSpec, comp: ToolCompletion) -> float:
    if spec.cost_in_per_mtok is None or spec.cost_out_per_mtok is None:
        return 0.0
    return (
        comp.prompt_tokens * spec.cost_in_per_mtok
        + comp.completion_tokens * spec.cost_out_per_mtok
    ) / 1_000_000.0


def build_pool(cfg: PoolConfig, workers: list[WorkerSpec]) -> WorkerPool:
    """Construct a live provider-backed pool from config."""
    if cfg.split_provider_routing:
        provider: Provider = RoutedOpenAIProvider(
            timeout_s=cfg.timeout_s,
            sort_by_model={w.model: w.provider_sort for w in workers},
        )
    else:
        provider = OpenRouterProvider(
            base_url=cfg.base_url, api_key=cfg.api_key(), timeout_s=cfg.timeout_s,
            sort_by_model={w.model: w.provider_sort for w in workers},
        )
    return WorkerPool(
        workers=workers,
        provider=provider,
        cache=CompletionCache(cfg.cache_dir),
        budget=BudgetTracker(cfg.budget_usd),
        gate=RateGate(cfg.max_concurrency, cfg.requests_per_minute),
        max_retries=cfg.max_retries,
        prompt_caching=cfg.prompt_caching,
    )
