"""RULER (Relative Universal LLM-Elicited Rewards) — group-level vf.Rubric backed by an LLM judge.

Sends all rollouts in a group to an OpenAI-compatible judge in one request and
returns relative scores in [0, 1] as per-rollout rewards. ``build_ruler_rubric``
is the orchestrator-side factory; ``JudgeClientPool`` round-robins judge endpoints
under a shared concurrency cap.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from itertools import cycle
from textwrap import dedent
from typing import Any

import httpx
import verifiers as vf
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    LengthFinishReasonError,
    RateLimitError,
)
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel, Field, ValidationError

from surogate.core.config.grpo_orch_config import GRPORulerConfig, GRPORulerJudgeConfig
from surogate.grpo.utils.logger import get_logger

DEFAULT_RUBRIC = dedent(
    """
    - A trajectory that achieves its goal should always get a significantly higher score than a trajectory that does not achieve its goal.
    - A trajectory that achieves its goal more efficiently (e.g. by avoiding unproductive detours, redundant tool calls, or wasted reasoning) should get a higher score than a less efficient one.
    - If one trajectory is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
    - Partial credit may be awarded for a trajectory that makes meaningful progress toward its goal but does not complete it.
    - Penalize trajectories that hallucinate, ignore the user's request, or violate explicit constraints in the system prompt.
    """
).strip()


class TrajectoryScore(BaseModel):
    """Score for a single trajectory in a RULER judging request."""

    trajectory_id: str = Field(description="Identifier of the trajectory being scored, matching the input id.")
    explanation: str = Field(description="One- or two-sentence justification for the assigned score.")
    score: float = Field(ge=0.0, le=1.0, description="Score between 0 and 1, with 1 being optimal.")


class RulerJudgeResponse(BaseModel):
    """Top-level structured response returned by the RULER judge."""

    scores: list[TrajectoryScore] = Field(description="One score entry per trajectory, in any order.")


class RulerJudgeError(RuntimeError):
    """Raised when the judge produces an unrecoverable response."""


def _to_json_str(obj: Any) -> str:
    """Serialize a verifiers Message/Tool (or plain dict) to a JSON string.

    Pydantic's ``model_dump_json`` handles its own nested types correctly, so we
    don't need json.dumps's ``default=str`` fallback (which silently converts
    unknown types to repr()).
    """
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json(exclude_none=True)
    if isinstance(obj, dict):
        return json.dumps(obj)
    raise TypeError(f"Unsupported type for RULER serialization: {type(obj).__name__}")


def _common_prefix_len(serialized: list[list[str]]) -> int:
    """Length of the longest JSON-serialized message list that is a prefix of every trajectory.

    String equality is O(message_size) once and avoids recursive Pydantic ``__eq__``
    which would scan every field of every nested type.
    """
    if not serialized or not serialized[0]:
        return 0
    first = serialized[0]
    n = len(first)
    for idx in range(n):
        ref = first[idx]
        if not all(len(other) > idx and other[idx] == ref for other in serialized[1:]):
            return idx
    return n


class JudgeClientPool:
    """Round-robin pool of ``AsyncOpenAI`` clients with a shared concurrency cap.

    A single pool is shared across all RULER rubrics in the orchestrator so that the
    ``max_concurrent_judges`` budget is enforced globally, not per-environment.

    Each judge endpoint gets a dedicated ``httpx.AsyncClient`` with generous keep-alive
    so judging traffic doesn't share connection budget with rollout-time chat completions.
    """

    def __init__(self, config: GRPORulerJudgeConfig, max_concurrent: int | None):
        if not config.base_url:
            raise ValueError("JudgeClientPool requires at least one base_url")
        api_key = os.getenv(config.api_key_var, "EMPTY") if config.api_key_var else "EMPTY"
        timeout = httpx.Timeout(config.timeout, connect=config.connect_timeout)
        limits = httpx.Limits(
            max_connections=config.max_connections,
            max_keepalive_connections=config.max_keepalive_connections,
        )
        self._http_clients: list[httpx.AsyncClient] = []
        self._clients: list[AsyncOpenAI] = []
        for base_url in config.base_url:
            http_client = httpx.AsyncClient(timeout=timeout, limits=limits, headers=dict(config.headers or {}))
            self._http_clients.append(http_client)
            self._clients.append(
                AsyncOpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    http_client=http_client,
                    max_retries=config.max_retries,
                    timeout=timeout,
                )
            )
        self._cycle = cycle(self._clients)
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent is not None else None
        )
        get_logger().info(
            f"Initialized RULER judge pool with {len(self._clients)} endpoint(s) "
            f"(max_concurrent={max_concurrent}, timeout={config.timeout}s)"
        )

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[AsyncOpenAI]:
        """Acquire a concurrency slot, then yield the next round-robin client.

        Round-robin selection happens *after* the semaphore is held so cycle order
        reflects actual execution order, giving even load distribution under bursty
        traffic. ``next(cycle)`` is atomic in single-threaded asyncio (no preemption
        between sync ops), so no additional lock is needed.
        """
        if self._semaphore is not None:
            await self._semaphore.acquire()
        try:
            yield next(self._cycle)
        finally:
            if self._semaphore is not None:
                self._semaphore.release()

    async def aclose(self) -> None:
        for http_client in self._http_clients:
            try:
                await http_client.aclose()
            except Exception as e:
                get_logger().warning(f"Error closing RULER judge http client: {e}")
        self._http_clients.clear()
        self._clients.clear()


class RulerRubric(vf.Rubric):
    """A verifiers ``Rubric`` that scores rollout groups with an LLM-as-judge.

    The rubric registers a single group reward function (``ruler_score``) and overrides
    ``score_group`` so per-rollout judge diagnostics (token usage, cost, latency)
    are written into ``state["metrics"]`` alongside the score itself. When this rubric
    is composed via ``vf.RubricGroup``, those diagnostics propagate through the
    aggregation logic since their keys are unique (``ruler_*`` namespace).
    """

    METRIC_PREFIX = "ruler"

    def __init__(
        self,
        *,
        judge_pool: JudgeClientPool,
        judge_model: str,
        rubric: str | None,
        weight: float,
        sampling: dict[str, Any],
        extra_body: dict[str, Any],
        request_timeout: float | None,
        max_retries_on_parse_error: int,
        swallow_exceptions: bool,
        debug: bool,
        cost_input_per_million: float | None,
        cost_output_per_million: float | None,
    ):
        super().__init__()
        self._pool = judge_pool
        self._judge_model = judge_model
        self._rubric_text = (rubric or DEFAULT_RUBRIC).strip()
        self._sampling = dict(sampling or {})
        self._extra_body = dict(extra_body or {})
        self._request_timeout = request_timeout
        self._max_retries_on_parse_error = int(max_retries_on_parse_error)
        self._swallow_exceptions = bool(swallow_exceptions)
        self._debug = bool(debug)
        self._cost_input_per_million = cost_input_per_million
        self._cost_output_per_million = cost_output_per_million

        # Register the group reward func so verifiers' auto-detection
        # (`task_uses_group_scoring`) sees this as a group-level rubric.
        # Weight is the single source of truth on self.weights[0].
        self.add_reward_func(self.ruler_score, weight=float(weight))

    @property
    def weight(self) -> float:
        return self.weights[0]

    # ------------------------------------------------------------------
    # Reward function
    # ------------------------------------------------------------------
    async def ruler_score(
        self,
        states: list[vf.State],
        prompts: list[Any],
        completions: list[Any],
        infos: list[dict[str, Any]],
    ) -> list[float]:
        """Group reward function: score every rollout in ``states`` with the judge.

        Mutates ``state["metrics"]`` in place to record diagnostics; returns the raw
        scores (in [0, 1]) so verifiers' base machinery can record them under the
        ``ruler_score`` key. Raises ``RulerJudgeError`` on unrecoverable failure
        unless ``swallow_exceptions=True``, in which case all rollouts receive 0.0
        and a ``ruler_judge_failed`` metric is set.
        """
        n = len(states)
        if n == 0:
            return []

        try:
            scores, response_meta = await self._invoke_judge(prompts, completions, infos, states)
        except Exception as e:
            get_logger().warning(f"RULER judge failed for group of {n} rollout(s): {type(e).__name__}: {e}")
            if not self._swallow_exceptions:
                raise
            for state in states:
                self._merge_metrics(state, {f"{self.METRIC_PREFIX}_judge_failed": 1.0})
            return [0.0] * n

        for state in states:
            metrics = {
                f"{self.METRIC_PREFIX}_judge_calls": response_meta["judge_calls_per_rollout"],
                f"{self.METRIC_PREFIX}_judge_failed": 0.0,
                f"{self.METRIC_PREFIX}_judge_latency_ms": response_meta["judge_latency_ms_per_rollout"],
                f"{self.METRIC_PREFIX}_input_tokens": float(response_meta["input_tokens_per_rollout"]),
                f"{self.METRIC_PREFIX}_output_tokens": float(response_meta["output_tokens_per_rollout"]),
                f"{self.METRIC_PREFIX}_judge_cost_usd": float(response_meta["cost_usd_per_rollout"]),
            }
            self._merge_metrics(state, metrics)

        return list(scores)

    # ------------------------------------------------------------------
    # score_group / score_rollout overrides
    # ------------------------------------------------------------------
    async def score_rollout(self, state: vf.State) -> None:
        """Score a single rollout by treating it as a group of size 1.

        RULER is fundamentally a group-level rubric (it relies on relative scoring
        across a group); a one-element group is degenerate but supported so this
        rubric can be used in code paths that call ``score_rollout`` defensively.
        """
        await self.score_group([state])

    async def score_group(self, states: list[vf.State]) -> None:
        """Score a group of rollouts and write reward, advantage, and metrics in place.

        Overrides ``vf.Rubric.score_group`` to preserve per-rollout judge diagnostics
        in ``state["metrics"]`` (the base implementation overwrites that field with
        only the registered reward-func names).
        """
        n = len(states)
        if n == 0:
            get_logger().warning("RulerRubric.score_group called with empty states list")
            return

        start = time.perf_counter()
        normalized_infos: list[dict[str, Any]] = []
        for s in states:
            info = s.get("info")
            normalized_infos.append(info if isinstance(info, dict) else {})
        scores = await self.ruler_score(
            states=states,
            prompts=[s["prompt"] for s in states],
            completions=[s["completion"] for s in states],
            infos=normalized_infos,
        )
        scoring_ms = (time.perf_counter() - start) * 1000.0

        weighted = [score * self.weight for score in scores]
        avg = sum(weighted) / n if n else 0.0

        for i, state in enumerate(states):
            state["reward"] = weighted[i]
            state["advantage"] = weighted[i] - avg
            self._merge_metrics(state, {f"{self.METRIC_PREFIX}_score": float(scores[i])})
            timing = state.get("timing")
            if isinstance(timing, dict):
                timing["scoring_ms"] = scoring_ms
                timing["total_ms"] = float(timing.get("total_ms", 0.0)) + scoring_ms
            for step in state.get("trajectory") or []:
                if step.get("advantage") is None:
                    step["advantage"] = state["advantage"]
                if step.get("reward") is None:
                    step["reward"] = state["reward"]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_metrics(state: vf.State, new_metrics: dict[str, float]) -> None:
        """Merge ``new_metrics`` into ``state["metrics"]`` without dropping existing keys."""
        existing = state.get("metrics")
        if not isinstance(existing, dict):
            existing = {}
        existing.update(new_metrics)
        state["metrics"] = existing

    def _build_payload(
        self,
        prompts: list[Any],
        completions: list[Any],
        infos: list[dict[str, Any]],
        states: list[vf.State],
    ) -> tuple[list[dict[str, str]], int]:
        """Render the judge user prompt + return the number of trajectories actually sent.

        Performs the longest-common-prefix optimization: when every trajectory starts
        with the same system/user messages, those are factored out into a ``<context>``
        block sent once. When all trajectories are bit-identical, only one is sent and
        its score is replicated.
        """
        # Serialize each message to JSON once. Subsequent prefix detection compares
        # strings (cheap) and the prompt is built by concatenating these strings into
        # a JSON array — avoids a second json.dumps pass.
        serialized: list[list[str]] = [
            [_to_json_str(m) for m in (list(prompt or []) + list(completion or []))]
            for prompt, completion in zip(prompts, completions)
        ]

        prefix_len = _common_prefix_len(serialized)
        all_identical = len(serialized) > 1 and all(len(s) == prefix_len for s in serialized)

        # Tools: env-level state, identical across all rollouts of a group. Read from
        # the first state; fall back to info.
        tools_source = states[0].get("tool_defs") or infos[0].get("tool_defs") or []
        tools_json = "[" + ",".join(_to_json_str(t) for t in tools_source) + "]" if tools_source else None

        def _join(parts: list[str]) -> str:
            return "[" + ",".join(parts) + "]"

        parts: list[str] = []
        if prefix_len > 0 and not all_identical:
            parts.append("<context>\n" + _join(serialized[0][:prefix_len]) + "\n</context>")
        if tools_json is not None:
            parts.append("<available_tools>\n" + tools_json + "\n</available_tools>")

        if all_identical:
            sent_trajectories = 1
            parts.append("Trajectories:\n\n" + '<trajectory id="1">\n' + _join(serialized[0]) + "\n</trajectory>")
        else:
            sent_trajectories = len(serialized)
            traj_blocks = [
                f'<trajectory id="{idx}">\n' + _join(full[prefix_len:]) + "\n</trajectory>"
                for idx, full in enumerate(serialized, start=1)
            ]
            parts.append("Trajectories:\n\n" + "\n\n".join(traj_blocks))

        user_text = "\n\n".join(parts)
        traj_id_list = ", ".join(f'"{i}"' for i in range(1, sent_trajectories + 1))
        system_text = dedent(
            f"""
            You are an impartial judge ranking {sent_trajectories} agent trajectories that were given the same goal.
            Every trajectory below must receive a score between 0.0 and 1.0 with a one- or two-sentence
            explanation. Use the relative differences between trajectories to set the spread of scores.

            Grading rubric:
            {self._rubric_text}

            HARD REQUIREMENT: respond with a JSON object whose `scores` array contains EXACTLY
            {sent_trajectories} entr{"y" if sent_trajectories == 1 else "ies"} — one per trajectory.
            The `trajectory_id` of each entry must be one of: [{traj_id_list}].
            Do not omit, merge, or duplicate any trajectory.
            """
        ).strip()

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        return messages, sent_trajectories

    async def _invoke_judge(
        self,
        prompts: list[Any],
        completions: list[Any],
        infos: list[dict[str, Any]],
        states: list[vf.State],
    ) -> tuple[list[float], dict[str, Any]]:
        """Issue the judge HTTP call(s) and return per-rollout scores plus diagnostics."""
        n = len(states)
        messages, sent = self._build_payload(prompts, completions, infos, states)

        sampling = dict(self._sampling)
        # Request kwargs assembled once per attempt
        request_kwargs: dict[str, Any] = {
            "model": self._judge_model,
            "messages": messages,
            "response_format": RulerJudgeResponse,
        }
        request_kwargs.update(sampling)
        if self._extra_body:
            request_kwargs["extra_body"] = self._extra_body
        if self._request_timeout is not None:
            request_kwargs["timeout"] = self._request_timeout

        attempts = self._max_retries_on_parse_error + 1
        last_error: Exception | None = None
        total_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency_ms = 0.0

        for attempt in range(attempts):
            t0 = time.perf_counter()
            try:
                async with self._pool.acquire() as client:
                    completion = await client.chat.completions.parse(**request_kwargs)
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                total_calls += 1
                total_latency_ms += (time.perf_counter() - t0) * 1000.0
                last_error = e
                if attempt + 1 >= attempts:
                    raise RulerJudgeError(f"Judge call failed after {attempts} attempts: {e}") from e
                backoff = min(2.0**attempt, 8.0) + random.uniform(0.0, 0.25)
                get_logger().warning(
                    f"RULER judge transient error (attempt {attempt + 1}/{attempts}): "
                    f"{type(e).__name__}: {e}. Retrying after {backoff:.1f}s."
                )
                await asyncio.sleep(backoff)
                continue
            except LengthFinishReasonError as e:
                # Judge ran out of tokens before finishing the JSON response.
                # Common with reasoning-mode models (e.g. Qwen3 with thinking
                # enabled). Charge the partial usage, then retry with a doubled
                # budget — capped to avoid runaway cost.
                total_calls += 1
                total_latency_ms += (time.perf_counter() - t0) * 1000.0
                last_error = e
                hit_limit = int(getattr(e.completion.usage, "completion_tokens", 0) or 0)
                total_output_tokens += hit_limit
                if hasattr(e.completion, "usage") and e.completion.usage is not None:
                    total_input_tokens += int(getattr(e.completion.usage, "prompt_tokens", 0) or 0)
                if attempt + 1 >= attempts:
                    raise RulerJudgeError(
                        f"Judge truncated by max_completion_tokens={hit_limit} after {attempts} attempts; "
                        "increase ruler.sampling.max_completion_tokens or use a non-reasoning judge model"
                    ) from e
                new_budget = min(max(hit_limit, 1024) * 2, 16384)
                request_kwargs["max_completion_tokens"] = new_budget
                get_logger().warning(
                    f"RULER judge truncated at {hit_limit} tokens (attempt {attempt + 1}/{attempts}); "
                    f"retrying with max_completion_tokens={new_budget}"
                )
                continue
            except APIError as e:
                total_calls += 1
                total_latency_ms += (time.perf_counter() - t0) * 1000.0
                raise RulerJudgeError(f"Judge API error: {e}") from e

            total_calls += 1
            total_latency_ms += (time.perf_counter() - t0) * 1000.0
            if completion.usage is not None:
                total_input_tokens += int(getattr(completion.usage, "prompt_tokens", 0) or 0)
                total_output_tokens += int(getattr(completion.usage, "completion_tokens", 0) or 0)

            try:
                parsed = self._extract_parsed_response(completion, expected=sent)
            except RulerJudgeError as e:
                last_error = e
                if attempt + 1 < attempts:
                    # Bump temperature on parse retries: the judge is otherwise
                    # deterministic under temperature=0 + structured output, so a
                    # plain re-issue would reproduce the same malformed response.
                    bumped_temp = 0.3 + 0.2 * attempt
                    request_kwargs["temperature"] = bumped_temp
                    get_logger().warning(
                        f"RULER judge parse error (attempt {attempt + 1}/{attempts}): {e}. "
                        f"Retrying with temperature={bumped_temp:.2f}."
                    )
                    continue
                raise

            scores = self._scores_from_parsed(parsed, n=n, sent=sent)
            cost_usd_total = self._compute_cost(total_input_tokens, total_output_tokens)
            # All per-rollout fields are divided by group size so the orchestrator
            # can recover step totals via metrics_df[col].sum().
            meta = {
                "judge_calls_per_rollout": total_calls / n,
                "judge_latency_ms_per_rollout": total_latency_ms / n,
                "input_tokens_per_rollout": total_input_tokens / n,
                "output_tokens_per_rollout": total_output_tokens / n,
                "cost_usd_per_rollout": cost_usd_total / n,
            }

            log = get_logger()
            (log.info if self._debug else log.debug)(
                f"RULER judged group of {n} (sent={sent}, calls={total_calls}, "
                f"latency={total_latency_ms:.0f}ms, in={total_input_tokens} out={total_output_tokens}): "
                f"scores={[f'{s:.3f}' for s in scores]}"
            )
            if self._debug:
                for idx, (score, expl) in enumerate(
                    zip(scores, self._explanations_from_parsed(parsed, n=n, sent=sent)), start=1
                ):
                    get_logger().info(f"  RULER traj {idx}: {score:.3f} — {expl}")
            return scores, meta

        raise AssertionError(f"unreachable: judge call loop exited without returning (last_error={last_error})")

    def _extract_parsed_response(
        self, completion: ParsedChatCompletion[RulerJudgeResponse], *, expected: int
    ) -> RulerJudgeResponse:
        if not completion.choices:
            raise RulerJudgeError("Judge returned no choices")
        choice = completion.choices[0]
        message = choice.message
        parsed = getattr(message, "parsed", None)
        if parsed is None:
            content = message.content or ""
            try:
                parsed = RulerJudgeResponse.model_validate_json(content)
            except (ValidationError, json.JSONDecodeError) as e:
                raise RulerJudgeError(
                    f"Judge response could not be parsed (finish_reason={choice.finish_reason}): {e}"
                ) from e
        if not isinstance(parsed, RulerJudgeResponse):
            raise RulerJudgeError(f"Judge response type mismatch: {type(parsed).__name__}")
        if len(parsed.scores) != expected:
            raise RulerJudgeError(f"Judge returned {len(parsed.scores)} scores, expected {expected}")
        return parsed

    @staticmethod
    def _scores_from_parsed(parsed: RulerJudgeResponse, *, n: int, sent: int) -> list[float]:
        if sent == 1 and n > 1:
            value = max(0.0, min(1.0, parsed.scores[0].score))
            return [value] * n
        # Order scores by trajectory_id so that mis-ordered judge responses still align.
        by_id: dict[str, float] = {}
        for entry in parsed.scores:
            by_id[entry.trajectory_id] = max(0.0, min(1.0, entry.score))
        ordered: list[float] = []
        for idx in range(1, n + 1):
            key = str(idx)
            if key not in by_id:
                raise RulerJudgeError(f"Judge response missing score for trajectory id={key}")
            ordered.append(by_id[key])
        return ordered

    @staticmethod
    def _explanations_from_parsed(parsed: RulerJudgeResponse, *, n: int, sent: int) -> list[str]:
        if sent == 1 and n > 1:
            return [parsed.scores[0].explanation] * n
        by_id = {entry.trajectory_id: entry.explanation for entry in parsed.scores}
        return [by_id.get(str(idx), "") for idx in range(1, n + 1)]

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        cost = 0.0
        if self._cost_input_per_million is not None:
            cost += (input_tokens / 1_000_000.0) * self._cost_input_per_million
        if self._cost_output_per_million is not None:
            cost += (output_tokens / 1_000_000.0) * self._cost_output_per_million
        return cost


def build_ruler_rubric(config: GRPORulerConfig, judge_pool: JudgeClientPool) -> RulerRubric:
    """Construct a ``RulerRubric`` from a validated ``GRPORulerConfig`` plus a shared pool.

    The same ``JudgeClientPool`` should be passed to every ``RulerRubric`` so that
    the global ``max_concurrent_judges`` budget is shared across environments.

    In ``metric`` mode the registered weight is 0.0 so RULER contributes only
    observability metrics; the env's existing rubric (or another rubric in a
    ``RubricGroup``) supplies the training signal.
    """
    if not config.enabled:
        raise ValueError("build_ruler_rubric called with config.enabled=False")
    weight = 0.0 if config.mode == "metric" else float(config.weight)
    return RulerRubric(
        judge_pool=judge_pool,
        judge_model=config.judge_model,
        rubric=config.rubric,
        weight=weight,
        sampling=config.sampling or {},
        extra_body=config.extra_body or {},
        request_timeout=config.request_timeout,
        max_retries_on_parse_error=config.max_retries_on_parse_error,
        swallow_exceptions=config.swallow_exceptions,
        debug=config.debug,
        cost_input_per_million=config.cost.input_per_million if config.cost else None,
        cost_output_per_million=config.cost.output_per_million if config.cost else None,
    )
