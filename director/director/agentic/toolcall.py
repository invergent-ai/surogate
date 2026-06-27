"""Per-step routed rollout for tool-use (function-calling) environments.

Reuses the router (per-step worker selection over the raw transcript), ``RolloutResult``,
and ``shape_fitness`` — only the action modality differs (tool calls vs bash). Produces
the same ``RolloutResult`` so the sep-CMA-ES harness (evolve/evolve_parallel) and fitness
shaping work unchanged.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..fugu.inference import select_worker
from ..shared.transcript import Transcript
from ..shared.types import Sampling
from .fitness import FitnessConfig, shape_fitness
from .rollout import RolloutResult
from .toolenv import RESPOND, ToolAction, ToolEnv, ToolEnvFactory

TOOL_SYSTEM = (
    "You are a customer-service agent. Use the provided tools to satisfy the user's "
    "request. Call one tool at a time; when the task is complete, respond to the user."
)


async def toolcall_rollout(
    router, pool, env: ToolEnv, *, max_turns: int = 30, sampling: Sampling | None = None,
    allowed: "set[str] | list[str] | None" = None,
) -> RolloutResult:
    sampling = sampling or Sampling(temperature=0.2, max_tokens=2048)
    user_msg, tools = env.reset()
    tx = Transcript()
    tx.add("system", TOOL_SYSTEM)
    tx.add("user", user_msg)
    messages = [{"role": "system", "content": TOOL_SYSTEM}, {"role": "user", "content": user_msg}]

    worker_sequence: list[str] = []
    cost = 0.0
    done = False
    turns = 0
    for turns in range(1, max_turns + 1):
        worker_id = select_worker(router, tx.render(), allowed=allowed)  # allowed -> solo worker for baselines
        worker_sequence.append(worker_id)
        resp = await pool.call_tools(worker_id, messages, tools, sampling)
        cost += resp.cost_usd
        tx.add("assistant", resp.as_text())

        if resp.tool_calls:
            tc = resp.tool_calls[0]
            messages.append({
                "role": "assistant", "content": resp.content or "",
                "tool_calls": [{
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }],
            })
            step = env.step(ToolAction(name=tc.name, arguments=tc.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": step.observation})
        else:
            messages.append({"role": "assistant", "content": resp.content or ""})
            step = env.step(ToolAction(name=RESPOND, arguments={"content": resp.content or ""}))
            messages.append({"role": "user", "content": step.observation})

        tx.add("user", step.observation)
        if step.done:
            done = True
            break

    return RolloutResult(
        reward=env.reward(), turns=turns, submitted=done,
        worker_sequence=worker_sequence, transcript=tx, cost_usd=cost,
    )


async def _run_all(router, pool, factories, *, max_turns, sampling, replicas, max_parallel):
    sem = asyncio.Semaphore(max_parallel)

    async def one(factory: ToolEnvFactory):
        async with sem:
            env = factory()
            try:
                return await toolcall_rollout(router, pool, env, max_turns=max_turns, sampling=sampling)
            finally:
                env.close()

    return await asyncio.gather(*[one(f) for f in factories for _ in range(replicas)])


@dataclass
class ToolEvalReport:
    n: int
    resolve_rate: float
    avg_turns: float
    worker_turn_share: dict[str, float] = field(default_factory=dict)
    spent_usd: float = 0.0

    def render(self) -> str:
        lines = [f"tool-use: instances={self.n} resolve_rate={self.resolve_rate:.3f} "
                 f"avg_turns={self.avg_turns:.1f} spent=${self.spent_usd:.4f}", "worker turn share:"]
        for w, s in sorted(self.worker_turn_share.items(), key=lambda x: -x[1]):
            lines.append(f"  {w:>10}: {s:.2%}")
        return "\n".join(lines)


async def toolcall_eval(
    router, pool, factories, *, max_turns=30, sampling=None, max_parallel=4
) -> ToolEvalReport:
    results = await _run_all(router, pool, factories, max_turns=max_turns,
                             sampling=sampling, replicas=1, max_parallel=max_parallel)
    turn_counts: Counter = Counter()
    for r in results:
        turn_counts.update(r.worker_sequence)
    total = sum(turn_counts.values()) or 1
    return ToolEvalReport(
        n=len(results),
        resolve_rate=sum(r.reward for r in results) / max(len(results), 1),
        avg_turns=sum(r.turns for r in results) / max(len(results), 1),
        worker_turn_share={w: c / total for w, c in turn_counts.items()},
        spent_usd=pool.budget.spent_usd,
    )


def make_toolcall_fitness_async(
    pool, factories, *, max_turns=30, sampling=None, replicas=1, max_parallel=4,
    cfg: FitnessConfig | None = None,
) -> Callable[[object], Awaitable[float]]:
    cfg = cfg or FitnessConfig()

    async def fitness(router) -> float:
        results = await _run_all(router, pool, factories, max_turns=max_turns,
                                 sampling=sampling, replicas=replicas, max_parallel=max_parallel)
        num_workers = len(getattr(router, "worker_ids", [])) or router.num_workers
        return shape_fitness(results, num_workers=num_workers, max_turns=max_turns, cfg=cfg)

    return fitness
