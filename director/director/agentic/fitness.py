"""Agentic fitness + evaluation over a set of environments.

``agentic_fitness`` builds an ``eval_fn()`` for ``fugu.cmaes.evolve`` that returns the
mean terminal reward (resolve rate) over the instances, reflecting the router's current
parameters. ``agentic_eval`` reports the resolve rate plus the per-worker turn share —
the "which worker, when" view the report highlights (e.g. Opus called at debug steps).
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from ..shared.types import Sampling
from .env import AgentEnv
from .rollout import RolloutResult, agentic_rollout

EnvFactory = Callable[[], AgentEnv]


def _sample_allowed(worker_ids: list[str], keep_prob: float, seed: int) -> set[str]:
    """Deterministic random worker subset (>=2 kept). Seeded by task index so every
    CMA-ES candidate faces the SAME subset per task — fitness differences then reflect
    the router, not the luck of the dropout draw. Trains pool-subset robustness."""
    rng = random.Random(seed)
    kept = [w for w in worker_ids if rng.random() < keep_prob]
    if len(kept) < 2:
        kept = rng.sample(worker_ids, min(2, len(worker_ids)))
    return set(kept)


@dataclass
class FitnessConfig:
    """Reward shaping for the evolutionary fitness.

    ``w_div`` rewards spreading turns across the worker pool (normalized entropy),
    which prevents the router from collapsing to one globally-strong worker — the key
    shaping term. ``w_turn``/``w_cost`` are efficiency penalties (normalized turns / USD).
    The reference inspection used w_div≈0.15, w_turn≈0.10, w_cost=0; we default the
    efficiency penalties off (opt-in) and keep the diversity bonus on.
    """

    w_div: float = 0.15
    w_turn: float = 0.0
    w_cost: float = 0.0
    cost_ref_usd: float = 1.0  # normalizer for the cost penalty


def _norm_entropy(seq: list[str], num_workers: int) -> float:
    """Shannon entropy of worker usage within a rollout, normalized to [0, 1]."""
    if not seq or num_workers <= 1:
        return 0.0
    counts = Counter(seq)
    n = len(seq)
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    return h / math.log(num_workers)


def shape_fitness(
    rollouts: list[RolloutResult], *, num_workers: int, max_turns: int, cfg: FitnessConfig
) -> float:
    """Mean terminal reward + diversity bonus − turn/cost penalties."""
    if not rollouts:
        return 0.0
    n = len(rollouts)
    mean_reward = sum(r.reward for r in rollouts) / n
    mean_entropy = sum(_norm_entropy(r.worker_sequence, num_workers) for r in rollouts) / n
    mean_norm_turns = sum(min(r.turns / max(max_turns, 1), 1.0) for r in rollouts) / n
    mean_cost = sum(r.cost_usd for r in rollouts) / n
    return (
        mean_reward
        + cfg.w_div * mean_entropy
        - cfg.w_turn * mean_norm_turns
        - cfg.w_cost * (mean_cost / max(cfg.cost_ref_usd, 1e-9))
    )


async def _run_all(
    router, pool, factories: list[EnvFactory], *, max_turns, sampling, replicas, max_parallel,
    pool_dropout: float = 0.0, dropout_seed: int = 0,
):
    sem = asyncio.Semaphore(max_parallel)
    worker_ids = list(getattr(router, "worker_ids", []))

    async def one(factory: EnvFactory, job_idx: int):
        allowed = None
        if pool_dropout > 0.0 and worker_ids:
            allowed = _sample_allowed(worker_ids, 1.0 - pool_dropout, dropout_seed + job_idx)
        async with sem:
            env = factory()
            try:
                return await agentic_rollout(
                    router, pool, env, max_turns=max_turns, sampling=sampling, allowed=allowed
                )
            finally:
                env.close()

    jobs = [one(f, i) for i, f in enumerate(factories) for _ in range(replicas)]
    return await asyncio.gather(*jobs)


async def agentic_fitness_async(
    router,
    pool,
    factories: list[EnvFactory],
    *,
    max_turns: int = 30,
    sampling: Sampling | None = None,
    replicas: int = 1,
    max_parallel: int = 4,
    cfg: FitnessConfig | None = None,
    pool_dropout: float = 0.0,
    dropout_seed: int = 0,
) -> float:
    """Shaped fitness for ``router`` over the env factories (used by the parallel
    replica evaluator — one router per concurrent candidate)."""
    cfg = cfg or FitnessConfig()
    results = await _run_all(
        router, pool, factories,
        max_turns=max_turns, sampling=sampling, replicas=replicas, max_parallel=max_parallel,
        pool_dropout=pool_dropout, dropout_seed=dropout_seed,
    )
    num_workers = len(getattr(router, "worker_ids", [])) or router.num_workers
    return shape_fitness(results, num_workers=num_workers, max_turns=max_turns, cfg=cfg)


def agentic_fitness(
    router,
    pool,
    factories: list[EnvFactory],
    *,
    max_turns: int = 30,
    sampling: Sampling | None = None,
    replicas: int = 1,
    max_parallel: int = 4,
    cfg: FitnessConfig | None = None,
    pool_dropout: float = 0.0,
    dropout_seed: int = 0,
) -> Callable[[], float]:
    """Sync ``eval_fn`` for sequential ``evolve`` (runs its own event loop)."""
    def eval_fn() -> float:
        return asyncio.run(
            agentic_fitness_async(
                router, pool, factories, max_turns=max_turns, sampling=sampling,
                replicas=replicas, max_parallel=max_parallel, cfg=cfg,
                pool_dropout=pool_dropout, dropout_seed=dropout_seed,
            )
        )

    return eval_fn


def make_agentic_fitness_async(
    pool,
    factories: list[EnvFactory],
    *,
    max_turns: int = 30,
    sampling: Sampling | None = None,
    replicas: int = 1,
    max_parallel: int = 4,
    cfg: FitnessConfig | None = None,
    pool_dropout: float = 0.0,
    dropout_seed: int = 0,
) -> Callable[[object], "asyncio.Future"]:
    """Bind everything except the router → ``async fitness(router) -> float`` for the
    replica-parallel evaluator in ``fugu.cmaes.evolve_parallel``."""
    cfg = cfg or FitnessConfig()

    async def fitness(router) -> float:
        return await agentic_fitness_async(
            router, pool, factories, max_turns=max_turns, sampling=sampling,
            replicas=replicas, max_parallel=max_parallel, cfg=cfg,
            pool_dropout=pool_dropout, dropout_seed=dropout_seed,
        )

    return fitness


@dataclass
class AgenticReport:
    n: int
    resolve_rate: float
    avg_turns: float
    worker_turn_share: dict[str, float] = field(default_factory=dict)
    spent_usd: float = 0.0

    def render(self) -> str:
        lines = [
            f"instances={self.n}  resolve_rate={self.resolve_rate:.3f}  "
            f"avg_turns={self.avg_turns:.1f}  spent=${self.spent_usd:.4f}",
            "worker turn share:",
        ]
        for w, s in sorted(self.worker_turn_share.items(), key=lambda x: -x[1]):
            lines.append(f"  {w:>10}: {s:.2%}")
        return "\n".join(lines)


async def agentic_eval(
    router, pool, factories: list[EnvFactory], *, max_turns: int = 30, sampling: Sampling | None = None,
    max_parallel: int = 4,
) -> AgenticReport:
    results = await _run_all(
        router, pool, factories,
        max_turns=max_turns, sampling=sampling, replicas=1, max_parallel=max_parallel,
    )
    turn_counts: Counter = Counter()
    for r in results:
        turn_counts.update(r.worker_sequence)
    total_turns = sum(turn_counts.values()) or 1
    return AgenticReport(
        n=len(results),
        resolve_rate=sum(r.reward for r in results) / max(len(results), 1),
        avg_turns=sum(r.turns for r in results) / max(len(results), 1),
        worker_turn_share={w: c / total_turns for w, c in turn_counts.items()},
        spent_usd=pool.budget.spent_usd,
    )
