"""Evaluation harness.

Runs the router over a dataset and reports orchestrator accuracy, each worker's solo
accuracy, the oracle (best-possible routing) accuracy, and the routing distribution —
replicating the report's headline analyses (orchestrator vs single worker, plus the
per-task routing histogram).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field

from ..fugu.inference import select_worker
from .tasks import Dataset
from .transcript import raw_query
from .types import Sampling
from .verifiers import get_grader


@dataclass
class EvalReport:
    n: int
    accuracy: float
    per_worker_accuracy: dict[str, float]
    oracle_accuracy: float
    routing_distribution: dict[str, float]
    spent_usd: float = 0.0
    notes: dict = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            f"n={self.n}  spent=${self.spent_usd:.4f}",
            f"orchestrator accuracy: {self.accuracy:.3f}",
            f"oracle (best routing): {self.oracle_accuracy:.3f}",
            "per-worker accuracy:",
        ]
        for w, a in sorted(self.per_worker_accuracy.items(), key=lambda x: -x[1]):
            frac = self.routing_distribution.get(w, 0.0)
            lines.append(f"  {w:>10}: acc={a:.3f}  routed={frac:.2%}")
        return "\n".join(lines)


async def run_eval(
    router,
    pool,
    dataset: Dataset,
    *,
    sampling: Sampling | None = None,
) -> EvalReport:
    sampling = sampling or Sampling(temperature=0.0)
    worker_ids = pool.worker_ids
    tasks = list(dataset)

    # Grade every worker on every task once (cached), so we can compute solo,
    # oracle, and orchestrator accuracy from the same calls.
    async def worker_reward(wid, task) -> float:
        comp = await pool.call(wid, task.messages(), sampling)
        return get_grader(task.grader)(comp.text, task.solution)

    grid = await asyncio.gather(
        *[worker_reward(w, t) for t in tasks for w in worker_ids]
    )
    L = len(worker_ids)
    rewards = [grid[i * L : (i + 1) * L] for i in range(len(tasks))]

    routed = [select_worker(router, raw_query(t.prompt)) for t in tasks]
    idx_of = {w: j for j, w in enumerate(worker_ids)}

    orch_correct = sum(rewards[i][idx_of[routed[i]]] for i in range(len(tasks)))
    oracle_correct = sum(max(r) for r in rewards)
    per_worker = {
        w: sum(rewards[i][j] for i in range(len(tasks))) / max(len(tasks), 1)
        for j, w in enumerate(worker_ids)
    }
    counts = Counter(routed)
    dist = {w: counts.get(w, 0) / max(len(tasks), 1) for w in worker_ids}

    return EvalReport(
        n=len(tasks),
        accuracy=orch_correct / max(len(tasks), 1),
        per_worker_accuracy=per_worker,
        oracle_accuracy=oracle_correct / max(len(tasks), 1),
        routing_distribution=dist,
        spent_usd=pool.budget.spent_usd,
    )
