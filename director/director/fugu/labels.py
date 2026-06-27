"""Soft-target label generation for the SFT stage.

For each verifiable question, every worker is sampled ``n`` times and graded against
the ground truth. The per-worker mean reward vector ``r̄`` is turned into a soft target
distribution ``p(j) ∝ exp(r̄_j / τ)``. The router is then trained to match ``p`` (see
``fugu.sft``). Labels are cached to JSONL so SFT re-runs never re-hit the API.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict, dataclass

from ..shared.tasks import Dataset, Task
from ..shared.types import Sampling
from ..shared.verifiers import get_grader

try:  # progress bar is optional
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


class _Progress:
    """Minimal progress shim that works with or without tqdm."""

    def __init__(self, total: int, desc: str):
        self._bar = tqdm(total=total, desc=desc) if tqdm is not None else None
        self._n = 0
        self._total = total

    def update(self, k: int = 1) -> None:
        self._n += k
        if self._bar is not None:
            self._bar.update(k)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


@dataclass
class SoftLabel:
    task_id: str
    prompt: str
    worker_ids: list[str]
    r_bar: list[float]
    p: list[float]
    grader: str


def _softmax(xs: list[float], tau: float) -> list[float]:
    m = max(xs)
    exps = [math.exp((x - m) / tau) for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


async def _score_task(pool, task: Task, n_samples: int, sampling: Sampling) -> list[float]:
    grader = get_grader(task.grader)
    worker_ids = pool.worker_ids

    async def worker_mean(wid: str) -> float:
        comps = await pool.sample(wid, task.messages(), n_samples, sampling)
        rewards = [grader(c.text, task.solution) for c in comps]
        return sum(rewards) / len(rewards) if rewards else 0.0

    return list(await asyncio.gather(*[worker_mean(w) for w in worker_ids]))


async def generate_soft_targets(
    pool,
    dataset: Dataset,
    *,
    n_samples: int = 4,
    tau: float = 0.1,
    sampling: Sampling | None = None,
    out_path: str | None = None,
    max_questions_in_flight: int | None = None,
) -> list[SoftLabel]:
    """Score every question against every worker and build soft targets.

    Questions are processed concurrently; the pool's own RateGate bounds the actual
    number of in-flight API calls, so the slow tail of one question overlaps with the
    others instead of blocking them. ``max_questions_in_flight`` optionally caps how
    many questions are open at once (memory bound for very large datasets).
    """
    sampling = sampling or Sampling()
    worker_ids = pool.worker_ids
    tasks = list(dataset)
    progress = _Progress(len(tasks), "labeling")
    sem = asyncio.Semaphore(max_questions_in_flight) if max_questions_in_flight else None

    async def label_one(task: Task) -> SoftLabel:
        if sem is not None:
            async with sem:
                r_bar = await _score_task(pool, task, n_samples, sampling)
        else:
            r_bar = await _score_task(pool, task, n_samples, sampling)
        progress.update()
        return SoftLabel(
            task_id=task.task_id,
            prompt=task.prompt,
            worker_ids=worker_ids,
            r_bar=r_bar,
            p=_softmax(r_bar, tau),
            grader=task.grader,
        )

    try:
        # return_exceptions so one failed item (e.g. a rate-limit storm exhausting
        # retries) doesn't discard the whole expensive run; skip + report failures.
        results = await asyncio.gather(*[label_one(t) for t in tasks], return_exceptions=True)
    finally:
        progress.close()
    labels = [r for r in results if isinstance(r, SoftLabel)]
    failed = len(results) - len(labels)
    if failed:
        print(f"[labels] WARNING: {failed}/{len(tasks)} items failed and were skipped")
    if out_path:
        save_labels(labels, out_path)
    return labels


def save_labels(labels: list[SoftLabel], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for lab in labels:
            f.write(json.dumps(asdict(lab), ensure_ascii=False) + "\n")


def load_labels(path: str) -> list[SoftLabel]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(SoftLabel(**json.loads(line)))
    return out
