"""sep-CMA-ES refinement on end-to-end reward, with checkpoint/resume and optional
replica-parallel candidate evaluation.

Wraps pycma's diagonal-covariance CMA-ES (``CMA_diagonal=True``) — sep-CMA-ES, the
Trinity/Fugu variant; default population ``4 + 3·floor(ln n)``. We warm-start from the
SFT vector and maximize mean end-to-end terminal reward (shaped, see agentic.fitness).

Two evaluation strategies share one checkpointed loop (``evolve_core``):
  - SequentialEvaluator: one router, candidates evaluated one at a time (rollouts inside
    a candidate may still run concurrently).
  - ReplicaEvaluator: K identical routers; candidates within a generation are evaluated
    concurrently, one per replica (cross-candidate parallelism without process pools —
    each replica loads its own candidate vector, so weights never race).

Checkpointing persists the CMA-ES state + best vector each generation so a multi-day
agentic run survives interruption (``checkpoint_dir`` + ``resume``).
"""

from __future__ import annotations

import asyncio
import os
import pickle
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from ..shared.tasks import Dataset
from ..shared.types import Sampling
from .inference import answer_single
from .model import SelectionRouter


class SepCMAES:
    def __init__(
        self,
        x0,
        sigma0: float = 0.03,
        popsize: int | None = None,
        seed: int = 0,
        tolerate_flat: bool = True,
    ):
        import cma

        opts: dict = {"CMA_diagonal": True, "verbose": -9, "seed": seed}
        if popsize is not None:
            opts["popsize"] = popsize
        if tolerate_flat:
            # Sparse binary terminal rewards make early generations flat (every candidate
            # scores the same, e.g. all-0). pycma stops after one flat generation by
            # default (tolflatfitness=1) and on tiny fitness range (tolfun); both kill
            # agentic runs prematurely. Disable those so the run uses its full generation
            # budget and keeps exploring until reward signal appears.
            opts.update({"tolflatfitness": 1e9, "tolfun": 0.0, "tolfunhist": 0.0})
        self.es = cma.CMAEvolutionStrategy(list(np.asarray(x0, dtype=float)), sigma0, opts)

    def ask(self) -> list[np.ndarray]:
        return self.es.ask()

    def tell(self, solutions, losses) -> None:
        self.es.tell(solutions, losses)  # pycma minimizes; callers pass -fitness

    def stop(self) -> bool:
        return bool(self.es.stop())

    @property
    def popsize(self) -> int:
        return self.es.popsize


@dataclass
class EvolveResult:
    best_x: np.ndarray
    best_fitness: float
    mean_history: list[float] = field(default_factory=list)
    best_history: list[float] = field(default_factory=list)
    generations_run: int = 0


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

_CKPT = "cmaes.pkl"


def _save_ckpt(d: str, es: SepCMAES, gen: int, res: EvolveResult) -> None:
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, _CKPT + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(
            {
                "es": es.es, "gen": gen,
                "best_x": np.asarray(res.best_x), "best_f": res.best_fitness,
                "mean_history": res.mean_history, "best_history": res.best_history,
            },
            f,
        )
    os.replace(tmp, os.path.join(d, _CKPT))  # atomic


def _load_ckpt(d: str):
    path = os.path.join(d, _CKPT)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# evaluators (candidate batch -> fitnesses; higher is better)
# ---------------------------------------------------------------------------


def _to_param(x) -> torch.Tensor:
    return torch.tensor(np.asarray(x), dtype=torch.float32)


class SequentialEvaluator:
    def __init__(self, router: SelectionRouter, eval_fn: Callable[[], float]):
        self.router = router
        self.eval_fn = eval_fn

    def __call__(self, candidates: list[np.ndarray]) -> list[float]:
        fits = []
        for x in candidates:
            self.router.load_vector(_to_param(x))
            fits.append(float(self.eval_fn()))
        return fits


class ReplicaEvaluator:
    """Evaluate a generation's candidates concurrently across K identical routers.

    Each candidate acquires a free replica, loads its vector, and runs its (async)
    fitness; the replica is returned to the pool afterwards. Concurrency = #replicas.
    """

    def __init__(self, routers: list[SelectionRouter], fitness_async: Callable[[object], Awaitable[float]]):
        if not routers:
            raise ValueError("ReplicaEvaluator needs at least one router")
        self.routers = routers
        self.fitness_async = fitness_async

    def __call__(self, candidates: list[np.ndarray]) -> list[float]:
        return asyncio.run(self._all(candidates))

    async def _all(self, candidates: list[np.ndarray]) -> list[float]:
        q: asyncio.Queue = asyncio.Queue()
        for r in self.routers:
            q.put_nowait(r)

        async def one(x) -> float:
            router = await q.get()
            try:
                router.load_vector(_to_param(x))
                return float(await self.fitness_async(router))
            finally:
                q.put_nowait(router)

        return list(await asyncio.gather(*[one(x) for x in candidates]))


# ---------------------------------------------------------------------------
# core loop
# ---------------------------------------------------------------------------


def evolve_core(
    es: SepCMAES,
    evaluate_batch: Callable[[list[np.ndarray]], list[float]],
    *,
    generations: int,
    checkpoint_dir: str | None = None,
    resume: bool = True,
    verbose: bool = False,
) -> EvolveResult:
    res = EvolveResult(best_x=np.zeros(0), best_fitness=-float("inf"))
    start = 0
    if checkpoint_dir and resume:
        ck = _load_ckpt(checkpoint_dir)
        if ck is not None:
            es.es = ck["es"]
            start = ck["gen"]
            res.best_x, res.best_fitness = ck["best_x"], ck["best_f"]
            res.mean_history, res.best_history = ck["mean_history"], ck["best_history"]
            if verbose:
                print(f"[cmaes] resumed from gen {start} (best={res.best_fitness:.4f})")

    for gen in range(start, generations):
        candidates = es.ask()
        fits = evaluate_batch(candidates)
        es.tell(candidates, [-f for f in fits])
        gen_best_i = int(np.argmax(fits))
        if fits[gen_best_i] > res.best_fitness:
            res.best_fitness = float(fits[gen_best_i])
            res.best_x = np.asarray(candidates[gen_best_i], dtype=float)
        res.mean_history.append(float(np.mean(fits)))
        res.best_history.append(float(np.max(fits)))
        res.generations_run = gen + 1
        if verbose:
            print(f"[cmaes] gen {gen:3d}  mean={np.mean(fits):.4f}  best={res.best_fitness:.4f}")
        if checkpoint_dir:
            _save_ckpt(checkpoint_dir, es, gen + 1, res)
        if es.stop():
            break
    return res


# ---------------------------------------------------------------------------
# public entrypoints
# ---------------------------------------------------------------------------


def evolve(
    router: SelectionRouter,
    eval_fn: Callable[[], float],
    *,
    generations: int = 60,
    sigma0: float = 0.03,
    popsize: int | None = None,
    seed: int = 0,
    x0: np.ndarray | None = None,
    checkpoint_dir: str | None = None,
    resume: bool = True,
    verbose: bool = False,
) -> EvolveResult:
    """Sequential evolution against a sync ``eval_fn`` (evaluates the router's current
    parameters). Warm-starts from the router's trainable vector unless ``x0`` is given."""
    if x0 is None:
        x0 = router.trainable_vector().numpy()
    es = SepCMAES(x0, sigma0=sigma0, popsize=popsize, seed=seed)
    res = evolve_core(
        es, SequentialEvaluator(router, eval_fn),
        generations=generations, checkpoint_dir=checkpoint_dir, resume=resume, verbose=verbose,
    )
    if res.best_x.size:
        router.load_vector(_to_param(res.best_x))
    return res


def evolve_parallel(
    routers: list[SelectionRouter],
    fitness_async: Callable[[object], Awaitable[float]],
    *,
    generations: int = 60,
    sigma0: float = 0.03,
    popsize: int | None = None,
    seed: int = 0,
    x0: np.ndarray | None = None,
    checkpoint_dir: str | None = None,
    resume: bool = True,
    verbose: bool = False,
) -> EvolveResult:
    """Replica-parallel evolution: candidates within a generation run concurrently, one
    per router replica. All replicas are loaded with the best vector at the end."""
    if not routers:
        raise ValueError("evolve_parallel needs at least one router replica")
    if x0 is None:
        x0 = routers[0].trainable_vector().numpy()
    es = SepCMAES(x0, sigma0=sigma0, popsize=popsize, seed=seed)
    res = evolve_core(
        es, ReplicaEvaluator(routers, fitness_async),
        generations=generations, checkpoint_dir=checkpoint_dir, resume=resume, verbose=verbose,
    )
    if res.best_x.size:
        for r in routers:
            r.load_vector(_to_param(res.best_x))
    return res


def pool_fitness(
    router: SelectionRouter,
    pool,
    dataset: Dataset,
    *,
    sampling: Sampling | None = None,
    replicas: int = 1,
) -> Callable[[], float]:
    """Single-step end-to-end fitness: mean terminal reward over ``dataset`` via ``pool``."""
    sampling = sampling or Sampling()
    tasks = list(dataset)

    async def _rollout() -> float:
        results = await asyncio.gather(
            *[answer_single(router, pool, t, sampling) for t in tasks for _ in range(replicas)]
        )
        return sum(r.reward for r in results) / max(len(results), 1)

    def eval_fn() -> float:
        return asyncio.run(_rollout())

    return eval_fn
