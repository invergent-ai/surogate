"""End-to-end offline proofs: routing learns from soft targets; CMA-ES improves
end-to-end reward; the eval harness reports sane numbers."""

from __future__ import annotations

import numpy as np
import torch

from director.fugu.cmaes import evolve
from director.fugu.inference import answer_single, select_worker
from director.fugu.labels import generate_soft_targets
from director.fugu.sft import train_sft
from director.shared.eval import run_eval
from director.shared.types import Sampling

from conftest import make_typed_tasks


async def test_soft_targets_reflect_worker_skill(pool):
    ds = make_typed_tasks(4)
    labels = await generate_soft_targets(pool, ds, n_samples=2, tau=0.1)
    by_id = {lab.task_id: lab for lab in labels}
    # worker "a" (index 0) should dominate the soft target on a TYPE_A task
    a_label = by_id["a0"]
    assert a_label.worker_ids == ["a", "b"]
    assert a_label.p[0] > a_label.p[1]
    b_label = by_id["b0"]
    assert b_label.p[1] > b_label.p[0]


async def test_sft_learns_to_route(pool, router):
    ds = make_typed_tasks(8)
    labels = await generate_soft_targets(pool, ds, n_samples=2, tau=0.05)
    stats = train_sft(router, labels, epochs=200, lr=0.05, batch_size=8)
    assert stats.final_loss < 0.1
    # routing now matches worker specialization
    assert select_worker(router, "TYPE_A question 99") == "a"
    assert select_worker(router, "TYPE_B question 99") == "b"

    # and orchestrator accuracy hits the oracle ceiling
    report = await run_eval(router, pool, ds, sampling=Sampling(temperature=0.0))
    assert report.accuracy == 1.0
    assert report.oracle_accuracy == 1.0
    # each single worker alone only solves its half
    assert abs(report.per_worker_accuracy["a"] - 0.5) < 1e-6


async def test_answer_single_grades(pool, router):
    ds = make_typed_tasks(2)
    # untrained router may misroute, but the call + grading must work
    res = await answer_single(router, pool, ds[0])
    assert res.worker_id in {"a", "b"}
    assert res.reward in {0.0, 1.0}


def test_cmaes_improves_synthetic_objective(router):
    # Optimize the router's trainable vector toward a fixed target via a quadratic
    # reward. Proves the ask/tell loop, vector load/save, and recombination work.
    n = router.n_trainable
    target = torch.full((n,), 0.5)

    def eval_fn() -> float:
        v = router.trainable_vector()
        return float(-((v - target) ** 2).sum())  # higher (closer to target) is better

    start = eval_fn()
    res = evolve(router, eval_fn, generations=40, sigma0=0.5, seed=0)
    assert res.best_fitness > start
    assert res.mean_history[-1] > res.mean_history[0]
    # router ends loaded with the best vector found
    assert np.allclose(router.trainable_vector().numpy(), res.best_x, atol=1e-5)
