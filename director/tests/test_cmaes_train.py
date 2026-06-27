"""Tests for the agentic training harness: fitness shaping, CMA-ES checkpoint/resume,
and replica-parallel evolution."""

from __future__ import annotations

import asyncio

import numpy as np
import torch

from director.agentic.fitness import FitnessConfig, shape_fitness
from director.agentic.rollout import RolloutResult
from director.fugu.cmaes import evolve, evolve_parallel
from director.fugu.inference import attach_worker_ids
from director.fugu.model import SelectionRouter

from conftest import FakeFeaturizer


# ---- fitness shaping ------------------------------------------------------

def _rollout(seq, reward=1.0, turns=5, cost=0.0):
    return RolloutResult(reward=reward, turns=turns, submitted=True, worker_sequence=seq, cost_usd=cost)


def test_diversity_bonus_rewards_spread():
    cfg = FitnessConfig(w_div=0.15)
    one_worker = [_rollout(["a"] * 6)]
    spread = [_rollout(["a", "b", "c", "a", "b", "c"])]
    f_one = shape_fitness(one_worker, num_workers=3, max_turns=10, cfg=cfg)
    f_spread = shape_fitness(spread, num_workers=3, max_turns=10, cfg=cfg)
    assert f_one == 1.0  # entropy 0 -> no bonus
    assert f_spread > f_one  # spreading earns the diversity bonus
    assert abs(f_spread - (1.0 + 0.15)) < 1e-6  # uniform over 3 -> full normalized entropy


def test_turn_and_cost_penalties():
    base = shape_fitness([_rollout(["a"], turns=2, cost=0.0)], num_workers=2, max_turns=10, cfg=FitnessConfig(w_div=0))
    turn_pen = shape_fitness([_rollout(["a"], turns=10, cost=0.0)], num_workers=2, max_turns=10,
                             cfg=FitnessConfig(w_div=0, w_turn=0.2))
    cost_pen = shape_fitness([_rollout(["a"], turns=2, cost=1.0)], num_workers=2, max_turns=10,
                             cfg=FitnessConfig(w_div=0, w_cost=0.3, cost_ref_usd=1.0))
    assert base == 1.0
    assert turn_pen < base and abs(turn_pen - (1.0 - 0.2)) < 1e-6
    assert abs(cost_pen - (1.0 - 0.3)) < 1e-6


# ---- checkpoint / resume --------------------------------------------------

def _quadratic_router():
    r = SelectionRouter(FakeFeaturizer(), num_workers=2)
    attach_worker_ids(r, ["a", "b"])
    return r


def test_checkpoint_resume(tmp_path):
    r = _quadratic_router()
    target = torch.full((r.n_trainable,), 0.3)

    def eval_fn() -> float:
        return float(-((r.trainable_vector() - target) ** 2).sum())

    ckpt = str(tmp_path / "ck")
    res1 = evolve(r, eval_fn, generations=8, sigma0=0.5, seed=0, checkpoint_dir=ckpt)
    assert res1.generations_run == 8
    import os
    assert os.path.exists(os.path.join(ckpt, "cmaes.pkl"))

    # resume: continue to 16 generations, should not restart from scratch
    res2 = evolve(r, eval_fn, generations=16, sigma0=0.5, seed=0, checkpoint_dir=ckpt, resume=True)
    assert res2.generations_run == 16
    assert len(res2.mean_history) == 16  # history carried across the resume
    assert res2.best_fitness >= res1.best_fitness


# ---- replica-parallel evolution ------------------------------------------

def test_evolve_parallel_improves():
    routers = [_quadratic_router() for _ in range(2)]
    n = routers[0].n_trainable
    target = torch.full((n,), 0.25)

    async def fitness_async(router) -> float:
        await asyncio.sleep(0)  # exercise the async path
        return float(-((router.trainable_vector() - target) ** 2).sum())

    start = fitness_async  # noqa
    res = evolve_parallel(routers, fitness_async, generations=40, sigma0=0.5, seed=0)
    # all replicas end loaded with the best vector
    best = res.best_x
    for r in routers:
        assert np.allclose(r.trainable_vector().numpy(), best, atol=1e-5)
    assert res.best_fitness > -((torch.zeros(n) - target) ** 2).sum().item()
    assert res.mean_history[-1] > res.mean_history[0]
