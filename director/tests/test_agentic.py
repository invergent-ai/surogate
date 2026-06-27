"""Offline tests for the agentic layer: action parsing, scripted env, routed rollout."""

from __future__ import annotations

import torch

from director.agentic.actions import SUBMIT_SENTINEL, parse_action
from director.agentic.env import ScriptedEnv
from director.agentic.fitness import agentic_eval, agentic_fitness
from director.agentic.rollout import agentic_rollout
from director.config import WorkerSpec
from director.fugu.inference import attach_worker_ids
from director.fugu.model import SelectionRouter
from director.shared.providers import FakeProvider, WorkerPool

from conftest import FakeFeaturizer


def agentic_answer(model, messages, sampling) -> str:
    """A 'good' worker fixes then submits; a 'bad' worker just lists files."""
    joined = " ".join(m["content"] for m in messages)
    if "good" in model:
        if "edit applied" in joined:
            return f"```bash\n{SUBMIT_SENTINEL}\n```"
        return "```bash\necho FIXED > calc.py\n```"
    return "```bash\nls\n```"


def _router(prefer_index: int) -> SelectionRouter:
    r = SelectionRouter(FakeFeaturizer(), num_workers=2)
    attach_worker_ids(r, ["good", "bad"])
    with torch.no_grad():
        r.head.weight.zero_()
        r.head.weight[prefer_index, :] = 10.0  # always route to this worker
    return r


def _pool() -> WorkerPool:
    workers = [
        WorkerSpec(worker_id="good", model="prov/good-worker"),
        WorkerSpec(worker_id="bad", model="prov/bad-worker"),
    ]
    return WorkerPool(workers, FakeProvider(agentic_answer))


def test_parse_action():
    assert parse_action("```bash\nls -la\n```").command == "ls -la"
    assert parse_action(f"```bash\n{SUBMIT_SENTINEL}\n```").submit is True
    assert parse_action("no code here").command is None


def test_scripted_env():
    env = ScriptedEnv(task="fix it", fix_token="FIXED")
    env.reset()
    assert env.step("echo FIXED").observation == "edit applied"
    assert env.step("pytest").observation == "PASSED"
    assert env.evaluate() == 1.0


async def test_rollout_routes_to_good_worker():
    env = ScriptedEnv(task="fix the bug", fix_token="FIXED")
    res = await agentic_rollout(_router(0), _pool(), env, max_turns=6)
    assert res.reward == 1.0
    assert res.submitted
    assert set(res.worker_sequence) == {"good"}


async def test_rollout_bad_worker_fails():
    env = ScriptedEnv(task="fix the bug", fix_token="FIXED")
    res = await agentic_rollout(_router(1), _pool(), env, max_turns=4)
    assert res.reward == 0.0
    assert set(res.worker_sequence) == {"bad"}


async def test_agentic_eval_report():
    factories = [lambda: ScriptedEnv(task="fix the bug", fix_token="FIXED") for _ in range(3)]
    report = await agentic_eval(_router(0), _pool(), factories, max_turns=6, max_parallel=2)
    assert report.resolve_rate == 1.0
    assert report.worker_turn_share.get("good", 0) == 1.0


def test_agentic_fitness_reflects_routing():
    # sync test: eval_fn runs its own event loop (as it does under the CLI / evolve)
    factories = [lambda: ScriptedEnv(task="fix the bug", fix_token="FIXED") for _ in range(3)]
    good_fit = agentic_fitness(_router(0), _pool(), factories, max_turns=6, max_parallel=2)()
    bad_fit = agentic_fitness(_router(1), _pool(), factories, max_turns=4, max_parallel=2)()
    assert good_fit == 1.0 and bad_fit == 0.0


# --- (2) multi-turn CMA-ES proof: per-step routing where specialization matters --------

def _specialized_answer(model, messages, sampling) -> str:
    """Worker 'a' solves TYPE_A tasks, worker 'b' solves TYPE_B; the wrong specialist
    just lists files and never fixes. Right specialist fixes, then submits next turn."""
    joined = " ".join(m["content"] for m in messages)
    is_a_task = "TYPE_A" in joined
    is_worker_a = "worker-a" in model
    if is_a_task == is_worker_a:  # the right specialist for this task
        if "edit applied" in joined:
            return f"```bash\n{SUBMIT_SENTINEL}\n```"
        return f"```bash\necho {'FIXA' if is_a_task else 'FIXB'} > f\n```"
    return "```bash\nls\n```"


def _specialist_pool() -> WorkerPool:
    workers = [
        WorkerSpec(worker_id="a", model="prov/worker-a"),
        WorkerSpec(worker_id="b", model="prov/worker-b"),
    ]
    return WorkerPool(workers, FakeProvider(_specialized_answer))


def _mixed_factories(n_per_type: int = 3):
    facs = []
    for _ in range(n_per_type):
        facs.append(lambda: ScriptedEnv(task="TYPE_A: fix the bug", fix_token="FIXA"))
        facs.append(lambda: ScriptedEnv(task="TYPE_B: fix the bug", fix_token="FIXB"))
    return facs


def _always_a_router() -> SelectionRouter:
    r = SelectionRouter(FakeFeaturizer(), num_workers=2)
    attach_worker_ids(r, ["a", "b"])
    with torch.no_grad():
        r.head.weight.zero_()  # all-zero logits -> argmax ties to index 0 ('a'): only TYPE_A solved
    return r


def test_cmaes_lifts_agentic_terminal_reward():
    from director.agentic.fitness import FitnessConfig
    from director.fugu.cmaes import evolve

    router = _always_a_router()
    pool = _specialist_pool()
    factories = _mixed_factories(3)
    eval_fn = agentic_fitness(
        router, pool, factories, max_turns=4, max_parallel=6, cfg=FitnessConfig(w_div=0.0)
    )
    baseline = eval_fn()
    assert 0.4 <= baseline <= 0.6  # collapsed router only solves half (the TYPE_A tasks)

    res = evolve(router, eval_fn, generations=40, sigma0=0.5, seed=0)
    final = eval_fn()
    assert final > baseline + 0.2
    assert final >= 0.9  # per-step routing learned: route TYPE_A->a, TYPE_B->b
    assert res.best_history[-1] >= res.best_history[0]


def test_agentic_pool_dropout_runs():
    # the pool-dropout fitness path executes and returns a valid resolve rate
    router = _always_a_router()
    fit = agentic_fitness(
        router, _specialist_pool(), _mixed_factories(2),
        max_turns=4, max_parallel=4, pool_dropout=0.34, dropout_seed=0,
    )()
    assert 0.0 <= fit <= 1.0
