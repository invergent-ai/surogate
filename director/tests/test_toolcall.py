"""Offline tests for the tool-use agentic modality: function-calling worker pool,
scripted tool env, per-step routed tool-call rollout, and evolution over it."""

from __future__ import annotations

import torch

from director.agentic.toolcall import make_toolcall_fitness_async, toolcall_eval, toolcall_rollout
from director.agentic.toolenv import ScriptedToolEnv
from director.config import WorkerSpec
from director.fugu.cmaes import evolve_parallel
from director.fugu.inference import attach_worker_ids
from director.fugu.model import SelectionRouter
from director.shared.providers import FakeProvider, WorkerPool
from director.shared.types import Sampling, ToolCall, ToolCompletion

from conftest import FakeFeaturizer

TOOLS = [
    {"type": "function", "function": {"name": "cancel_order",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}},
]


def tool_answer(model, messages, tools, sampling) -> ToolCompletion:
    """'good' cancels the right order then responds; 'bad' just responds."""
    already_cancelled = any(m.get("role") == "tool" for m in messages)
    if "good" in model and not already_cancelled:
        return ToolCompletion(content=None, model=model,
                              tool_calls=[ToolCall(id="c1", name="cancel_order", arguments={"order_id": "O123"})])
    return ToolCompletion(content="Done, anything else?", tool_calls=[], model=model)


def _pool() -> WorkerPool:
    workers = [WorkerSpec(worker_id="good", model="prov/good"), WorkerSpec(worker_id="bad", model="prov/bad")]
    return WorkerPool(workers, FakeProvider(tool_fn=tool_answer))


def _router(prefer: int) -> SelectionRouter:
    r = SelectionRouter(FakeFeaturizer(), num_workers=2)
    attach_worker_ids(r, ["good", "bad"])
    with torch.no_grad():
        r.head.weight.zero_()
        r.head.weight[prefer, :] = 10.0
    return r


def _env() -> ScriptedToolEnv:
    return ScriptedToolEnv(task="cancel order O123", tools=TOOLS,
                           success_tool="cancel_order", success_args={"order_id": "O123"})


async def test_pool_call_tools_caches():
    pool = _pool()
    msgs = [{"role": "user", "content": "hi"}]
    a = await pool.call_tools("good", msgs, TOOLS, Sampling())
    b = await pool.call_tools("good", msgs, TOOLS, Sampling())
    assert a.tool_calls[0].name == "cancel_order"
    assert b.cached and not a.cached


async def test_toolcall_rollout_routes():
    good = await toolcall_rollout(_router(0), _pool(), _env(), max_turns=6)
    assert good.reward == 1.0 and set(good.worker_sequence) == {"good"}
    bad = await toolcall_rollout(_router(1), _pool(), _env(), max_turns=4)
    assert bad.reward == 0.0 and set(bad.worker_sequence) == {"bad"}


async def test_toolcall_eval_report():
    report = await toolcall_eval(_router(0), _pool(), [_env for _ in range(3)], max_turns=6, max_parallel=2)
    assert report.resolve_rate == 1.0
    assert report.worker_turn_share.get("good", 0) == 1.0


def test_evolve_over_tooluse():
    # CMA-ES discovers routing tool-use tasks to the capable worker (start neutral)
    torch.manual_seed(0)
    routers = []
    for _ in range(2):
        r = SelectionRouter(FakeFeaturizer(), num_workers=2)
        attach_worker_ids(r, ["good", "bad"])
        routers.append(r)
    fitness = make_toolcall_fitness_async(_pool(), [_env for _ in range(2)], max_turns=6, max_parallel=2)
    res = evolve_parallel(routers, fitness, generations=30, sigma0=0.6, seed=0)
    assert res.best_fitness >= 0.99  # found routing that resolves (reward 1, single-worker -> entropy 0)
