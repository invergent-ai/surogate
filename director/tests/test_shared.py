"""Tests for the shared worker-pool layer: cache, budget, pool, graders."""

from __future__ import annotations

import pytest

from director.shared.budget import BudgetExceeded, BudgetTracker
from director.shared.cache import CompletionCache, completion_key
from director.shared.providers import FakeProvider, WorkerPool
from director.shared.types import Completion, Sampling
from director.shared.verifiers import (
    code_exec,
    extract_boxed,
    gsm8k_exact,
    math_equal,
    mc_letter,
)
from director.config import WorkerSpec


def test_completion_key_stable_and_distinct():
    s = Sampling(seed=1)
    k1 = completion_key("m", [{"role": "user", "content": "hi"}], s)
    k2 = completion_key("m", [{"role": "user", "content": "hi"}], s)
    k3 = completion_key("m", [{"role": "user", "content": "bye"}], s)
    assert k1 == k2 and k1 != k3


def test_cache_roundtrip():
    c = CompletionCache(None)
    assert c.get("x") is None
    c.set("x", Completion(text="hello", model="m"))
    got = c.get("x")
    assert got is not None and got.text == "hello"


def test_budget_cap():
    b = BudgetTracker(cap_usd=0.10)
    b.add(0.05)
    b.check()  # still under
    b.add(0.06)
    with pytest.raises(BudgetExceeded):
        b.check()


async def test_pool_caches_identical_calls():
    fp = FakeProvider()
    pool = WorkerPool([WorkerSpec(worker_id="a", model="prov/a")], fp)
    msgs = [{"role": "user", "content": "q"}]
    c1 = await pool.call("a", msgs, Sampling(seed=0))
    c2 = await pool.call("a", msgs, Sampling(seed=0))
    assert fp.calls == 1  # second served from cache
    assert c2.cached and not c1.cached


async def test_pool_sample_n_distinct():
    fp = FakeProvider(lambda m, msgs, s: f"seed={s.seed}")
    pool = WorkerPool([WorkerSpec(worker_id="a", model="prov/a")], fp)
    comps = await pool.sample("a", [{"role": "user", "content": "q"}], 3, Sampling())
    assert fp.calls == 3
    assert {c.text for c in comps} == {"seed=0", "seed=1", "seed=2"}


def test_graders():
    assert gsm8k_exact("the answer is #### 42", "#### 42") == 1.0
    assert gsm8k_exact("answer 41", "#### 42") == 0.0
    assert math_equal("so \\boxed{\\frac{1}{2}}", "\\boxed{\\frac{1}{2}}") == 1.0
    assert mc_letter("Answer: C", "C") == 1.0
    assert mc_letter("Answer: A", "C") == 0.0
    assert extract_boxed("x \\boxed{a+\\frac{b}{c}} y") == "a+\\frac{b}{c}"


def test_code_exec():
    sol = {"test": "def check(f):\n    assert f(2) == 4\n", "entry_point": "sq"}
    good = "```python\ndef sq(x):\n    return x*x\n```"
    bad = "def sq(x):\n    return x+1\n"
    assert code_exec(good, sol) == 1.0
    assert code_exec(bad, sol) == 0.0


def test_code_exec_stdio():
    from director.shared.verifiers import code_exec_stdio

    sol = {"tests": [{"input": "2 3\n", "output": "5"}, {"input": "10 5\n", "output": "15"}]}
    good = "```python\na,b=map(int,input().split())\nprint(a+b)\n```"
    assert code_exec_stdio(good, sol) == 1.0
    assert code_exec_stdio("```python\nprint(0)\n```", sol) == 0.0


def test_grid_exact():
    from director.shared.verifiers import grid_exact, parse_grid

    gold = [[1, 2], [3, 4]]
    assert grid_exact("the answer is [[1, 2], [3, 4]]", gold) == 1.0  # json literal
    assert grid_exact("Test output:\n1 2\n3 4", gold) == 1.0  # digit rows
    assert grid_exact("1 2\n3 5", gold) == 0.0
    assert parse_grid("1 2\n3 4") == [[1, 2], [3, 4]]
