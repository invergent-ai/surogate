"""Maskable selection + pool-dropout training (the configurable-pool product feature).

A user can opt workers out at request time, so (a) selection must fall back to the best
*available* worker, and (b) training must teach the full ranking — not just the top pick
— so fallbacks are good. Uses a 3-worker stub (fallback only matters with >=3)."""

from __future__ import annotations

import pytest
import torch

from director.agentic.fitness import _sample_allowed
from director.fugu.inference import attach_worker_ids, select_worker
from director.fugu.labels import SoftLabel
from director.fugu.model import SelectionRouter
from director.fugu.sft import train_sft

from conftest import FakeFeaturizer


def _router3() -> SelectionRouter:
    r = SelectionRouter(FakeFeaturizer(), num_workers=3)
    attach_worker_ids(r, ["a", "b", "c"])
    return r


def test_select_worker_respects_mask():
    r = _router3()
    with torch.no_grad():
        # TYPE_A feature is [1,0] -> logits = head column 0 -> rank a>b>c
        r.head.weight.copy_(torch.tensor([[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]))
    q = "TYPE_A question"
    assert select_worker(r, q) == "a"
    assert select_worker(r, q, allowed={"b", "c"}) == "b"   # top pick opted out -> fallback
    assert select_worker(r, q, allowed={"c"}) == "c"        # only one left
    assert select_worker(r, q, allowed=["a", "c"]) == "a"   # list works too


def test_empty_or_unknown_allowed_raises():
    r = _router3()
    with pytest.raises(ValueError):
        select_worker(r, "TYPE_A q", allowed=set())
    with pytest.raises(ValueError):
        select_worker(r, "TYPE_A q", allowed={"nope"})


def _ranked_labels(n=8):
    # TYPE_A ranks a>b>c ; TYPE_B ranks c>b>a. Soft targets carry the FULL ranking.
    out = []
    for i in range(n):
        out.append(SoftLabel(f"a{i}", f"TYPE_A question {i}", ["a", "b", "c"],
                             r_bar=[1.0, 0.6, 0.2], p=[0.6, 0.3, 0.1], grader="fake_right"))
        out.append(SoftLabel(f"b{i}", f"TYPE_B question {i}", ["a", "b", "c"],
                             r_bar=[0.2, 0.6, 1.0], p=[0.1, 0.3, 0.6], grader="fake_right"))
    return out


def test_pool_dropout_sft_recovers_full_ranking():
    r = _router3()
    stats = train_sft(r, _ranked_labels(), epochs=400, lr=0.1, batch_size=8, pool_dropout=0.5)
    assert stats.final_loss == stats.final_loss  # not NaN (dropout-masked CE is finite)
    # full ranking learned -> masking the top pick falls back to the next-best, in order
    assert select_worker(r, "TYPE_A q") == "a"
    assert select_worker(r, "TYPE_A q", allowed={"b", "c"}) == "b"
    assert select_worker(r, "TYPE_A q", allowed={"a", "c"}) == "a"
    assert select_worker(r, "TYPE_B q") == "c"
    assert select_worker(r, "TYPE_B q", allowed={"a", "b"}) == "b"
    assert select_worker(r, "TYPE_B q", allowed={"a"}) == "a"


def test_sample_allowed_deterministic_and_min_two():
    ids = ["a", "b", "c", "d"]
    s1 = _sample_allowed(ids, 0.5, seed=7)
    s2 = _sample_allowed(ids, 0.5, seed=7)
    assert s1 == s2 and s1.issubset(set(ids)) and len(s1) >= 2
    # different seeds generally give different subsets across the index range
    subsets = {frozenset(_sample_allowed(ids, 0.5, seed=k)) for k in range(20)}
    assert len(subsets) > 1
