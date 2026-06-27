"""Offline tests for the curation pipeline: classify, probe (resumable), balance, split."""

from __future__ import annotations

from collections import Counter

from director.data.manifest import (
    ProbeRecord,
    classify,
    probed_ids,
    read_manifest,
    read_probes,
)
from director.shared.curate import (
    accuracy_matrix,
    balance,
    curate,
    probe,
    rer,
    select_agents,
    split,
    train_dataset,
)
from director.shared.tasks import Dataset, Task


def _typed_candidates(n_per_type=6, domain="d"):
    tasks = []
    for i in range(n_per_type):
        for t in ("A", "B"):
            tasks.append(
                Task(
                    task_id=f"{t}{i}",
                    prompt=f"TYPE_{t} question {i}",
                    solution=None,
                    grader="fake_right",
                    metadata={"domain": domain, "source": "x"},
                )
            )
    return Dataset(tasks)


def test_classify():
    wids = ["a", "b", "c"]
    assert classify([1, 0, 0], wids) == ("discriminative", ["a"])
    assert classify([1, 1, 1], wids) == ("saturated", ["a", "b", "c"])
    assert classify([0, 0, 0], wids) == ("dead", [])
    assert classify([1, 1, 0], wids)[0] == "discriminative"


async def test_probe_and_resume(pool, tmp_path):
    md = str(tmp_path / "m")
    recs = await probe(pool, _typed_candidates(4), md, max_in_flight=8)
    assert len(recs) == 8
    # every typed item has exactly one winner (a on TYPE_A, b on TYPE_B)
    assert all(r.verdict == "discriminative" for r in recs)
    wins = Counter(r.winners[0] for r in recs)
    assert wins == {"a": 4, "b": 4}
    # resume: re-probing the same pool does no new work
    assert len(probed_ids(md)) == 8
    recs2 = await probe(pool, _typed_candidates(4), md, max_in_flight=8)
    assert len(recs2) == 8 and len(read_probes(md)) == 8


def _disc(winner, domain="d", i=0):
    return ProbeRecord(
        task_id=f"{winner}{i}", domain=domain, source="x", prompt="p", solution=None,
        grader="g", system=None, rewards=[1.0], winners=[winner], verdict="discriminative",
    )


def test_balance_flattens_sole_winners():
    # 20 'a'-winners, 4 'b'-winners; target 8 should not be all 'a'
    items = [_disc("a", i=i) for i in range(20)] + [_disc("b", i=100 + i) for i in range(4)]
    chosen = balance(items, per_domain_target=8, seed=0)
    counts = Counter(c.winners[0] for c in chosen)
    assert len(chosen) == 8
    assert counts["b"] == 4  # all scarce winners included
    assert counts["a"] == 4  # round-robin caps the dominant winner


def test_split_stratified():
    items = [_disc("a", domain="d1", i=i) for i in range(10)] + [
        _disc("b", domain="d2", i=i) for i in range(10)
    ]
    out = split(items, train_ratio=0.8, seed=0)
    by = Counter((it.domain, it.split) for it in out)
    assert by[("d1", "train")] == 8 and by[("d1", "test")] == 2
    assert by[("d2", "train")] == 8 and by[("d2", "test")] == 2


def _probe(rewards, dataset, i):
    return ProbeRecord(
        task_id=f"{dataset}{i}", domain=dataset, source="x", prompt="p", solution=None,
        grader="g", system=None, rewards=rewards, winners=[], verdict="discriminative",
        dataset=dataset,
    )


def test_rer_error_normalized_headroom():
    import numpy as np
    # two datasets, two complementary agents: each solves one -> oracle 1.0, best single 0.5
    E = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert abs(rer(E) - 1.0) < 1e-9                 # (1.0 - 0.5)/(1 - 0.5)
    assert rer(np.array([[1.0, 1.0], [1.0, 1.0]])) == 0.0  # saturated -> no headroom


def test_select_agents_prunes_dominated():
    # a wins all 'code', b wins all 'math'; c never wins -> dominated, must be dropped
    probes = (
        [_probe([1.0, 0.0, 0.0], "code", i) for i in range(5)]
        + [_probe([0.0, 1.0, 0.0], "math", i) for i in range(5)]
    )
    datasets, E = accuracy_matrix(probes, ["a", "b", "c"])
    assert datasets == ["code", "math"]
    agents, r = select_agents(probes, ["a", "b", "c"])
    assert set(agents) == {"a", "b"}   # complementary pair; c pruned (adds no RER)
    assert abs(r - 1.0) < 1e-9


async def test_curate_end_to_end(pool, tmp_path):
    md = str(tmp_path / "m")
    await probe(pool, _typed_candidates(6), md, max_in_flight=8)
    items = curate(md, per_domain_target=100, worker_ids=["a", "b"], sources=["x"], seed=0)
    assert items and read_manifest(md)
    # both winners represented; train split materializes as a Dataset
    wins = Counter(it.winners[0] for it in items)
    assert wins["a"] > 0 and wins["b"] > 0
    assert len(train_dataset(items)) > 0
