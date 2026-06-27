"""Source registry invariants: held-out benchmarks stay out of the training pool."""

from __future__ import annotations

from director.shared.sources import DOMAINS, EVAL_ONLY, SOURCES, train_sources


def test_gpqa_is_eval_only():
    # GPQA-Diamond is a reported benchmark: available as a loader, never trained on.
    assert "gpqa" in SOURCES
    assert "gpqa" in EVAL_ONLY
    assert "gpqa" not in train_sources()


def test_science_training_supply_without_gpqa():
    # science domain must still have training supply once gpqa is held out
    science_train = [s for s in DOMAINS["science"] if s in train_sources()]
    assert set(science_train) >= {"supergpqa_sci", "mmlu_sci"}


def test_train_sources_cover_all_domains():
    covered = {SOURCES[s][1] for s in train_sources()}
    assert {"math", "code", "science", "general", "reasoning"} <= covered


import pytest  # noqa: E402


@pytest.mark.network
def test_all_loaders_respect_limit():
    # regression: load_mmlu_pro once ignored limit (missing n += 1) and flooded the pool
    for name, (fn, _dom) in SOURCES.items():
        assert len(list(fn(limit=3))) <= 3, f"{name} ignores limit"
