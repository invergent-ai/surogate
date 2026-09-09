"""Markers, and the two switches that decide what a bare `pytest tests/` runs.

Most of this suite is fast and needs nothing but a checkout. A minority builds real
trainers from real checkpoints on real cards, and takes minutes per test rather than
milliseconds -- the onboarding gates, the dispatch-PP stages, the distillation capture. Left
ungated those decide the runtime of every full run, which is how a suite stops being run.

So `slow` is opt-in: it is skipped unless `--slow` is passed. `gpu` stays opt-out through
`--no-gpu`, because that switch answers a different question -- not "is this worth the
minutes" but "does this machine have a card at all".
"""

import pytest


def pytest_addoption(parser):
    parser.addoption("--no-gpu", action="store_true", default=False,
                     help="Skip tests that require GPU hardware")
    parser.addoption("--slow", action="store_true", default=False,
                     help="Also run tests marked slow (real checkpoints, minutes each)")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: tests that require GPU hardware")
    config.addinivalue_line("markers", "multinode: tests that require multiple nodes")
    config.addinivalue_line("markers", "slow: tests that are slow to run")
    config.addinivalue_line(
        "markers", "network: test needs internet access (skipped/failing offline is expected)"
    )


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="--no-gpu option passed")
    skip_slow = pytest.mark.skip(reason="slow: real checkpoints and minutes per test; pass --slow")
    no_gpu = config.getoption("--no-gpu", default=False)
    slow = config.getoption("--slow", default=False)
    for item in items:
        if no_gpu and "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        if not slow and "slow" in item.keywords:
            item.add_marker(skip_slow)
