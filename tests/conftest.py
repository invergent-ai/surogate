import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: tests that require GPU hardware"
    )
    config.addinivalue_line(
        "markers", "multinode: tests that require multiple nodes"
    )
    config.addinivalue_line(
        "markers", "slow: tests that are slow to run"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if --no-gpu is passed."""
    if config.getoption("--no-gpu", default=False):
        skip_gpu = pytest.mark.skip(reason="--no-gpu option passed")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


def pytest_addoption(parser):
    parser.addoption(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Skip tests that require GPU"
    )
