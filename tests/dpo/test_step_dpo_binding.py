"""The native DPO step + metrics must be exposed on the trainer extension.

A full forward/backward needs a model + GPU (covered by the end-to-end smoke
test in Task 7); here we assert the binding surface exists after `make build`.
"""

import pytest

pytest.importorskip("surogate")
from surogate import _surogate  # noqa: E402


def test_step_dpo_native_is_bound():
    assert hasattr(_surogate.SurogateTrainer, "step_dpo_native")


def test_get_dpo_native_metrics_is_bound():
    assert hasattr(_surogate.SurogateTrainer, "get_dpo_native_metrics")
