"""The batch receiver must resume from the TRAINER's step, not the orchestrator's.

`multi_run_manager.progress[idx].step` is loaded from the ORCHESTRATOR's
checkpoints and means "the next batch to collect". The orchestrator runs ahead
of the trainer by up to `max_async_level`, so a restarted trainer that seeded
its receiver from that value skipped every batch in between — permanently
dropped, never trained on.
"""

from unittest.mock import patch

from surogate.grpo.transport import setup_training_batch_receiver
from surogate.grpo.transport.filesystem import FileSystemTrainingBatchReceiver


class _Progress:
    def __init__(self, step):
        self.step = step


class _MRM:
    def __init__(self, step):
        self.progress = {0: _Progress(step)}


def test_start_step_overrides_orchestrator_progress():
    with patch("surogate.grpo.transport.filesystem.get_multi_run_manager", return_value=_MRM(9)):
        r = FileSystemTrainingBatchReceiver(start_step=6)
    assert r._get_received_step(0) == 6, "trainer resume point must win over orch progress"


def test_without_start_step_falls_back_to_progress():
    with patch("surogate.grpo.transport.filesystem.get_multi_run_manager", return_value=_MRM(9)):
        r = FileSystemTrainingBatchReceiver()
    assert r._get_received_step(0) == 9


def test_factory_forwards_start_step():
    class _Cfg:
        type = "filesystem"

    with patch("surogate.grpo.transport.filesystem.get_multi_run_manager", return_value=_MRM(9)):
        r = setup_training_batch_receiver(_Cfg(), start_step=4)
    assert r._get_received_step(0) == 4
