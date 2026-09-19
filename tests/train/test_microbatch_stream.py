"""All eager objectives consume the same batches, including epoch boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest

from surogate.train.trainer import SurogateTrainerWrapper


class Loader:
    def __init__(self):
        self.position = 0
        self.current_epoch = 0
        self.reads = []

    def has_next(self):
        return self.position < 3

    def advance_epoch(self):
        assert self.position == 3, "An epoch must not be abandoned with unread batches"
        self.position = 0
        self.current_epoch += 1

    def epoch(self):
        return self.current_epoch

    def load_batch(self, inputs, targets, positions, *sidecars):
        assert inputs.shape == (2, 4), "Native loader chunks must not be split into smaller rows"
        value = self.current_epoch * 100 + self.position
        inputs.fill(value)
        targets.fill(value + 1)
        positions.fill(0)
        for sidecar in sidecars:
            sidecar.fill(value)
        self.reads.append(value)
        self.position += 1


@pytest.mark.parametrize("candidate", [False, True])
def test_batch_audit_preserves_native_geometry_and_stream(candidate, monkeypatch, tmp_path):
    monkeypatch.setenv("SUROGATE_AUDIT_TRAIN_BATCHES", "1")
    wrapper = SurogateTrainerWrapper.__new__(SurogateTrainerWrapper)
    wrapper.config = SimpleNamespace(output_dir=str(tmp_path))
    wrapper.train_loader = Loader()
    arrays = [np.empty((2, 4), np.int32) for _ in range(3)]
    sidecars = [np.empty((2, 4, 3), np.int32), np.empty((2, 4, 3), np.float32)] if candidate else []
    consumed = []
    for _ in range(8):
        if not wrapper.train_loader.has_next():
            wrapper.train_loader.advance_epoch()
        wrapper._load_training_microbatch(*arrays, *sidecars)
        consumed.append(int(arrays[0][0, 0]))
    assert consumed == [0, 1, 2, 100, 101, 102, 200, 201]
    assert wrapper.train_loader.reads == consumed
    assert len((tmp_path / "batch-audit.jsonl").read_text().splitlines()) == 8
