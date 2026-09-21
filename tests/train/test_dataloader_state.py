"""Checkpoint resume: the loader state that checkpoint.json carries must round-trip exactly.

`save_checkpoint` records (seed, epoch, file_index, chunk_index) and `load_checkpoint` feeds
them to `DataLoader.set_state`; a resumed run must then produce the same batches the original
run would have produced next, from any position: inside a file, at the lazy end-of-file index,
past a file boundary and in a later epoch. A bad entry must fail without touching the loader.
"""

import numpy as np
import pytest

from tests.distill.test_dataloader_kd import write_token_shard

LENGTH = 16
SEED = 7


def _state(loader):
    return (loader.seed, loader.epoch(), loader.file_index(), loader.chunk_index())


def _take(loader):
    inputs = np.empty((1, LENGTH), dtype=np.int32)
    targets = np.empty_like(inputs)
    loader.load_batch(inputs, targets)
    return inputs, targets


def _trainer_step_load(loader):
    """The training loop's consumption pattern: advance the epoch when the file list is spent."""
    if not loader.has_next():
        loader.advance_epoch()
    return _take(loader)


@pytest.fixture
def shards(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    write_token_shard(first, np.arange(6 * LENGTH + 1, dtype=np.int32), 4096)
    write_token_shard(second, np.arange(1000, 1000 + 5 * LENGTH + 1, dtype=np.int32), 4096)
    return [str(first), str(second)]


def _fresh(shards):
    from surogate import _surogate as ext

    return ext.DataLoader(shards, LENGTH, seed=SEED)


def _assert_same_future(original, restored, loads):
    for _ in range(loads):
        assert restored.has_next() == original.has_next()
        assert restored.has_next(2) == original.has_next(2)
        a_in, a_tg = _trainer_step_load(original)
        b_in, b_tg = _trainer_step_load(restored)
        np.testing.assert_array_equal(b_in, a_in)
        np.testing.assert_array_equal(b_tg, a_tg)
        assert _state(restored) == _state(original)


def test_state_round_trips_mid_file(shards):
    original = _fresh(shards)
    for _ in range(2):
        _trainer_step_load(original)
    saved = _state(original)
    restored = _fresh(shards)
    restored.set_state(*saved)
    assert _state(restored) == saved
    _assert_same_future(original, restored, 3)


@pytest.mark.parametrize("loads_before_save", [6, 7, 11, 12, 14])
def test_state_round_trips_across_file_and_epoch_boundaries(shards, loads_before_save):
    # 6: lazy end-of-file index of the first file; 7: inside the second file;
    # 11: every sequence of epoch 0 consumed; 12 and 14: inside epoch 1.
    original = _fresh(shards)
    for _ in range(loads_before_save):
        _trainer_step_load(original)
    saved = _state(original)
    assert saved[1] == (1 if loads_before_save >= 12 else 0)
    restored = _fresh(shards)
    for _ in range(3):  # a restore does not depend on the restoring loader's own history
        _trainer_step_load(restored)
    restored.set_state(*saved)
    assert _state(restored) == saved
    _assert_same_future(original, restored, 4)


def test_state_is_a_function_of_consumption(shards):
    first, second = _fresh(shards), _fresh(shards)
    for _ in range(9):
        _trainer_step_load(first)
        _trainer_step_load(second)
    assert _state(first) == _state(second)


def test_bad_state_raises_without_mutating(shards):
    original = _fresh(shards)
    _trainer_step_load(original)
    before = _state(original)
    with pytest.raises(RuntimeError):
        original.set_state(SEED, 0, 2, 0)  # only two files
    with pytest.raises(RuntimeError):
        original.set_state(SEED, 0, 0, -1)
    assert _state(original) == before
    witness = _fresh(shards)
    witness.set_state(*before)
    _assert_same_future(original, witness, 2)
