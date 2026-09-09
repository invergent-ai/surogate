"""A shard with exactly one distributed batch must be usable across epochs."""

import numpy as np
import pytest

from tests.distill.test_dataloader_kd import write_token_shard


@pytest.mark.parametrize("world_size", [1, 2])
def test_last_complete_batch_is_loadable(tmp_path, world_size):
    from surogate import _surogate as ext

    length = 16
    path = tmp_path / "tiny.bin"
    tokens = np.arange(world_size * length + 1, dtype=np.int32)
    write_token_shard(path, tokens, 128)
    loaders = [ext.DataLoader([str(path)], length, rank=r, world_size=world_size) for r in range(world_size)]
    for epoch in range(2):
        starts = set()
        for loader in loaders:
            if epoch:
                loader.advance_epoch()
            assert loader.has_next()
            assert not loader.has_next(2)
            inputs = np.empty((1, length), dtype=np.int32)
            targets = np.empty_like(inputs)
            loader.load_batch(inputs, targets)
            np.testing.assert_array_equal(targets, inputs + 1)
            starts.add(int(inputs[0, 0]))
            assert not loader.has_next()
        assert starts == set(range(0, world_size * length, length))
