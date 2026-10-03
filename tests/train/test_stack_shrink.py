"""The one-shot post-warmup stack shrink runs only when the first step's peak bounds the run."""

from types import SimpleNamespace

import pytest

pytest.importorskip("surogate._surogate")

from surogate.train.trainer import SurogateTrainerWrapper  # noqa: E402

MIB = 1024 * 1024


def _wrapper(*, sequence_chunks=1, row_packer=None, dispatch_pp=False):
    w = SurogateTrainerWrapper.__new__(SurogateTrainerWrapper)
    w.config = SimpleNamespace(sequence_chunks=sequence_chunks)
    w._stack_shrunk = False
    w._row_packer = row_packer
    w._dispatch_pp = dispatch_pp
    calls = []

    def shrink_stack_after_warmup():
        calls.append(1)
        return [(1142 * MIB, 4126 * MIB)]  # (new, old) per GPU

    w.trainer = SimpleNamespace(shrink_stack_after_warmup=shrink_stack_after_warmup)
    return w, calls


@pytest.mark.parametrize("sequence_chunks", [1, None])
def test_dense_training_shrinks_once(sequence_chunks):
    w, calls = _wrapper(sequence_chunks=sequence_chunks)
    w._maybe_shrink_stack_after_warmup()
    w._maybe_shrink_stack_after_warmup()
    assert calls == [1] and w._stack_shrunk


@pytest.mark.parametrize("sequence_chunks", [2, 128])
def test_chunked_training_keeps_the_upfront_stack(sequence_chunks):
    # A chunk's attention backward allocates dK/dV scratch for its whole KV prefix, so a later,
    # longer row needs more stack than the first step measured (#265).
    w, calls = _wrapper(sequence_chunks=sequence_chunks)
    w._maybe_shrink_stack_after_warmup()
    w._maybe_shrink_stack_after_warmup()
    assert calls == [] and w._stack_shrunk


@pytest.mark.parametrize("kwargs", [{"row_packer": object()}, {"dispatch_pp": True}])
def test_row_packing_and_dispatch_pp_keep_the_upfront_stack(kwargs):
    w, calls = _wrapper(**kwargs)
    w._maybe_shrink_stack_after_warmup()
    assert calls == []


def test_config_without_sequence_chunks_is_dense():
    w, calls = _wrapper()
    w.config = SimpleNamespace()
    w._maybe_shrink_stack_after_warmup()
    assert calls == [1]
