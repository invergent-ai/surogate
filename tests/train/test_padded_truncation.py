"""#277: a truncated example in a padded shard must not train on the next window's first token.

The loss mask is input-aligned (``mask[t]`` trains the prediction of ``tokens[t + 1]``), so after
a cut its last bit trains the token that was cut off, and the loader's target there is the shard's
next token: the first of the next example's window. The writers clear that bit; the loader also
ignores it, for shards written before they did.
"""

import numpy as np
import pytest

from surogate.train.tokenize import TokenizeDatasets, TokenizedDataFileWriter, write_padded

T = 16
PAD = 0


def read_shard(path):
    """(tokens, position ids, mask bits, non_overlapping) of a version-3 shard with masks."""
    raw = np.fromfile(path, dtype=np.uint8)
    header = raw[:1024].view(np.int32)
    n, non_overlapping = int(header[4]), bool(header[7])
    body = raw[1024:]
    tokens = body[: 4 * n].view(np.int32)
    positions = body[4 * n : 8 * n].view(np.int32)
    mask = np.unpackbits(body[8 * n :], bitorder="little")[:n]
    return tokens, positions, mask, non_overlapping


def examples():
    """A short example, one cut by the window (supervised up to its end), one exactly a window."""
    def input_mask(n, prompt):
        m = np.zeros(n, dtype=np.int32)
        m[prompt : n - 1] = 1  # input-aligned: the last token predicts nothing
        return m

    tokens = [np.arange(3, 13, dtype=np.int32), np.arange(100, 120, dtype=np.int32),
              np.arange(200, 216, dtype=np.int32)]
    masks = [input_mask(10, 2), input_mask(20, 3), input_mask(16, 4)]
    return tokens, masks


def check_windows(path, tokens, masks):
    x, p, m, non_overlapping = read_shard(path)
    assert non_overlapping
    assert x.size == 3 * T
    np.testing.assert_array_equal(p, np.tile(np.arange(T), 3))
    short, cut, exact = (slice(k * T, (k + 1) * T) for k in range(3))
    np.testing.assert_array_equal(x[short][:10], tokens[0])
    np.testing.assert_array_equal(m[short], np.r_[masks[0], np.zeros(6, np.int32)])
    np.testing.assert_array_equal(x[cut], tokens[1][:T])
    # The cut example keeps every target it has inside the window, and not the one that was cut off.
    np.testing.assert_array_equal(m[cut][: T - 1], masks[1][: T - 1])
    assert masks[1][T - 1] == 1 and m[cut][T - 1] == 0
    np.testing.assert_array_equal(m[exact], masks[2])


def test_padded_writer_does_not_train_a_cut_example_on_the_next_window(tmp_path):
    tokens, masks = examples()
    before = [m.copy() for m in masks]
    TokenizeDatasets._write_padded_vectorized(
        None, tokens, masks, [t.size for t in tokens], str(tmp_path), "train", 512, T, PAD,
        max_tokens_per_file=1 << 20, non_overlapping=True,
    )
    check_windows(tmp_path / "train-000.bin", tokens, masks)
    for m, b in zip(masks, before):
        np.testing.assert_array_equal(m, b)  # the caller's masks are left as they were


def test_write_padded_does_not_train_a_cut_example_on_the_next_window(tmp_path):
    tokens, masks = examples()
    path = tmp_path / "eval-000.bin"
    with TokenizedDataFileWriter(str(path), 512, masking=True, non_overlapping=True) as writer:
        write_padded(writer, ({"tokens": t, "mask": m} for t, m in zip(tokens, masks)), T, PAD)
    check_windows(path, tokens, masks)


def write_stale_shard(path, n_windows, non_overlapping):
    """Every bit set, the last of each window included: what a pre-fix padded writer left after a cut."""
    tokens = np.arange(1, n_windows * T + (0 if non_overlapping else 1) + 1, dtype=np.int32)
    with TokenizedDataFileWriter(str(path), 4096, masking=True, non_overlapping=non_overlapping) as writer:
        writer.add_document(tokens=tokens, position_ids=np.arange(tokens.size), mask=np.ones(tokens.size, np.int32))
    return tokens


def load_all(path):
    from surogate import _surogate

    loader = _surogate.DataLoader([str(path)], T, seed=3)
    rows = []
    while loader.has_next(1):
        x = np.empty((1, T), np.int32)
        y = np.empty((1, T), np.int32)
        loader.load_batch(x, y)
        rows.append((x[0], y[0]))
    return rows


def test_loader_never_trains_the_last_position_of_a_padded_window(tmp_path):
    pytest.importorskip("surogate._surogate", reason="needs the built extension")
    path = tmp_path / "train-000.bin"
    write_stale_shard(path, 3, non_overlapping=True)
    rows = load_all(path)
    assert len(rows) == 3
    for x, y in rows:
        np.testing.assert_array_equal(y[: T - 1], x[1:])
        assert y[T - 1] == -100


def test_loader_keeps_the_last_target_of_a_packed_window(tmp_path):
    """A packed (overlapping) shard is one stream: a window's last target is its real next token."""
    pytest.importorskip("surogate._surogate", reason="needs the built extension")
    path = tmp_path / "train-000.bin"
    tokens = write_stale_shard(path, 2, non_overlapping=False)
    rows = load_all(path)
    assert len(rows) == 2
    for x, y in rows:
        np.testing.assert_array_equal(y[: T - 1], x[1:])
        assert y[T - 1] == tokens[int(np.flatnonzero(tokens == x[0])[0]) + T]
