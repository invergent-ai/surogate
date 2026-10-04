"""Bin packing (``sample_packing: bin``): CPU only, no GPU, no compiled extension.

Every document lands whole in exactly one window, starting at position 0 (the engine's
document boundary), with its loss mask at the same offsets and nothing supervised across a
document or window boundary.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from surogate.core.config.train_dataset_config import _parse_sample_packing
from surogate.train.bin_packing import MAX_DOCS_PER_WINDOW, materialize_window, next_k_fit, padding_share


def doc_boundaries(pos_row: np.ndarray) -> list[tuple[int, int]]:
    """Mirror of CausalLMExecutionProfile::compute_doc_masking for one row: [start, end) spans."""
    spans, start = [], 0
    for t in range(1, len(pos_row)):
        if pos_row[t] - pos_row[t - 1] != 1:
            spans.append((start, t))
            start = t
    spans.append((start, len(pos_row)))
    return spans


def chat_lengths(n, mean, T, seed=0):
    rng = random.Random(seed)
    return [min(T, max(16, int(rng.lognormvariate(math.log(mean) - 0.5, 1.0)))) for _ in range(n)]


@pytest.mark.parametrize("k", [1, 4, 64])
def test_every_document_lands_once_within_capacity(k):
    T = 512
    lengths = chat_lengths(500, 120, T, seed=k) + [0, 0, T, T + 100]
    windows = next_k_fit(lengths, T, k)
    placed = sorted(i for w in windows for i in w)
    assert placed == [i for i, n in enumerate(lengths) if n > 0]
    for w in windows:
        assert w
        assert sum(min(lengths[i], T) for i in w) <= T
        assert len(w) <= MAX_DOCS_PER_WINDOW


def test_k1_is_next_fit():
    # Next-Fit closes the open window whenever the next document does not fit.
    assert next_k_fit([6, 3, 2, 5, 4], 10, 1) == [[0, 1], [2, 3], [4]]


def test_oldest_window_with_room_wins():
    # Windows [6] and [7] are open: 3 fits both and goes to the older one; then 2 fits only [7].
    assert next_k_fit([6, 7, 3, 2], 10, 2) == [[0, 2], [1, 3]]


def test_fullest_window_is_written_out_when_k_are_open():
    # 9 fits neither [6] nor [7]; with k=2 the fuller [7] is written out first.
    assert next_k_fit([6, 7, 9], 10, 2) == [[1], [0], [2]]


def test_max_docs_bounds_documents_per_window():
    windows = next_k_fit([1] * 10, 100, 4, max_docs=3)
    assert [len(w) for w in windows] == [3, 3, 3, 1]


def test_more_open_windows_pad_less():
    T = 4096
    lengths = chat_lengths(4000, 1100, T)
    shares = [padding_share(lengths, next_k_fit(lengths, T, k), T) for k in (1, 8, 64)]
    assert shares[0] > shares[1] > shares[2]
    assert shares[2] < 0.02


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        next_k_fit([1], 0, 1)
    with pytest.raises(ValueError):
        next_k_fit([1], 10, 0)


def test_window_keeps_documents_whole_and_isolated():
    T, pad = 64, 0
    rng = np.random.default_rng(0)
    lengths = [10, 20, 5, 70]
    tokens = [rng.integers(3, 1000, n).astype(np.int32) for n in lengths]
    # Input-aligned masks as tokenize builds them: last bit 0, the rest supervised here.
    masks = []
    for n in lengths:
        m = np.ones(n, dtype=np.int32)
        m[-1] = 0
        masks.append(m)
    masks[3] = np.ones(70, dtype=np.int32)  # truncated to 64: its bit at 63 would point past the cut

    windows = next_k_fit(lengths, T, 4)
    assert sorted(map(sorted, windows)) == [[0, 1, 2], [3]]
    for docs in windows:
        x, p, y = materialize_window(tokens, masks, docs, T, pad)
        assert x.shape == p.shape == y.shape == (T,)
        assert p.max() < T
        spans = doc_boundaries(p)
        o = 0
        for span, i in zip(spans, docs):
            n = min(lengths[i], T)
            assert span[0] == o
            np.testing.assert_array_equal(x[o : o + n], tokens[i][:n])
            np.testing.assert_array_equal(p[o : o + n], np.arange(n))
            assert y[o + n - 1] == 0  # nothing predicts across a document or window boundary
            assert y[o : o + n - 1].all()
            o += n
        # Padding joins the last document (no extra boundary) and carries no target.
        assert len(spans) == len(docs)
        assert (x[o:] == pad).all() and not y[o:].any()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("false", False), ("True", True), ("bin", "bin"), (" BIN ", "bin"), (None, None)],
)
def test_sample_packing_values(value, expected):
    assert _parse_sample_packing(value) == expected


def test_sample_packing_rejects_unknown_mode():
    with pytest.raises(ValueError):
        _parse_sample_packing("ffd")
