"""Contract tests for reorder_micros_for_vtc_guard (upstream issue #74).

In chunked GRPO the engine's step-end ValidTokenCount is the LAST micro's
CHUNK-0 valid count; the guard pins the max-count micro last so (a) an
all-masked-chunk-0 sample can never corrupt the step and (b) the grad
token-scale denominator is stable across steps.
"""

from __future__ import annotations

import numpy as np

from surogate.grpo.trainer import reorder_micros_for_vtc_guard

CHUNK = 1536


def _c0(mb) -> int:
    return int(np.asarray(mb["loss_mask"]).reshape(-1)[:CHUNK].sum())


def _mb(prompt: int, completion: int) -> dict:
    # (1, T) — the packer's real shape; the guard must flatten before slicing
    return {"loss_mask": np.array([[0] * prompt + [1] * completion], dtype=np.int8)}


def test_all_masked_chunk0_last_micro_is_replaced():
    mbs = [_mb(100, 200), _mb(50, 150), _mb(CHUNK, 80)]  # last: prompt fills chunk 0
    idx = reorder_micros_for_vtc_guard(mbs, CHUNK)
    assert idx == 0
    assert _c0(mbs[-1]) == 200
    assert _c0(mbs[idx]) == 0  # swapped into interior


def test_max_count_micro_is_pinned_last_even_when_last_is_nonzero():
    mbs = [_mb(0, 300), _mb(100, 120), _mb(200, 30)]  # last has 30 valid — small denominator
    idx = reorder_micros_for_vtc_guard(mbs, CHUNK)
    assert idx == 0
    assert _c0(mbs[-1]) == 300


def test_noop_when_max_already_last_and_on_empty_inputs():
    mbs = [_mb(200, 30), _mb(0, 300)]
    assert reorder_micros_for_vtc_guard(mbs, CHUNK) is None
    assert _c0(mbs[-1]) == 300
    assert reorder_micros_for_vtc_guard([], CHUNK) is None
    assert reorder_micros_for_vtc_guard(mbs, 0) is None


def test_2d_mask_never_selects_empty_chunk0_by_total_count():
    """Regression: with (1, T) masks, a row-slice sums the WHOLE sample and a
    long-completion multi-chunk micro (huge total, empty chunk 0) wins the
    max — placing it last manufactures vtc=0. Observed live before the fix."""
    mbs = [_mb(50, 900), _mb(CHUNK, CHUNK), _mb(100, 200)]  # chunk0: 900, 0, 200
    idx = reorder_micros_for_vtc_guard(mbs, CHUNK)
    assert idx == 0
    assert int(np.asarray(mbs[-1]["loss_mask"]).reshape(-1)[:CHUNK].sum()) == 900
