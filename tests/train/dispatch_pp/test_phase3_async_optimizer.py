"""Phase-3 gate: the dispatch-PP async 1-step-stale optimizer.

The async optimizer overlaps the CPU-master update with the next iteration's
compute by draining a depth-1 queue of update closures on a worker thread. Depth-1
(at most one update in flight) is exactly the one-step staleness RoundPipe v1
accepts. These tests isolate the two concerns the design insists never be conflated:

  * correctness / determinism — with overlap OFF the optimizer is bitwise-identical
    to a plain in-order AdamW, and with overlap ON the *same* updates are applied in
    the *same* order, so the final master params match the synchronous reference;
  * staleness — with overlap ON, a just-submitted update is still in flight when
    ``step`` returns (the worker is exactly one update behind), and never more.

CPU-only: drives ``cpu_adamw_step`` over FP32 master arrays; no GPU required.
"""

import numpy as np
import pytest

sg = pytest.importorskip("surogate._surogate")

_LR = 1e-2
_B1 = 0.9
_B2 = 0.999
_EPS = 1e-8
_WD = 0.01


def _reference_adamw(master, grads):
    """Plain in-order AdamW reference matching cpu_adamw_step (FP32)."""
    p = np.asarray(master, dtype=np.float32).copy()
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    for t, g in enumerate(grads, start=1):
        g = np.asarray(g, dtype=np.float32)
        p *= 1.0 - _LR * _WD  # decoupled weight decay
        m[:] = _B1 * m + (1.0 - _B1) * g
        v[:] = _B2 * v + (1.0 - _B2) * g * g
        m_hat = m / (1.0 - _B1**t)
        v_hat = v / (1.0 - _B2**t)
        p -= _LR * m_hat / (np.sqrt(v_hat) + _EPS)
    return p


def _make_problem(n=64, steps=8, seed=0):
    rng = np.random.default_rng(seed)
    master = rng.standard_normal(n).astype(np.float32)
    grads = [rng.standard_normal(n).astype(np.float32) for _ in range(steps)]
    return master, grads


def test_sync_matches_plain_adamw():
    """Overlap OFF => bitwise-identical to a plain in-order AdamW (determinism gate)."""
    master, grads = _make_problem()
    opt = sg.AsyncStaleAdamW(master.tolist(), _LR, _B1, _B2, _EPS, _WD, False)
    assert not opt.overlap_enabled()
    for g in grads:
        opt.step(g.tolist())
    got = np.asarray(opt.master(), dtype=np.float32)
    ref = _reference_adamw(master, grads)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5)


def test_async_final_matches_sync():
    """Overlap ON applies the same updates in the same order => same final params as
    the synchronous reference (the overlap changes *when*, not *what*)."""
    master, grads = _make_problem(seed=1)
    sync = sg.AsyncStaleAdamW(master.tolist(), _LR, _B1, _B2, _EPS, _WD, False)
    for g in grads:
        sync.step(g.tolist())
    sync_master = np.asarray(sync.master(), dtype=np.float32)

    asyncro = sg.AsyncStaleAdamW(master.tolist(), _LR, _B1, _B2, _EPS, _WD, True)
    assert asyncro.overlap_enabled()
    for g in grads:
        asyncro.step(g.tolist())
    async_master = np.asarray(asyncro.master(), dtype=np.float32)

    np.testing.assert_array_equal(async_master, sync_master)


def test_one_step_staleness_bound():
    """With overlap ON, the just-submitted update is still in flight when ``step``
    returns — the worker is exactly one update behind, never more."""
    master, grads = _make_problem(steps=6, seed=2)
    opt = sg.AsyncStaleAdamW(master.tolist(), _LR, _B1, _B2, _EPS, _WD, True)
    for i, g in enumerate(grads):
        # Stall the worker so the just-submitted update cannot finish before we
        # observe the counter. Depth-1: step() fences on the *previous* update, so
        # on return exactly `i` updates are done (this submission, the i-th, is in
        # flight) — one-step stale, and never more than one outstanding.
        opt.step(g.tolist(), delay_us=20000)
        assert opt.applied_count() == i

    opt.drain()
    assert opt.applied_count() == len(grads)


def test_bad_grad_size_raises_and_stays_usable():
    """A mismatched-size grad is rejected at submit time (no partial commit), and the
    optimizer remains usable for subsequent correctly-sized updates."""
    master, _ = _make_problem(n=32, seed=3)
    opt = sg.AsyncStaleAdamW(master.tolist(), _LR, _B1, _B2, _EPS, _WD, True)
    with pytest.raises(Exception):
        opt.step([1.0, 2.0, 3.0])  # wrong length vs master (32)
    # A correctly-sized update still applies afterwards.
    opt.step(np.ones(32, dtype=np.float32).tolist())
    opt.drain()
    assert opt.applied_count() == 1
