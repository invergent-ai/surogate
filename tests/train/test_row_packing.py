"""Row packing layout: CPU only, no GPU, no downloads.

The engine side (document masking, batch-invariant kernels) is covered by
test_row_packing_gpu.py. Here: every row lands in exactly one window, intact, with its
position ids restarting at 0 (the engine's document boundary), its targets and its teacher
sidecar rows at the same offsets, and nothing supervised outside the rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from surogate.train.row_packing import RowPacker, effective_length, plan_windows


def doc_boundaries(pos_row: np.ndarray) -> list[tuple[int, int]]:
    """Mirror of CausalLMExecutionProfile::compute_doc_masking for one row: [start, end) spans."""
    spans, start = [], 0
    for t in range(1, len(pos_row)):
        if pos_row[t] - pos_row[t - 1] != 1:
            spans.append((start, t))
            start = t
    spans.append((start, len(pos_row)))
    return spans


def padded_rows(lengths, T, K=None, seed=0, pad=0):
    """What a padded shard hands the trainer: one row per window, answer supervised at L-1."""
    rng = np.random.default_rng(seed)
    n = len(lengths)
    x = np.full((n, T), pad, np.int32)
    y = np.full((n, T), -100, np.int32)
    p = np.tile(np.arange(T, dtype=np.int32), (n, 1))
    ids = lps = None
    if K:
        ids = np.zeros((n, T, K), np.int32)
        lps = np.zeros((n, T, K), np.float32)
    for r, L in enumerate(lengths):
        x[r, : L + 1] = rng.integers(3, 1000, L + 1)  # the answer token follows the last input
        y[r, L - 1] = x[r, L]
        if K:
            ids[r, L - 1, :3] = [x[r, L], 5, 6]
            ids[r, L - 1, 3:] = -1
            lps[r, L - 1, :3] = [-0.1, -2.0, -3.0]
    return x, y, p, ids, lps


def test_effective_length_ends_after_last_supervised_input():
    assert effective_length(np.array([-100, -100, 7, -100])) == 3
    assert effective_length(np.array([5, -100, -100])) == 1
    assert effective_length(np.full(4, -100)) == 0


@pytest.mark.parametrize("slots", [1, 3, 8])
def test_plan_windows_places_every_row_once_within_capacity(slots):
    rng = np.random.default_rng(slots)
    T = 512
    lengths = [int(v) for v in rng.integers(1, T, 40)] + [0, 0]
    windows = plan_windows(lengths, T, slots)
    assert len(windows) % slots == 0
    placed = sorted(i for w in windows for i in w)
    assert placed == [i for i, n in enumerate(lengths) if n > 0]
    assert all(sum(lengths[i] for i in w) <= T for w in windows)
    # Fewest waves first-fit decreasing reaches (an independent FFD count).
    bins = []
    for n in sorted((n for n in lengths if n > 0), reverse=True):
        for b, used in enumerate(bins):
            if used + n <= T:
                bins[b] += n
                break
        else:
            bins.append(n)
    assert len(windows) // slots == -(-len(bins) // slots)
    assert plan_windows(lengths, T, slots) == windows  # deterministic


def test_plan_windows_caps_rows_per_window():
    windows = plan_windows([3] * 50, 4096, 1, max_rows=16)
    assert max(len(w) for w in windows) <= 16
    assert sorted(i for w in windows for i in w) == list(range(50))


def test_plan_windows_spreads_rows_over_the_waves_slots():
    # Eight short rows, eight slots: one wave, one row per window (least cross-row coupling).
    windows = plan_windows([30] * 8, 4096, 8)
    assert sorted(len(w) for w in windows) == [1] * 8


def test_plan_windows_rejects_rows_longer_than_the_window():
    with pytest.raises(ValueError, match="more than sequence_len"):
        plan_windows([10, 600], 512, 1)


def test_packed_windows_carry_rows_sidecars_and_document_boundaries():
    T, K, slots = 256, 8, 2
    lengths = [40, 17, 100, 3, 60, 90, 25]
    x, y, p, ids, lps = padded_rows(lengths, T, K)
    packer = RowPacker(T, slots, pad_token_id=0, kd_top_k=K)
    packer.add_batch(x[:4], y[:4], p[:4], ids[:4], lps[:4])
    packer.add_batch(x[4:], y[4:], p[4:], ids[4:], lps[4:])
    step = packer.build()
    assert step.rows == len(lengths) and step.packed_rows == len(lengths)
    assert step.waves == len(step.inputs) and all(a.shape == (slots, T) for a in step.inputs)
    assert sum(lengths) == step.real_tokens
    seen = []
    for w in range(step.waves):
        for s in range(slots):
            rows = step.windows[w * slots + s]
            spans = doc_boundaries(step.positions[w][s])
            o = 0
            for k, r in enumerate(rows):
                L = lengths[r]
                assert spans[k] == (o, o + L), "each row must be exactly one document"
                np.testing.assert_array_equal(step.inputs[w][s, o : o + L], x[r, :L])
                np.testing.assert_array_equal(step.positions[w][s, o : o + L], np.arange(L))
                np.testing.assert_array_equal(step.targets[w][s, o : o + L], y[r, :L])
                np.testing.assert_array_equal(step.kd_ids[w][s, o : o + L], ids[r, :L])
                np.testing.assert_array_equal(step.kd_logprobs[w][s, o : o + L], lps[r, :L])
                seen.append(r)
                o += L
            assert (step.targets[w][s, o:] == -100).all() and (step.kd_ids[w][s, o:] == 0).all()
            if o < T and rows:
                assert spans[len(rows)] == (o, T), "the unused tail is its own document"
    assert sorted(seen) == list(range(len(lengths)))
    # The sidecar still sits on the supervised position: targets[i] <-> ids[i].
    for w in range(step.waves):
        sup = step.targets[w] != -100
        assert (step.kd_ids[w][sup][:, 0] == step.targets[w][sup]).all()


def test_rows_without_supervision_are_dropped_and_empty_slots_are_dummies():
    T = 128
    x, y, p, _, _ = padded_rows([10, 20], T)
    y[1] = -100
    packer = RowPacker(T, slots=4)
    packer.add_batch(x, y, p)
    step = packer.build()
    assert step.rows == 2 and step.packed_rows == 1 and step.waves == 1
    dummies = [s for s in range(4) if not step.windows[s]]
    assert len(dummies) == 3
    for s in dummies:
        assert (step.targets[0][s] == -100).all()
        np.testing.assert_array_equal(step.positions[0][s], np.arange(T))


def test_rows_must_start_at_position_zero():
    T = 64
    x, y, p, _, _ = padded_rows([10], T)
    p[0] += 5  # a continuing stream would merge into its predecessor's document
    packer = RowPacker(T, slots=1)
    with pytest.raises(ValueError, match="position 0"):
        packer.add_batch(x, y, p)


def test_sidecar_arrays_required_exactly_when_kd_is_on():
    T = 64
    x, y, p, ids, lps = padded_rows([10], T, K=4)
    with pytest.raises(ValueError, match="sidecar"):
        RowPacker(T, 1, kd_top_k=4).add_batch(x, y, p)
    with pytest.raises(ValueError, match="sidecar"):
        RowPacker(T, 1).add_batch(x, y, p, ids, lps)


def test_each_step_packs_only_its_own_rows():
    """A step packs exactly the rows it was handed (the padded step's rows), nothing carried over."""
    T = 128
    lengths = [12, 30, 7, 50]
    x, y, p, _, _ = padded_rows(lengths, T)
    packer = RowPacker(T, slots=1)

    def docs(step):
        out = []
        for w in range(step.waves):
            spans = doc_boundaries(step.positions[w][0])
            out += [tuple(step.inputs[w][0, a:b]) for (a, b), _ in zip(spans, step.windows[w])]
        return sorted(out)

    packer.add_batch(x[:2], y[:2], p[:2])
    first = packer.build()
    packer.add_batch(x[2:], y[2:], p[2:])
    second = packer.build()
    assert docs(first) == sorted([tuple(x[0, :12]), tuple(x[1, :30])])
    assert docs(second) == sorted([tuple(x[2, :7]), tuple(x[3, :50])])


class _FakeLoader:
    """A padded shard as the trainer sees it: load_batch fills `slots` rows per call, in order, and an
    epoch is exhausted after `rows_per_epoch` rows (has_next / advance_epoch as in the native loader)."""

    def __init__(self, x, y, p, rows_per_epoch, rows_per_call):
        self.x, self.y, self.p = x, y, p
        self.rows_per_epoch = rows_per_epoch
        self.rows_per_call = rows_per_call
        self.cursor = 0
        self.epochs = 0
        self.served = []

    def has_next(self, n=1):
        return self.cursor + n * self.rows_per_call <= self.rows_per_epoch

    def advance_epoch(self):
        self.epochs += 1
        self.cursor = 0

    def load_batch(self, inputs, targets, positions, *sidecars):
        k = inputs.shape[0]
        idx = [(self.cursor + i) % len(self.x) for i in range(k)]
        inputs[:], targets[:], positions[:] = self.x[idx], self.y[idx], self.p[idx]
        self.served += idx
        self.cursor += k


def _wrapper(slots, ga, T, loader):
    from types import SimpleNamespace

    from surogate.train.trainer import SurogateTrainerWrapper

    w = SurogateTrainerWrapper.__new__(SurogateTrainerWrapper)
    w.config = SimpleNamespace(
        gpus=slots,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=ga,
        sequence_len=T,
        output_dir="/nonexistent",
    )
    w.train_loader = loader
    w._row_packer = RowPacker(T, slots)
    w._row_packing_totals = {"steps": 0, "rows": 0, "micro_steps": 0, "real_tokens": 0}
    w.trainer = SimpleNamespace(ga=[], set_grad_accumulation=lambda n: w.trainer.ga.append(n))
    return w


def test_trainer_packs_exactly_the_rows_a_padded_step_loads(monkeypatch):
    pytest.importorskip("surogate._surogate")
    monkeypatch.delenv("SUROGATE_AUDIT_TRAIN_BATCHES", raising=False)
    T, slots, ga = 256, 2, 3
    lengths = [30, 7, 120, 64, 5, 90, 33, 12, 40, 18, 77, 9]
    x, y, p, _, _ = padded_rows(lengths, T)
    loader = _FakeLoader(x, y, p, rows_per_epoch=len(lengths) - 2, rows_per_call=slots)  # epoch ends mid-way
    w = _wrapper(slots, ga, T, loader)
    buf = [np.empty((slots * ga, T), np.int32) for _ in range(3)]
    steps = [w._load_packed_step(*buf) for _ in range(2)]
    # Two steps x GA loader calls x slots rows, in loader order, with the epoch advanced where the
    # padded loop advances it (before a call that would run past the epoch).
    assert len(loader.served) == 2 * ga * slots and loader.epochs == 1
    for k, step in enumerate(steps):
        served = loader.served[k * ga * slots : (k + 1) * ga * slots]
        packed_docs = sorted(
            tuple(step.inputs[wv][s][a:b])
            for wv in range(step.waves)
            for s in range(slots)
            for (a, b), _ in zip(doc_boundaries(step.positions[wv][s]), step.windows[wv * slots + s])
        )
        assert packed_docs == sorted(tuple(x[r, : lengths[r]]) for r in served)
        assert w.trainer.ga[k] == step.waves <= ga
    assert w._row_packing_totals["rows"] == 2 * ga * slots


# ---------------------------------------------------------------------------------------------
# Linear-attention models: row packing is admitted only where every token mixer is document-aware.


def _ir(*ops):
    import json

    return json.dumps({"modules": [{"forward": {"operations": [{"op": op} for op in ops]}}]})


def test_this_build_restarts_linear_attention_at_document_boundaries():
    from surogate.train import row_packing

    assert row_packing.LINEAR_ATTENTION_DOC_BOUNDARIES is True
    assert row_packing.linear_attention_doc_boundaries() is True
    assert row_packing.document_isolation_problem(_ir("matmul", "chunk_gated_delta_rule", "mamba_conv1d")) is None
    assert row_packing.document_isolation_problem(_ir("matmul", "flash_attention")) is None
    assert row_packing.document_isolation_problem(None) is None


def test_row_packing_refuses_linear_attention_on_a_build_without_document_boundaries(monkeypatch):
    from surogate.train import row_packing

    monkeypatch.setattr(row_packing, "linear_attention_doc_boundaries", lambda: False)
    problem = row_packing.document_isolation_problem(_ir("chunk_gated_delta_rule", "mamba_conv1d"))
    assert problem is not None and "LINEAR_ATTENTION_DOC_BOUNDARIES" in problem
    # Attention-only models do not depend on it.
    assert row_packing.document_isolation_problem(_ir("flash_attention")) is None


def test_row_packing_refuses_unverified_token_mixers(monkeypatch):
    from surogate.train import row_packing

    monkeypatch.delenv("SUROGATE_ALLOW_UNVERIFIED_ROW_PACKING", raising=False)
    for op in ("mamba_ssm_scan", "glm_causal_conv1d", "chunk_kimi_delta_rule"):
        problem = row_packing.document_isolation_problem(_ir("matmul", op))
        assert problem is not None and op in problem
    monkeypatch.setenv("SUROGATE_ALLOW_UNVERIFIED_ROW_PACKING", "1")
    assert row_packing.document_isolation_problem(_ir("mamba_ssm_scan")) is None
