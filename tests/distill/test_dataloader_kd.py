"""Native DataLoader KD sidecar tests.

Builds a synthetic BIN.TOK token shard plus a matching `.kd` sidecar with a
per-position pattern that makes misalignment detectable, then checks that
`load_batch(..., kd_ids, kd_logprobs)` serves teacher rows exactly aligned
with `targets` (sidecar row i = distribution predicting tokens[i+1]).

Runs on CPU; only needs the built `_surogate` extension.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import surogate._surogate as _surogate
except ImportError:
    pytest.skip("surogate._surogate C++ extension not built", allow_module_level=True)

from surogate.distill.sidecar import SidecarWriter

SEQ_LEN = 64
N_TOKENS = 8 * SEQ_LEN + 5  # deliberately not a multiple of SEQ_LEN
VOCAB = 50000
TOP_K = 4
HASH = "0123456789abcdef"


def write_token_shard(path, tokens: np.ndarray, vocab_size: int) -> None:
    """Minimal BIN.TOK version-3 writer (no masks): header + tokens + position_ids."""
    header = np.zeros(256, dtype=np.int32)
    header[0:2] = np.frombuffer(b"BIN.TOK\n", dtype=np.int32)
    header[2] = 3  # version
    header[3] = 4  # bytes per token
    header[4] = len(tokens)
    header[5] = vocab_size
    header[6] = 0  # has_masks
    header[7] = 0  # non_overlapping
    with open(path, "wb") as f:
        header.tofile(f)
        tokens.astype(np.int32).tofile(f)
        np.arange(len(tokens), dtype=np.int32).tofile(f)  # position ids


def expected_logprob(row: int, k: int) -> np.ndarray:
    # Row-dependent pattern, distinct per (row, k), representable in fp16.
    return np.float16(-0.125 * ((row % 97) + 1) - 0.5 * np.arange(k))


@pytest.fixture()
def kd_shard(tmp_path):
    rng = np.random.default_rng(7)
    tokens = rng.integers(0, VOCAB, size=N_TOKENS, dtype=np.int32)
    shard = tmp_path / "train-000.bin"
    write_token_shard(shard, tokens, VOCAB)

    # Sidecar row i predicts tokens[i+1]: encode that token as the top-1 id so
    # alignment with `targets` is directly checkable. Remaining ids are
    # (tokens[i+1] + j) % VOCAB.
    with SidecarWriter(str(shard) + ".kd", N_TOKENS, TOP_K, VOCAB, HASH) as writer:
        ids = np.zeros((N_TOKENS, TOP_K), dtype=np.uint32)
        lps = np.zeros((N_TOKENS, TOP_K), dtype=np.float16)
        for i in range(N_TOKENS - 1):
            ids[i] = (int(tokens[i + 1]) + np.arange(TOP_K)) % VOCAB
            lps[i] = expected_logprob(i, TOP_K)
        writer.write_rows(ids, lps)
    return shard, tokens


class TestDataLoaderKd:
    def test_kd_rows_align_with_targets(self, kd_shard):
        shard, tokens = kd_shard
        loader = _surogate.DataLoader([str(shard)], SEQ_LEN, seed=3)
        loader.enable_kd(TOP_K)
        assert loader.has_kd

        inputs = np.empty((1, SEQ_LEN), dtype=np.int32)
        targets = np.empty((1, SEQ_LEN), dtype=np.int32)
        pos = np.empty((1, SEQ_LEN), dtype=np.int32)
        kd_ids = np.empty((1, SEQ_LEN, TOP_K), dtype=np.int32)
        kd_lps = np.empty((1, SEQ_LEN, TOP_K), dtype=np.float32)

        checked = 0
        while loader.has_next():
            loader.load_batch(inputs, targets, pos, kd_ids, kd_lps)
            # Recover the chunk's absolute position from the token content:
            # inputs[0, 0] == tokens[chunk_pos] and the shard has unique-enough
            # random content — recover via the position ids instead (arange).
            chunk_pos = int(pos[0, 0])
            for j in range(SEQ_LEN):
                row = chunk_pos + j
                assert targets[0, j] == tokens[row + 1]
                # top-1 teacher id was written as the true next token
                assert kd_ids[0, j, 0] == targets[0, j], (
                    f"KD row misaligned at chunk_pos={chunk_pos} j={j}: "
                    f"kd top-1 {kd_ids[0, j, 0]} != target {targets[0, j]}"
                )
                np.testing.assert_allclose(
                    kd_lps[0, j],
                    expected_logprob(row, TOP_K).astype(np.float32),
                    rtol=0,
                    atol=0,
                    err_msg=f"KD logprobs mismatch at row {row}",
                )
            checked += 1
        # has_next(1) is conservative on the last file (strict `+ world_size`
        # bound): the epoch's final chunk is withheld. Pre-existing loader
        # behavior, independent of KD; with world_size=1 that is one chunk.
        total_chunks = (N_TOKENS - 1) // SEQ_LEN
        assert checked == total_chunks - 1
        assert checked > 0

    def test_enable_kd_rejects_wrong_k(self, kd_shard):
        shard, _ = kd_shard
        loader = _surogate.DataLoader([str(shard)], SEQ_LEN, seed=3)
        with pytest.raises(RuntimeError, match="top_k"):
            loader.enable_kd(TOP_K + 1)

    def test_enable_kd_rejects_missing_sidecar(self, tmp_path):
        rng = np.random.default_rng(11)
        tokens = rng.integers(0, VOCAB, size=N_TOKENS, dtype=np.int32)
        shard = tmp_path / "train-001.bin"
        write_token_shard(shard, tokens, VOCAB)
        loader = _surogate.DataLoader([str(shard)], SEQ_LEN, seed=3)
        with pytest.raises(RuntimeError, match="distill-capture"):
            loader.enable_kd(TOP_K)

    def test_kd_buffers_without_enable_raise(self, kd_shard):
        shard, _ = kd_shard
        loader = _surogate.DataLoader([str(shard)], SEQ_LEN, seed=3)
        inputs = np.empty((1, SEQ_LEN), dtype=np.int32)
        targets = np.empty((1, SEQ_LEN), dtype=np.int32)
        pos = np.empty((1, SEQ_LEN), dtype=np.int32)
        kd_ids = np.empty((1, SEQ_LEN, TOP_K), dtype=np.int32)
        kd_lps = np.empty((1, SEQ_LEN, TOP_K), dtype=np.float32)
        with pytest.raises(RuntimeError, match="enable_kd"):
            loader.load_batch(inputs, targets, pos, kd_ids, kd_lps)

    def test_plain_load_batch_still_works(self, kd_shard):
        shard, tokens = kd_shard
        loader = _surogate.DataLoader([str(shard)], SEQ_LEN, seed=3)
        loader.enable_kd(TOP_K)
        inputs = np.empty((1, SEQ_LEN), dtype=np.int32)
        targets = np.empty((1, SEQ_LEN), dtype=np.int32)
        loader.load_batch(inputs, targets)
        pos0 = np.where(tokens[:-SEQ_LEN] == inputs[0, 0])[0]
        assert len(pos0) >= 1
