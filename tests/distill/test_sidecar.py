"""CPU tests for the .kd sidecar reader/writer (pure numpy, no torch)."""

import numpy as np
import pytest

from surogate.distill.sidecar import (
    KD_LOGPROB_DTYPE_FP16,
    KD_MAGIC,
    KD_VERSION,
    SidecarWriter,
    read_sidecar,
    read_sidecar_header,
    read_token_shard_header,
    sidecar_path_for,
    validate_sidecar,
)

TOKEN_MAGIC = b"BIN.TOK\n"
HASH = "0123456789abcdef"


def _write_token_shard(path, tokens, vocab_size, version=3):
    """Minimal BIN.TOK writer mirroring TokenizedDataFileWriter's layout:
    [1024-byte header][tokens int32][position_ids int32 (version 3)].
    """
    tokens = np.asarray(tokens, dtype="<i4")
    position_ids = np.arange(len(tokens), dtype="<i4")
    header = bytearray(1024)
    header[0:8] = TOKEN_MAGIC
    fields = np.array([version, 4, len(tokens), vocab_size, 0, 0], dtype="<i4")
    header[8 : 8 + fields.nbytes] = fields.tobytes()
    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(tokens.tobytes())
        if version >= 3:
            f.write(position_ids.tobytes())


def _make_pair(tmp_path, n_tokens=64, k=4, vocab_size=100, tokenize_hash=HASH):
    shard = str(tmp_path / "train-000.bin")
    tokens = np.arange(n_tokens, dtype=np.int32) % vocab_size
    _write_token_shard(shard, tokens, vocab_size)
    sidecar = shard + ".kd"
    ids = (np.arange(n_tokens * k, dtype=np.uint32) % vocab_size).reshape(n_tokens, k)
    logprobs = (-0.25 * (1 + np.arange(n_tokens * k, dtype=np.float32) % 16)).reshape(n_tokens, k)
    with SidecarWriter(sidecar, n_tokens=n_tokens, k=k, vocab_size=vocab_size, tokenize_hash=tokenize_hash) as w:
        w.write_rows(ids[: n_tokens // 2], logprobs[: n_tokens // 2])
        w.write_rows(ids[n_tokens // 2 :], logprobs[n_tokens // 2 :])
    return shard, sidecar, tokens, ids, logprobs


def test_token_shard_header_roundtrip(tmp_path):
    shard = str(tmp_path / "train-000.bin")
    _write_token_shard(shard, np.arange(10), vocab_size=50)
    h = read_token_shard_header(shard)
    assert h.version == 3
    assert h.bytes_per_token == 4
    assert h.n_tokens == 10
    assert h.vocab_size == 50
    assert not h.has_masks
    assert not h.non_overlapping
    assert h.tokens_offset == 1024
    assert h.position_ids_offset == 1024 + 4 * 10


def test_sidecar_roundtrip(tmp_path):
    shard, sidecar, _, ids, logprobs = _make_pair(tmp_path)
    header = read_sidecar_header(sidecar)
    assert header.version == KD_VERSION
    assert header.k == 4
    assert header.n_tokens == 64
    assert header.vocab_size == 100
    assert header.logprob_dtype == KD_LOGPROB_DTYPE_FP16
    assert header.tokenize_hash == HASH

    read_header, read_ids, read_logprobs = read_sidecar(sidecar)
    assert read_header == header
    np.testing.assert_array_equal(np.asarray(read_ids), ids)
    # Values chosen fp16-representable, so the roundtrip is exact.
    np.testing.assert_array_equal(np.asarray(read_logprobs, dtype=np.float32), logprobs)


def test_sidecar_block_layout(tmp_path):
    # ids block (uint32) precedes the logprobs block (fp16): byte offsets are
    # 1024 and 1024 + 4*k*n_tokens.
    _, sidecar, _, ids, logprobs = _make_pair(tmp_path, n_tokens=8, k=2)
    with open(sidecar, "rb") as f:
        raw = f.read()
    assert raw[:8] == KD_MAGIC
    assert len(raw) == 1024 + 6 * 2 * 8
    ids_block = np.frombuffer(raw, dtype="<u4", count=8 * 2, offset=1024).reshape(8, 2)
    lp_block = np.frombuffer(raw, dtype="<f2", count=8 * 2, offset=1024 + 4 * 2 * 8).reshape(8, 2)
    np.testing.assert_array_equal(ids_block, ids)
    np.testing.assert_array_equal(lp_block.astype(np.float32), logprobs)


def test_tail_rows_zero_filled(tmp_path):
    shard = str(tmp_path / "train-000.bin")
    _write_token_shard(shard, np.arange(64), vocab_size=100)
    sidecar = shard + ".kd"
    with SidecarWriter(sidecar, n_tokens=64, k=4, vocab_size=100, tokenize_hash=HASH) as w:
        w.write_rows(np.ones((40, 4), np.uint32), np.full((40, 4), -1.0, np.float16))
        assert w.rows_written == 40
    _, ids, logprobs = read_sidecar(sidecar)
    np.testing.assert_array_equal(np.asarray(ids[40:]), 0)
    np.testing.assert_array_equal(np.asarray(logprobs[40:], dtype=np.float32), 0.0)
    np.testing.assert_array_equal(np.asarray(ids[:40]), 1)


def test_alignment_row_i_predicts_token_i_plus_1(tmp_path):
    # Contract: sidecar row i holds the teacher distribution predicting
    # tokens[i+1], i.e. aligned with targets[i]. The DataLoader serves a chunk
    # at chunk_pos as inputs=tokens[chunk_pos:chunk_pos+S] and
    # targets=tokens[chunk_pos+1:chunk_pos+S+1], reading sidecar rows
    # [chunk_pos, chunk_pos+S) with NO extra shift — including the last row of
    # each capture window, whose target is the first token of the next window.
    n_tokens, k, vocab, S = 64, 2, 100, 16
    shard = str(tmp_path / "train-000.bin")
    tokens = ((np.arange(n_tokens) * 7 + 3) % vocab).astype(np.int32)
    _write_token_shard(shard, tokens, vocab)
    sidecar = shard + ".kd"
    ids = np.zeros((n_tokens, k), np.uint32)
    ids[: n_tokens - 1, 0] = tokens[1:]  # perfect teacher: top-1 = the true next token
    with SidecarWriter(sidecar, n_tokens=n_tokens, k=k, vocab_size=vocab, tokenize_hash=HASH) as w:
        w.write_rows(ids, np.zeros((n_tokens, k), np.float16))

    _, read_ids, _ = read_sidecar(sidecar)
    for chunk_pos in (0, S, 2 * S):
        targets = tokens[chunk_pos + 1 : chunk_pos + S + 1]
        np.testing.assert_array_equal(np.asarray(read_ids[chunk_pos : chunk_pos + S, 0]), targets)


def test_validate_ok(tmp_path):
    shard, sidecar, _, _, _ = _make_pair(tmp_path)
    header = validate_sidecar(sidecar, shard, expect_k=4, expect_hash=HASH)
    assert header.k == 4


def test_validate_missing_sidecar(tmp_path):
    shard = str(tmp_path / "train-000.bin")
    _write_token_shard(shard, np.arange(16), vocab_size=100)
    with pytest.raises(ValueError, match="distill-capture"):
        validate_sidecar(shard + ".kd", shard)


def test_validate_bad_magic(tmp_path):
    shard, sidecar, _, _, _ = _make_pair(tmp_path)
    with open(sidecar, "r+b") as f:
        f.write(b"XXXXXXXX")
    with pytest.raises(ValueError, match="bad magic"):
        validate_sidecar(sidecar, shard)


def test_validate_unsupported_version(tmp_path):
    shard, sidecar, _, _, _ = _make_pair(tmp_path)
    with open(sidecar, "r+b") as f:
        f.seek(8)
        f.write(np.array([99], dtype="<i4").tobytes())
    with pytest.raises(ValueError, match="version"):
        validate_sidecar(sidecar, shard)


def test_validate_n_tokens_mismatch(tmp_path):
    shard, sidecar, _, _, _ = _make_pair(tmp_path, n_tokens=64)
    other_shard = str(tmp_path / "train-001.bin")
    _write_token_shard(other_shard, np.arange(32), vocab_size=100)
    with pytest.raises(ValueError, match="re-tokenized"):
        validate_sidecar(sidecar, other_shard)


def test_validate_k_mismatch(tmp_path):
    shard, sidecar, _, _, _ = _make_pair(tmp_path, k=4)
    with pytest.raises(ValueError, match="top_k"):
        validate_sidecar(sidecar, shard, expect_k=8)


def test_validate_hash_mismatch(tmp_path):
    shard, sidecar, _, _, _ = _make_pair(tmp_path, tokenize_hash="deadbeefdeadbeef")
    with pytest.raises(ValueError, match="re-tokenized"):
        validate_sidecar(sidecar, shard, expect_hash=HASH)


def test_validate_truncated_file(tmp_path):
    shard, sidecar, _, _, _ = _make_pair(tmp_path)
    size = (tmp_path / "train-000.bin.kd").stat().st_size
    with open(sidecar, "r+b") as f:
        f.truncate(size - 8)
    with pytest.raises(ValueError, match="truncated"):
        validate_sidecar(sidecar, shard)


def test_writer_rejects_bad_shapes_and_overflow(tmp_path):
    shard = str(tmp_path / "train-000.bin")
    _write_token_shard(shard, np.arange(8), vocab_size=100)
    with pytest.raises(ValueError, match="k must be"):
        SidecarWriter(shard + ".kd", n_tokens=8, k=0, vocab_size=100, tokenize_hash=HASH)
    with pytest.raises(ValueError, match="k must be"):
        SidecarWriter(shard + ".kd", n_tokens=8, k=2048, vocab_size=100, tokenize_hash=HASH)
    with SidecarWriter(shard + ".kd", n_tokens=8, k=2, vocab_size=100, tokenize_hash=HASH) as w:
        with pytest.raises(ValueError, match="shape"):
            w.write_rows(np.zeros((4, 3), np.uint32), np.zeros((4, 3), np.float16))
        with pytest.raises(ValueError, match="does not match"):
            w.write_rows(np.zeros((4, 2), np.uint32), np.zeros((3, 2), np.float16))
        w.write_rows(np.zeros((8, 2), np.uint32), np.zeros((8, 2), np.float16))
        with pytest.raises(ValueError, match="overflows"):
            w.write_rows(np.zeros((1, 2), np.uint32), np.zeros((1, 2), np.float16))


def test_sidecar_path_for(tmp_path):
    assert sidecar_path_for("/data/train-000.bin") == "/data/train-000.bin.kd"
    assert sidecar_path_for("/data/train-000.bin", kd_dir="/kd") == "/kd/train-000.bin.kd"
