"""Reader/writer for the `.kd` teacher-logprob sidecar format.

A sidecar `train-NNN.bin.kd` stores, for every token position of the matching
`train-NNN.bin` token shard, the teacher's top-K next-token log-probabilities.

File layout::

    [Header 1024 bytes]
      bytes 0..7   : magic "KD.LOGP\\n"
      int32 fields (little-endian, int32 index = byte offset / 4):
        [2] version = 1
        [3] k                (top-K per token, 1..1024)
        [4] n_tokens         (must equal the token shard's n_tokens)
        [5] vocab_size       (teacher vocab)
        [6] logprob_dtype    (0 = fp16; only fp16 in v1)
        [7] reserved = 0
      bytes 64..79 : tokenize hash (16 ASCII hex chars from .tokenize_hash,
                     zero-padded) — validated by the Python layer before
                     training (C++ ignores it)
    [ids      : n_tokens x k uint32]   at byte 1024
    [logprobs : n_tokens x k fp16]     at byte 1024 + 4*k*n_tokens

Alignment: row i holds the teacher's next-token distribution at input
position i (predicting tokens[i+1], i.e. aligned with targets[i]). The
DataLoader reads sidecar rows [chunk_pos, chunk_pos + S) for a chunk at
chunk_pos — no extra shift. Rows past the last captured window are
zero-filled.

This module is pure numpy — it must stay importable without torch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

KD_MAGIC = b"KD.LOGP\n"
KD_VERSION = 1
KD_HEADER_BYTES = 1024
KD_HASH_OFFSET = 64
KD_HASH_BYTES = 16
KD_LOGPROB_DTYPE_FP16 = 0
KD_MAX_TOP_K = 1024

TOKEN_MAGIC = b"BIN.TOK\n"
TOKEN_HEADER_BYTES = 1024

SIDECAR_SUFFIX = ".kd"


def sidecar_path_for(token_shard_path: str, kd_dir: str | None = None) -> str:
    """Sidecar path for a token shard; `kd_dir` overrides the directory."""
    if kd_dir:
        return os.path.join(kd_dir, os.path.basename(token_shard_path) + SIDECAR_SUFFIX)
    return token_shard_path + SIDECAR_SUFFIX


@dataclass
class SidecarHeader:
    version: int
    k: int
    n_tokens: int
    vocab_size: int
    logprob_dtype: int
    tokenize_hash: str

    @property
    def ids_offset(self) -> int:
        return KD_HEADER_BYTES

    @property
    def logprobs_offset(self) -> int:
        return KD_HEADER_BYTES + 4 * self.k * self.n_tokens

    @property
    def file_size(self) -> int:
        return KD_HEADER_BYTES + 6 * self.k * self.n_tokens


@dataclass
class TokenShardHeader:
    version: int
    bytes_per_token: int
    n_tokens: int
    vocab_size: int
    has_masks: bool
    non_overlapping: bool

    @property
    def tokens_offset(self) -> int:
        return TOKEN_HEADER_BYTES

    @property
    def position_ids_offset(self) -> int:
        # Version 3 shards store one int32 position id per token right after
        # the token block.
        return TOKEN_HEADER_BYTES + 4 * self.n_tokens


def read_token_shard_header(path: str) -> TokenShardHeader:
    """Parse the 1024-byte BIN.TOK header of a token shard."""
    with open(path, "rb") as f:
        raw = f.read(TOKEN_HEADER_BYTES)
    if len(raw) < TOKEN_HEADER_BYTES:
        raise ValueError(f"Token shard '{path}' is too small to contain a BIN.TOK header.")
    if raw[:8] != TOKEN_MAGIC:
        raise ValueError(f"Token shard '{path}' has bad magic {raw[:8]!r}; expected {TOKEN_MAGIC!r}.")
    fields = np.frombuffer(raw, dtype="<i4")
    return TokenShardHeader(
        version=int(fields[2]),
        bytes_per_token=int(fields[3]),
        n_tokens=int(fields[4]),
        vocab_size=int(fields[5]),
        has_masks=bool(fields[6]),
        non_overlapping=bool(fields[7]),
    )


def _encode_hash(tokenize_hash: str) -> bytes:
    encoded = (tokenize_hash or "").encode("ascii")
    if len(encoded) > KD_HASH_BYTES:
        raise ValueError(f"tokenize hash '{tokenize_hash}' exceeds {KD_HASH_BYTES} bytes.")
    return encoded.ljust(KD_HASH_BYTES, b"\x00")


def write_sidecar_header(f, header: SidecarHeader) -> None:
    """Write the 1024-byte KD.LOGP header at the current start of file `f`."""
    raw = bytearray(KD_HEADER_BYTES)
    raw[0:8] = KD_MAGIC
    fields = np.array(
        [header.version, header.k, header.n_tokens, header.vocab_size, header.logprob_dtype, 0],
        dtype="<i4",
    )
    raw[8 : 8 + fields.nbytes] = fields.tobytes()
    raw[KD_HASH_OFFSET : KD_HASH_OFFSET + KD_HASH_BYTES] = _encode_hash(header.tokenize_hash)
    f.seek(0)
    f.write(bytes(raw))


def read_sidecar_header(path: str) -> SidecarHeader:
    """Parse the 1024-byte KD.LOGP header of a sidecar file."""
    with open(path, "rb") as f:
        raw = f.read(KD_HEADER_BYTES)
    if len(raw) < KD_HEADER_BYTES:
        raise ValueError(f"KD sidecar '{path}' is too small to contain a header.")
    if raw[:8] != KD_MAGIC:
        raise ValueError(
            f"KD sidecar '{path}' has bad magic {raw[:8]!r}; expected {KD_MAGIC!r}. "
            "The file is not a KD sidecar — re-run `surogate distill-capture`."
        )
    fields = np.frombuffer(raw, dtype="<i4")
    version = int(fields[2])
    if version != KD_VERSION:
        raise ValueError(f"KD sidecar '{path}' has unsupported version {version} (expected {KD_VERSION}).")
    hash_bytes = raw[KD_HASH_OFFSET : KD_HASH_OFFSET + KD_HASH_BYTES]
    return SidecarHeader(
        version=version,
        k=int(fields[3]),
        n_tokens=int(fields[4]),
        vocab_size=int(fields[5]),
        logprob_dtype=int(fields[6]),
        tokenize_hash=hash_bytes.rstrip(b"\x00").decode("ascii", errors="replace"),
    )


class SidecarWriter:
    """Streaming writer for a `.kd` sidecar.

    Preallocates the full file (header + ids block + logprobs block) and
    memory-maps both blocks, so rows can be appended in order while the ids
    block physically precedes the logprobs block. Unwritten tail rows stay
    zero-filled (the preallocation writes zeros).
    """

    def __init__(self, path: str, n_tokens: int, k: int, vocab_size: int, tokenize_hash: str):
        if not 1 <= k <= KD_MAX_TOP_K:
            raise ValueError(f"k must be in [1, {KD_MAX_TOP_K}], got {k}.")
        if n_tokens <= 0:
            raise ValueError(f"n_tokens must be positive, got {n_tokens}.")
        if not 0 < vocab_size < 2**31:
            raise ValueError(f"vocab_size must be in (0, 2^31), got {vocab_size}.")
        self.path = path
        self.header = SidecarHeader(
            version=KD_VERSION,
            k=k,
            n_tokens=n_tokens,
            vocab_size=vocab_size,
            logprob_dtype=KD_LOGPROB_DTYPE_FP16,
            tokenize_hash=tokenize_hash,
        )
        with open(path, "wb") as f:
            write_sidecar_header(f, self.header)
            f.truncate(self.header.file_size)
        self._ids = np.memmap(path, dtype="<u4", mode="r+", offset=self.header.ids_offset, shape=(n_tokens, k))
        self._logprobs = np.memmap(
            path, dtype="<f2", mode="r+", offset=self.header.logprobs_offset, shape=(n_tokens, k)
        )
        self.rows_written = 0

    def write_rows(self, ids: np.ndarray, logprobs: np.ndarray) -> None:
        """Append `(rows, k)` teacher top-k ids and logprobs at the next row."""
        ids = np.asarray(ids)
        logprobs = np.asarray(logprobs)
        if ids.ndim != 2 or ids.shape[1] != self.header.k:
            raise ValueError(f"ids must have shape (rows, {self.header.k}), got {ids.shape}.")
        if logprobs.shape != ids.shape:
            raise ValueError(f"logprobs shape {logprobs.shape} does not match ids shape {ids.shape}.")
        end = self.rows_written + ids.shape[0]
        if end > self.header.n_tokens:
            raise ValueError(
                f"write_rows overflows the sidecar: {end} rows > n_tokens={self.header.n_tokens}."
            )
        self._ids[self.rows_written : end] = ids.astype(np.uint32, copy=False)
        self._logprobs[self.rows_written : end] = logprobs.astype(np.float16, copy=False)
        self.rows_written = end

    def close(self) -> None:
        if self._ids is None:
            return
        self._ids.flush()
        self._logprobs.flush()
        self._ids = None
        self._logprobs = None

    def __enter__(self) -> SidecarWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def read_sidecar(path: str) -> tuple[SidecarHeader, np.ndarray, np.ndarray]:
    """Read a sidecar; returns (header, ids [n_tokens, k] uint32, logprobs [n_tokens, k] fp16).

    The arrays are read-only memory maps.
    """
    header = read_sidecar_header(path)
    ids = np.memmap(path, dtype="<u4", mode="r", offset=header.ids_offset, shape=(header.n_tokens, header.k))
    logprobs = np.memmap(
        path, dtype="<f2", mode="r", offset=header.logprobs_offset, shape=(header.n_tokens, header.k)
    )
    return header, ids, logprobs


def validate_sidecar(
    path: str,
    token_shard_path: str,
    expect_k: int | None = None,
    expect_hash: str | None = None,
) -> SidecarHeader:
    """Validate a sidecar against its token shard; raises ValueError with an actionable message."""
    if not os.path.exists(path):
        raise ValueError(
            f"KD sidecar '{path}' is missing for token shard '{token_shard_path}'. "
            "Run `surogate distill-capture <config.yaml>` to capture teacher logprobs."
        )
    header = read_sidecar_header(path)
    token_header = read_token_shard_header(token_shard_path)

    if not 1 <= header.k <= KD_MAX_TOP_K:
        raise ValueError(f"KD sidecar '{path}' has invalid k={header.k} (expected 1..{KD_MAX_TOP_K}).")
    if header.logprob_dtype != KD_LOGPROB_DTYPE_FP16:
        raise ValueError(
            f"KD sidecar '{path}' has unsupported logprob_dtype={header.logprob_dtype} (only fp16=0 in v1)."
        )
    if not 0 < header.vocab_size < 2**31:
        raise ValueError(f"KD sidecar '{path}' has invalid vocab_size={header.vocab_size}.")
    if header.n_tokens != token_header.n_tokens:
        raise ValueError(
            f"KD sidecar '{path}' has n_tokens={header.n_tokens} but token shard "
            f"'{token_shard_path}' has n_tokens={token_header.n_tokens}. The token shards were "
            "re-tokenized after capture — re-run `surogate distill-capture`."
        )
    if expect_k is not None and header.k != expect_k:
        raise ValueError(
            f"KD sidecar '{path}' was captured with top_k={header.k} but the config requests "
            f"top_k={expect_k}. Re-run `surogate distill-capture` with the desired distillation.top_k."
        )
    if expect_hash is not None and header.tokenize_hash != expect_hash:
        raise ValueError(
            f"KD sidecar '{path}' embeds tokenize hash '{header.tokenize_hash}' but the current "
            f"token shards have hash '{expect_hash}'. The token shards were re-tokenized — "
            "re-run `surogate distill-capture`."
        )
    actual_size = os.path.getsize(path)
    if actual_size != header.file_size:
        raise ValueError(
            f"KD sidecar '{path}' is truncated or oversized: {actual_size} bytes on disk, header "
            f"implies {header.file_size}. The capture likely did not finish — re-run "
            "`surogate distill-capture`."
        )
    return header
