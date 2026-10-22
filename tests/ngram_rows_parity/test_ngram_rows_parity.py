"""The trainer's n-gram row selection against the serving engine's rows_kernel, bit for bit, on fixed ids.

ngram_rows_parity (built by build.sh from ngram_rows_parity.cu: the trainer's ngram_memory::rows linked from
libsurogate-common, the engine's rows_kernel extracted verbatim from csrc/src/serve) runs both kernels on the same
deterministic 2^20-token stream (EOS 2 %, conversation starts; the trainer in 4096-column chunks with the two-token
carry, the engine in segments of 1 / 3 / <= 700 columns reading slot history) and exits 0 iff every row agrees; this
test runs it for Swift-1.5's hash constants and the ngram-poc's MEM and CTRL tables, and checks the trainer's rows
against the NumPy reference (surogate.dsl.modules.ngram_memory.ngram_rows_reference) too.

    NGRAM_ROWS_PARITY_BIN=/path/ngram_rows_parity pytest tests/ngram_rows_parity/test_ngram_rows_parity.py
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from surogate.dsl.modules.ngram_memory import TableMeta, ngram_rows_reference

SWIFT15 = {"multipliers": [23703573157769, 20109073645365, 8052911324071],
           "head_vocab_sizes": [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                                20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]}
POC_MEM = [330017, 330019, 330037, 330041, 330047, 330053, 330061, 330067,
           440009, 440023, 440039, 440047, 440087, 440093, 440101, 440131]
CASES = {"swift15": SWIFT15["head_vocab_sizes"], "poc_mem": POC_MEM, "poc_ctrl": [1] * 16}


def _meta(sizes):
    return TableMeta.from_hash({"multipliers": SWIFT15["multipliers"], "head_vocab_sizes": sizes, "ngram_size": 3,
                                "heads_per_ngram": 8, "head_dim": 128, "eos_token_id": 248044})


@pytest.mark.gpu
@pytest.mark.parametrize("case", sorted(CASES))
def test_trainer_rows_equal_the_engine_rows_kernel(case, tmp_path):
    binary = os.environ.get("NGRAM_ROWS_PARITY_BIN")
    if not binary or not Path(binary).exists():
        pytest.skip("NGRAM_ROWS_PARITY_BIN not set (build tests/ngram_rows_parity/build.sh)")
    meta = _meta(CASES[case])
    hash_txt = tmp_path / "hash.txt"
    hash_txt.write_text(" ".join(str(v) for v in [3, 16, 248044, *meta.multipliers, *meta.head_offsets,
                                                  *meta.head_vocab_sizes]))
    n = 1 << 20
    proc = subprocess.run([binary, str(hash_txt), str(n), "2026", str(tmp_path)], capture_output=True, text=True,
                          timeout=600)
    print(proc.stdout, proc.stderr)
    assert proc.returncode == 0, proc.stdout + proc.stderr  # trainer kernel == engine rows_kernel, every row
    ids = np.fromfile(tmp_path / "ids.i32", dtype=np.int32).astype(np.int64)
    pos = np.fromfile(tmp_path / "pos.i32", dtype=np.int32).astype(np.int64)
    rows = np.fromfile(tmp_path / "rows_trainer.i32", dtype=np.int32).reshape(-1, 16)
    assert len(ids) == n and (pos == 0).sum() > 500 and (ids == 248044).mean() > 0.01
    # the NumPy reference: chunks of 4096 with the two tokens before each chunk as the carry
    T = 4096
    ref = np.concatenate([ngram_rows_reference(ids[None, c:c + T], pos[None, c:c + T], meta,
                                               carry=None if c == 0 else ids[None, c - 2:c])
                          for c in range(0, n, T)])
    assert np.array_equal(rows, ref)
    if case == "poc_ctrl":
        assert (rows == np.arange(16)[None, :]).all()
    assert (rows >= 0).all() and (rows < meta.row_count).all()
