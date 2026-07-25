"""Local HF capture correctness: per-document attention isolation and alignment.

Builds a tiny packed token shard (multiple docs with position-id resets), runs
`capture_shard` with a cached small teacher, and compares each captured row
against a reference computed by feeding EVERY DOCUMENT AS ITS OWN SEQUENCE.
The per-doc reference makes this test sensitive to cross-document attention
leaks: if the capture forward did not isolate packed documents (flash-attention
varlen), rows after the first doc boundary diverge.

Requirements: 1 GPU, flash-attn installed, cached Qwen3 weights
(QWEN3_MODEL_PATH or HF cache for Qwen/Qwen3-0.6B / Qwen/Qwen3-1.7B /
Qwen/Qwen3.5-0.8B).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("flash_attn", reason="capture's document isolation requires flash-attention-2")

from surogate.distill.capture import capture_shard, _load_teacher
from surogate.distill.sidecar import read_sidecar, read_token_shard_header

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

CANDIDATE_MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3.5-0.8B"]
ENV_VAR = "QWEN3_MODEL_PATH"
SEQ_LEN = 96
TOP_K = 8
HASH = "fedcba9876543210"
# Docs per window chosen so windows contain internal doc boundaries AND one
# doc crosses a window boundary (truncated at the window edge like tokenize
# packing does not allow, but position ids simply keep counting — the capture
# splits at window starts regardless, matching DataLoader chunk semantics).
DOC_LENS = [40, 56, 33, 63, 30]  # = 222 tokens -> 2 full windows + 30 tail


def resolve_model_path() -> Path | None:
    env = os.environ.get(ENV_VAR)
    if env and Path(env).exists():
        return Path(env)
    cache_root = Path("~/.cache/huggingface/hub").expanduser()
    for model_id in CANDIDATE_MODELS:
        snaps = cache_root / f"models--{model_id.replace('/', '--')}" / "snapshots"
        if snaps.exists():
            for snap in sorted(snaps.iterdir(), reverse=True):
                if (snap / "config.json").exists():
                    return snap
    return None


def write_token_shard(path, tokens: np.ndarray, position_ids: np.ndarray, vocab_size: int) -> None:
    header = np.zeros(256, dtype=np.int32)
    header[0:2] = np.frombuffer(b"BIN.TOK\n", dtype=np.int32)
    header[2] = 3
    header[3] = 4
    header[4] = len(tokens)
    header[5] = vocab_size
    with open(path, "wb") as f:
        header.tofile(f)
        tokens.astype(np.int32).tofile(f)
        position_ids.astype(np.int32).tofile(f)


@pytest.fixture(scope="module")
def teacher():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip(f"No cached teacher. Set {ENV_VAR} or cache one of {CANDIDATE_MODELS}")
    model = _load_teacher(str(snapshot), "cuda", allow_cross_doc_attention=False)
    yield snapshot, model
    del model
    torch.cuda.empty_cache()


def test_capture_matches_per_document_reference(tmp_path, teacher):
    snapshot, model = teacher
    vocab_size = model.config.vocab_size

    rng = np.random.default_rng(99)
    tokens = rng.integers(0, min(vocab_size, 50000), size=sum(DOC_LENS), dtype=np.int32)
    position_ids = np.concatenate([np.arange(n, dtype=np.int32) for n in DOC_LENS])
    n_tokens = len(tokens)

    shard = tmp_path / "train-000.bin"
    write_token_shard(shard, tokens, position_ids, vocab_size)

    capture_shard(
        model,
        str(shard),
        str(shard) + ".kd",
        top_k=TOP_K,
        sequence_len=SEQ_LEN,
        teacher_batch_size=2,
        teacher_vocab_size=vocab_size,
        tokenize_hash=HASH,
        device="cuda",
    )
    header, cap_ids, cap_lps = read_sidecar(str(shard) + ".kd")
    assert header.k == TOP_K and header.n_tokens == n_tokens

    # Reference: independent forward per (window-truncated) document segment.
    # Segments are what the student attends over: doc boundaries reset at
    # position-id drops AND at every window boundary.
    n_windows = n_tokens // SEQ_LEN
    boundaries = sorted(
        {i for i in range(n_windows * SEQ_LEN) if i % SEQ_LEN == 0 or position_ids[i] == 0}
    ) + [n_windows * SEQ_LEN]
    top1_match = 0
    checked = 0
    with torch.inference_mode():
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            seg = torch.tensor(tokens[a:b].astype(np.int64), device="cuda").unsqueeze(0)
            logits = model(input_ids=seg).logits[0].float()
            ref_lp = torch.log_softmax(logits, dim=-1).cpu()
            # Row a+i predicts tokens[a+i+1]; the segment's last row predicts
            # the next segment's first token — captured, and comparable, since
            # both use only the segment's context.
            for i in range(b - a):
                row = a + i
                if row >= n_windows * SEQ_LEN:
                    break
                ref_row = ref_lp[i]
                ref_vals, ref_ids = torch.topk(ref_row, TOP_K)
                cap_row_ids = cap_ids[row].astype(np.int64)
                cap_row_lps = cap_lps[row].astype(np.float32)
                checked += 1
                if int(ref_ids[0]) == int(cap_row_ids[0]):
                    top1_match += 1
                # Mass agreement over the union of both top-k sets: tight when
                # attention contexts match, badly off when the capture leaked
                # attention across documents.
                union = sorted(set(cap_row_ids.tolist()) | set(int(x) for x in ref_ids))
                ref_p = {int(t): float(np.exp(ref_row[t])) for t in union}
                cap_p = {int(t): 0.0 for t in union}
                for t, lp in zip(cap_row_ids, cap_row_lps):
                    cap_p[int(t)] = float(np.exp(lp))
                l1 = sum(abs(ref_p[t] - cap_p.get(t, 0.0)) for t in union)
                assert l1 < 0.15, (
                    f"row {row}: captured top-k distribution diverges from the per-document "
                    f"reference (L1 over union = {l1:.3f}) — cross-document attention leak?"
                )

    assert checked == n_windows * SEQ_LEN
    assert top1_match / checked > 0.98, f"top-1 agreement only {top1_match}/{checked}"

    # Tail rows past the last full window stay zero-filled.
    tail = cap_lps[n_windows * SEQ_LEN :]
    assert np.all(tail == 0)
