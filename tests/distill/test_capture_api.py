"""CPU tests for the OpenAI-compatible API capture backend (mocked vLLM server).

Uses httpx.MockTransport, so no network or GPU is involved; the fake teacher is
deterministic: at every position the top candidates predicting the next token
are (prev_token + 1 + r) % VOCAB with logprob -0.5 - r for rank r.
"""

import json
import threading

import numpy as np
import pytest

httpx = pytest.importorskip("httpx")
capture = pytest.importorskip("surogate.distill.capture", reason="requires the built _surogate native module")

from surogate.distill.capture import _doc_spans, _make_api_client, capture_shard_api  # noqa: E402
from surogate.distill.sidecar import read_sidecar, validate_sidecar  # noqa: E402

VOCAB = 50
HASH = "0123456789abcdef"
API_BASE = "http://fake-teacher:8000/v1"


def _write_token_shard(path, tokens, position_ids, vocab_size=VOCAB):
    tokens = np.asarray(tokens, dtype="<i4")
    position_ids = np.asarray(position_ids, dtype="<i4")
    assert len(tokens) == len(position_ids)
    header = bytearray(1024)
    header[0:8] = b"BIN.TOK\n"
    header[8:32] = np.array([3, 4, len(tokens), vocab_size, 0, 0], dtype="<i4").tobytes()
    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(tokens.tobytes())
        f.write(position_ids.tobytes())


def _packed_shard(tmp_path, doc_lengths, n_tokens=None):
    """Shard with per-doc position resets (0..len-1 per doc), like sample packing."""
    pos = np.concatenate([np.arange(l, dtype=np.int32) for l in doc_lengths])
    if n_tokens is not None:
        assert len(pos) == n_tokens
    tokens = ((np.arange(len(pos)) * 3 + 1) % VOCAB).astype(np.int32)
    shard = str(tmp_path / "train-000.bin")
    _write_token_shard(shard, tokens, pos)
    return shard, tokens, pos


class FakeVLLM:
    """MockTransport handler mimicking vLLM's /completions prompt_logprobs response."""

    def __init__(self, extra_candidates=0, max_candidates=None, fail_first=0, omit_prompt_logprobs=False,
                 error_response=None):
        self.extra_candidates = extra_candidates
        self.max_candidates = max_candidates
        self.fail_first = fail_first
        self.omit_prompt_logprobs = omit_prompt_logprobs
        self.error_response = error_response
        self.calls = 0
        self.requests = []
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        with self._lock:
            self.calls += 1
            call_index = self.calls
            self.requests.append(
                {"path": request.url.path, "body": body, "auth": request.headers.get("authorization")}
            )
        if self.error_response is not None:
            return httpx.Response(self.error_response[0], json=self.error_response[1])
        if call_index <= self.fail_first:
            return httpx.Response(500, json={"error": "transient boom"})
        if self.omit_prompt_logprobs:
            # Plain OpenAI completions response (e.g. an aggregator like OpenRouter).
            return httpx.Response(200, json={"choices": [{"index": 0, "text": "x"}]})
        k = int(body["prompt_logprobs"])
        n_cands = self.max_candidates if self.max_candidates is not None else k + self.extra_candidates
        prompt = body["prompt"]
        plp = [None]
        for i in range(1, len(prompt)):
            prev = int(prompt[i - 1])
            plp.append(
                {str((prev + 1 + r) % VOCAB): {"logprob": -0.5 - r, "rank": r + 1} for r in range(n_cands)}
            )
        return httpx.Response(200, json={"choices": [{"index": 0, "text": "", "prompt_logprobs": plp}]})


def _client(handler):
    # Mirrors _make_api_client's construction (unset key env -> "EMPTY") with a
    # mock transport; _make_api_client itself is covered by test_api_key_from_env_var.
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=API_BASE,
        headers={"Authorization": "Bearer EMPTY"},
    )


def _run(handler, shard, top_k=2, sequence_len=8, concurrency=2):
    with _client(handler) as client:
        capture_shard_api(
            client,
            "teacher-x",
            shard,
            shard + ".kd",
            top_k=top_k,
            sequence_len=sequence_len,
            concurrency=concurrency,
            tokenize_hash=HASH,
            retry_backoff=0.0,
        )


def test_doc_spans():
    pos = np.concatenate([np.arange(5), np.arange(6), np.arange(9)]).astype(np.int32)  # docs at 0, 5, 11
    assert _doc_spans(pos, 0, 8) == [(0, 5), (5, 8)]
    # Window start is always a span start even mid-document.
    assert _doc_spans(pos, 8, 16) == [(8, 11), (11, 16)]
    assert _doc_spans(None, 8, 16) == [(8, 16)]


def test_request_shape_doc_split_and_alignment(tmp_path):
    shard, tokens, _ = _packed_shard(tmp_path, [5, 6, 9], n_tokens=20)  # S=8 -> 2 windows, tail 4
    handler = FakeVLLM()
    _run(handler, shard)

    # (a) request shape: token-id prompt, prompt_logprobs=K, served model name.
    for req in handler.requests:
        assert req["path"] == "/v1/completions"
        body = req["body"]
        assert body["model"] == "teacher-x"
        assert isinstance(body["prompt"], list) and all(isinstance(t, int) for t in body["prompt"])
        assert body["prompt_logprobs"] == 2
        assert body["max_tokens"] == 1
        assert body["temperature"] == 1.0
        assert req["auth"] == "Bearer EMPTY"

    # (b) per-doc splitting at position resets, with one-token lookahead. The doc
    # spanning the window boundary [8,16) restarts at the window start.
    expected_prompts = {
        tuple(tokens[0:6].tolist()),   # doc [0,5) + lookahead
        tuple(tokens[5:9].tolist()),   # doc [5,8) (cut at window end) + lookahead across the boundary
        tuple(tokens[8:12].tolist()),  # doc fragment [8,11) + lookahead
        tuple(tokens[11:17].tolist()),  # doc [11,16) (cut at window end) + lookahead
    }
    assert {tuple(r["body"]["prompt"]) for r in handler.requests} == expected_prompts

    # (c) row alignment: prompt_logprobs[i] predicts prompt position i, stored at
    # sidecar row a+i-1; with the fake teacher, row r's top-1 is (tokens[r]+1)%V.
    validate_sidecar(shard + ".kd", shard, expect_k=2, expect_hash=HASH)
    _, ids, lps = read_sidecar(shard + ".kd")
    captured = 16  # 2 full windows; rows 16..19 are tail
    np.testing.assert_array_equal(np.asarray(ids[:captured, 0]), (tokens[:captured] + 1) % VOCAB)
    np.testing.assert_array_equal(np.asarray(ids[:captured, 1]), (tokens[:captured] + 2) % VOCAB)
    # fp16 roundtrip is exact for -0.5 / -1.5.
    np.testing.assert_array_equal(np.asarray(lps[:captured, 0], dtype=np.float32), -0.5)
    np.testing.assert_array_equal(np.asarray(lps[:captured, 1], dtype=np.float32), -1.5)
    np.testing.assert_array_equal(np.asarray(ids[captured:]), 0)


def test_no_lookahead_at_shard_end(tmp_path):
    # Single doc ending exactly at n_tokens == n_windows*S: the final row has no
    # next token, so it stays zero-filled (existing local-backend semantics).
    shard, tokens, _ = _packed_shard(tmp_path, [16], n_tokens=16)
    handler = FakeVLLM()
    _run(handler, shard, sequence_len=8)
    prompts = sorted(tuple(r["body"]["prompt"]) for r in handler.requests)
    assert prompts == sorted([tuple(tokens[0:9].tolist()), tuple(tokens[8:16].tolist())])
    _, ids, _ = read_sidecar(shard + ".kd")
    np.testing.assert_array_equal(np.asarray(ids[:15, 0]), (tokens[:15] + 1) % VOCAB)
    np.testing.assert_array_equal(np.asarray(ids[15]), 0)


def test_k_plus_one_candidates_truncated_by_logprob(tmp_path):
    # vLLM may return K+1 entries when the actual token is outside the top-K;
    # capture keeps the top K by logprob.
    shard, tokens, _ = _packed_shard(tmp_path, [8], n_tokens=8)
    handler = FakeVLLM(extra_candidates=1)
    _run(handler, shard, top_k=2)
    _, ids, lps = read_sidecar(shard + ".kd")
    np.testing.assert_array_equal(np.asarray(ids[:7, 0]), (tokens[:7] + 1) % VOCAB)
    np.testing.assert_array_equal(np.asarray(ids[:7, 1]), (tokens[:7] + 2) % VOCAB)
    assert float(np.asarray(lps[:7]).min()) >= -1.5  # rank-3 candidate (-2.5) was dropped


def test_abort_when_server_max_logprobs_below_k(tmp_path):
    shard, _, _ = _packed_shard(tmp_path, [8], n_tokens=8)
    handler = FakeVLLM(max_candidates=1)
    with pytest.raises(RuntimeError, match=r"--max-logprobs >= 2"):
        _run(handler, shard, top_k=2)


def test_abort_when_prompt_logprobs_missing(tmp_path):
    # A plain OpenAI-compatible provider (e.g. OpenRouter) that ignores the
    # prompt_logprobs extension must abort with actionable guidance.
    shard, _, _ = _packed_shard(tmp_path, [8], n_tokens=8)
    handler = FakeVLLM(omit_prompt_logprobs=True)
    with pytest.raises(RuntimeError, match="OpenRouter"):
        _run(handler, shard)
    assert not (tmp_path / "train-000.bin.kd").exists()


def test_4xx_unknown_field_surfaces_guidance(tmp_path):
    shard, _, _ = _packed_shard(tmp_path, [8], n_tokens=8)
    handler = FakeVLLM(error_response=(400, {"error": {"message": "unknown field 'prompt_logprobs'"}}))
    with pytest.raises(RuntimeError, match="vLLM-compatible"):
        _run(handler, shard)


def test_unrelated_4xx_is_not_rewritten(tmp_path):
    shard, _, _ = _packed_shard(tmp_path, [8], n_tokens=8)
    handler = FakeVLLM(error_response=(404, {"error": {"message": "model not found"}}))
    with pytest.raises(httpx.HTTPStatusError):
        _run(handler, shard)
    assert handler.calls == 1  # non-transient 4xx is not retried


def test_retry_recovers_from_transient_failures(tmp_path):
    shard, tokens, _ = _packed_shard(tmp_path, [8], n_tokens=8)
    handler = FakeVLLM(fail_first=2)
    _run(handler, shard, concurrency=1)
    assert handler.calls == 3  # 2 transient 500s + 1 success for the single doc
    _, ids, _ = read_sidecar(shard + ".kd")
    np.testing.assert_array_equal(np.asarray(ids[:7, 0]), (tokens[:7] + 1) % VOCAB)


def test_retry_gives_up_after_three_attempts(tmp_path):
    shard, _, _ = _packed_shard(tmp_path, [8], n_tokens=8)
    handler = FakeVLLM(error_response=(500, {"error": "down"}))
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        _run(handler, shard, concurrency=1)
    assert handler.calls == 3


def test_api_key_from_env_var(tmp_path, monkeypatch):
    from surogate.core.config.sft_config import DistillationConfig

    dist = DistillationConfig(teacher_model="t", teacher_api_base=API_BASE, teacher_api_key_var="MY_TEACHER_KEY")
    monkeypatch.delenv("MY_TEACHER_KEY", raising=False)
    with _make_api_client(dist) as client:
        assert client.headers["authorization"] == "Bearer EMPTY"
    monkeypatch.setenv("MY_TEACHER_KEY", "sekrit")
    with _make_api_client(dist) as client:
        assert client.headers["authorization"] == "Bearer sekrit"
