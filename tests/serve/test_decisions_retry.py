"""Decisions retry (--decision-attempts): a request whose option logits come out non-finite runs again
from scratch inside the engine, and only an exhausted request returns the error.

GPU. Skipped unless SUROGATE_DECISIONS_RETRY_ARTIFACT names a prepared chat model (.sinfer); select
one free GPU with CUDA_VISIBLE_DEVICES, and add engine flags with SUROGATE_DECISIONS_RETRY_ARGS
(a JSON list).

The faults are the engine's test injections (family/impl/runtime/target_support.h):
- SUROGATE_SERVE_FAULT_POISON_GPU_PREFIX=N: the first N saved GPU prefixes have their KV pages
  overwritten with NaN, so every question read on them is non-finite. A retry that reused the
  failed attempt's prefix would fail again; only a fresh prefill succeeds.
- SUROGATE_SERVE_FAULT_NAN_READOUT=N: the first N candidate readouts come back as NaN. This covers
  the path with no shared prefix (one question, whole prompts).

Every answer is compared with a clean server's answer for the same request, bit for bit.
"""

import concurrent.futures
import json
import os
import re
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(not os.getenv("SUROGATE_DECISIONS_RETRY_ARTIFACT"),
                                reason="needs a prepared chat model and a free GPU")

ARGS = json.loads(os.getenv("SUROGATE_DECISIONS_RETRY_ARGS", "[]"))
ROUTE = "/v1/decisions"

# A shared state long enough that its questions are read on a saved GPU prefix (past the
# prefill floor), with several questions, and a one-question request that runs as a whole prompt.
STATE = "\n".join(
    f"Ticket {i}: the customer reports that order #{1000 + i} arrived late and the package was "
    f"damaged; they ask for a refund of {10 + i} dollars and mention they have been a customer "
    f"for {i % 7 + 1} years." for i in range(60))
SHARED = json.dumps({"model": "rune", "state": STATE, "questions": {
    "sentiment": {"type": "choice", "instructions": "What is the overall sentiment of the tickets?",
                  "criteria": {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}},
    "refund": {"type": "choice", "instructions": "Should the refunds be granted?",
               "criteria": {"yes": "Yes", "no": "No"}},
    "urgent": {"type": "choice", "instructions": "Is this urgent?", "criteria": {"yes": "Yes", "no": "No"}},
    "loyal": {"type": "choice", "instructions": "Are these long-time customers?",
              "criteria": {"yes": "Yes", "no": "No"}},
}})
SINGLE = json.dumps({"model": "rune", "state": "The package arrived two weeks late and was crushed.",
                     "questions": {"sentiment": {"type": "choice", "instructions": "Sentiment?",
                                                 "criteria": {"positive": "Positive", "negative": "Negative"}}}})
# Two questions under different system prompts (one needs the extended codebook past 26
# options) share no prefix, so they run as whole prompts, each prepared again for a retry.
WHOLE = json.dumps({"model": "rune", "state": "Order 7 arrived late.", "questions": {
    "late": {"type": "choice", "instructions": "Was it late?", "criteria": {"yes": "Yes", "no": "No"}},
    "which": {"type": "choice", "instructions": "Which option?",
              "criteria": {f"o{i}": f"Option {i}" for i in range(30)}},
}})
NEIGHBOURS = [json.dumps({"model": "rune", "state": f"Order {i} arrived {i} days late.",
                          "questions": {"late": {"type": "choice", "instructions": "Was it late?",
                                                 "criteria": {"yes": "Yes", "no": "No"}}}})
              for i in range(1, 7)]


@contextmanager
def serve(tmp_path, name, env=None, args=()):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    log = tmp_path / f"{name}.log"
    cmd = [_resolve_binary("server"), os.environ["SUROGATE_DECISIONS_RETRY_ARTIFACT"],
           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "rune",
           "--max-model-len", "8192", "--max-num-seqs", "8", *ARGS, *args]
    url = f"http://127.0.0.1:{port}"
    with log.open("w") as output:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                assert process.poll() is None, log.read_text()
                try:
                    if requests.get(url + "/v1/models", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.2)
            else:
                pytest.fail(log.read_text())
            yield url, log
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def post(url, body):
    return requests.post(url + ROUTE, data=body.encode(), timeout=600, headers={"Content-Type": "application/json"})


def answers(url, body):
    response = post(url, body)
    assert response.ok, response.text
    out = response.json()
    return {"answers": out["answers"], "usage": out["usage"]}


def pages_in_use(url):
    return requests.get(url + "/kv_stats", timeout=10).json()["models"][0]["pages_in_use"]


def settle(url, expected):
    """KV pages in use once a one-question request has run after the faults: exactly what a clean
    server holds after the same requests (finished lanes stay retained, so the count depends on
    what ran; nothing a failed attempt held -- its saved prefix, its lanes' pages -- may add to it)."""
    answers(url, SINGLE)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if pages_in_use(url) == expected:
            return
        time.sleep(0.5)
    assert pages_in_use(url) == expected


_EXPECTED_PAGES = {}


def clean_pages(tmp_path_factory, *bodies):
    """Pages in use on a fresh clean server after `bodies` and then SINGLE."""
    if bodies not in _EXPECTED_PAGES:
        with serve(tmp_path_factory.mktemp("clean-pages"), "clean-pages") as (url, log):
            for body in bodies:
                post(url, body)
            answers(url, SINGLE)
            time.sleep(1)
            _EXPECTED_PAGES[bodies] = pages_in_use(url)
    return _EXPECTED_PAGES[bodies]


def warnings(log):
    return re.findall(r"\[req \d+\] decisions attempt (\d+) of (\d+) returned non-finite logits",
                      Path(log).read_text())


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("decisions-retry")
    with serve(tmp, "clean") as (url, log):
        recorded = {"shared": answers(url, SHARED), "single": answers(url, SINGLE), "whole": answers(url, WHOLE),
                    "neighbours": [answers(url, body) for body in NEIGHBOURS]}
        # Run to run on an idle server: the same answers again.
        assert answers(url, SHARED) == recorded["shared"]
        assert not warnings(log)
    return recorded


def test_poisoned_prefix_is_run_again_from_scratch(clean, tmp_path, tmp_path_factory):
    expected = clean_pages(tmp_path_factory, SHARED)  # before this test's server takes the GPU
    with serve(tmp_path, "poison-once", {"SUROGATE_SERVE_FAULT_POISON_GPU_PREFIX": "1"}) as (url, log):
        assert answers(url, SHARED) == clean["shared"]
        assert warnings(log) == [("1", "3")]
        assert re.search(r"done finish=\S+ .* questions=4 shared_prefix=\d+ attempts=2", Path(log).read_text())
        settle(url, expected)
        # The fault is spent: the next request runs once.
        assert answers(url, SHARED) == clean["shared"]
        assert warnings(log) == [("1", "3")]


def test_neighbours_are_unaffected(clean, tmp_path):
    with serve(tmp_path, "poison-neighbours", {"SUROGATE_SERVE_FAULT_POISON_GPU_PREFIX": "1"}) as (url, log):
        # The one request with a shared prefix takes the fault; single-question neighbours run
        # beside its attempts.
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            victim = pool.submit(answers, url, SHARED)
            neighbours = [pool.submit(answers, url, body) for body in NEIGHBOURS]
            assert victim.result() == clean["shared"]
            assert [f.result() for f in neighbours] == clean["neighbours"]
        assert warnings(log) == [("1", "3")]


def test_single_question_readout_is_run_again(clean, tmp_path, tmp_path_factory):
    expected = clean_pages(tmp_path_factory, SINGLE)  # before this test's server takes the GPU
    with serve(tmp_path, "nan-readout", {"SUROGATE_SERVE_FAULT_NAN_READOUT": "1"}) as (url, log):
        assert answers(url, SINGLE) == clean["single"]
        assert warnings(log) == [("1", "3")]
        settle(url, expected)


def test_whole_prompt_questions_are_prepared_again(clean, tmp_path, tmp_path_factory):
    expected = clean_pages(tmp_path_factory, WHOLE)  # before this test's server takes the GPU
    with serve(tmp_path, "nan-readout-whole", {"SUROGATE_SERVE_FAULT_NAN_READOUT": "1"}) as (url, log):
        assert answers(url, WHOLE) == clean["whole"]
        assert warnings(log) == [("1", "3")]
        assert re.search(r"questions=2 shared_prefix=0 attempts=2", Path(log).read_text())
        settle(url, expected)


def test_persistent_fault_returns_the_error_after_every_attempt(clean, tmp_path, tmp_path_factory):
    expected = clean_pages(tmp_path_factory, SHARED)  # before this test's server takes the GPU
    records = tmp_path / "requests.jsonl"
    with serve(tmp_path, "poison-always", {"SUROGATE_SERVE_FAULT_POISON_GPU_PREFIX": "1000000"},
               ["--request-log-jsonl", str(records)]) as (url, log):
        response = post(url, SHARED)
        assert response.status_code == 500, response.text
        assert "model returned non-finite logits" in response.json()["error"]["message"]
        assert warnings(log) == [("1", "3"), ("2", "3")]
        assert re.search(r"\] error model returned non-finite logits attempts=3", Path(log).read_text())
        errors = [r for r in map(json.loads, records.read_text().splitlines()) if r.get("event") == "request_error"]
        assert [r["request"]["decisions"]["attempts"] for r in errors] == [3], errors
        settle(url, expected)
        # The server serves on: a request without a shared prefix is unaffected by this fault.
        assert answers(url, SINGLE) == clean["single"]


def test_one_attempt_turns_retrying_off(clean, tmp_path):
    with serve(tmp_path, "one-attempt", {"SUROGATE_SERVE_FAULT_NAN_READOUT": "1"},
               ["--decision-attempts", "1"]) as (url, log):
        response = post(url, SINGLE)
        assert response.status_code == 500, response.text
        assert not warnings(log)
        assert answers(url, SINGLE) == clean["single"]
