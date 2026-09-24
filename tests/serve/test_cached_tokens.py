"""Cached prompt tokens in Chat Completions, Completions and Anthropic Messages usage. GPU; pending.

Skipped unless SUROGATE_CACHED_TOKENS_TEST_ARTIFACT names a prepared chat model (.sinfer); select one
free GPU with CUDA_VISIBLE_DEVICES, and add engine flags with SUROGATE_CACHED_TOKENS_TEST_ARGS (a JSON
list). Set SUROGATE_SERVE_BIN to the engine under test (from a worktree, the resolver can otherwise
find another checkout's build). The model must reuse a repeated prompt's prefix: a dense-attention
model does at any token boundary; a hybrid linear-attention model, MTP or DFlash only at the append
frontier or a checkpoint, and may legitimately report 0 on /v1/completions.

Each endpoint gets the same long prompt twice, streamed and not. The second answer must report
cached tokens, and the number must equal `prefix_cache_hit_tokens` in that request's `request_done`
record of --request-log-jsonl. OpenAI endpoints keep the whole prompt in prompt_tokens and report
the cached part in prompt_tokens_details.cached_tokens. Anthropic Messages report
input_tokens + cache_read_input_tokens == the whole prompt.
"""

import json
import os
import socket
import subprocess
import time
import uuid

import pytest
import requests

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(not os.getenv("SUROGATE_CACHED_TOKENS_TEST_ARTIFACT"),
                                reason="needs a prepared chat model and a free GPU")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    root = tmp_path_factory.mktemp("cached-tokens")
    log, jsonl = root / "server.log", root / "requests.jsonl"
    cmd = [_resolve_binary("server"), os.environ["SUROGATE_CACHED_TOKENS_TEST_ARTIFACT"],
           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "test",
           "--max-model-len", "4096", "--max-num-seqs", "2", "--request-log-jsonl", str(jsonl),
           *json.loads(os.getenv("SUROGATE_CACHED_TOKENS_TEST_ARGS", "[]"))]
    url = f"http://127.0.0.1:{port}"
    with log.open("w") as output:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
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
            yield url, jsonl
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def long_prompt():
    # A unique opening per test so nothing from an earlier test is cached, then enough text to
    # span several cache pages.
    lines = [f"Record {i}: the shipment for order {1000 + i} left the depot on day {i % 28 + 1}." for i in range(80)]
    return f"Session {uuid.uuid4()}.\n" + "\n".join(lines) + "\nHow many records are there? Answer briefly."


def last_done(jsonl):
    records = []
    for line in jsonl.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:  # a record still being written
            continue
    return [r for r in records if r["event"] == "request_done"][-1]["result"]["prefix_cache_hit_tokens"]


def sse_events(response):
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("data: ") and line != "data: [DONE]":
            yield json.loads(line[len("data: "):])


def openai_usage(url, path, body, stream):
    body = {**body, "model": "test", "max_tokens": 8, "temperature": 0}
    if not stream:
        response = requests.post(url + path, json=body, timeout=300)
        assert response.ok, response.text
        return response.json()["usage"]
    body.update(stream=True, stream_options={"include_usage": True})
    with requests.post(url + path, json=body, timeout=300, stream=True) as response:
        assert response.ok, response.text
        usages = [e["usage"] for e in sse_events(response) if e.get("usage")]
    assert len(usages) == 1
    return usages[0]


def messages_usage(url, prompt, stream):
    body = {"model": "test", "max_tokens": 8, "messages": [{"role": "user", "content": prompt}]}
    if not stream:
        response = requests.post(url + "/v1/messages", json=body, timeout=300)
        assert response.ok, response.text
        return response.json()["usage"]
    body["stream"] = True
    with requests.post(url + "/v1/messages", json=body, timeout=300, stream=True) as response:
        assert response.ok, response.text
        events = list(sse_events(response))
    start = next(e for e in events if e["type"] == "message_start")["message"]["usage"]
    final = next(e for e in events if e["type"] == "message_delta")["usage"]
    # message_start is sent once the prompt is prefilled and already carries the real split,
    # because gateways read input tokens from it alone.
    assert start["input_tokens"] == final["input_tokens"]
    assert start["cache_read_input_tokens"] == final["cache_read_input_tokens"]
    assert start["output_tokens"] == 0
    return final


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_openai_endpoints_report_cached_tokens(server, path, stream):
    url, jsonl = server
    prompt = long_prompt()
    body = {"messages": [{"role": "user", "content": prompt}]} if "chat" in path else {"prompt": prompt}
    first = openai_usage(url, path, body, stream)
    second = openai_usage(url, path, body, stream)
    cached = second["prompt_tokens_details"]["cached_tokens"]
    assert second["prompt_tokens"] == first["prompt_tokens"]  # the whole prompt either way
    if path == "/v1/completions" and cached == 0 == last_done(jsonl):
        pytest.skip("this model reuses no prefix on /v1/completions (hybrid, MTP or DFlash)")
    assert 0 < cached <= second["prompt_tokens"]
    assert cached == last_done(jsonl)
    assert first["prompt_tokens_details"]["cached_tokens"] < cached


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_messages_report_cache_reads(server, stream):
    url, jsonl = server
    prompt = long_prompt()
    first = messages_usage(url, prompt, stream)
    second = messages_usage(url, prompt, stream)
    whole = first["input_tokens"] + first["cache_read_input_tokens"]
    assert second["input_tokens"] + second["cache_read_input_tokens"] == whole
    assert 0 < second["cache_read_input_tokens"] <= whole
    assert second["cache_read_input_tokens"] == last_done(jsonl)
    assert second["cache_creation_input_tokens"] == 0
