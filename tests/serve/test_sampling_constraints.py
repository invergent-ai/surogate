"""Opt-in serving checks; reserve GPUs and set SUROGATE_SAMPLING_TEST_ARTIFACT."""

import json
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(
    not os.getenv("SUROGATE_SAMPLING_TEST_ARTIFACT"), reason="requires a prepared GPU artifact"
)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    binary = _resolve_binary("server")
    assert binary
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log = tmp_path_factory.mktemp("sampling-server") / "server.log"
    command = [
        binary, os.environ["SUROGATE_SAMPLING_TEST_ARTIFACT"], "--port", str(port),
        "--served-model-name", "test", "--max-model-len", "512", "--kv-capacity", "2048",
        "--max-num-seqs", "4", "--no-thinking", "--enable-sleep-mode",
    ]
    if devices := os.getenv("SUROGATE_SAMPLING_TEST_DEVICES"):
        command += ["--devices", devices]
    if backend := os.getenv("SUROGATE_SAMPLING_TEST_SPEC"):
        command += ["--spec", backend, "--draft-tokens", "3", "--kv-cache-dtype", "bf16"]
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while True:
                assert process.poll() is None, log.read_text()
                try:
                    if requests.get(base + "/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                assert time.monotonic() < deadline, log.read_text()
                time.sleep(0.1)
            yield base
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def chat(server, **extra):
    body = {
        "model": "test", "messages": [{"role": "user", "content": "Say hello in plain text."}],
        "max_tokens": 128, "temperature": 0, "return_token_ids": True,
    }
    body.update(extra)
    return requests.post(server + "/v1/chat/completions", json=body, timeout=90)


def schema_format(schema):
    return {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}}


@pytest.mark.parametrize("temperature", [0, 1])
def test_logit_bias_greedy_and_stochastic(server, temperature):
    for token in (100, 101):
        response = chat(server, logit_bias={str(token): 100}, temperature=temperature,
                        top_k=0, top_p=1, max_tokens=12, ignore_eos=True)
        assert response.ok, response.text
        assert response.json()["choices"][0]["token_ids"] == [token] * 12
    baseline = chat(server, max_tokens=1).json()["choices"][0]["token_ids"][0]
    response = chat(server, max_tokens=1, logit_bias={str(baseline): -100})
    assert response.ok, response.text
    assert response.json()["choices"][0]["token_ids"][0] != baseline


def test_logit_bias_concurrent_isolation(server):
    def ask(token):
        response = chat(server, logit_bias={str(token): 100}, max_tokens=12, ignore_eos=True)
        assert response.ok, response.text
        return response.json()["choices"][0]["token_ids"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(ask, [100, 101, 100, 101])) == [[t] * 12 for t in [100, 101, 100, 101]]


@pytest.mark.parametrize("bias", [{"-1": 1}, {"100x": 1}, {"999999999": 1}, {"100": 101}, {"100": -101}])
def test_invalid_bias_rejected(server, bias):
    response = chat(server, logit_bias=bias)
    assert response.status_code == 400, response.text


@pytest.mark.parametrize("temperature", [0, 1])
def test_json_schema_nested_output(server, temperature):
    schema = {
        "type": "object", "properties": {
            "label": {"type": "string", "enum": ["yes", "no"]},
            "count": {"type": "integer", "minimum": 1, "maximum": 3},
            "flags": {"type": "array", "items": {"type": "boolean"}, "minItems": 2, "maxItems": 2},
        }, "required": ["label", "count", "flags"], "additionalProperties": False,
    }
    response = chat(server, response_format=schema_format(schema), temperature=temperature, top_k=0, top_p=1)
    assert response.ok, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop", response.text
    value = json.loads(choice["message"]["content"])
    assert set(value) == {"label", "count", "flags"}
    assert value["label"] in {"yes", "no"}
    assert type(value["count"]) is int and 1 <= value["count"] <= 3
    assert len(value["flags"]) == 2 and all(type(flag) is bool for flag in value["flags"])


def test_constraint_isolation_cache_and_sleep(server):
    def ask(value):
        response = chat(server, response_format=schema_format({"const": value}))
        assert response.ok, response.text
        return json.loads(response.json()["choices"][0]["message"]["content"])
    values = [{"answer": "first"}, {"answer": "second"}, [1, True], "quoted\"text", {"emoji": "🦊", "line": "a\nb"}]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(ask, values)) == values
    assert requests.post(server + "/sleep", timeout=30).ok
    assert requests.post(server + "/wake_up", timeout=30).ok
    for value in reversed(values):
        assert ask(value) == value
    # The reused lane must drop its previous mask when the next request has no constraint.
    response = chat(server, logit_bias={"100": 100}, max_tokens=4, ignore_eos=True)
    assert response.ok, response.text
    assert response.json()["choices"][0]["token_ids"] == [100] * 4


def test_json_object_and_streaming(server):
    response = chat(server, response_format={"type": "json_object"}, messages=[
        {"role": "user", "content": 'Return a JSON object with "ok": true.'}
    ])
    assert response.ok, response.text
    assert isinstance(json.loads(response.json()["choices"][0]["message"]["content"]), dict)
    expected = {"streamed": True}
    response = chat(server, stream=True, response_format=schema_format({"const": expected}))
    assert response.ok, response.text
    pieces = []
    for line in response.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            for choice in json.loads(line[6:])["choices"]:
                pieces.append(choice["delta"].get("content", ""))
    assert json.loads("".join(pieces)) == expected


@pytest.mark.parametrize("schema", [
    {"type": "integer", "multipleOf": 3}, {"not": {"type": "null"}},
    {"allOf": [{"type": "string"}, {"const": "a"}]},
    {"$ref": "https://example.com/schema"}, {"type": "nonsense"},
])
def test_unsupported_schema_rejected(server, schema):
    response = chat(server, response_format=schema_format(schema))
    assert response.status_code == 400, response.text


def test_mixed_constrained_and_biased_requests(server):
    def ask(constrained):
        expected = {"safe": True}
        extra = {"response_format": schema_format({"const": expected})} if constrained else {"max_tokens": 8, "ignore_eos": True}
        response = chat(server, logit_bias={"100": 100}, **extra)
        assert response.ok, response.text
        choice = response.json()["choices"][0]
        if constrained:
            assert json.loads(choice["message"]["content"]) == expected
        else:
            assert choice["token_ids"] == [100] * 8
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(ask, [True, False, True, False]))


@pytest.mark.parametrize("extra", [{"ignore_eos": True}, {"min_tokens": 1}, {"stop": ["\""]}])
def test_conflicting_structured_options_rejected(server, extra):
    response = chat(server, response_format=schema_format({"const": {"ok": True}}), **extra)
    assert response.status_code == 400, response.text
