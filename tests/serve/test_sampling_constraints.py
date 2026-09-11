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


class TestServer(str):
    __test__ = False


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    binary = _resolve_binary("server")
    assert binary
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log = tmp_path_factory.mktemp("sampling-server") / "server.log"
    records = log.with_name("requests.jsonl")
    command = [
        binary, os.environ["SUROGATE_SAMPLING_TEST_ARTIFACT"], "--port", str(port),
        "--served-model-name", "test", "--max-model-len", "512", "--kv-capacity", "2048",
        "--max-num-seqs", "4", "--no-thinking", "--enable-sleep-mode",
        "--request-log-jsonl", str(records), "--enable-auto-tool-choice",
        "--tool-call-parser", os.getenv("SUROGATE_SAMPLING_TEST_TOOL_PARSER", "qwen3_xml"),
    ]
    if devices := os.getenv("SUROGATE_SAMPLING_TEST_DEVICES"):
        command += ["--devices", devices]
    if backend := os.getenv("SUROGATE_SAMPLING_TEST_SPEC"):
        command += ["--spec", backend, "--draft-tokens",
                    os.getenv("SUROGATE_SAMPLING_TEST_DRAFT_TOKENS", "3"),
                    "--kv-cache-dtype", os.getenv("SUROGATE_SAMPLING_TEST_KV_DTYPE", "bf16")]
    elif dtype := os.getenv("SUROGATE_SAMPLING_TEST_KV_DTYPE"):
        command += ["--kv-cache-dtype", dtype]
    if os.getenv("SUROGATE_SAMPLING_TEST_ADAPTIVE") == "1":
        command += ["--spec-adaptive"]
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
            handle = TestServer(base)
            handle.records = records
            yield handle
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
@pytest.mark.parametrize(("top_k", "top_p"), [(0, 1), (0, 0.9), (64, 0.9)])
def test_json_schema_nested_output(server, temperature, top_k, top_p):
    schema = {
        "type": "object", "properties": {
            "label": {"type": "string", "enum": ["yes", "no"]},
            "count": {"type": "integer", "minimum": 1, "maximum": 3},
            "flags": {"type": "array", "items": {"type": "boolean"}, "minItems": 2, "maxItems": 2},
        }, "required": ["label", "count", "flags"], "additionalProperties": False,
    }
    response = chat(server, response_format=schema_format(schema), temperature=temperature, top_k=top_k, top_p=top_p)
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
    {"type": "array", "uniqueItems": True}, {"not": {"type": "null"}},
    {"oneOf": [{"type": "integer"}, {"type": "number"}]},
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


@pytest.mark.parametrize("schema,expected", [
    ({"allOf": [{"type": "integer", "minimum": 9}, {"maximum": 9}]}, 9),
    ({"type": "integer", "minimum": 2, "maximum": 4, "multipleOf": 3}, 3),
    ({"type": "string", "pattern": "^[A-Z]+$", "minLength": 3, "maxLength": 3}, None),
    ({"type": "string", "format": "date"}, None),
])
def test_extended_schemas(server, schema, expected):
    response = chat(server, response_format=schema_format(schema))
    assert response.ok, response.text
    value = json.loads(response.json()["choices"][0]["message"]["content"])
    if expected is not None:
        assert value == expected
    elif "pattern" in schema:
        assert len(value) == 3 and value.isascii() and value.isalpha() and value.isupper()
    else:
        import datetime
        datetime.date.fromisoformat(value)


def test_responses_structured_output(server):
    expected = {"answer": "structured", "ok": True}
    fmt = {"type": "json_schema", "name": "result", "strict": True, "schema": {"const": expected}}
    payload = {"model": "test", "input": "Say hello.", "text": {"format": fmt}, "max_output_tokens": 128}
    response = requests.post(server + "/v1/responses", json=payload, timeout=90)
    assert response.ok, response.text
    result = response.json()
    assert result["text"]["format"] == fmt
    text = "".join(part["text"] for item in result["output"] if item["type"] == "message" for part in item["content"])
    assert json.loads(text) == expected
    assert result["store"] is False
    response = requests.post(server + "/v1/responses", json={**payload, "stream": True}, timeout=90)
    assert response.ok, response.text
    pieces = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event["type"] == "response.output_text.delta":
                pieces.append(event["delta"])
    assert json.loads("".join(pieces)) == expected


@pytest.mark.skipif(not os.getenv("SUROGATE_SAMPLING_TEST_SPEC"), reason="requires speculative backend")
def test_constrained_drafts_are_accepted(server):
    expected = {"numbers": list(range(20))}
    response = chat(server, messages=[{"role": "user", "content": "Return exactly this JSON: " + json.dumps(expected)}],
                    response_format=schema_format({"const": expected}))
    assert response.ok, response.text
    assert json.loads(response.json()["choices"][0]["message"]["content"]) == expected
    deadline = time.monotonic() + 5
    while True:
        done = [json.loads(line) for line in server.records.read_text().splitlines() if '"request_done"' in line]
        if done and done[-1]["speculative"]["accepted_tokens"] > 0:
            assert done[-1]["speculative"]["rounds"] > 0
            break
        assert time.monotonic() < deadline, done[-1] if done else "no request log"
        time.sleep(0.05)


@pytest.mark.skipif(os.getenv("SUROGATE_SAMPLING_TEST_ADAPTIVE") != "1", reason="requires adaptive DFlash")
def test_adaptive_dflash_uses_multiple_windows_and_target_only_steps(server):
    response = chat(server, logit_bias={"100": 100}, max_tokens=96, ignore_eos=True)
    assert response.ok, response.text
    ids = response.json()["choices"][0]["token_ids"]
    assert len(ids) == 96 and set(ids) == {100}
    done = [json.loads(line) for line in server.records.read_text().splitlines()
            if json.loads(line).get("event") == "request_done"][-1]
    windows = done["speculative"]["rounds_per_draft_window"]
    assert windows[0] > 0, windows
    assert sum(count > 0 for count in windows) > 1, windows
