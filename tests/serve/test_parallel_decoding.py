"""Native PCD integration. Set SUROGATE_PCD_TEST_ARTIFACT; select GPUs with CUDA_VISIBLE_DEVICES."""

import json
import math
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(not os.getenv("SUROGATE_PCD_TEST_ARTIFACT"), reason="needs a prepared text model")


@pytest.fixture(scope="module", params=[1, 4])
def server(request, tmp_path_factory):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    root = tmp_path_factory.mktemp(f"pcd-{request.param}")
    log = root / "server.log"
    cmd = [_resolve_binary("server"), os.environ["SUROGATE_PCD_TEST_ARTIFACT"],
           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "test",
           "--max-model-len", "2048", "--kv-capacity", str(2048 * request.param),
           "--max-num-seqs", str(request.param), "--no-thinking", "--enable-sleep-mode"]
    cmd += json.loads(os.getenv("SUROGATE_PCD_TEST_ARGS", "[]"))
    base = f"http://127.0.0.1:{port}"
    with log.open("w") as output:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                assert process.poll() is None, log.read_text()
                try:
                    if requests.get(base + "/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.1)
            else:
                pytest.fail(log.read_text())
            yield base
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def body(properties=None):
    properties = properties or {
        "happy": {"type": "boolean", "description": "The customer is happy."},
        "priority": {"type": "string", "enum": ["urgent", "not urgent", "not applicable"]},
        "number": {"type": "integer", "enum": [1, 10, 100]},
        "unicode": {"type": "string", "enum": ["日本語", "quote\"newline\n", "🦊"]},
    }
    return {"model": "test", "messages": [{"role": "user", "content": "The customer is upset. Fix their server now."}],
            "max_tokens": 512, "temperature": 0, "parallel_decoding": True,
            "response_format": {"type": "json_schema", "json_schema": {"name": "classification", "strict": True,
                "schema": {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}}}}


def ask(server, payload):
    response = requests.post(server + "/v1/chat/completions", json=payload, timeout=120)
    assert response.ok, response.text
    return response.json()


def validate(result, payload):
    assert result["choices"][0]["finish_reason"] == "stop"
    value = json.loads(result["choices"][0]["message"]["content"])
    fields = result["parallel_decoding"]["fields"]
    properties = payload["response_format"]["json_schema"]["schema"]["properties"]
    assert set(fields) == set(value) == set(properties)
    for key, spec in properties.items():
        choices = spec.get("enum", [False, True])
        assert value[key] in choices
        probabilities = fields[key]["probabilities"]
        assert [p["value"] for p in probabilities] == choices
        assert all(math.isfinite(p["probability"]) and 0 <= p["probability"] <= 1 for p in probabilities)
        assert sum(p["probability"] for p in probabilities) == pytest.approx(1)
        assert fields[key]["value"] == value[key] == max(probabilities, key=lambda p: p["probability"])["value"]
    assert result["usage"]["completion_tokens"] > 0
    return fields


def test_colliding_choices_and_reuse(server):
    payload = body()
    first = validate(ask(server, payload), payload)
    second = validate(ask(server, payload), payload)
    for field in first:
        for a, b in zip(first[field]["probabilities"], second[field]["probabilities"]):
            assert a["probability"] == pytest.approx(b["probability"], abs=.03)
    # A following regular request must not inherit classifier readouts or masks.
    regular = ask(server, {"model": "test", "messages": [{"role": "user", "content": "Hello"}],
                          "max_tokens": 2, "temperature": 0})
    assert "parallel_decoding" not in regular


def test_streaming_and_constants(server):
    payload = body({"constant": {"enum": ["quote\" 日本語"]}, "bool": {"enum": [True]}})
    payload.update(stream=True, stream_options={"include_usage": True})
    response = requests.post(server + "/v1/chat/completions", json=payload, timeout=120)
    assert response.ok, response.text
    response.encoding = "utf-8"
    chunks = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert json.loads(content) == {"constant": "quote\" 日本語", "bool": True}
    final = next(c for c in chunks if "parallel_decoding" in c)
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["parallel_decoding"]["fields"]["bool"]["probabilities"] == [{"value": True, "probability": 1}]
    assert chunks[-1]["usage"]["completion_tokens"] > 0
    assert response.text.endswith("data: [DONE]\n\n")


@pytest.mark.parametrize("extra", [
    {"logprobs": True}, {"return_token_ids": True}, {"min_tokens": 1}, {"stop": "x"},
    {"top_p": .9}, {"logit_bias": {"1": 2}}, {"parallel_decoding": "true"}, {"max_tokens": 1},
    {"response_format": {"type": "json_object"}}, {"response_format": None},
])
def test_incompatible_options(server, extra):
    payload = body(); payload.update(extra)
    response = requests.post(server + "/v1/chat/completions", json=payload, timeout=30)
    assert response.status_code == 400, response.text
    assert requests.get(server + "/health", timeout=5).ok


def test_more_than_twenty_choices(server):
    payload = body({"label": {"type": "integer", "enum": list(range(32))}})
    validate(ask(server, payload), payload)


def test_concurrent_and_sleep(server):
    payloads = [body({f"flag_{i}": {"type": "boolean"}}) for i in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda p: ask(server, p), payloads))
    for result, payload in zip(results, payloads):
        validate(result, payload)
    assert requests.post(server + "/sleep", timeout=30).ok
    assert requests.post(server + "/wake_up", timeout=30).ok
    validate(ask(server, body()), body())


def test_regular_json_after_classification(server):
    # Includes SentencePiece tokenizers with an automatic BOS/dummy space; the
    # grammar's continuation tokenizer must not insert either into forced bytes.
    payload = body({"done": {"type": "boolean"}})
    payload["parallel_decoding"] = False
    result = ask(server, payload)
    value = json.loads(result["choices"][0]["message"]["content"])
    assert set(value) == {"done"} and type(value["done"]) is bool
    assert "parallel_decoding" not in result
