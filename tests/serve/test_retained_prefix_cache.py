"""Completed-prefix isolation across unrelated requests, including hybrid/MTP state.

Set SUROGATE_PREFIX_CACHE_TEST_ARTIFACT to a prepared artifact and reserve its GPUs.
SUROGATE_PREFIX_CACHE_TEST_ARGS may enable MTP, DFlash, adapters, or a pipeline.
"""
import json
import os
import re
import socket
import subprocess
import time

import pytest
import requests

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(
    not os.getenv("SUROGATE_PREFIX_CACHE_TEST_ARTIFACT"), reason="requires a GPU artifact"
)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    log = tmp_path_factory.mktemp("retained-prefix") / "server.log"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    args = [
        _resolve_binary("server"), os.environ["SUROGATE_PREFIX_CACHE_TEST_ARTIFACT"],
        "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "test",
        "--max-model-len", "8192", "--kv-capacity", "8192", "--max-num-seqs", "1",
    ] + json.loads(os.getenv("SUROGATE_PREFIX_CACHE_TEST_ARGS", "[]"))
    with log.open("w") as output:
        process = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 240
            while True:
                assert process.poll() is None, log.read_text()[-6000:]
                try:
                    if requests.get(url + "/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                assert time.monotonic() < deadline, log.read_text()[-6000:]
                time.sleep(.2)
            yield url, log
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def ask(server, tokens):
    response = requests.post(server[0] + "/v1/chat/completions", json={
        "model": "test", "messages": [{"role": "user", "content": "Continue."}],
        "tokens": tokens, "max_tokens": 24, "temperature": 0, "seed": 42,
        "ignore_eos": True, "return_token_ids": True, "logprobs": True, "top_logprobs": 0,
    }, timeout=180)
    assert response.ok, response.text + "\n" + server[1].read_text()[-4000:]
    result = response.json()
    assert "error" not in result, result
    return result["choices"][0]


def test_unrelated_request_preserves_completed_prefix(server):
    # Cross a prefill chunk and the Qwen/GLM sparse-indexer threshold.
    prompt = [101, 110, 120, 130] * (int(os.getenv("SUROGATE_PREFIX_CACHE_TEST_TOKENS", "4096")) // 4)
    unrelated = [201, 210, 220, 230] * 16
    first = ask(server, prompt)
    suffix = prompt + first["token_ids"] + [140, 150, 160] * 11
    resident = ask(server, suffix)
    ask(server, unrelated)
    repeated = ask(server, prompt)
    assert repeated["token_ids"] == first["token_ids"]
    ask(server, unrelated)
    restored = ask(server, suffix)
    assert restored["token_ids"] == resident["token_ids"]
    expected = [item["logprob"] for item in resident["logprobs"]["content"]]
    actual = [item["logprob"] for item in restored["logprobs"]["content"]]
    assert actual == pytest.approx(expected, abs=1e-5)
    done = [line for line in server[1].read_text().splitlines() if "] done finish=" in line]
    assert int(re.search(r" cache=(\d+)", done[-1])[1]) >= len(prompt), done[-1]
    assert requests.get(server[0] + "/health", timeout=5).ok


def test_rewritten_turn_restores_after_unrelated_request(server):
    # A new user turn can remove earlier assistant thinking from the rendered
    # prompt. Reuse must restore the saved state at that boundary, not a state
    # belonging to the end of the longer, original conversation.
    messages = [
        {"role": "system", "content": "These are background notes. " * 256},
        {"role": "user", "content": "Name the capital of France."},
    ]

    def chat(history):
        response = requests.post(server[0] + "/v1/chat/completions", json={
            "model": "test", "messages": history, "max_tokens": 24,
            "temperature": 0, "seed": 42, "ignore_eos": True,
            "return_token_ids": True, "logprobs": True, "top_logprobs": 0,
            "chat_template_kwargs": {"preserve_thinking": False},
        }, timeout=180)
        assert response.ok, response.text + "\n" + server[1].read_text()[-4000:]
        result = response.json()
        assert "error" not in result, result
        return result["choices"][0]

    first = chat(messages)
    continuation = messages + [first["message"],
                               {"role": "user", "content": "Name its country as well."}]
    resident = chat(continuation)
    ask(server, [301, 310, 320, 330] * 16)
    restored = chat(continuation)
    assert restored["token_ids"] == resident["token_ids"]
    assert [item["logprob"] for item in restored["logprobs"]["content"]] == pytest.approx(
        [item["logprob"] for item in resident["logprobs"]["content"]], abs=1e-5)
    done = [line for line in server[1].read_text().splitlines() if "] done finish=" in line]
    assert int(re.search(r" cache=(\d+)", done[-1])[1]) > 256, done[-1]
    assert "reuse=full_reset" not in done[-1], done[-1]


@pytest.mark.skipif(not os.getenv("SUROGATE_PREFIX_CACHE_TEST_AGENTIC"),
                    reason="requires a tool-capable model with a 32K context")
def test_long_tool_conversation_survives_detours(server):
    tools = [{"type": "function", "function": {
        "name": "get_weather" if i == 0 else f"lookup_record_{i}",
        "description": "Get current weather for a city." if i == 0 else "Look up an archived record.",
        "parameters": {"type": "object", "properties": {
            "city" if i == 0 else "record_id": {"type": "string"}},
            "required": ["city" if i == 0 else "record_id"]},
    }} for i in range(38)]
    messages = [{"role": "system", "content":
                 "Use tools for current weather. The notes below are historical background.\n" +
                 "An archived example mentions a mild breeze across a city. It is not current weather.\n" * 1100},
                {"role": "user", "content": "What is the current weather in Paris? Use get_weather."}]

    def chat():
        response = requests.post(server[0] + "/v1/chat/completions", json={
            "model": "test", "messages": messages, "tools": tools, "tool_choice": "auto",
            "max_tokens": 256, "temperature": .3, "top_p": .95, "seed": 42,
            "chat_template_kwargs": {"preserve_thinking": False},
        }, timeout=240)
        assert response.ok, response.text
        result = response.json()
        assert "error" not in result, result
        done = [line for line in server[1].read_text().splitlines() if "] done finish=" in line]
        print(done[-1], flush=True)
        return result, done[-1]

    first, _ = chat()
    assert first["usage"]["prompt_tokens"] > 20000
    message = first["choices"][0]["message"]
    assert message.get("tool_calls"), message
    messages.append(message)
    for call in message["tool_calls"]:
        messages.append({"role": "tool", "tool_call_id": call["id"],
                         "content": '{"city":"Paris","temperature_c":22,"condition":"clear"}'})
    ask(server, [401, 410, 420, 430] * 16)
    second, log = chat()
    assert int(re.search(r" cache=(\d+)", log)[1]) > 20000, log
    messages.append(second["choices"][0]["message"])
    for call in messages[-1].get("tool_calls", []):
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": "No additional data."})
    messages.append({"role": "user", "content": "Summarize that weather in one sentence."})
    ask(server, [501, 510, 520, 530] * 16)
    _, log = chat()
    assert int(re.search(r" cache=(\d+)", log)[1]) > 20000, log
