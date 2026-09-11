"""Opt-in mixed-round isolation checks; reserve a GPU and set SUROGATE_MIXED_TEST_ARTIFACT.

SUROGATE_MIXED_TEST_ARGS is a JSON array of additional server options. It can
enable LoRA, speculation, eager execution or a pipeline on the reserved devices.
SUROGATE_MIXED_TEST_MODELS is a JSON array of model/adapter names (default: test).
"""

import json
import math
import os
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(
    not os.getenv("SUROGATE_MIXED_TEST_ARTIFACT"), reason="requires a prepared GPU artifact"
)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("mixed-rounds")
    log = root / "server.log"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = [
        _resolve_binary("server"), os.environ["SUROGATE_MIXED_TEST_ARTIFACT"],
        "--port", str(port), "--served-model-name", "test", "--no-thinking",
        "--max-model-len", "2048", "--kv-capacity", "8192", "--max-num-seqs", "4",
        "--max-num-batched-tokens", "256", "--no-prefix-reuse",
    ] + json.loads(os.getenv("SUROGATE_MIXED_TEST_ARGS", "[]"))
    env = dict(os.environ, SUROGATE_SERVE_ROUND_TRACE="1")
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 180
            while True:
                assert process.poll() is None, log.read_text()
                try:
                    if requests.get(url + "/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                assert time.monotonic() < deadline, log.read_text()
                time.sleep(.1)
            yield url, log
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def body(model, index, length, output):
    return {
        "model": model, "tokens": [100 + index, 110 + index, 120 + index] * length,
        "messages": [{"role": "user", "content": "Continue."}],
        "max_tokens": output, "temperature": 0, "ignore_eos": True,
        "return_token_ids": True, "logprobs": True, "top_logprobs": 0,
    }


def ask(server, payload):
    response = requests.post(server[0] + "/v1/chat/completions", json=payload, timeout=180)
    assert response.ok, response.text + "\n" + server[1].read_text()[-4000:]
    return response.json()


def streaming(server, payload, started, cancel=False):
    chunks = []
    times = []
    with requests.post(server[0] + "/v1/chat/completions", json=dict(payload, stream=True),
                       stream=True, timeout=180) as response:
        assert response.ok, response.text
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            assert "error" not in chunk, chunk
            chunks.append(chunk)
            if chunk["choices"] and chunk["choices"][0].get("logprobs"):
                times.append(time.monotonic())
                started.set()
                if cancel:
                    return chunks, times
    return chunks, times


def check_scores(server, payload, prompt, tokens, generated):
    replay = ask(server, dict(payload, tokens=prompt + tokens, max_tokens=1, prompt_logprobs=0))
    assert len(tokens) == len(generated)
    errors = []
    for index, score in enumerate(generated):
        expected = replay["prompt_logprobs"][len(prompt) + index][str(tokens[index])]["logprob"]
        errors.append(abs(score["logprob"] - expected))
    assert all(math.isfinite(error) for error in errors), errors
    options = json.loads(os.getenv("SUROGATE_MIXED_TEST_ARGS", "[]"))
    if "--spec-adaptive" in options:
        # Use the same peak and average bounds as the existing adaptive vision
        # regression: changing batch/verification widths changes rounding.
        peak = .25 if "fp8" in options else .15
        assert max(errors) <= peak, errors
        assert sum(errors) / len(errors) <= .035, errors
    else:
        assert max(errors) <= .12, errors


def test_overlapping_prefill_preserves_decode_and_adapters(server, record_property):
    models = json.loads(os.getenv("SUROGATE_MIXED_TEST_MODELS", '["test"]'))
    long_decode = body(models[0], 0, 8, 192)
    incoming = [body(models[i % len(models)], i + 1, 170 + i * 7, 24) for i in range(3)]
    started = threading.Event()
    offset = server[1].stat().st_size
    with ThreadPoolExecutor(max_workers=4) as pool:
        running = pool.submit(streaming, server, long_decode, started)
        assert started.wait(60), server[1].read_text()[-4000:]
        pending = [pool.submit(ask, server, payload) for payload in incoming]
        results = [future.result() for future in pending]
        chunks, times = running.result()
    log = server[1].read_text()[offset:]
    assert "round-trace: mixed staged lane=" in log, log[-4000:]
    options = json.loads(os.getenv("SUROGATE_MIXED_TEST_ARGS", "[]"))
    if "dflash" in options and "--spec-adaptive" not in options:
        width = int(options[options.index("--draft-tokens") + 1]) + 1
        assert f"mixed graph_hit=0 verify_width={width}" in log, log[-4000:]
    tokens = [token for chunk in chunks for choice in chunk["choices"] for token in choice.get("token_ids", [])]
    scores = [entry for chunk in chunks for choice in chunk["choices"]
              for entry in (choice.get("logprobs") or {}).get("content", [])]
    assert len(tokens) == long_decode["max_tokens"], chunks
    check_scores(server, long_decode, long_decode["tokens"], tokens, scores)
    for payload, result in zip(incoming, results):
        choice = result["choices"][0]
        check_scores(server, payload, result["prompt_token_ids"], choice["token_ids"], choice["logprobs"]["content"])
    gaps = sorted(b - a for a, b in zip(times, times[1:]))
    record_property("p95_stream_chunk_gap_ms", gaps[int(.95 * (len(gaps) - 1))] * 1000)


def test_cancelled_lane_does_not_change_next_request(server):
    models = json.loads(os.getenv("SUROGATE_MIXED_TEST_MODELS", '["test"]'))
    payload = body(models[-1], 4, 16, 16)
    expected = ask(server, payload)
    started = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(streaming, server, body(models[0], 5, 8, 512), started, True)
        assert started.wait(60)
        running.result()
        actual = ask(server, payload)
    assert actual["choices"][0]["token_ids"] == expected["choices"][0]["token_ids"]
    if "--spec-adaptive" in json.loads(os.getenv("SUROGATE_MIXED_TEST_ARGS", "[]")):
        # Calibration can change the draft width between these two requests.
        choice = actual["choices"][0]
        check_scores(server, payload, actual["prompt_token_ids"], choice["token_ids"], choice["logprobs"]["content"])
        return
    assert [s["logprob"] for s in actual["choices"][0]["logprobs"]["content"]] == pytest.approx(
        [s["logprob"] for s in expected["choices"][0]["logprobs"]["content"]], abs=.02)


def test_images_arriving_during_decode_keep_their_own_content(server):
    if "--vision" not in json.loads(os.getenv("SUROGATE_MIXED_TEST_ARGS", "[]")):
        pytest.skip("requires a vision artifact and --vision")
    from tests.serve.test_dflash_vision import image_part

    models = json.loads(os.getenv("SUROGATE_MIXED_TEST_MODELS", '["test"]'))
    prompts = []
    for index, (color, size) in enumerate([("red", 64), ("blue", 512), ("green", 256)]):
        prompts.append({
            "model": models[index % len(models)], "messages": [{"role": "user", "content": [
                image_part(color, size),
                {"type": "text", "text": "Name the background color. Answer with one color word."},
            ]}], "max_tokens": 16, "temperature": 0, "return_token_ids": True,
            "logprobs": True, "top_logprobs": 0,
        })
    expected = [ask(server, prompt) for prompt in prompts]
    started = threading.Event()
    offset = server[1].stat().st_size
    with ThreadPoolExecutor(max_workers=4) as pool:
        running = pool.submit(streaming, server, body("test", 6, 8, 192), started)
        assert started.wait(60)
        results = list(pool.map(lambda prompt: ask(server, prompt), prompts))
        running.result()
    assert "round-trace: mixed staged lane=" in server[1].read_text()[offset:]
    for actual, reference in zip(results, expected):
        assert actual["prompt_token_ids"] == reference["prompt_token_ids"]
        assert actual["choices"][0]["token_ids"] == reference["choices"][0]["token_ids"]
        assert [s["logprob"] for s in actual["choices"][0]["logprobs"]["content"]] == pytest.approx(
            [s["logprob"] for s in reference["choices"][0]["logprobs"]["content"]], abs=.12)


def test_dflash_rotates_more_than_eight_active_requests(server):
    options = json.loads(os.getenv("SUROGATE_MIXED_TEST_ARGS", "[]"))
    if "dflash" not in options or "--max-num-seqs" not in options or int(options[options.index("--max-num-seqs") + 1]) <= 8:
        pytest.skip("requires DFlash with more than eight request lanes")
    ready = threading.Barrier(12)

    def run(index):
        ready.wait(timeout=30)
        return ask(server, body("test", index, 8, 256))

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(run, range(12)))
    for result in results:
        assert len(result["choices"][0]["token_ids"]) == 256
        assert all(math.isfinite(s["logprob"]) for s in result["choices"][0]["logprobs"]["content"])
