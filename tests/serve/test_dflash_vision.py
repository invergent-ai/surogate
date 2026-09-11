"""Opt-in real-model checks; set SUROGATE_DFLASH_VISION_TEST_ARTIFACT and reserve a GPU.

The artifact must contain Qwen3.5 vision weights and its matching DFlash drafter.
SUROGATE_DFLASH_VISION_TEST_KV_DTYPE selects bf16 (default) or fp8.
SUROGATE_DFLASH_VISION_TEST_DRAFT_TOKENS defaults to the maximum window of 15.
"""

import base64
import io
import json
import os
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from PIL import Image

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(
    not os.getenv("SUROGATE_DFLASH_VISION_TEST_ARTIFACT"), reason="requires a paired vision/DFlash artifact"
)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    artifact = os.environ["SUROGATE_DFLASH_VISION_TEST_ARTIFACT"]
    draft_tokens = os.getenv("SUROGATE_DFLASH_VISION_TEST_DRAFT_TOKENS", "15")
    adaptive = os.getenv("SUROGATE_DFLASH_VISION_TEST_ADAPTIVE", "false")
    root = tmp_path_factory.mktemp("dflash-vision")
    log, records = root / "server.log", root / "requests.jsonl"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = [
        _resolve_binary("server"), artifact, "--port", str(port), "--served-model-name", "base",
        "--vision", "--no-thinking", "--enable-sleep-mode",
        "--max-model-len", "4096", "--kv-capacity", "8192", "--max-num-seqs", "4",
        "--max-num-batched-tokens", "128", "--media-cache-mib", "16", "--media-live-mib", "64",
        "--kv-cache-dtype", os.getenv("SUROGATE_DFLASH_VISION_TEST_KV_DTYPE", "bf16"),
        "--model", f"draft={artifact},spec=dflash,draft-tokens={draft_tokens},spec-adaptive={adaptive}",
        "--request-log-jsonl", str(records),
    ]
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
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
            yield url, records
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def image_part(color, size=512):
    data = io.BytesIO()
    Image.new("RGB", (size, size), color).save(data, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": uri}}


def conversation(color="red", size=512):
    return [{"role": "user", "content": [
        image_part(color, size),
        {"type": "text", "text": "What is the background color? Reply with one color only."},
    ]}]


def ask(server, messages, model="draft", **extra):
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 32,
            "ignore_eos": False, "return_token_ids": True, "logprobs": True, "top_logprobs": 3}
    body.update(extra)
    response = requests.post(server[0] + "/v1/chat/completions", json=body, timeout=120)
    assert response.ok, response.text
    return response.json()


def same_policy(actual, expected):
    a, b = [r["choices"][0] for r in (actual, expected)]
    assert actual["prompt_token_ids"] == expected["prompt_token_ids"]
    assert a["token_ids"] == b["token_ids"]
    assert [v["logprob"] for v in a["logprobs"]["content"]] == pytest.approx(
        [v["logprob"] for v in b["logprobs"]["content"]], abs=.12)


def latest_done(server):
    return [r for r in map(json.loads, server[1].read_text().splitlines()) if r["event"] == "request_done"][-1]


@pytest.mark.parametrize("color,size", [("red", 64), ("blue", 64), ("red", 512), ("blue", 512)])
def test_image_scores_match_ordinary_decode(server, color, size):
    messages = conversation(color, size)
    expected = ask(server, messages, "base")
    actual = ask(server, messages)
    same_policy(actual, expected)
    assert color in actual["choices"][0]["message"]["content"].lower()
    done = latest_done(server)
    assert done["speculative"]["rounds"] + done["speculative"]["fallback_steps"] > 0
    assert done["speculative"]["draft_window"] == int(os.getenv("SUROGATE_DFLASH_VISION_TEST_DRAFT_TOKENS", "15"))
    if size == 512:
        assert actual["usage"]["prompt_tokens"] > 128


def test_mixed_batch_preserves_visual_positions(server):
    prompts = [conversation("red", 64), conversation("blue", 512),
               [{"role": "user", "content": "Reply with the word hello."}],
               [{"role": "user", "content": [image_part("red", 128), image_part("blue", 256),
                   {"type": "text", "text": "Name the colors of the first and second images, in order. Reply with two color words only."}]}]]
    expected = [ask(server, prompt, "base", max_tokens=16) for prompt in prompts]
    assert len({r["usage"]["prompt_tokens"] for r in expected}) >= 3
    for _ in range(2):
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda p: ask(server, p, max_tokens=16), prompts))
        for result, reference in zip(actual, expected):
            same_policy(result, reference)


@pytest.mark.skipif(os.getenv("SUROGATE_DFLASH_VISION_TEST_KV_DTYPE") != "fp8",
                    reason="FP8 configuration exercises the small conversation snapshot budget")
def test_checkpoint_pressure_preserves_prefill_scores(server):
    messages = [{"role": "user", "content": [
        image_part("red", 128), image_part("blue", 256),
        {"type": "text", "text": "Name the colors of the first and second images, in order. Reply with two color words only."},
    ]}]
    expected = ask(server, messages, "base", max_tokens=16)

    def occupy(index, started):
        payload = {
            "model": "draft", "messages": [{"role": "user", "content":
                f"Task {index}: count from 1 to 1000 in order, separated by commas."}],
            "max_tokens": 1024, "ignore_eos": True, "temperature": 0,
            "stream": True, "return_token_ids": True, "logprobs": True, "top_logprobs": 0,
        }
        count = 0
        with requests.post(server[0] + "/v1/chat/completions", json=payload,
                           stream=True, timeout=120) as response:
            assert response.ok, response.text
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                assert "error" not in chunk, chunk
                for choice in chunk["choices"]:
                    tokens = choice.get("token_ids", [])
                    count += len(tokens)
                    if (choice.get("logprobs") or {}).get("content"):
                        started.set()
        assert count == 1024

    # Keep the snapshot owners active while another conversation is admitted.
    # Its first-token score must agree even when it cannot retain a snapshot.
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = []
        for index in range(2):
            started = threading.Event()
            running.append(pool.submit(occupy, index, started))
            assert started.wait(60)
        assert all(not future.done() for future in running)
        actual = ask(server, messages, max_tokens=16)
        same_policy(actual, expected)
        for future in running:
            future.result()


@pytest.mark.parametrize("color,size", [("red", 64), ("blue", 512)])
def test_long_image_answer_scores_match_teacher_forcing(server, color, size):
    messages = conversation(color, size)
    messages[0]["content"][-1]["text"] = (
        "Describe the image in detail, discussing its color, texture, shapes, and composition. "
        "Write a complete paragraph."
    )
    generated = ask(server, messages, max_tokens=32)
    choice = generated["choices"][0]
    assert len(choice["token_ids"]) == 32
    expected_tokens = generated["prompt_token_ids"] + choice["token_ids"]
    history = messages + [choice["message"]]
    # Compare probabilities for exactly the same history. Independent greedy
    # continuations may choose different words when BF16 scores are nearly tied.
    for model in ("base", "draft"):
        for _ in range(2):
            replay = ask(server, history, model, max_tokens=1, prompt_logprobs=5)
            assert replay["prompt_token_ids"][:len(expected_tokens)] == expected_tokens
            n = len(generated["prompt_token_ids"])
            errors = [abs(replay["prompt_logprobs"][n + i][str(token)]["logprob"] -
                          choice["logprobs"]["content"][i]["logprob"])
                      for i, token in enumerate(choice["token_ids"])]
            if os.getenv("SUROGATE_DFLASH_VISION_TEST_ADAPTIVE") == "true":
                # Changing verification widths changes BF16 rounding; FP8 cache
                # quantization can amplify individual differences. Bound both
                # peak and average error. Native GQA tests check cache rounding
                # equivalence exactly, independently of these model-level checks.
                peak = .25 if os.getenv("SUROGATE_DFLASH_VISION_TEST_KV_DTYPE") == "fp8" else .15
                assert max(errors) <= peak, errors
                assert sum(errors) / len(errors) <= .035, errors
            else:
                assert max(errors) <= .12, errors


def test_completed_image_turn_reuse_and_edited_history(server):
    initial = conversation("red", 256)
    first = ask(server, initial, max_tokens=8, ignore_eos=False)
    followup = initial + [first["choices"][0]["message"],
                          {"role": "user", "content": "What color did you see? Answer briefly."}]
    actual = ask(server, followup, max_tokens=16)
    assert latest_done(server)["result"]["prefix_cache_hit_tokens"] > 0
    same_policy(actual, ask(server, followup, "base", max_tokens=16))
    # Edit the assistant after the image, then replace the image while keeping its token shape.
    # The first can restore the saved boundary; the second must invalidate visual prefix reuse.
    edited = initial + [{"role": "assistant", "content": "I saw a solid background."}, followup[-1]]
    actual = ask(server, edited, max_tokens=16)
    same_policy(actual, ask(server, edited, "base", max_tokens=16))
    changed_image = conversation("blue", 256)
    actual = ask(server, changed_image, max_tokens=16)
    same_policy(actual, ask(server, changed_image, "base", max_tokens=16))
    assert "blue" in actual["choices"][0]["message"]["content"].lower()


def test_streamed_image_scores(server):
    messages = conversation("blue", 128)
    expected = ask(server, messages, max_tokens=16)
    response = requests.post(server[0] + "/v1/chat/completions", json={
        "model": "draft", "messages": messages, "temperature": 0, "max_tokens": 16,
        "stream": True, "logprobs": True, "top_logprobs": 3,
    }, timeout=120)
    assert response.ok, response.text
    chunks = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    scores = [entry for chunk in chunks for choice in chunk["choices"]
              for entry in (choice.get("logprobs") or {}).get("content", [])]
    reference = expected["choices"][0]["logprobs"]["content"]
    assert [s["bytes"] for s in scores] == [s["bytes"] for s in reference]
    assert [s["logprob"] for s in scores] == pytest.approx([s["logprob"] for s in reference], abs=.12)


def test_video_chat_and_responses(server, tmp_path, monkeypatch):
    from tests.serve.test_qwen3_vl_http import test_video_color_order_through_chat_and_responses
    monkeypatch.setenv("SUROGATE_QWEN3_VL_TEST_URL", server[0])
    monkeypatch.setenv("SUROGATE_QWEN3_VL_TEST_MODEL", "draft")
    for responses in (False, True):
        path = tmp_path / str(responses)
        path.mkdir()
        test_video_color_order_through_chat_and_responses(path, responses)


def test_sleep_wake_preserves_image_scores(server):
    messages = conversation("red", 128)
    expected = ask(server, messages, max_tokens=16)
    assert requests.post(server[0] + "/sleep?model=draft", timeout=60).ok
    assert requests.get(server[0] + "/is_sleeping?model=draft", timeout=5).json()["is_sleeping"]
    assert requests.post(server[0] + "/wake_up?model=draft", timeout=60).ok
    same_policy(ask(server, messages, max_tokens=16), expected)
