"""Opt-in image scheduling checks on a tower with more than four layers.

Use the SUROGATE_MIXED_TEST_* settings with --vision. The same checks run on a
single GPU or a physical pipeline selected through --devices.
Set SUROGATE_IMAGE_TEXT_TEST=1 for whole-image text attention. The sleep check
also needs --enable-sleep-mode and SUROGATE_SLEEP_PREEMPT=1.
"""

import json
import os
import re
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import pytest
import requests

from tests.serve.test_dflash_vision import image_part
from tests.serve.test_mixed_prefill_decode import ask, body, server, streaming  # noqa: F401

pytestmark = pytest.mark.skipif(
    not os.getenv("SUROGATE_MIXED_TEST_ARTIFACT") or
    "--vision" not in json.loads(os.getenv("SUROGATE_MIXED_TEST_ARGS", "[]")),
    reason="requires a prepared vision artifact and --vision",
)


def image_prompt():
    models = json.loads(os.getenv("SUROGATE_MIXED_TEST_MODELS", '["test"]'))
    return {
        "model": models[-1], "messages": [{"role": "user", "content": [
            image_part("blue", 768),
            {"type": "text", "text": "Name the background color. Answer with one color word."},
        ]}], "max_tokens": 16, "temperature": 0, "return_token_ids": True,
        "logprobs": True, "top_logprobs": 0,
    }


def wait_log(server, offset, pattern):
    deadline = time.monotonic() + 60
    while True:
        log = server[1].read_text()[offset:]
        match = re.search(pattern, log)
        if match:
            return match
        assert time.monotonic() < deadline, log[-4000:]
        time.sleep(.001)


def phase_start(phase):
    return r"vision-step lane=(\d+) complete=0" if phase == "encoder" else r"image-text-step lane=(\d+) cursor=(\d+)"


def phase_end(phase, first):
    return (rf"vision-step lane={first[1]} complete=1" if phase == "encoder" else
            rf"prefill lane={first[1]} cursor={first[2]} nominal=")


def same_image_response(actual, expected):
    assert actual["prompt_token_ids"] == expected["prompt_token_ids"]
    a, b = actual["choices"][0], expected["choices"][0]
    assert a["token_ids"] == b["token_ids"]
    assert [v["logprob"] for v in a["logprobs"]["content"]] == pytest.approx(
        [v["logprob"] for v in b["logprobs"]["content"]], abs=.12)


@pytest.mark.parametrize("phase", ["encoder", pytest.param("text", marks=pytest.mark.skipif(
    not os.getenv("SUROGATE_IMAGE_TEXT_TEST"), reason="requires whole-image text attention"))])
def test_text_progresses_during_image_processing(server, phase):
    payload = image_prompt()
    expected = ask(server, payload)
    started = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(streaming, server, body("test", 6, 8, 512), started)
        assert started.wait(60)
        offset = server[1].stat().st_size
        actual = pool.submit(ask, server, payload)
        first = wait_log(server, offset, phase_start(phase))
        lane = first[1]
        last = wait_log(server, offset, phase_end(phase, first))
        between = server[1].read_text()[offset:][first.end():last.start()]
        commits = re.findall(r"decode-commit lane=(\d+) tokens=(\d+)", between)
        assert any(other != lane and int(tokens) > 0 for other, tokens in commits), between
        same_image_response(actual.result(), expected)
        chunks, _ = running.result()
        assert sum(len(c.get("token_ids", [])) for chunk in chunks for c in chunk["choices"]) == 512


@pytest.mark.parametrize("phase", ["encoder", pytest.param("text", marks=pytest.mark.skipif(
    not os.getenv("SUROGATE_IMAGE_TEXT_TEST"), reason="requires whole-image text attention"))])
def test_cancelling_partial_image_processing_releases_the_request(server, phase):
    payload = image_prompt()
    expected = ask(server, payload)
    started = threading.Event()
    address = urlsplit(server[0])
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(streaming, server, body("test", 6, 8, 512), started)
        assert started.wait(60)
        offset = server[1].stat().st_size
        # Keep control of the socket before the image's first output token.
        with socket.create_connection((address.hostname, address.port), timeout=60) as connection:
            encoded = json.dumps(dict(payload, stream=True)).encode()
            headers = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {address.netloc}\r\n"
                       f"Content-Type: application/json\r\nContent-Length: {len(encoded)}\r\n\r\n")
            connection.sendall(headers.encode() + encoded)
            first = wait_log(server, offset, phase_start(phase))
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        wait_log(server, offset, r"done finish=cancelled .* gen=0 ")
        assert not re.search(phase_end(phase, first), server[1].read_text()[offset:])
        same_image_response(ask(server, payload), expected)
        chunks, _ = running.result()
        assert sum(len(c.get("token_ids", [])) for chunk in chunks for c in chunk["choices"]) == 512


@pytest.mark.skipif(not os.getenv("SUROGATE_IMAGE_TEXT_TEST") or not os.getenv("SUROGATE_SLEEP_PREEMPT"),
                    reason="requires image text slicing and preemptive sleep mode")
def test_sleep_restores_partial_image_text_processing(server):
    payload = image_prompt()
    expected = ask(server, payload)
    started = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(streaming, server, body("test", 6, 8, 512), started)
        assert started.wait(60)
        offset = server[1].stat().st_size
        actual = pool.submit(ask, server, payload)
        first = wait_log(server, offset, phase_start("text"))
        try:
            response = requests.post(server[0] + "/sleep", timeout=60)
            assert response.ok, response.text
            assert not re.search(phase_end("text", first), server[1].read_text()[offset:])
        finally:
            response = requests.post(server[0] + "/wake_up", timeout=60)
            assert response.ok, response.text
        same_image_response(actual.result(), expected)
        chunks, _ = running.result()
        assert sum(len(c.get("token_ids", [])) for chunk in chunks for c in chunk["choices"]) == 512
