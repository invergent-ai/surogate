"""Opt-in encoder scheduling checks on a tower with more than four layers.

Use the SUROGATE_MIXED_TEST_* settings with --vision. The same checks run on a
single GPU or a physical pipeline selected through --devices.
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


def same_image_response(actual, expected):
    assert actual["prompt_token_ids"] == expected["prompt_token_ids"]
    a, b = actual["choices"][0], expected["choices"][0]
    assert a["token_ids"] == b["token_ids"]
    assert [v["logprob"] for v in a["logprobs"]["content"]] == pytest.approx(
        [v["logprob"] for v in b["logprobs"]["content"]], abs=.12)


def test_text_progresses_before_image_encoding_finishes(server):
    payload = image_prompt()
    expected = ask(server, payload)
    started = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(streaming, server, body("test", 6, 8, 512), started)
        assert started.wait(60)
        offset = server[1].stat().st_size
        actual = pool.submit(ask, server, payload)
        first = wait_log(server, offset, r"vision-step lane=(\d+) complete=0")
        lane = first[1]
        last = wait_log(server, offset, rf"vision-step lane={lane} complete=1")
        between = server[1].read_text()[offset:][first.end():last.start()]
        commits = re.findall(r"decode-commit lane=(\d+) tokens=(\d+)", between)
        assert any(other != lane and int(tokens) > 0 for other, tokens in commits), between
        same_image_response(actual.result(), expected)
        chunks, _ = running.result()
        assert sum(len(c.get("token_ids", [])) for chunk in chunks for c in chunk["choices"]) == 512


def test_cancelling_partial_image_encoding_releases_the_request(server):
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
            first = wait_log(server, offset, r"vision-step lane=(\d+) complete=0")
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        wait_log(server, offset, r"done finish=cancelled .* gen=0 ")
        assert f"vision-step lane={first[1]} complete=1" not in server[1].read_text()[offset:]
        same_image_response(ask(server, payload), expected)
        chunks, _ = running.result()
        assert sum(len(c.get("token_ids", [])) for chunk in chunks for c in chunk["choices"]) == 512
