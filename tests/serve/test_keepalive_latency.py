"""A reused keep-alive connection is as fast as a new one (SUROGATE-CHANGES #12). GPU; pending.

Skipped unless SUROGATE_KEEPALIVE_TEST_ARTIFACT names a prepared chat model (.sinfer); select one free
GPU with CUDA_VISIBLE_DEVICES, and add engine flags with SUROGATE_KEEPALIVE_TEST_ARGS (a JSON list).

Sequential two-question decisions, like the agentgateway spike's, are timed on a fresh connection per
request and on one pooled keep-alive connection. With Nagle's algorithm on the server, every reused
request waited about 40 ms for the client's delayed ACK (64 ms against 104 ms on Rune). With
TCP_NODELAY the reused connection is no slower. Python's requests sends a small request in one write
and sets TCP_NODELAY on its own sockets, as the gateway and the SDKs do.
"""

import json
import os
import socket
import statistics
import subprocess
import time

import pytest
import requests

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(not os.getenv("SUROGATE_KEEPALIVE_TEST_ARTIFACT"),
                                reason="needs a prepared chat model and a free GPU")

DECISION = {"model": "test",
            "state": {"ticket": "Comanda 8812 a ajuns cu întârziere și cutia era strivită. Vreau banii înapoi."},
            "questions": {
                "refund": {"type": "noul", "instructions": "Is a refund requested?",
                           "criteria": {"true": "A refund is requested", "false": "No refund is requested"}},
                "tone": {"type": "choice", "instructions": "What is the tone?",
                         "criteria": {"calm": "Calm", "annoyed": "Annoyed", "furious": "Furious"}}}}


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    log = tmp_path_factory.mktemp("keepalive") / "server.log"
    cmd = [_resolve_binary("server"), os.environ["SUROGATE_KEEPALIVE_TEST_ARTIFACT"],
           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "test",
           "--max-model-len", "4096", "--max-num-seqs", "4",
           *json.loads(os.getenv("SUROGATE_KEEPALIVE_TEST_ARGS", "[]"))]
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
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def timed(post, url):
    began = time.perf_counter()
    response = post(url + "/v1/decisions", json=DECISION, timeout=60)
    elapsed = (time.perf_counter() - began) * 1000
    assert response.ok, response.text
    return elapsed


def test_reused_connection_is_as_fast_as_a_new_one(base):
    for _ in range(5):  # warm the engine and the prefix-free path
        timed(requests.post, base)
    fresh = [timed(lambda *a, **k: requests.post(*a, headers={"Connection": "close"}, **k), base) for _ in range(20)]
    with requests.Session() as session:
        pooled = [timed(session.post, base) for _ in range(21)][1:]  # the first request opens the connection
    new, reused = statistics.median(fresh), statistics.median(pooled)
    print(f"median decision latency: new connection {new:.1f} ms, reused keep-alive connection {reused:.1f} ms")
    # The delayed-ACK stall was ~40 ms per reused request; allow noise, not the stall.
    assert reused <= new + 10.0, (new, reused)
