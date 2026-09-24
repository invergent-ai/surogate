"""The engine survives a GPU out-of-memory, and a dead engine says so (SUROGATE-CHANGES #16). GPU.

Skipped unless SUROGATE_OOM_TEST_ARTIFACT names a prepared chat model (.sinfer). Select one free GPU
with CUDA_VISIBLE_DEVICES, and use SUROGATE_SERVE_BIN to pick the engine binary under test.
SUROGATE_OOM_TEST_ARGS may add engine flags as a JSON list. Each case starts its own server:

- An out-of-memory in a round (forced with SUROGATE_SERVE_FAULT_KV_OOM) fails only the requests in
  that round, with a retryable 429. The next request succeeds, and /health stays 200.
- A worker that dies (forced with SUROGATE_SERVE_FAULT_WORKER_FATAL) turns /health to 503, and the
  process exits with status 1, so a supervisor restarts it and health checks eject it.
- Under a load that fills the cache, with --kv-capacity auto, the cache never maps more pages than
  the cap it was sized for. In the 2026-09-24 fuzz run it mapped past its cap and filled the
  device.
"""

import concurrent.futures as cf
import contextlib
import json
import math
import os
import random
import re
import socket
import subprocess
import threading
import time

import pytest
import requests

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(not os.getenv("SUROGATE_OOM_TEST_ARTIFACT"),
                                reason="needs a prepared chat model and a free GPU")

WORDS = ("order delivery refund invoice parcel courier warehouse customer ticket account balance "
         "payment receipt return exchange voucher discount shipment tracking address").split()


def prompt(rng, words):
    return " ".join(rng.choice(WORDS) for _ in range(words))


@contextlib.contextmanager
def server(root, name, args, env=None):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    log = root / f"{name}.log"
    cmd = [_resolve_binary("server"), os.environ["SUROGATE_OOM_TEST_ARTIFACT"],
           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "test", *args,
           *json.loads(os.getenv("SUROGATE_OOM_TEST_ARGS", "[]"))]
    base = f"http://127.0.0.1:{port}"
    with log.open("w") as output:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT,
                                   env={**os.environ, **(env or {})})
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                assert process.poll() is None, log.read_text()
                try:
                    if requests.get(base + "/v1/models", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.2)
            else:
                pytest.fail(log.read_text())
            yield base, process, log
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def chat(base, text, max_tokens, **extra):
    body = {"model": "test", "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "temperature": 0, **extra}
    return requests.post(base + "/v1/chat/completions", json=body, timeout=600)


def test_out_of_memory_fails_the_round_and_serving_goes_on(tmp_path):
    # No reserve, so every granule is mapped on demand, and the third on-demand map fails as if
    # the device were full: after the warm-up's, inside one of the first requests below.
    env = {"SUROGATE_SERVE_ELASTIC_KV_RESERVE": "0", "SUROGATE_SERVE_FAULT_KV_OOM": "3"}
    args = ["--max-model-len", "8192", "--max-num-seqs", "4", "--kv-capacity", "auto"]
    rng = random.Random(16)
    with server(tmp_path, "kv-oom", args, env) as (base, process, log):
        statuses = []
        for _ in range(4):
            response = chat(base, prompt(rng, 2500), 8)
            statuses.append(response.status_code)
            if response.status_code == 429:
                error = response.json()["error"]
                assert error["code"] == "server_overloaded", error
                assert "ran out of memory" in error["message"], error
            else:
                assert response.status_code == 200, response.text
        assert statuses.count(429) == 1, statuses
        assert statuses[-1] == 200 and statuses.index(429) < len(statuses) - 1, statuses
        assert process.poll() is None, "the engine exited"
        assert requests.get(base + "/health", timeout=5).status_code == 200
        text = log.read_text()
        assert "injected out-of-memory" in text
        assert "ran out of device memory" in text and "continuing" in text
        assert "worker loop fatal" not in text


def test_a_dead_worker_turns_health_to_503_and_the_process_exits_1(tmp_path):
    # Once 40 rounds have run the worker loop throws something it cannot survive: after the
    # warm-up, inside the long request below (a decode round can emit several tokens, so 1,000
    # tokens still take well over 40 rounds).
    env = {"SUROGATE_SERVE_FAULT_WORKER_FATAL": "40"}
    args = ["--max-model-len", "4096", "--max-num-seqs", "2"]
    with server(tmp_path, "worker-fatal", args, env) as (base, process, log):
        assert requests.get(base + "/health", timeout=5).status_code == 200
        response = chat(base, "Count from one to one thousand in words.", 1000, ignore_eos=True)
        assert response.status_code >= 500, response.text
        health = requests.get(base + "/health", timeout=5)
        assert health.status_code == 503, health.text
        assert health.json()["status"] == "unavailable"
        assert process.wait(timeout=60) == 1
        text = log.read_text()
        assert "injected worker fault" in text
        assert "exiting with status 1" in text


def test_the_cache_stays_within_its_cap_under_load(tmp_path):
    """The spec's shape (--max-model-len 32768 --max-num-seqs 16 --kv-capacity auto), 16 clients
    with distinct long prompts: more than the cache holds, so it fills to its cap."""
    args = ["--max-model-len", "32768", "--max-num-seqs", "16", "--kv-capacity", "auto"]
    with server(tmp_path, "cap", args) as (base, process, log):
        cap_pages = int(re.search(r"KV capacity auto resolved=\d+ tokens pages=(\d+)/", log.read_text()).group(1))
        stats = requests.get(base + "/kv_stats", timeout=5).json()["models"][0]
        granule = stats["granule_pages"]
        bound = math.ceil(cap_pages / granule) * granule
        peak = {"mapped": 0, "in_use": 0}
        done = threading.Event()

        def sample():
            while not done.is_set():
                try:
                    model = requests.get(base + "/kv_stats", timeout=2).json()["models"][0]
                    peak["mapped"] = max(peak["mapped"], model["pages_mapped"])
                    peak["in_use"] = max(peak["in_use"], model["pages_in_use"])
                except requests.RequestException:
                    pass
                time.sleep(0.05)

        sampler = threading.Thread(target=sample)
        sampler.start()
        try:
            rng = random.Random(24)
            texts = [prompt(rng, rng.randrange(3000, 7000)) for _ in range(64)]
            with cf.ThreadPoolExecutor(16) as pool:
                statuses = list(pool.map(lambda text: chat(base, text, 4).status_code, texts))
        finally:
            done.set()
            sampler.join()
        assert all(status == 200 for status in statuses), statuses
        assert peak["in_use"] > bound // 2, f"the load did not fill the cache: {peak}, cap {cap_pages}"
        assert peak["mapped"] <= bound, f"mapped {peak['mapped']} pages, cap {cap_pages} ({bound} in granules)"
        text = log.read_text()
        assert "out of device memory" not in text
        assert "worker loop fatal" not in text
