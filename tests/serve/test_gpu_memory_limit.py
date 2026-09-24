"""Several servers share one GPU, each within --gpu-memory-limit-mib (SUROGATE-CHANGES #8). GPU.

Skipped unless SUROGATE_LIMIT_TEST_ARTIFACT names a prepared chat model (.sinfer) small enough for two
copies to fit on the selected GPU (CUDA_VISIBLE_DEVICES, one free card). SUROGATE_LIMIT_TEST_MIB is
each server's limit (default 15800), SUROGATE_LIMIT_TEST_ARGS adds engine flags as a JSON list, and
SUROGATE_SERVE_BIN picks the engine binary under test.

Two servers start on the same card, each with the limit, and both are driven with more long,
distinct prompts than their caches hold, until each cache has filled close to its capacity.
Neither may run out of memory, every request must succeed, and each process's own GPU usage (NVML,
as nvidia-smi reports it) must stay within its limit the whole time. The card's total usage is
checked against the sum of the processes', so memory NVML failed to attribute would show. A limit
below the weights is refused at startup.
"""

import concurrent.futures as cf
import contextlib
import json
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

pytestmark = pytest.mark.skipif(not os.getenv("SUROGATE_LIMIT_TEST_ARTIFACT"),
                                reason="needs a prepared chat model and a free GPU")

LIMIT_MIB = int(os.getenv("SUROGATE_LIMIT_TEST_MIB", "15800"))
WORDS = ("order delivery refund invoice parcel courier warehouse customer ticket account balance "
         "payment receipt return exchange voucher discount shipment tracking address").split()


@contextlib.contextmanager
def server(root, name):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    log = root / f"{name}.log"
    cmd = [_resolve_binary("server"), os.environ["SUROGATE_LIMIT_TEST_ARTIFACT"],
           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "test",
           "--max-model-len", "4096", "--max-num-seqs", "8", "--kv-capacity", "auto",
           "--gpu-memory-limit-mib", str(LIMIT_MIB),
           *json.loads(os.getenv("SUROGATE_LIMIT_TEST_ARGS", "[]"))]
    base = f"http://127.0.0.1:{port}"
    with log.open("w") as output:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                assert process.poll() is None, log.read_text()
                try:
                    if requests.get(base + "/v1/models", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            else:
                pytest.fail(log.read_text())
            yield base, process, log
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def visible_gpu():
    """The nvidia-smi index of the card the servers run on (the first of CUDA_VISIBLE_DEVICES)."""
    return os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


def usage_mib(pids):
    """Each process's GPU memory, as nvidia-smi reports it (NVML), and the card's total."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    found = {}
    for line in out.splitlines():
        pid, used = (part.strip() for part in line.split(","))
        if int(pid) in pids:
            found[int(pid)] = int(used)
    total = subprocess.run(["nvidia-smi", "-i", visible_gpu(), "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return found, int(total.stdout.strip())


def cap_pages(log):
    return int(re.search(r"KV capacity \w+ resolved=\d+ tokens pages=(\d+)/", log.read_text()).group(1))


def test_two_servers_share_a_gpu_within_their_limits(tmp_path):
    with server(tmp_path, "a") as (a, pa, log_a), server(tmp_path, "b") as (b, pb, log_b):
        pids = {pa.pid, pb.pid}
        peak = {pid: 0 for pid in pids}
        in_use = {a: 0, b: 0}
        unattributed = [0]  # the card's usage beyond the two processes', at its largest
        errors = []
        done = threading.Event()

        def sample():
            while not done.is_set():
                try:
                    found, total = usage_mib(pids)
                    for pid, used in found.items():
                        peak[pid] = max(peak[pid], used)
                    if len(found) == 2:
                        unattributed[0] = max(unattributed[0], total - sum(found.values()))
                    for base in (a, b):
                        model = requests.get(base + "/kv_stats", timeout=2).json()["models"][0]
                        in_use[base] = max(in_use[base], model["pages_in_use"])
                except Exception as error:  # noqa: BLE001 -- recorded, reported below
                    errors.append(repr(error))
                time.sleep(0.2)

        sampler = threading.Thread(target=sample)
        sampler.start()
        try:
            rng = random.Random(8)
            jobs = [(base, " ".join(rng.choice(WORDS) for _ in range(rng.randrange(1500, 3000))))
                    for _ in range(48) for base in (a, b)]

            def chat(job):
                base, text = job
                body = {"model": "test", "messages": [{"role": "user", "content": text}],
                        "max_tokens": 16, "temperature": 0}
                return requests.post(base + "/v1/chat/completions", json=body, timeout=600).status_code

            with cf.ThreadPoolExecutor(32) as pool:
                statuses = list(pool.map(chat, jobs))
        finally:
            done.set()
            sampler.join()
        assert all(status == 200 for status in statuses), statuses
        assert pa.poll() is None and pb.poll() is None, "a server exited"
        assert len(errors) < 5, errors[:5]
        for pid, used in peak.items():
            assert 0 < used <= LIMIT_MIB, f"process {pid} peaked at {used} MiB, limit {LIMIT_MIB} MiB"
        # The load filled both caches, so the limits were tested at their worst.
        for base, log in ((a, log_a), (b, log_b)):
            assert in_use[base] >= 0.8 * cap_pages(log), (in_use, cap_pages(log))
        # The card holds little beyond what NVML attributes to the two servers (the driver's
        # reserve): the servers' memory, VMM-mapped cache included, is counted against them.
        assert unattributed[0] <= 1536, f"{unattributed[0]} MiB of the card is not attributed"
        for log in (log_a, log_b):
            text = log.read_text()
            assert "gpu-memory-limit=" in text, "the limit is not in the startup log"
            assert "out of device memory" not in text and "worker loop fatal" not in text
        stats = [requests.get(base + "/kv_stats", timeout=5).json() for base in (a, b)]
        for stat in stats:
            assert stat["models"][0]["kv_capacity_tokens"] > 0


def test_a_limit_below_the_weights_is_refused(tmp_path):
    port = socket.socket()
    port.bind(("127.0.0.1", 0))
    number = port.getsockname()[1]
    port.close()
    cmd = [_resolve_binary("server"), os.environ["SUROGATE_LIMIT_TEST_ARTIFACT"], "--host", "127.0.0.1",
           "--port", str(number), "--max-model-len", "4096", "--gpu-memory-limit-mib", "2048",
           *json.loads(os.getenv("SUROGATE_LIMIT_TEST_ARGS", "[]"))]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode != 0
    assert "--gpu-memory-limit-mib 2048" in result.stdout + result.stderr, result.stdout + result.stderr
