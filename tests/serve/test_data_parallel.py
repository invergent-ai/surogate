"""`surogate serve --devices A,B --data-parallel` (#261): one engine per GPU behind one model id.

Opt-in: set SUROGATE_DATA_PARALLEL_TEST_ARTIFACT to a prepared artifact small enough for one GPU,
and SUROGATE_DATA_PARALLEL_TEST_DEVICES to two or more GPUs (default "0,1").
"""

import os
import re
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from surogate.cli.serve import _resolve_binary

MODEL = "dp"


def _decode_counters(base):
    response = requests.get(base + "/metrics", timeout=5)
    response.raise_for_status()
    return {
        int(replica): int(value)
        for replica, value in re.findall(
            rf'surogate_decode_tokens_total\{{model="{MODEL}",replica="(\d+)"\}} (\d+)', response.text
        )
    }


def _served_by(base, request):
    """Runs `request` with nothing else in flight and returns the replica that decoded it."""
    before = _decode_counters(base)
    result = request()
    after = _decode_counters(base)
    grown = [replica for replica in after if after[replica] > before.get(replica, 0)]
    assert len(grown) == 1, (before, after)
    return grown[0], result


@pytest.mark.skipif(not os.getenv("SUROGATE_DATA_PARALLEL_TEST_ARTIFACT"), reason="requires a prepared GPU artifact")
def test_data_parallel_replicas_share_one_model_id(tmp_path):
    artifact = os.environ["SUROGATE_DATA_PARALLEL_TEST_ARTIFACT"]
    devices = os.getenv("SUROGATE_DATA_PARALLEL_TEST_DEVICES", "0,1")
    replicas = len(devices.split(","))
    assert replicas >= 2
    binary = _resolve_binary("server")
    assert binary
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    log = tmp_path / "server.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                binary, artifact, "--port", str(port), "--devices", devices, "--data-parallel",
                "--served-model-name", MODEL, "--enable-sleep-mode",
                "--max-model-len", "1024", "--kv-capacity", "4096", "--max-num-seqs", "4",
                "--no-thinking", "--log-stats-interval-ms", "0",
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 180
            while True:
                assert process.poll() is None, log.read_text()
                try:
                    if requests.get(base + "/health", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                assert time.monotonic() < deadline, log.read_text()
                time.sleep(0.2)

            def chat(messages, rank=None, max_tokens=16):
                response = requests.post(
                    base + "/v1/chat/completions",
                    headers={} if rank is None else {"X-data-parallel-rank": str(rank)},
                    json={
                        "model": MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
                        "ignore_eos": True, "return_token_ids": True,
                    },
                    timeout=120,
                )
                assert response.status_code == 200, response.text
                return response.json()["choices"][0]

            # One model id, one replica per GPU.
            models = requests.get(base + "/v1/models", timeout=5).json()
            assert [item["id"] for item in models["data"]] == [MODEL]
            kv = requests.get(base + "/kv_stats", timeout=5).json()["models"]
            assert [(m["model"], m["replica"]) for m in kv] == [(MODEL, i) for i in range(replicas)]
            assert sorted(_decode_counters(base)) == list(range(replicas))

            # Every replica runs the same model: pinned greedy requests agree token for token.
            prompt = [{"role": "user", "content": "Continue: 1, 2, 3,"}]
            pinned = []
            for rank in range(replicas):
                served, choice = _served_by(base, lambda rank=rank: chat(prompt, rank=rank))
                assert served == rank
                pinned.append(choice["token_ids"])
            assert all(tokens == pinned[0] for tokens in pinned)
            bad = requests.post(base + "/v1/chat/completions", headers={"X-data-parallel-rank": str(replicas)},
                                json={"model": MODEL, "messages": prompt, "max_tokens": 4}, timeout=30)
            assert bad.status_code == 400, bad.text

            # Unrelated conversations arriving together reach every replica.
            before = _decode_counters(base)
            with ThreadPoolExecutor(max_workers=4 * replicas) as pool:
                list(pool.map(lambda i: chat([{"role": "user", "content": f"Topic {i}: tell me a fact."}]),
                              range(4 * replicas)))
            after = _decode_counters(base)
            assert all(after[r] > before[r] for r in range(replicas)), (before, after)

            # A conversation's next turn goes back to the replica holding its prefix.
            conversation = [{"role": "system", "content": "You are terse. " * 40},
                            {"role": "user", "content": "Name a prime number."}]
            home, first = _served_by(base, lambda: chat(conversation))
            conversation += [{"role": "assistant", "content": first["message"]["content"]},
                             {"role": "user", "content": "And another one."}]
            for _ in range(3):
                served, reply = _served_by(base, lambda: chat(conversation))
                assert served == home
                conversation += [{"role": "assistant", "content": reply["message"]["content"]},
                                 {"role": "user", "content": "One more."}]

            # Sleep and wake reach every replica.
            assert requests.post(base + "/sleep", timeout=60).status_code == 200
            assert requests.get(base + "/is_sleeping", timeout=5).json()["is_sleeping"]
            assert all(m["sleeping"] for m in requests.get(base + "/kv_stats", timeout=5).json()["models"])
            assert requests.post(base + "/wake_up", timeout=60).status_code == 200
            assert not requests.get(base + "/is_sleeping", timeout=5).json()["is_sleeping"]
            assert chat(prompt, rank=replicas - 1)["token_ids"] == pinned[0]
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
