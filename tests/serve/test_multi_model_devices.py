"""Opt-in HTTP coverage; set SUROGATE_MULTI_MODEL_TEST_ARTIFACT and reserve CUDA devices."""

import os
import re
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from surogate.cli.serve import _resolve_binary


@pytest.mark.skipif(not os.getenv("SUROGATE_MULTI_MODEL_TEST_ARTIFACT"), reason="requires a prepared GPU artifact")
def test_named_replicas_route_and_sleep_independently(tmp_path):
    artifact = os.environ["SUROGATE_MULTI_MODEL_TEST_ARTIFACT"]
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
                binary,
                artifact,
                "--port",
                str(port),
                "--device",
                "0",
                "--served-model-name",
                "primary",
                "--model",
                f"replica={artifact},device=0",
                "--enable-sleep-mode",
                "--max-model-len",
                "512",
                "--kv-capacity",
                "2048",
                "--max-num-seqs",
                "4",
                "--no-thinking",
                "--log-stats-interval-ms",
                "0",
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 90
            while True:
                assert process.poll() is None, log.read_text()
                try:
                    if requests.get(base + "/health", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                assert time.monotonic() < deadline, log.read_text()
                time.sleep(0.1)

            def ask(model):
                response = requests.post(
                    base + "/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "Continue: 1, 2, 3,"}],
                        "temperature": 0,
                        "max_tokens": 32,
                        "ignore_eos": True,
                        "return_token_ids": True,
                    },
                    timeout=60,
                )
                assert response.status_code == 200, response.text
                data = response.json()
                assert data["model"] == model
                return data["choices"][0]["token_ids"]

            models = requests.get(base + "/v1/models", timeout=5).json()
            assert {item["id"] for item in models["data"]} == {"primary", "replica"}
            expected = ask("primary")
            assert ask("replica") == expected
            response = requests.post(base + "/sleep?model=replica", timeout=30)
            assert response.status_code == 200, response.text
            assert requests.get(base + "/is_sleeping?model=replica", timeout=5).json()["is_sleeping"]
            assert not requests.get(base + "/is_sleeping?model=primary", timeout=5).json()["is_sleeping"]
            assert ask("primary") == expected
            # Routing a request to a sleeping model also exercises automatic wake by the scheduler.
            assert ask("replica") == expected

            def decode_counters():
                response = requests.get(base + "/metrics", timeout=5)
                response.raise_for_status()
                return {
                    name: int(value)
                    for name, value in re.findall(
                        r'surogate_decode_tokens_total\{model="([^"]+)"\} (\d+)', response.text
                    )
                }

            before = decode_counters()
            with ThreadPoolExecutor(max_workers=8) as pool:
                outcomes = list(pool.map(ask, ["primary", "replica"] * 4))
            assert all(len(result) == 32 and all(token >= 0 for token in result) for result in outcomes)
            after = decode_counters()
            for name in ("primary", "replica"):
                assert after[name] > before[name], "requests did not reach both independent engines"
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
