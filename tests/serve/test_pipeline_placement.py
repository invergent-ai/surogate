"""Opt-in startup and correctness checks for a caller-supplied MoE pipeline.

SUROGATE_PIPELINE_PLACEMENT_ARTIFACT supplies a prepared artifact; reserve GPUs through
CUDA_VISIBLE_DEVICES and set SUROGATE_PIPELINE_PLACEMENT_DEVICES to their logical IDs.
SUROGATE_PIPELINE_PLACEMENT_ARGS is a JSON array of additional server options.
This checks placement and generation, without collecting throughput measurements.
"""

import json
import math
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from surogate.cli.serve import _resolve_binary


@pytest.mark.skipif(not os.getenv("SUROGATE_PIPELINE_PLACEMENT_ARTIFACT"),
                    reason="requires a prepared MoE artifact and reserved GPUs")
def test_pipeline_plans_before_allocating_expert_cache(tmp_path):
    devices = os.getenv("SUROGATE_PIPELINE_PLACEMENT_DEVICES", "0,1")
    assert len(devices.split(",")) > 1
    policy = os.getenv("SUROGATE_PIPELINE_PLACEMENT_POLICY", "auto")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log = tmp_path / "pipeline.log"
    command = [
        _resolve_binary("server"), os.environ["SUROGATE_PIPELINE_PLACEMENT_ARTIFACT"],
        "--devices", devices, "--port", str(port), "--served-model-name", "test",
        "--max-model-len", "256", "--kv-capacity", "1024", "--max-num-seqs", "4",
        "--max-num-batched-tokens", "128", "--host-moe-layers", policy,
        "--no-thinking", "--no-prefix-reuse",
        *json.loads(os.getenv("SUROGATE_PIPELINE_PLACEMENT_ARGS", "[]")),
    ]
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
            while True:
                assert process.poll() is None, log.read_text()[-12000:]
                try:
                    if requests.get(url + "/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                assert time.monotonic() < deadline, log.read_text()[-12000:]
                time.sleep(.1)

            startup = log.read_text()
            planned = startup.index(f"pipeline: {len(devices.split(','))} stages,")
            allocated = startup.find("expert cache: reserving")
            assert allocated == -1 or allocated > planned, startup
            if policy not in {"auto", "0"}:
                # An explicit offload policy must survive preflight into actual loading.
                assert allocated != -1, startup

            def generate(index):
                prompt = [100 + index, 110 + index, 120 + index] * 64
                response = requests.post(url + "/v1/chat/completions", json={
                    "model": "test", "tokens": prompt,
                    "messages": [{"role": "user", "content": "Continue."}],
                    "max_tokens": 16, "temperature": 0, "ignore_eos": True,
                    "return_token_ids": True, "logprobs": True, "top_logprobs": 0,
                }, timeout=240)
                assert response.ok, response.text + "\n" + log.read_text()[-4000:]
                result = response.json()
                assert result["usage"]["prompt_tokens"] == len(prompt)
                assert result["usage"]["completion_tokens"] == 16
                choice = result["choices"][0]
                assert len(choice["token_ids"]) == 16
                scores = choice["logprobs"]["content"]
                assert len(scores) == 16
                assert all(math.isfinite(score["logprob"]) for score in scores)

            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(generate, range(4)))
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
