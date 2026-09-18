"""Real CPU HTTP checks; opt in with SUROGATE_TTS_TEST_MODEL (HF ID or native directory)."""

import hashlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


def test_released_voices_over_http(tmp_path):
    model = os.environ.get("SUROGATE_TTS_TEST_MODEL")
    if not model:
        pytest.skip("set SUROGATE_TTS_TEST_MODEL to run the published native CPU model")
    root = Path(__file__).resolve().parents[2]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "surogate.cli.serve",
        "--tts",
        model,
        "--port",
        str(port),
        "--api-key",
        "tts-integration-test",
        "--served-model-name",
        "tts-test",
    ]
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(root)},
        )
    try:
        url = f"http://127.0.0.1:{port}"
        with httpx.Client(
            base_url=url, headers={"Authorization": "Bearer tts-integration-test"}, timeout=120
        ) as client:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                assert process.poll() is None, log_path.read_text()[-4000:]
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.2)
            else:
                pytest.fail("TTS server did not become ready")
            assert httpx.get(url + "/health").status_code == 401
            assert client.get("/v1/models").json()["data"][0]["id"] == "tts-test"
            names = [v["id"] for v in client.get("/v1/audio/voices").json()["data"]]
            assert {"Doina", "Tudor", "Radu"} <= set(names)
            expected = {
                "Doina": "72fc2280ef9256a0637d67a9e5a78983324354a1e61e45d9a9c01e717e274206",
                "Tudor": "75c8d92d8d2d8b684f34cc6e46ac2f47c652bd6e18e5542ddc8f5f4a37e58d8e",
                "Radu": "018c5b37b7fd93b5726151e926820799832eec3d8ffe1e30a220d4f59d7baa21",
            }
            for voice, digest in expected.items():
                response = client.post(
                    "/v1/audio/speech",
                    json={
                        "model": "tts-test",
                        "voice": voice,
                        "response_format": "wav",
                        "seed": 9,
                        "input": "Bună, ce faci? Eu tocmai am ajuns acasă și încerc să-mi dau seama ce să mănânc.",
                    },
                )
                assert response.status_code == 200, response.text[:500] if response.is_error else ""
                assert hashlib.sha256(response.content).hexdigest() == digest
                (tmp_path / f"{voice.lower()}.wav").write_bytes(response.content)
            assert "libcuda.so" not in Path(f"/proc/{process.pid}/maps").read_text()
            assert "/torch/" not in Path(f"/proc/{process.pid}/maps").read_text()
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            pytest.fail("TTS server failed to shut down")
    assert log_path.read_text().count("loaded MagpieTTS GGUF:") == 1
