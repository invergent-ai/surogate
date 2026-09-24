"""Servers read their API key from a file, and the key stays off the command line (SUROGATE-CHANGES #17).

Each case starts a real server with --api-key-file and is skipped unless its model is named:
SUROGATE_KEYFILE_TEST_STT (a prepared STT model, served on the CPU), SUROGATE_KEYFILE_TEST_TTS (a
prepared TTS voice directory) and SUROGATE_KEYFILE_TEST_SERVER (a prepared chat model, which needs a
free GPU selected with CUDA_VISIBLE_DEVICES). SUROGATE_STT_BIN, SUROGATE_TTS_BIN and SUROGATE_SERVE_BIN
pick the binaries under test.

The server must authenticate with the key from the file (401 without it, 200 with it), and
/proc/<pid>/cmdline must not contain the key. Giving both --api-key and --api-key-file is refused,
and so is an empty key-file path (an unset variable in a unit file), which must never leave a
server without authentication.
"""

import os
import secrets
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

from surogate.cli.serve import _resolve_binary

CASES = {
    # mode: (environment variable naming the model, extra flags)
    "stt": ("SUROGATE_KEYFILE_TEST_STT", ["--device", "cpu", "--threads", "2"]),
    "tts": ("SUROGATE_KEYFILE_TEST_TTS", ["--threads", "2"]),
    "server": ("SUROGATE_KEYFILE_TEST_SERVER", ["--max-model-len", "2048", "--max-num-seqs", "1"]),
}


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def key_file(tmp_path):
    key = "sk-" + secrets.token_hex(16)
    path = tmp_path / "api.key"
    path.write_text(key + "\n")
    path.chmod(0o600)
    return key, path


@pytest.mark.parametrize("mode", list(CASES))
def test_key_from_a_file_authenticates_and_stays_off_the_command_line(tmp_path, mode):
    variable, extra = CASES[mode]
    model = os.getenv(variable)
    if not model:
        pytest.skip(f"set {variable}")
    key, path = key_file(tmp_path)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log = tmp_path / "server.log"
    cmd = [_resolve_binary(mode), model, "--host", "127.0.0.1", "--port", str(port),
           "--api-key-file", str(path), *extra]
    with log.open("w") as output:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
        try:
            auth = {"Authorization": f"Bearer {key}"}
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                assert process.poll() is None, log.read_text()
                try:
                    status = requests.get(base + "/v1/models", headers=auth, timeout=1).status_code
                    assert status != 401, "the server does not accept the key from its file"
                    if status == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            else:
                pytest.fail(log.read_text())
            assert requests.get(base + "/v1/models", timeout=5).status_code == 401
            wrong = {"Authorization": "Bearer sk-wrong"}
            assert requests.get(base + "/v1/models", headers=wrong, timeout=5).status_code == 401
            assert requests.get(base + "/v1/models", headers=auth, timeout=5).status_code == 200
            cmdline = Path(f"/proc/{process.pid}/cmdline").read_bytes()
            assert key.encode() not in cmdline
            assert key not in log.read_text()
        finally:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.mark.parametrize("mode", list(CASES))
def test_a_key_and_a_key_file_together_are_refused(tmp_path, mode):
    variable, extra = CASES[mode]
    model = os.getenv(variable)
    if not model:
        pytest.skip(f"set {variable}")
    key, path = key_file(tmp_path)
    cmd = [_resolve_binary(mode), model, "--host", "127.0.0.1", "--port", str(free_port()),
           "--api-key", "sk-other", "--api-key-file", str(path), *extra]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode != 0
    assert "not both" in result.stdout + result.stderr, result.stdout + result.stderr
    assert key not in result.stdout + result.stderr


@pytest.mark.parametrize("mode", list(CASES))
def test_an_empty_key_file_path_is_refused(tmp_path, mode):
    variable, extra = CASES[mode]
    model = os.getenv(variable)
    if not model:
        pytest.skip(f"set {variable}")
    cmd = [_resolve_binary(mode), model, "--host", "127.0.0.1", "--port", str(free_port()),
           "--api-key-file", "", *extra]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode != 0
    assert "cannot read --api-key-file" in result.stdout + result.stderr, result.stdout + result.stderr
