"""Embedding API regression on CPU and GPU with a prepared local encoder.

Set SUROGATE_EMBED_TEST_ARTIFACT; GPU tests use SUROGATE_EMBED_TEST_DEVICE (default 0).
SUROGATE_EMBED_TEST_SANITIZER=1 wraps the GPU server in Compute Sanitizer.
"""

import base64
import math
import os
import signal
from pathlib import Path
import socket
import struct
import subprocess
import time

import pytest
import requests


pytestmark = pytest.mark.skipif(
    not os.getenv("SUROGATE_EMBED_TEST_ARTIFACT"), reason="requires a prepared embedding artifact"
)


@pytest.fixture(scope="module", params=["cpu", "gpu"])
def encoder(request, tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    binary = Path(os.getenv("SUROGATE_EMBED_BIN", str(root / "csrc/build-serve/surogate-embed")))
    if not binary.is_file():
        pytest.skip("embedding server is not built")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [str(binary), os.environ["SUROGATE_EMBED_TEST_ARTIFACT"], "--port", str(port),
               "--device", "cpu" if request.param == "cpu" else os.getenv("SUROGATE_EMBED_TEST_DEVICE", "0"),
               "--served-model-name", "test-encoder"]
    sanitizer = request.param == "gpu" and os.getenv("SUROGATE_EMBED_TEST_SANITIZER") == "1"
    if sanitizer:
        command = ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99"] + command
    log_path = tmp_path_factory.mktemp("embedding-http") / "server.log"
    env = dict(os.environ, OMP_NUM_THREADS="4", OMP_WAIT_POLICY="ACTIVE")
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(log_path.read_text())
                try:
                    if requests.get(url + "/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail("encoder did not become ready: " + log_path.read_text())
            yield url
        finally:
            # Include the sanitizer's launcher and target, which outlive a signal
            # sent only to the outer process.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    output = log_path.read_text()
    assert "Invalid __global__" not in output and "cudaError" not in output, output
    if sanitizer:
        # Servers are terminated after completed requests, so the sanitizer may
        # not print its normal-exit summary. All requests must finish successfully
        # and no device errors may have been emitted during those synchronized calls.
        assert "COMPUTE-SANITIZER" in output, output


def post(encoder, **fields):
    return requests.post(encoder + "/v1/embeddings",
                         json={"model": "test-encoder", "input": [1, 2, 3], **fields}, timeout=60)


def vector(response):
    assert response.status_code == 200, response.text
    return response.json()["data"][0]["embedding"]


def test_dimensions_and_encoding(encoder):
    full = vector(post(encoder))
    for dimension in [1, 128, len(full)]:
        shortened = vector(post(encoder, dimensions=dimension, encoding_format="float"))
        assert len(shortened) == dimension
        expected = full[:dimension]
        if dimension != len(full):
            norm = math.sqrt(sum(v * v for v in expected))
            expected = [v / norm for v in expected]
        assert shortened == pytest.approx(expected, abs=1e-7, rel=1e-6)
        encoded = vector(post(encoder, dimensions=dimension, encoding_format="base64"))
        decoded = base64.b64decode(encoded, validate=True)
        assert len(decoded) == dimension * 4
        assert struct.unpack(f"<{dimension}f", decoded) == tuple(shortened)


def test_model_identity(encoder):
    models = requests.get(encoder + "/v1/models", timeout=5).json()["data"]
    assert [model["id"] for model in models] == ["test-encoder"]
    response = post(encoder, model="unknown-encoder")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
    assert post(encoder).json()["model"] == "test-encoder"
    default = requests.post(encoder + "/v1/embeddings", json={"input": [1, 2, 3]}, timeout=60)
    assert default.ok and default.json()["model"] == "test-encoder", default.text


def test_invalid_inputs_preserve_server(encoder):
    full = vector(post(encoder))
    invalid = [
        {"input": [-1]}, {"input": [2147483647]}, {"input": [4294967296]},
        {"input": [18446744073709551615]}, {"input": [1.5]}, {"input": [True]},
        {"input": [[1], []]}, {"input": []}, {"input": ""}, {"model": False},
        {"dimensions": 0}, {"dimensions": len(full) + 1}, {"dimensions": True},
        {"dimensions": 1.5}, {"encoding_format": "unknown"}, {"encoding_format": 1},
    ]
    for fields in invalid:
        response = post(encoder, **fields)
        assert response.status_code == 400, (fields, response.text)
        assert response.json()["error"]["param"] == next(iter(fields))
        assert vector(post(encoder)) == full


def test_batch_and_text_input(encoder):
    response = post(encoder, input=[[1], [1, 2, 3]], dimensions=128)
    assert response.ok, response.text
    body = response.json()
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert all(len(item["embedding"]) == 128 for item in body["data"])
    assert body["usage"]["prompt_tokens"] == 4
    for text in ["Hello", ["Hello", "World"]]:
        response = post(encoder, input=text, dimensions=128)
        assert response.ok, response.text


def test_many_single_token_sequences(encoder, request):
    if request.node.callspec.params["encoder"] != "gpu":
        pytest.skip("GPU arena regression")
    expected = vector(post(encoder, input=[42], dimensions=128))
    # Fill the entire batch budget, then cross it to exercise chunking. These
    # requests previously exhausted the arena when allocating the output head.
    for count in (7000, 8193):
        response = post(encoder, input=[[42]] * count, dimensions=128)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["usage"]["prompt_tokens"] == count
        assert len(body["data"]) == count
        for index, item in enumerate(body["data"]):
            assert item["index"] == index
            actual = item["embedding"]
            assert len(actual) == len(expected)
            assert sum(v * v for v in actual) == pytest.approx(1.0, abs=1e-5)
            # BF16 projections select different kernels for a wide batch.
            assert sum(a * b for a, b in zip(actual, expected)) > 0.995


def test_batch_preserves_sequence_boundaries(encoder, request):
    if request.node.callspec.params["encoder"] != "cpu":
        pytest.skip("CPU batch packing regression")
    # Different lengths, query-tile tails, and more than one 2,048-token batch.
    inputs = [[42, 71, 93], [137, 53] * 257 + [93], [71] * 1531, [42], [93] * 128]
    expected = [vector(post(encoder, input=ids)) for ids in inputs]
    for order in [list(range(len(inputs))), list(reversed(range(len(inputs))))]:
        response = post(encoder, input=[inputs[i] for i in order])
        assert response.ok, response.text
        payload = response.json()
        assert payload["usage"]["prompt_tokens"] == sum(map(len, inputs))
        assert len(payload["data"]) == len(inputs)
        for index, item in enumerate(payload["data"]):
            assert item["index"] == index
            actual = item["embedding"]
            assert all(math.isfinite(value) for value in actual)
            assert sum(a * b for a, b in zip(actual, expected[order[index]])) > 0.999
