"""Decisions v1, end to end on a served model: the answers to a fixed request set are compared with a recording.

GPU. Skipped unless SUROGATE_DECISIONS_V1_ARTIFACT names a prepared chat model (.sinfer); select one
free GPU with CUDA_VISIBLE_DEVICES, and add engine flags with SUROGATE_DECISIONS_V1_ARGS (a JSON list).

The requests are the model-free golden set (csrc/src/testing/serve/fixtures/serve/decisions_v1/
golden.json), sent to /v1/decisions on an otherwise idle server.

- SUROGATE_DECISIONS_V1_RECORD=<file> writes the served answers there, with the artifact and the server
  arguments they were served with. Record from a v1 engine.
- SUROGATE_DECISIONS_V1_GOLDEN=<file> requires the same answers again: the same response envelope,
  every choice identical, the same usage.input_tokens (which pins the rendered prompts' token
  counts), and every probability, confidence, noul and score within SUROGATE_DECISIONS_V1_TOLERANCE.
  The default, 1e-6, is for a different engine build or GPU, whose kernels may round the logits
  differently in the last bits; set it to 0 to require the same build to reproduce its own
  recording exactly. The recording's artifact and server arguments must match this run's.

Every request is also sent to the two aliases, which are the same endpoint: on an idle server they
must answer identically, bit for bit (run-to-run determinism).
"""

import json
import math
import os
import re
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(not os.getenv("SUROGATE_DECISIONS_V1_ARTIFACT"),
                                reason="needs a prepared chat model and a free GPU")

GOLDEN = (Path(__file__).resolve().parents[2]
          / "csrc/src/testing/serve/fixtures/serve/decisions_v1/golden.json")
ROUTES = ("/v1/decisions", "/api/alpha/decisions", "/api/v1/decisions")
ARGS = json.loads(os.getenv("SUROGATE_DECISIONS_V1_ARGS", "[]"))


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    log = tmp_path_factory.mktemp("decisions-v1") / "server.log"
    cmd = [_resolve_binary("server"), os.environ["SUROGATE_DECISIONS_V1_ARTIFACT"],
           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "rune",
           "--max-model-len", "8192", "--max-num-seqs", "8", *ARGS]
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


def ask(url, body, questions, route=ROUTES[0]):
    response = requests.post(url + route, data=body.encode(), timeout=300,
                             headers={"Content-Type": "application/json"})
    assert response.ok, response.text
    out = response.json()
    # The v1 response envelope.
    assert list(out) == ["id", "model", "provider", "answers", "usage"], list(out)
    assert re.fullmatch(r"dec-[0-9a-f]{16}", out["id"]), out["id"]
    assert out["model"] == "rune" and out["provider"] == "surogate"
    assert list(out["usage"]) == ["input_tokens", "output_tokens", "cost"]
    assert out["usage"]["output_tokens"] == questions and out["usage"]["cost"] == 0
    assert list(out["answers"]) == list(json.loads(body)["questions"])
    return {"answers": out["answers"], "input_tokens": out["usage"]["input_tokens"]}


NUMBERS = ("confidence", "noul", "score")


def compare(name, got, want, tolerance):
    assert got["input_tokens"] == want["input_tokens"], (name, "rendered prompt length changed")
    assert list(got["answers"]) == list(want["answers"]), name
    for question, answer in got["answers"].items():
        expected = want["answers"][question]
        where = (name, question)
        assert list(answer) == list(expected), where
        assert answer["type"] == expected["type"], where
        if "choice" in expected:
            assert answer["choice"] == expected["choice"], where
        if "legend" in expected:
            assert answer["legend"] == expected["legend"], where
        for field in NUMBERS:
            if field in expected:
                assert math.isclose(answer[field], expected[field], rel_tol=0.0, abs_tol=tolerance), (*where, field)
        if "probabilities" in expected:
            assert list(answer["probabilities"]) == list(expected["probabilities"]), where
            for key, value in expected["probabilities"].items():
                assert math.isclose(answer["probabilities"][key], value, rel_tol=0.0, abs_tol=tolerance), (*where, key)


def test_served_v1_answers_match_the_recording(base):
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]
    count = {case["name"]: len(json.loads(case["body"])["questions"]) for case in cases}
    served = {case["name"]: ask(base, case["body"], count[case["name"]]) for case in cases}
    for case in cases:  # the aliases are the same endpoint: identical answers on an idle server
        for route in ROUTES[1:]:
            assert ask(base, case["body"], count[case["name"]], route) == served[case["name"]], (case["name"], route)

    setup = {"artifact": os.environ["SUROGATE_DECISIONS_V1_ARTIFACT"], "args": ARGS}
    record = os.getenv("SUROGATE_DECISIONS_V1_RECORD")
    if record:
        Path(record).write_text(json.dumps({"version": "v1", "setup": setup, "answers": served}, indent=1) + "\n")
        return
    golden_path = os.getenv("SUROGATE_DECISIONS_V1_GOLDEN")
    if not golden_path:
        pytest.skip("set SUROGATE_DECISIONS_V1_GOLDEN to compare, or SUROGATE_DECISIONS_V1_RECORD to record")
    golden = json.loads(Path(golden_path).read_text())
    assert golden["setup"] == setup, "the recording was made with another model or other server arguments"
    tolerance = float(os.getenv("SUROGATE_DECISIONS_V1_TOLERANCE", "1e-6"))
    assert list(served) == list(golden["answers"])
    for name, got in served.items():
        compare(name, got, golden["answers"][name], tolerance)
