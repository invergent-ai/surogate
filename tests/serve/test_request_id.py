"""The caller's X-Request-Id in the request log, and dropped streams settled from it. GPU; pending.

Skipped unless SUROGATE_REQUEST_ID_TEST_ARTIFACT names a prepared chat model (.sinfer); select one free
GPU with CUDA_VISIBLE_DEVICES, and add engine flags with SUROGATE_REQUEST_ID_TEST_ARGS (a JSON list).

- Every endpoint echoes X-Request-Id, refusals included, and writes it into its request records:
  chat, completions, Anthropic Messages, Responses and decisions.
- A stream the client drops mid-response still ends in a request_done record that carries the
  caller's id and the tokens actually generated, and in no request_error, so the caller can settle
  it (SUROGATE-CHANGES #2).
- Every request ends in exactly one terminal record, however early the client hangs up: a
  request_rejected when it left while the request was being prepared, a request_done once the
  prompt was submitted (parallel decoding included).
"""

import json
import os
import socket
import struct
import subprocess
import time
import uuid

import pytest
import requests

from surogate.cli.serve import _resolve_binary

pytestmark = pytest.mark.skipif(not os.getenv("SUROGATE_REQUEST_ID_TEST_ARTIFACT"),
                                reason="needs a prepared chat model and a free GPU")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    root = tmp_path_factory.mktemp("request-id")
    log, jsonl = root / "server.log", root / "requests.jsonl"
    cmd = [_resolve_binary("server"), os.environ["SUROGATE_REQUEST_ID_TEST_ARTIFACT"],
           "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "test",
           "--max-model-len", "4096", "--max-num-seqs", "2", "--request-log-jsonl", str(jsonl), "--cors",
           *json.loads(os.getenv("SUROGATE_REQUEST_ID_TEST_ARGS", "[]"))]
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
            yield url, jsonl
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


TERMINAL = ("request_done", "request_error", "request_rejected")


def records(jsonl, rid, deadline=10.0, settle=1.0):
    """The request records carrying `rid`, once its request has finished.

    After the first terminal record it keeps reading for `settle` seconds, so a second terminal
    record for the same request -- which a caller would double-count -- shows up in the result.
    """
    def read():
        found = []
        for line in jsonl.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:  # a line still being written
                continue
            if record["event"].startswith("request_") and record["request"].get("client_request_id") == rid:
                found.append(record)
        return found

    end = time.monotonic() + deadline
    while True:
        found = read()
        if any(r["event"] in TERMINAL for r in found):
            time.sleep(settle)
            return read()
        if time.monotonic() > end:
            return found
        time.sleep(0.1)


PROMPT = "Count slowly from one to five hundred, writing every number as a word, one per line."

CASES = {
    "chat": ("/v1/chat/completions", {"model": "test", "messages": [{"role": "user", "content": PROMPT}]}),
    "completions": ("/v1/completions", {"model": "test", "prompt": PROMPT}),
    "messages": ("/v1/messages", {"model": "test", "messages": [{"role": "user", "content": PROMPT}]}),
    "responses": ("/v1/responses", {"model": "test", "input": PROMPT}),
}


# The SSE data lines that carry generated text, per protocol: hanging up after a few of them drops
# the stream mid-answer. Counting bytes instead is not enough: a Responses stream opens with ~700
# bytes of lifecycle events before its first token. Only `data:` lines count, so an `event:` line
# naming the same event is not counted twice.
DELTA_MARKERS = {
    "chat": '"delta"',
    "completions": '"text"',
    "messages": "content_block_delta",
    "responses": '_text.delta"',
}


def limited(name, body, tokens):
    """`body` capped at `tokens` output tokens (Responses accepts no fewer than 16)."""
    body = dict(body)
    if name == "responses":
        body["max_output_tokens"] = max(tokens, 16)
    else:
        body["max_tokens"] = tokens
    return body


@pytest.mark.parametrize("name", list(CASES))
def test_id_is_echoed_and_logged(server, name):
    url, jsonl = server
    path, body = CASES[name]
    rid = f"gw-{uuid.uuid4()}"
    response = requests.post(url + path, json=limited(name, body, 16), headers={"X-Request-Id": rid}, timeout=300)
    assert response.ok, response.text
    assert response.headers["X-Request-Id"] == rid
    events = [r["event"] for r in records(jsonl, rid)]
    assert events == ["request_start", "request_done"], events


def test_decisions_and_refusals_carry_the_id(server):
    url, jsonl = server
    rid = f"gw-{uuid.uuid4()}"
    body = {"model": "test", "state": "The parcel arrived broken.",
            "questions": {"refund": {"type": "noul", "instructions": "Is a refund due?",
                                     "criteria": {"true": "yes", "false": "no"}}}}
    response = requests.post(url + "/v1/decisions", json=body, headers={"X-Request-Id": rid}, timeout=300)
    assert response.ok and response.headers["X-Request-Id"] == rid
    assert [r["event"] for r in records(jsonl, rid)] == ["request_start", "request_done"]
    # A refused request is echoed and logged as rejected.
    rid = f"gw-{uuid.uuid4()}"
    bad = dict(body, questions={"q": {"type": "guess", "instructions": "i", "criteria": {"a": "a", "b": "b"}}})
    response = requests.post(url + "/v1/decisions", json=bad, headers={"X-Request-Id": rid}, timeout=60)
    assert response.status_code == 400 and response.headers["X-Request-Id"] == rid
    assert [r["event"] for r in records(jsonl, rid)] == ["request_rejected"]
    # An unsafe id is neither echoed nor logged.
    response = requests.post(url + "/v1/decisions", json=body, headers={"X-Request-Id": "has spaces"}, timeout=300)
    assert response.ok and "X-Request-Id" not in response.headers


def test_browsers_may_send_and_read_the_id(server):
    url, _ = server
    preflight = requests.options(url + "/v1/chat/completions", timeout=30, headers={
        "Origin": "https://example.test", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,x-request-id"})
    assert "x-request-id" in preflight.headers["Access-Control-Allow-Headers"].lower()
    response = requests.get(url + "/v1/models", headers={"X-Request-Id": "cors-1"}, timeout=30)
    assert response.headers["X-Request-Id"] == "cors-1"
    assert "x-request-id" in response.headers["Access-Control-Expose-Headers"].lower()


@pytest.mark.parametrize("name", list(CASES))
def test_dropped_stream_is_settled_from_request_done(server, name):
    url, jsonl = server
    path, body = CASES[name]
    rid = f"gw-{uuid.uuid4()}"
    body = dict(limited(name, body, 1024), stream=True)
    with requests.post(url + path, json=body, headers={"X-Request-Id": rid}, stream=True, timeout=300) as response:
        assert response.ok and response.headers["X-Request-Id"] == rid
        deltas = 0
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data:") and DELTA_MARKERS[name] in line:
                deltas += 1
                if deltas >= 4:  # a few tokens into the answer, then hang up
                    break
        assert deltas >= 4, "the stream ended before four text deltas"
    found = records(jsonl, rid, deadline=60)
    events = [r["event"] for r in found]
    assert events == ["request_start", "request_done"], events
    done = found[-1]["result"]
    assert done["finish_reason"] == "cancelled"
    assert 0 < done["completion_tokens"] < 1024
    assert done["prompt_tokens"] > 0


PARALLEL = {"model": "test", "parallel_decoding": True, "max_tokens": 64, "stream": True,
            "messages": [{"role": "user", "content": "Our production server is down. Please help."}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "triage", "strict": True, "schema": {
                    "type": "object",
                    "properties": {"urgent": {"type": "boolean"},
                                   "category": {"type": "string", "enum": ["outage", "billing", "other"]}},
                    "required": ["urgent", "category"], "additionalProperties": False}}}}


def wait_for_start(jsonl, rid, deadline=60.0):
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        for line in jsonl.read_text().splitlines():
            if rid in line and '"request_start"' in line:
                return
        time.sleep(0.005)
    pytest.fail(f"no request_start for {rid}")


@pytest.mark.parametrize("when", ["at_once", "after_start"])
@pytest.mark.parametrize("name", [*CASES, "parallel"])
def test_early_hang_up_ends_in_one_terminal_record(server, name, when):
    """The client sends its request and resets the connection, either at once or as soon as the
    request is logged as started. At once, the server usually sees the reset while preparing the
    request and rejects it; once started, the request is settled from request_done. Either way
    there is exactly one terminal record."""
    url, jsonl = server
    if name == "parallel":
        path, body = "/v1/chat/completions", PARALLEL
    else:
        path, body = CASES[name]
        body = dict(limited(name, body, 1024), stream=True)
    rid = f"gw-{uuid.uuid4()}"
    payload = json.dumps(body).encode()
    host, port = url.removeprefix("http://").split(":")
    head = (f"POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n"
            f"X-Request-Id: {rid}\r\nContent-Length: {len(payload)}\r\n\r\n").encode()
    with socket.create_connection((host, int(port))) as client:
        client.sendall(head + payload)
        if when == "after_start":
            wait_for_start(jsonl, rid)
        # SO_LINGER 0: close sends a reset, so the server's next read or write fails.
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    found = records(jsonl, rid, deadline=60)
    events = [r["event"] for r in found]
    if events == ["request_rejected"]:
        assert when == "at_once", "a started request was rejected"
        assert found[0]["error"]["code"] == "client_disconnected", found[0]
        return
    assert events == ["request_start", "request_done"], events
    done = found[-1]["result"]
    assert done["prompt_tokens"] > 0
    assert done["completion_tokens"] < body.get("max_tokens", body.get("max_output_tokens"))
