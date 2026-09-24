"""Realtime transcription over one WebSocket (SUROGATE-CHANGES #9).

Starts a real STT server on the CPU; skipped unless SUROGATE_WS_TEST_STT names a prepared streaming
STT model and SUROGATE_WS_TEST_AUDIO a mono 16 kHz PCM16 WAV file. SUROGATE_STT_BIN picks the binary.

A client that streams audio over one WebSocket receives exactly the events the HTTP interface
returns for the same chunks, and the stream is counted and cleaned up like an HTTP one. Protocol
errors get a close frame with their code, and no message, however malformed, stops the server.
The framing itself is unit-tested without a model (csrc/src/testing/serve/test_websocket.cpp).
"""

import base64
import json
import os
import socket
import subprocess
import time
import wave

import pytest
import requests

from surogate.cli.serve import _resolve_binary

websockets_sync = pytest.importorskip("websockets.sync.client")
websockets_exceptions = pytest.importorskip("websockets.exceptions")
from websockets.exceptions import ConnectionClosed, InvalidStatus  # noqa: E402

CHUNK = 3200  # 100 ms of PCM16 at 16 kHz


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    model, audio = os.getenv("SUROGATE_WS_TEST_STT"), os.getenv("SUROGATE_WS_TEST_AUDIO")
    binary = _resolve_binary("stt")
    if not model or not audio or binary is None:
        pytest.skip("set SUROGATE_WS_TEST_STT and SUROGATE_WS_TEST_AUDIO, and build surogate-stt")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    log = tmp_path_factory.mktemp("ws") / "stt.log"
    with log.open("w") as output:
        process = subprocess.Popen([binary, model, "--host", "127.0.0.1", "--port", str(port), "--device", "cpu",
                                    "--threads", "2", "--max-num-seqs", "2"], stdout=output, stderr=subprocess.STDOUT)
    base = f"127.0.0.1:{port}"
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        assert process.poll() is None, log.read_text()
        try:
            if requests.get(f"http://{base}/v1/models", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(0.3)
    with wave.open(audio) as source:
        pcm = source.readframes(source.getnframes())
    yield base, pcm
    process.terminate()
    process.wait(timeout=60)


def open_streams(base):
    for line in requests.get(f"http://{base}/metrics", timeout=10).text.splitlines():
        if line.startswith("surogate_streams_open"):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError("no surogate_streams_open")


def http_events(base, pcm):
    stream = requests.post(f"http://{base}/v1/audio/streams", timeout=60).json()["id"]
    events = []
    for offset in range(0, len(pcm), CHUNK):
        r = requests.post(f"http://{base}/v1/audio/streams/{stream}", data=pcm[offset:offset + CHUNK], timeout=120)
        events += r.json()["events"]
    events += requests.post(f"http://{base}/v1/audio/streams/{stream}?finish=true", timeout=120).json()["events"]
    return events


def websocket_events(base, pcm):
    events = []
    with websockets_sync.connect(f"ws://{base}/v1/audio/streams", max_size=None) as ws:
        ready = json.loads(ws.recv())
        assert ready == {"type": "ready", "sample_rate": 16000, "channels": 1, "encoding": "pcm_s16le"}
        assert open_streams(base) == 1
        for offset in range(0, len(pcm), CHUNK):
            ws.send(pcm[offset:offset + CHUNK])  # no waiting for a reply between chunks
        ws.send(json.dumps({"type": "finish"}))
        while True:
            event = json.loads(ws.recv())
            if event["type"] == "done":
                break
            events.append(event)
        with pytest.raises(ConnectionClosed):
            ws.recv()
    return events


def without_timings(events):
    """The events without their wall-clock measurements (finalize_ms and the like)."""
    return [{k: v for k, v in event.items() if not k.endswith("_ms")} for event in events]


def test_websocket_events_match_the_http_interface(server):
    base, pcm = server
    over_http = http_events(base, pcm)
    over_websocket = websocket_events(base, pcm)
    assert without_timings(over_websocket) == without_timings(over_http)
    assert any(event["type"] == "final" and event.get("text") for event in over_websocket)
    assert open_streams(base) == 0


def test_a_websocket_closed_without_finishing_discards_its_stream(server):
    base, pcm = server
    with websockets_sync.connect(f"ws://{base}/v1/audio/streams") as ws:
        json.loads(ws.recv())
        ws.send(pcm[:CHUNK])
        assert open_streams(base) == 1
        started = time.monotonic()
        ws.close()
        assert time.monotonic() - started < 0.8  # the server answers the close at once, not after 1 s
    deadline = time.monotonic() + 10
    while open_streams(base) != 0:
        assert time.monotonic() < deadline
        time.sleep(0.1)


def test_protocol_errors(server):
    base, _ = server
    plain = requests.get(f"http://{base}/v1/audio/streams", timeout=10)
    assert plain.status_code == 426 and plain.headers["Upgrade"] == "websocket"
    wrong_version = requests.get(f"http://{base}/v1/audio/streams", timeout=10, headers={
        "Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Version": "8",
        "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ=="})
    assert wrong_version.status_code == 426 and wrong_version.headers["Sec-WebSocket-Version"] == "13"
    no_key = requests.get(f"http://{base}/v1/audio/streams", timeout=10, headers={
        "Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Version": "13"})
    assert no_key.status_code == 400
    with websockets_sync.connect(f"ws://{base}/v1/audio/streams") as ws:
        json.loads(ws.recv())
        ws.send(json.dumps({"type": "unknown"}))
        assert json.loads(ws.recv())["type"] == "error"
        with pytest.raises(ConnectionClosed) as closed:
            ws.recv()
        assert closed.value.rcvd.code == 1003
    with websockets_sync.connect(f"ws://{base}/v1/audio/streams") as ws:
        json.loads(ws.recv())
        ws.send(b"\x00")  # half a sample
        assert json.loads(ws.recv())["type"] == "error"
        with pytest.raises(ConnectionClosed) as closed:
            ws.recv()
        assert closed.value.rcvd.code == 1007
    # Malformed control messages are refused, and never stop the server.
    for control in ('{"type": 1}', '{"type": null}', '[]', '"finish"', '{"type": {"finish": 1}}'):
        with websockets_sync.connect(f"ws://{base}/v1/audio/streams") as ws:
            json.loads(ws.recv())
            ws.send(control)
            assert json.loads(ws.recv())["type"] == "error"
            with pytest.raises(ConnectionClosed) as closed:
                ws.recv()
            assert closed.value.rcvd.code == 1003
    # A message over the size limit (10 s of audio) is a protocol error with its own close code.
    with websockets_sync.connect(f"ws://{base}/v1/audio/streams", max_size=None) as ws:
        json.loads(ws.recv())
        ws.send(b"\x00\x00" * 160001)
        assert json.loads(ws.recv())["type"] == "error"
        with pytest.raises(ConnectionClosed) as closed:
            ws.recv()
        assert closed.value.rcvd.code == 1009
    assert requests.get(f"http://{base}/health", timeout=10).status_code == 200
    assert open_streams(base) == 0


def test_an_abandoned_handshake_frees_its_stream(server):
    """A client that sends the handshake and leaves before the 101 still gives its stream back."""
    base, _ = server
    host, port = base.split(":")
    for _ in range(2):
        with socket.create_connection((host, int(port))) as raw:
            raw.sendall(("GET /v1/audio/streams HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
                         "Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: "
                         + base64.b64encode(os.urandom(16)).decode() + "\r\n\r\n").encode())
            raw.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
    deadline = time.monotonic() + 10
    while open_streams(base) != 0:
        assert time.monotonic() < deadline, "an abandoned handshake still holds its stream"
        time.sleep(0.1)
    # The server may not have seen the handshakes yet: it takes them, then gives them back. A stream
    # that stays taken fails both places here for good.
    def connect():
        while True:
            try:
                return websockets_sync.connect(f"ws://{base}/v1/audio/streams")
            except websockets_exceptions.InvalidStatus as refused:
                assert refused.response.status_code == 429 and time.monotonic() < deadline, \
                    "an abandoned handshake still holds its stream"
                time.sleep(0.1)
    deadline = time.monotonic() + 10
    with connect() as first, connect() as second:
        assert json.loads(first.recv())["type"] == json.loads(second.recv())["type"] == "ready"


def test_a_websocket_stream_lives_as_long_as_its_connection(server):
    """Pings keep a quiet connection open, and its stream does not expire like an HTTP one (2 min)."""
    base, pcm = server
    with websockets_sync.connect(f"ws://{base}/v1/audio/streams", ping_interval=20) as ws:
        json.loads(ws.recv())
        ws.send(pcm[:CHUNK])
        deadline = time.monotonic() + 130
        while time.monotonic() < deadline:
            assert open_streams(base) == 1
            time.sleep(5)
        ws.send(pcm[CHUNK:2 * CHUNK])
        ws.send(json.dumps({"type": "finish"}))
        while (event := json.loads(ws.recv()))["type"] != "done":
            assert event["type"] != "error", event
    assert open_streams(base) == 0


def test_the_live_stream_limit_is_an_http_refusal(server):
    base, _ = server
    with websockets_sync.connect(f"ws://{base}/v1/audio/streams") as first, \
         websockets_sync.connect(f"ws://{base}/v1/audio/streams") as second:
        json.loads(first.recv())
        json.loads(second.recv())
        with pytest.raises(InvalidStatus) as refused:
            websockets_sync.connect(f"ws://{base}/v1/audio/streams")
        assert refused.value.response.status_code == 429
