"""The STT and TTS servers report their requests in flight on /metrics (SUROGATE-CHANGES #18).

Each case starts a real server on the CPU and is skipped unless its model is named:
SUROGATE_METRICS_TEST_STT (a prepared streaming STT model) with SUROGATE_METRICS_TEST_AUDIO (a mono
16 kHz PCM16 WAV file), and SUROGATE_METRICS_TEST_TTS (a prepared TTS voice directory).
SUROGATE_STT_BIN and SUROGATE_TTS_BIN pick the binaries under test.

- STT: a realtime stream is counted open from creation until it is finished or deleted; a file
  transcription is counted running while it is served; requests and audio seconds are counted.
- TTS: a speech request is counted running until its response ends, including one whose client
  disconnects early; requests, characters and audio seconds are counted.
"""

import os
import re
import socket
import subprocess
import threading
import time
import wave
from pathlib import Path

import pytest
import requests

from surogate.cli.serve import _resolve_binary


RUNNING = 'surogate_requests{state="running"}'


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class Server:
    def __init__(self, mode, model, tmp_path, *extra):
        binary = _resolve_binary(mode)
        if binary is None:
            pytest.skip(f"the {mode} server is not built")
        self.base = f"http://127.0.0.1:{free_port()}"
        port = self.base.rsplit(":", 1)[1]
        self.log = tmp_path / f"{mode}.log"
        self.output = self.log.open("w")
        self.process = subprocess.Popen([binary, model, "--host", "127.0.0.1",
                                         "--port", port, "--threads", "2", *extra],
                                        stdout=self.output, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            assert self.process.poll() is None, self.log.read_text()
            try:
                if requests.get(self.base + "/v1/models", timeout=1).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.3)
        pytest.fail(self.log.read_text())

    def metrics(self):
        text = requests.get(self.base + "/metrics", timeout=10).text
        values = {}
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            name, value = line.rsplit(" ", 1)
            values[re.sub(r"model=\"[^\"]*\",?", "", name).replace("{}", "")] = float(value)
        return values

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.output.close()


@pytest.fixture
def stt(tmp_path):
    model = os.getenv("SUROGATE_METRICS_TEST_STT")
    if not model or not os.getenv("SUROGATE_METRICS_TEST_AUDIO"):
        pytest.skip("set SUROGATE_METRICS_TEST_STT and SUROGATE_METRICS_TEST_AUDIO")
    server = Server("stt", model, tmp_path, "--device", "cpu")
    yield server
    server.close()


@pytest.fixture
def tts(tmp_path):
    model = os.getenv("SUROGATE_METRICS_TEST_TTS")
    if not model:
        pytest.skip("set SUROGATE_METRICS_TEST_TTS")
    server = Server("tts", model, tmp_path)
    yield server
    server.close()


def long_wav(tmp_path, seconds):
    """The test audio repeated to about `seconds`, as a WAV file."""
    with wave.open(os.environ["SUROGATE_METRICS_TEST_AUDIO"]) as source:
        params, frames = source.getparams(), source.readframes(source.getnframes())
    repeats = max(1, int(seconds * params.framerate / params.nframes))
    path = tmp_path / "long.wav"
    with wave.open(str(path), "wb") as target:
        target.setparams(params)
        target.writeframes(frames * repeats)
    return path, repeats * params.nframes / params.framerate


def watch(server, name, until):
    """The largest value `name` takes on /metrics until `until()` is true."""
    peak = 0.0
    while not until():
        peak = max(peak, server.metrics().get(name, 0.0))
        time.sleep(0.02)
    return peak


def test_stt_streams_are_counted_from_creation_until_finished_or_deleted(stt):
    before = stt.metrics()
    assert before["surogate_up"] == 1 and before["surogate_streams_open"] == 0
    first = requests.post(stt.base + "/v1/audio/streams", timeout=30).json()["id"]
    second = requests.post(stt.base + "/v1/audio/streams", timeout=30).json()["id"]
    assert stt.metrics()["surogate_streams_open"] == 2
    with wave.open(os.environ["SUROGATE_METRICS_TEST_AUDIO"]) as source:
        pcm = source.readframes(source.getnframes())
    chunk = pcm[: 16000 * 2]  # one second
    assert requests.post(f"{stt.base}/v1/audio/streams/{first}", data=chunk, timeout=120).ok
    assert stt.metrics()["surogate_streams_open"] == 2
    assert requests.post(f"{stt.base}/v1/audio/streams/{first}?finish=true", data=chunk, timeout=120).ok
    assert stt.metrics()["surogate_streams_open"] == 1
    assert requests.delete(f"{stt.base}/v1/audio/streams/{second}", timeout=30).ok
    after = stt.metrics()
    assert after["surogate_streams_open"] == 0
    assert after['surogate_requests_total{endpoint="streams",outcome="ok"}'] == 2
    assert after["surogate_audio_seconds_total"] == pytest.approx(2.0, abs=0.01)
    assert "surogate_characters_total" not in after  # STT bills no characters


def test_stt_file_transcription_is_counted_while_served(stt, tmp_path):
    path, seconds = long_wav(tmp_path, 40)
    done = threading.Event()
    result = {}

    def transcribe():
        with path.open("rb") as audio:
            result["response"] = requests.post(stt.base + "/v1/audio/transcriptions", timeout=600,
                                               files={"file": ("long.wav", audio, "audio/wav")})
        done.set()

    worker = threading.Thread(target=transcribe)
    worker.start()
    peak = watch(stt, RUNNING, done.is_set)
    worker.join()
    assert result["response"].ok, result["response"].text
    assert peak == 1
    after = stt.metrics()
    assert after[RUNNING] == 0
    assert after['surogate_requests_total{endpoint="transcriptions",outcome="ok"}'] == 1
    assert after["surogate_audio_seconds_total"] == pytest.approx(seconds, abs=0.01)


TEXT = ("Bună ziua. Aceasta este o propoziție mai lungă, citită pentru a verifica cât timp durează "
        "sinteza vocală și dacă serverul o raportează corect cât timp lucrează. ") * 4


def test_tts_request_is_counted_until_its_response_ends(tts):
    done = threading.Event()
    result = {}

    def speak():
        result["response"] = requests.post(tts.base + "/v1/audio/speech", timeout=600,
                                           json={"input": TEXT, "response_format": "wav"})
        done.set()

    worker = threading.Thread(target=speak)
    worker.start()
    peak = watch(tts, RUNNING, done.is_set)
    worker.join()
    response = result["response"]
    assert response.ok, response.text
    assert peak == 1
    after = tts.metrics()
    assert after[RUNNING] == 0
    assert after['surogate_requests_total{endpoint="speech",outcome="ok"}'] == 1
    assert after["surogate_characters_total"] == int(response.headers["X-Usage-Characters"])
    audio_seconds = (len(response.content) - 44) / 2 / 22050
    assert after["surogate_audio_seconds_total"] == pytest.approx(audio_seconds, abs=0.01)


def test_tts_request_ends_when_its_client_disconnects_early(tts):
    started = time.monotonic()
    with pytest.raises(requests.exceptions.ReadTimeout):
        requests.post(tts.base + "/v1/audio/speech", timeout=(5, 1.0),
                      json={"input": TEXT, "response_format": "wav"})
    # The client has gone; the server notices, stops, and no longer counts the request.
    deadline = time.monotonic() + 120
    while tts.metrics()[RUNNING] != 0:
        assert time.monotonic() < deadline, "a request whose client left is still counted"
        time.sleep(0.1)
    after = tts.metrics()
    assert after["surogate_characters_total"] == 0  # nothing was billed
    assert after['surogate_requests_total{endpoint="speech",outcome="error"}'] == 1
    assert time.monotonic() - started < 120


def test_stt_expired_streams_stop_counting_without_new_stream_requests(stt):
    """A drain sends no stream requests; /metrics itself prunes expired streams."""
    stream = requests.post(stt.base + "/v1/audio/streams", timeout=30).json()["id"]
    assert stream and stt.metrics()["surogate_streams_open"] == 1
    # Streams expire after two minutes without a request (the server's fixed policy).
    deadline = time.monotonic() + 150
    while stt.metrics()["surogate_streams_open"] != 0:
        assert time.monotonic() < deadline, "an expired stream is still counted"
        time.sleep(1)
