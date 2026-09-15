"""Real speech-model HTTP checks; opt in with SUROGATE_STT_TEST_MODEL/AUDIO."""

import io
import os
import socket
import subprocess
import time
import wave
from pathlib import Path

import pytest
import requests

from surogate.cli.serve import _resolve_binary


@pytest.fixture(scope="module")
def server():
    model = os.environ.get("SUROGATE_STT_TEST_MODEL")
    audio = os.environ.get("SUROGATE_STT_TEST_AUDIO")
    if not model or not audio:
        pytest.skip("set SUROGATE_STT_TEST_MODEL and SUROGATE_STT_TEST_AUDIO for real-model validation")
    binary = os.environ.get("SUROGATE_STT_TEST_BIN") or _resolve_binary("stt")
    if not binary:
        pytest.skip("build surogate-stt first")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    device = os.environ.get("SUROGATE_STT_TEST_DEVICE", "cpu")
    process = subprocess.Popen(
        [
            binary,
            model,
            "--device",
            device,
            "--port",
            str(port),
            "--max-num-seqs",
            "2",
            "--served-model-name",
            "speech-test",
            "--api-key",
            "test-key",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    client = requests.Session()
    client.headers["Authorization"] = "Bearer test-key"
    try:
        for _ in range(300):
            if process.poll() is not None:
                pytest.fail(process.stderr.read().decode())
            try:
                if client.get(base + "/health", timeout=0.2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.1)
        else:
            pytest.fail("speech server did not start")
        yield client, base, Path(audio)
    finally:
        client.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stderr.close()


def test_file_and_validation(server):
    client, base, audio = server
    assert requests.get(base + "/health", timeout=5).status_code == 401
    assert client.get(base + "/v1/models", timeout=5).json()["data"][0]["id"] == "speech-test"
    r = client.post(base + "/v1/audio/transcriptions", files={"file": audio.read_bytes()}, timeout=120)
    assert r.ok, r.text
    assert r.json()["text"].strip()
    verbose = client.post(
        base + "/v1/audio/transcriptions",
        files={"file": audio.read_bytes()},
        data={"response_format": "verbose_json"},
        timeout=120,
    )
    assert verbose.ok, verbose.text
    assert verbose.json()["task"] == "transcribe"
    assert verbose.json()["duration"] > 0
    assert verbose.json()["text"] == r.json()["text"]
    for field, value in [("model", "unknown"), ("language", "en"), ("prompt", "ignored?"), ("response_format", "srt")]:
        r = client.post(
            base + "/v1/audio/transcriptions", files={"file": audio.read_bytes()}, data={field: value}, timeout=10
        )
        assert r.status_code == 400, r.text
    r = client.post(base + "/v1/audio/transcriptions", files={"file": b"bad file"}, timeout=10)
    assert r.status_code == 400
    assert client.get(base + "/health", timeout=5).ok


def test_streams_are_independent_and_packet_sizes_do_not_change_text(server):
    import numpy as np
    import soundfile as sf

    client, base, audio = server
    samples, rate = sf.read(audio, dtype="float32")
    assert rate == 16000
    # Include silence to check pause finalization and segment reset.
    samples = np.concatenate((samples, np.zeros(16000, dtype=np.float32)))
    pcm = np.clip(samples * 32768, -32768, 32767).astype("<i2").tobytes()
    ids = [client.post(base + "/v1/audio/streams", timeout=10).json()["id"] for _ in range(2)]
    assert client.post(base + "/v1/audio/streams", timeout=10).status_code == 429
    events = [[], []]
    offsets = [0, 0]
    sizes = [1024, 3074]
    while min(offsets) < len(pcm):
        for i in range(2):
            if offsets[i] >= len(pcm):
                continue
            r = client.post(
                base + "/v1/audio/streams/" + ids[i], data=pcm[offsets[i] : offsets[i] + sizes[i]], timeout=120
            )
            assert r.ok, r.text
            events[i].extend(r.json()["events"])
            offsets[i] += sizes[i]
    for i in range(2):
        r = client.post(base + "/v1/audio/streams/" + ids[i] + "?finish=true", timeout=120)
        assert r.ok, r.text
        events[i].extend(r.json()["events"])
        assert client.post(base + "/v1/audio/streams/" + ids[i], timeout=5).status_code == 404
    # Timing and finalization duration are intentionally excluded.
    transcripts = [[(event["type"], event["text"]) for event in stream] for stream in events]
    assert transcripts[0] == transcripts[1]
    assert any(e["type"] == "partial" for e in events[0])
    assert any(e.get("reason") == "pause" for e in events[0])
    final_text = " ".join(e["text"] for e in events[0] if e["type"] == "final").strip()
    wav = io.BytesIO()
    with wave.open(wav, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(pcm)
    r = client.post(base + "/v1/audio/transcriptions", files={"file": wav.getvalue()}, timeout=120)
    assert r.ok, r.text
    assert r.json()["text"] == final_text


def test_cancel_malformed_chunks_and_short_tail(server):
    client, base, _ = server
    ident = client.post(base + "/v1/audio/streams", timeout=10).json()["id"]
    url = base + "/v1/audio/streams/" + ident
    assert client.post(url, data=b"x", timeout=5).status_code == 400
    assert client.delete(url, timeout=5).ok
    assert client.post(url, timeout=5).status_code == 404
    for length in (1, 80, 160, 255, 256, 511, 512):
        ident = client.post(base + "/v1/audio/streams", timeout=10).json()["id"]
        r = client.post(base + "/v1/audio/streams/" + ident + "?finish=true", data=b"\0\0" * length, timeout=120)
        assert r.ok, (length, r.text)
    assert client.get(base + "/health", timeout=5).ok
