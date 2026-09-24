"""Real speech-model HTTP checks; opt in with SUROGATE_STT_TEST_MODEL/AUDIO."""

import io
import json
import os
import shutil
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
            "--threads",
            os.environ.get("SUROGATE_STT_TEST_THREADS", "4"),
            "--cpu-kernels",
            os.environ.get("SUROGATE_STT_TEST_KERNELS", "auto"),
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
    oracle = os.environ.get("SUROGATE_STT_TEST_ORACLE")
    if oracle:
        assert r.json()["text"] == json.loads(Path(oracle).read_text())["final"].strip()
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
    if os.environ.get("SUROGATE_STT_TEST_OFFLINE"):
        response = client.post(base + "/v1/audio/streams", timeout=10)
        assert response.status_code == 400
        assert "/v1/audio/transcriptions" in response.json()["error"]["message"]
        return
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
    if os.environ.get("SUROGATE_STT_TEST_OFFLINE"):
        for length in (1, 80, 160, 255, 256, 319, 320, 511, 512):
            wav = io.BytesIO()
            with wave.open(wav, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(b"\0\0" * length)
            r = client.post(base + "/v1/audio/transcriptions", files={"file": wav.getvalue()}, timeout=120)
            assert r.ok, (length, r.text)
            if length < 320:
                assert r.json()["text"] == ""
        return
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


def test_every_transcription_reports_its_duration(server):
    """The billed seconds reach the gateway on every format: usage in JSON, a header on all."""
    client, base, audio = server
    try:  # a 16 kHz mono WAV has an exactly known length; other formats are decoded by the server
        with wave.open(str(audio), "rb") as f:
            exact = f.getnframes() / 16000 if f.getframerate() == 16000 and f.getnchannels() == 1 else None
    except (wave.Error, EOFError):
        exact = None
    seconds = {}
    for fmt in ("json", "text", "verbose_json"):
        r = client.post(base + "/v1/audio/transcriptions", files={"file": audio.read_bytes()},
                        data={"response_format": fmt}, timeout=120)
        assert r.ok, r.text
        header = float(r.headers["X-Audio-Duration-Seconds"])
        seconds[fmt] = header
        if fmt == "text":
            assert r.headers["Content-Type"].startswith("text/plain")
            text_body = r.text  # the transcript alone, as before
            continue
        body = r.json()
        assert body["usage"] == {"type": "duration", "seconds": header}
        if fmt == "verbose_json":
            assert body["duration"] == header
    assert len(set(seconds.values())) == 1 and seconds["json"] > 0
    assert text_body == client.post(base + "/v1/audio/transcriptions", files={"file": audio.read_bytes()},
                                    timeout=120).json()["text"]
    if exact is not None:
        assert seconds["json"] == exact
        # Compressed uploads are billed for the audio, not for the codec's padding: the MP3's
        # gapless header is honoured, and an M4A with its index after the audio (ffmpeg's and
        # most phones' default layout) is readable. AAC keeps up to one frame of padding at the
        # end (1024 samples at the file's rate: 64 ms at 16 kHz), as FFmpeg's own decoder does.
        encoder = shutil.which("ffmpeg")
        if encoder:
            for suffix, codec, tolerance in ((".mp3", "libmp3lame", 0.002), (".m4a", "aac", 1024 / 16000)):
                encoded = audio.with_name(audio.stem + "-duration-test" + suffix)
                subprocess.run([encoder, "-loglevel", "error", "-y", "-i", str(audio), "-c:a", codec, str(encoded)],
                               check=True)
                try:
                    r = client.post(base + "/v1/audio/transcriptions", files={"file": encoded.read_bytes()}, timeout=120)
                    assert r.ok, (suffix, r.text)
                    billed = float(r.headers["X-Audio-Duration-Seconds"])
                    assert r.json()["usage"]["seconds"] == billed
                    assert exact <= billed <= exact + tolerance, (suffix, billed, exact)
                finally:
                    encoded.unlink(missing_ok=True)
    # Silence shorter than a frame has no text and is still billed for its length.
    wav = io.BytesIO()
    with wave.open(wav, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\0\0" * 300)
    r = client.post(base + "/v1/audio/transcriptions", files={"file": wav.getvalue()}, timeout=120)
    assert r.ok, r.text
    assert r.json() == {"text": "", "usage": {"type": "duration", "seconds": 300 / 16000}}
    assert r.headers["X-Audio-Duration-Seconds"] == "0.01875"
    # A failed transcription bills nothing.
    r = client.post(base + "/v1/audio/transcriptions", files={"file": b"bad file"}, timeout=10)
    assert r.status_code == 400 and "X-Audio-Duration-Seconds" not in r.headers
