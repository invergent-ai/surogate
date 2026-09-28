"""A pocket-tts package (voices.json "method": "pocket_tts_python"): the server runs the worker program the
package names, hands it the request text as UTF-8 bytes and serves its audio at the package's sample rate."""

import io
import json
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from surogate.cli import serve
from surogate.serve.tts import assets

from .test_tts import native_server

WORKER = """import os, sys
out = sys.stdout.buffer
out.write(b'0 ready\\n')
out.flush()
pending = b''
while True:
    while b'\\n' not in pending:
        data = os.read(0, 65536)
        if not data:
            sys.exit(0)
        pending += data
    line, _, pending = pending.partition(b'\\n')
    line = line.decode()
    if line == 'cancel':
        continue
    identity, voice, seed, temperature, cfg, chunks = line.split('\\t')
    text = bytes(int(b) for b in chunks.split()).decode('utf-8')
    with open(os.environ['FIXTURE_TEXT_LOG'], 'a', encoding='utf-8') as log:
        log.write(f'{voice}\\t{temperature}\\t{text}\\n')
    pcm = bytes([int(voice), 0]) * len(text)  # one sample per code point: the length shows what arrived
    out.write(f'{identity} pcm {len(pcm)}\\n'.encode() + pcm)
    out.write(f'{identity} done 24000 0.01 0.01 1.0\\n'.encode())
    out.flush()
"""


def make_package(root, **changes):
    (root / "voices").mkdir(parents=True, exist_ok=True)
    for name in ("ana", "radu"):
        (root / f"voices/{name}.wav").write_bytes(b"RIFF fixture")
    (root / "model").mkdir(exist_ok=True)
    (root / "model/config.yaml").write_text("flow_lm: {}\n")
    worker = root / "bin/worker"
    worker.parent.mkdir(exist_ok=True)
    worker.write_text(f"#!{sys.executable}\n" + WORKER)
    worker.chmod(0o755)
    profile = {
        "schema": 1, "method": "pocket_tts_python", "sample_rate": 24000, "worker": "bin/worker",
        "model": {"config": "model/config.yaml"},
        "decoding": {"temperature": 0.3, "code_attempts": 3},
        "voices": {"Ana": {"id": 1, "prompt": "voices/ana.wav"},
                   "Radu": {"id": 2, "prompt": "voices/radu.wav", "decoding": {"temperature": 0.5}}},
    }
    for key, value in changes.items():
        profile[key] = value
    (root / "voices.json").write_text(json.dumps(profile))
    return SimpleNamespace(root=root)


@pytest.fixture
def pocket(tmp_path):
    return make_package(tmp_path / "package")


def served(pocket, tmp_path, *options):
    log = tmp_path / "texts.log"
    return native_server(pocket, tmp_path, *options,
                         env={"SUROGATE_TTS_WORKER_BIN": None, "FIXTURE_TEXT_LOG": str(log)}), log


def test_text_jobs_at_the_packages_sample_rate(pocket, tmp_path):
    server, log = served(pocket, tmp_path)
    with server as (client, _):
        assert [v["id"] for v in client.get("/v1/audio/voices").json()["data"]] == ["Ana", "Radu"]
        text = "IBAN-ul este RO49 AAAA 1B31 0075 9384 0000. Mulțumesc!"
        r = client.post("/v1/audio/speech", json={"input": text})
        assert r.status_code == 200, r.text
        assert r.headers["x-audio-sample-rate"] == "24000"
        with wave.open(io.BytesIO(r.content)) as w:
            assert w.getframerate() == 24000
            assert w.readframes(w.getnframes()) == bytes([1, 0]) * len(text)
        pcm = client.post("/v1/audio/speech", json={"input": "Bună", "voice": "radu", "response_format": "pcm"})
        assert pcm.content == bytes([2, 0]) * 4 and pcm.headers["x-audio-sample-rate"] == "24000"
        with client.stream("POST", "/v1/audio/speech", json={"input": "Ștefan", "stream_format": "audio"}) as s:
            body = b"".join(s.iter_bytes())
            assert s.headers["x-audio-sample-rate"] == "24000"
        assert int.from_bytes(body[24:28], "little") == 24000  # the streamed WAV header's rate
        sse = client.post("/v1/audio/speech", json={"input": "Da.", "stream_format": "sse"})
        done = [json.loads(e[6:]) for e in sse.text.split("\n\n") if e.startswith("data: ")][-1]
        assert done["type"] == "speech.audio.done" and done["usage"]["audio_seconds"] == pytest.approx(3 * 2 / 48000)
    # The worker got the text exactly as sent (diacritics, digits, punctuation) with each voice's temperature;
    # the first line is the warm-up.
    lines = [l.split("\t") for l in log.read_text(encoding="utf-8").splitlines()]
    assert lines[0] == ["1", "0.3", "Bună."]
    assert lines[1:] == [["1", "0.3", text], ["2", "0.5", "Bună"], ["1", "0.3", "Ștefan"], ["1", "0.3", "Da."]]


def test_input_limits(pocket, tmp_path):
    server, _ = served(pocket, tmp_path, "--max-input-characters", "5")
    with server as (client, _):
        assert client.post("/v1/audio/speech", json={"input": "Ștefăn"}).status_code == 400  # 6 code points
        assert client.post("/v1/audio/speech", json={"input": "Ștefă"}).status_code == 200
        r = client.post("/v1/audio/speech", json={"input": " \n "})
        assert r.status_code == 400 and "spoken text" in r.text


# Each refused by the server (its message first) and by `surogate serve --tts` before it starts one (second).
INVALID = [
    ({"worker": "../outside"}, "outside the package", "inside the package"),
    ({"worker": "bin/missing"}, "missing", "missing"),
    ({"sample_rate": 1000}, "sample rate", "sample rate"),
    ({"decoding": {"temperature": 0}}, "Invalid TTS decoding value", "temperature"),
    ({"decoding": {"temperature": 0.3, "code_attempts": 9}}, "code_attempts", "code_attempts"),
    ({"voices": {"Ana": {"id": 1, "prompt": "voices/none.wav"}}}, "missing", "missing"),
    ({"method": "pocket_tts_other"}, "Unsupported native TTS package", "Expected an exported native CPU"),
]


@pytest.mark.parametrize("change,message,_", INVALID)
def test_invalid_packages_are_refused(tmp_path, change, message, _):
    binary = serve._resolve_binary("tts")
    if not binary:
        pytest.skip("make serve-tts-build first")
    package = make_package(tmp_path / "package", **change)
    r = subprocess.run([binary, str(package.root), "--port", "1"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and message in r.stderr, r.stderr


@pytest.mark.parametrize("change,_,message", INVALID)
def test_the_launcher_refuses_what_the_server_refuses(tmp_path, change, _, message):
    package = make_package(tmp_path / "package", **change)
    with pytest.raises(ValueError, match=message):
        assets.prepare_bundle(str(package.root))


def test_the_launcher_checks_the_manifest_and_keeps_to_the_cpu(pocket):
    root = pocket.root.resolve()
    (root / "bin/worker").chmod(0o644)
    names = ["bin/worker", "model/config.yaml", "voices/ana.wav", "voices/radu.wav"]
    (root / "MANIFEST.sha256").write_text("".join(f"{assets.sha256(root / n)}  {n}\n" for n in names))
    bundle = assets.prepare_bundle(str(root / "voices.json"))
    assert bundle.root == root and bundle.profile["method"] == "pocket_tts_python"
    assert (root / "bin/worker").stat().st_mode & 0o100  # made runnable, as a download may lose the bit
    with pytest.raises(ValueError, match="CPU only"):
        assets.prepare_bundle(str(root), device="0")
    (root / "voices/radu.wav").write_bytes(b"RIFF changed")
    with pytest.raises(ValueError, match="checksum mismatch: voices/radu.wav"):
        assets.prepare_bundle(str(root))


SAY = '<say-as interpret-as="characters">{}</say-as>'


def test_marked_codes_reach_the_worker_and_the_tags_are_not_billed(pocket, tmp_path):
    server, log = served(pocket, tmp_path, "--max-input-characters", "20")
    with server as (client, _):
        text = "Codul este " + SAY.format("RO49 AB-1") + "."  # 21 characters spoken, 57 with the tags
        r = client.post("/v1/audio/speech", json={"input": text})
        assert r.status_code == 400 and "exceeds 20" in r.text
        text = "Codul: " + SAY.format("RO49 AB-1") + "."  # 17 spoken
        sse = client.post("/v1/audio/speech", json={"input": text, "stream_format": "sse"})
        done = [json.loads(e[6:]) for e in sse.text.split("\n\n") if e.startswith("data: ")][-1]
        assert done["usage"]["input_characters"] == 17
    assert log.read_text(encoding="utf-8").splitlines()[-1].split("\t")[2] == text  # the markup is the worker's


@pytest.mark.parametrize("markup", [
    '<say-as interpret-as="date">2026</say-as>',  # only characters
    '<say-as interpret-as="characters">RO49 Ș</say-as>',  # a letter with no clip
    '<say-as interpret-as="characters">RO49',  # never closed
    'RO49</say-as>',  # closed, never opened
    '<SAY-AS interpret-as="characters">RO49</SAY-AS>',  # not the tag the worker reads
    '<say-as interpret-as="characters">' + "7" * 65 + '</say-as>',  # longer than any code
    '<say-as interpret-as="characters"> - </say-as>',  # nothing to say
])
def test_malformed_markup_is_refused_not_read_as_text(pocket, tmp_path, markup):
    server, _ = served(pocket, tmp_path)
    with server as (client, _):
        r = client.post("/v1/audio/speech", json={"input": "Codul este " + markup + "."})
        assert r.status_code == 400 and "say-as" in r.text, r.text
