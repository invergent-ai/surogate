"""Native TTS asset integrity, Romanian token parity and HTTP lifecycle."""

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from surogate.cli import serve
from surogate.serve.tts.assets import sha256, validate_bundle

FIXTURES = Path(__file__).parent / "fixtures"


def wav_bytes(value=0):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(bytes([value, 0]) * 100)
    return output.getvalue()


@pytest.fixture
def bundle(tmp_path):
    """A deterministic protocol worker exercises lifecycle without loading model weights."""
    shutil.copytree(FIXTURES / "tts-tokenizer", tmp_path / "tokenizer")
    for name in (
        "model.gguf",
        "codec.gguf",
        "lib/libnemo_speech_tts.so.1",
        "lib/libggml.so.0",
        "lib/libggml-base.so.0",
        "lib/libggml-cpu.so.0",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    binary = tmp_path / "bin/synthesize"
    binary.parent.mkdir()
    binary.write_text(
        f"#!{sys.executable}\n"
        + """import os, sys, time, wave
from pathlib import Path
root = Path(sys.argv[4])
assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ.get('EXPECT_CUDA_VISIBLE_DEVICES', '')
for line in open(sys.argv[3]):
    identity, voice, seed, temperature, cfg, chunks = line.rstrip().split('\\t')
    if seed == '101':
        sys.exit(3)
    if seed == '102':
        time.sleep(5)
    with wave.open(str(root / (identity + '.wav')), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(bytes([int(voice), 0]) * 100)
    print(identity + ' completed', flush=True)
"""
    )
    profile = {
        "schema": 1,
        "method": "experimental_magpie_native_cpu",
        "voices": {"Doina": {"id": 11, "decoding": {"temperature": 0.4}}, "Tudor": {"id": 12}},
        "decoding": {"temperature": 0.5, "cfg_scale": 1.5, "max_steps": 900, "topk": 80, "longform_mode": "auto"},
        "tokenizer": {"profile": "v2607", "offset": 96, "eos": 3358},
        "files": {str(p.relative_to(tmp_path)): sha256(p) for p in tmp_path.rglob("*") if p.is_file()},
    }
    (tmp_path / "voices.json").write_text(json.dumps(profile))
    return validate_bundle(tmp_path)


CPU_RUNTIME = "139c84abd9973d48c26112e51b310680ac13e1df4d4386cd875f40a7893d44a2"
CPU_BASE = "5a46b8f5f84dfd5f1e86730ee9fd280e0acf6649e77c6519a3182f44b5b196de"


@pytest.mark.parametrize("variant,device,message", [
    ("unknown", "cpu", "unsupported native runtime ABI"),
    ("cpu", "cuda", "only the CPU runtime"),
    ("cuda-wrong-backend", "cuda", "not the one its runtime was built with"),
    ("cuda-no-backend", "cpu", "not the one its runtime was built with"),
])
def test_the_real_worker_checks_the_runtime_before_loading_it(tmp_path, variant, device, message):
    """surogate-tts-worker's ABI and device checks run before any library is loaded."""
    from surogate.serve.tools.tts.gpu_variant import PINNED

    binary = serve._resolve_binary("tts")
    if not binary or not (Path(binary).parent / "surogate-tts-worker").is_file():
        pytest.skip("make serve-tts-build first")
    files = {
        "unknown": {"lib/libnemo_speech_tts.so.1": "0" * 64, "lib/libggml-base.so.0": CPU_BASE},
        "cpu": {"lib/libnemo_speech_tts.so.1": CPU_RUNTIME, "lib/libggml-base.so.0": CPU_BASE},
        "cuda-wrong-backend": {"lib/libnemo_speech_tts.so.1": PINNED["libnemo_speech_tts.so.1"],
                               "lib/libggml-base.so.0": PINNED["libggml-base.so.0"],
                               "lib/libggml-cuda.so.0": "1" * 64},
        "cuda-no-backend": {"lib/libnemo_speech_tts.so.1": PINNED["libnemo_speech_tts.so.1"],
                            "lib/libggml-base.so.0": PINNED["libggml-base.so.0"]},
    }[variant]
    (tmp_path / "voices.json").write_text(json.dumps({"files": files}))
    for name in ("model.gguf", "codec.gguf"):
        (tmp_path / name).write_bytes(b"fixture")
    result = subprocess.run(
        [str(Path(binary).parent / "surogate-tts-worker"), str(tmp_path / "model.gguf"), str(tmp_path / "codec.gguf"),
         "/dev/null", "-", "1", "1", "auto", device], capture_output=True, text=True, timeout=30,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode != 0 and message in result.stderr, result.stderr


def test_the_gpu_variant_tool_and_the_worker_pin_the_same_cuda_runtime():
    from surogate.serve.tools.tts.gpu_variant import PINNED

    source = (Path(__file__).resolve().parents[2] / "csrc/src/serve/tts/native_worker.cpp").read_text()
    for name, digest in PINNED.items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest), name
        assert f'"{digest}"' in source, name


def test_a_gpu_needs_the_packages_gpu_variant(bundle, monkeypatch):
    from surogate.serve.tts import assets

    # The CPU-only package (and the published one, which is CPU-only so far) is refused for a GPU
    # before anything starts; the CPU keeps working.
    assert assets.prepare_bundle(str(bundle.root), device="cpu") == bundle
    with pytest.raises(ValueError, match="only the CPU runtime"):
        assets.prepare_bundle(str(bundle.root), device="0")
    monkeypatch.setattr("huggingface_hub.snapshot_download", Mock(side_effect=AssertionError("no download")))
    with pytest.raises(ValueError, match="only the CPU runtime"):
        assets.prepare_bundle(assets.MODEL_ID, device="0")
    # A GPU variant carries the CUDA runtime in lib/.
    cuda = bundle.root / "lib/libggml-cuda.so.0"
    cuda.write_bytes(b"fixture")
    profile = json.loads((bundle.root / "voices.json").read_text())
    profile["files"]["lib/libggml-cuda.so.0"] = sha256(cuda)
    (bundle.root / "voices.json").write_text(json.dumps(profile))
    assert assets.prepare_bundle(str(bundle.root), device="0").root == bundle.root


def test_gpu_variant_tool_assembles_a_package_the_launcher_accepts_for_a_gpu(bundle, tmp_path_factory):
    from surogate.serve.tools.tts.gpu_variant import LIBRARIES, assemble
    from surogate.serve.tts import assets

    work = tmp_path_factory.mktemp("gpu-variant")  # outside the CPU package
    cuda_bin = work / "cuda-bin"
    cuda_bin.mkdir()
    for names in LIBRARIES.values():
        (cuda_bin / names[-1]).write_bytes(b"cuda " + names[-1].encode())
    (bundle.root / "build_info.json").write_text(json.dumps({"native_commit": "07003daa"}))
    with pytest.raises(SystemExit, match="inside the CPU package"):
        assemble(bundle.root, cuda_bin, bundle.root / "gpu-variant", check_pins=False)
    with pytest.raises(SystemExit, match="not the pinned CUDA runtime"):
        assemble(bundle.root, cuda_bin, work / "unpinned")
    variant = assemble(bundle.root, cuda_bin, work / "package", check_pins=False)
    assert json.loads((variant / "build_info.json").read_text())["cmake"]["GGML_CUDA"] == "ON"
    prepared = assets.prepare_bundle(str(variant), device="0")
    assert prepared.profile["voices"] == bundle.profile["voices"]
    assert (variant / "lib/libggml-cuda.so.0").read_bytes() == b"cuda libggml-cuda.so.0.12.0"
    assert (variant / "model.gguf").read_bytes() == (bundle.root / "model.gguf").read_bytes()


@pytest.mark.parametrize("fault", ["checksum", "escape", "missing_hash", "speaker", "policy", "ambiguous_voice"])
def test_corrupt_bundle_is_rejected(bundle, fault):
    path = bundle.root / "voices.json"
    profile = json.loads(path.read_text())
    if fault == "checksum":
        (bundle.root / "model.gguf").write_bytes(b"changed")
    elif fault == "escape":
        profile["files"]["../outside"] = "0" * 64
    elif fault == "missing_hash":
        del profile["files"]["model.gguf"]
    elif fault == "speaker":
        profile["voices"]["Doina"]["id"] = True
    elif fault == "policy":
        profile["decoding"]["max_steps"] = 901
    else:
        profile["voices"]["doina"] = {"id": 13}
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        validate_bundle(bundle.root)


def test_hf_download_is_pinned_and_cached(bundle, tmp_path, monkeypatch):
    from surogate.serve.tts import assets

    monkeypatch.setenv("SUROGATE_SERVE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(assets, "PROFILE_SHA256", sha256(bundle.root / "voices.json"))

    def download(model, *, revision, allow_patterns, local_dir, force_download):
        assert model == assets.MODEL_ID and revision == assets.REVISION
        assert allow_patterns == [assets.PREFIX + "/*"] and not force_download
        destination = local_dir / assets.PREFIX
        destination.mkdir(parents=True)
        for name in ["voices.json", *bundle.profile["files"]]:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle.root / name, target)

    download_mock = Mock(side_effect=download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download_mock)
    first = assets.prepare_bundle(assets.MODEL_ID)
    second = assets.prepare_bundle(assets.MODEL_ID)
    assert first == second and download_mock.call_count == 1
    assert assets.prepare_bundle(str(first.root / "voices.json")) == first


def test_cli_executes_native_tts_without_importing_ingest(monkeypatch, bundle):
    monkeypatch.setattr(sys, "argv", ["surogate", "serve", "--tts", "--voice=Tudor", "model", "--no-cache"])
    monkeypatch.setitem(sys.modules, "surogate.serve.ingest", None)
    monkeypatch.setattr(serve, "_resolve_binary", lambda mode: "/native-tts")
    prepare = Mock(return_value=bundle)
    monkeypatch.setattr("surogate.serve.tts.assets.prepare_bundle", prepare)
    execute = Mock()
    monkeypatch.setattr(serve.os, "execv", execute)
    serve.maybe_exec_serve()
    prepare.assert_called_once()
    assert prepare.call_args.kwargs["reuse_cache"] is False
    execute.assert_called_once_with(
        "/native-tts", ["/native-tts", str(bundle.root), "--voice", "Tudor", "--served-model-name", "model"]
    )


def frontend_binary():
    root = Path(__file__).resolve().parents[2]
    for directory in ("build-serve", "build-tts"):
        binary = root / "csrc" / directory / "test_tts_frontend"
        if binary.is_file():
            return binary
    pytest.skip("make serve-tts-build first")


def frontend_results(inputs):
    result = subprocess.run(
        [str(frontend_binary())],
        input="\n".join(json.dumps(text) for text in inputs) + "\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    return [json.loads(line) for line in result.stdout.splitlines()]


def test_native_frontend_matches_all_301_romanian_benchmark_requests():
    cases = json.loads((FIXTURES / "tts-stress331.json").read_text())["rows"]
    assert len(cases) == 331
    cases = [case for case in cases if case["language"] == "ro"]
    assert len(cases) == 301
    for case, output in zip(cases, frontend_results([case["input"] for case in cases]), strict=True):
        assert "error" not in output, (case["id"], output)
        digest = hashlib.sha256(json.dumps(output["chunks"], separators=(",", ":")).encode()).hexdigest()
        assert digest == case["chunks_sha256"], case["id"]


def test_native_frontend_rejects_invalid_structured_values_and_limits():
    invalid = [
        "",
        "[pause]",
        "x" * 4097,
        "2026-02-30",
        "ora 25:10",
        "\ue100",
        "<extra_id_0>",
        "a\0b",
        "99999999999999999999",
    ]
    assert all("error" in item for item in frontend_results(invalid))
    outputs = frontend_results(["Ştefan şi Ţara", "Ștefan și Țara", "Ștefan și Țara"])
    assert outputs[0]["chunks"] == outputs[1]["chunks"] == outputs[2]["chunks"]


def child_pids(pid):
    children = set()
    for task in Path(f"/proc/{pid}/task").glob("*/children"):
        try:
            children.update(int(p) for p in task.read_text().split())
        except FileNotFoundError:
            pass
    return children


@contextlib.contextmanager
def native_server(bundle, tmp_path, *options, env=None):
    binary = serve._resolve_binary("tts")
    if not binary:
        pytest.skip("make serve-tts-build first")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = tmp_path / "http.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                binary,
                str(bundle.root),
                "--port",
                str(port),
                "--served-model-name",
                "test",
                "--api-key",
                "test-key",
                *options,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={name: value for name, value in {
                **os.environ, "SUROGATE_TTS_WORKER_BIN": str(bundle.root / "bin/synthesize"), **(env or {})
            }.items() if value is not None},  # None removes a variable
        )
    children = set()
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", headers={"Authorization": "Bearer test-key"}, timeout=10
        ) as client:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                assert process.poll() is None, log_path.read_text()
                try:
                    if client.get("/health", timeout=0.2).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.03)
            else:
                pytest.fail("native TTS failed to start")
            children = child_pids(process.pid)
            yield client, process
            children |= child_pids(process.pid)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            pytest.fail("Native TTS shutdown did not finish")
        assert all(not Path(f"/proc/{pid}").exists() for pid in children), "Native worker leaked after shutdown"


def test_native_http_auth_voice_switching_pcm_and_cleanup(bundle, tmp_path):
    with native_server(bundle, tmp_path, "--voice", "Tudor") as (client, process):
        assert client.get("/health", headers={"Authorization": "wrong"}).status_code == 401
        assert client.get("/v1/models").json()["data"][0]["id"] == "test"
        assert client.get("/v1/audio/voices").json()["default_voice"] == "Tudor"
        children = child_pids(process.pid)
        assert len(children) == 1
        child = next(iter(children))
        args = Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")
        work = Path(args[args.index(b"/dev/stdin") + 1].decode())
        first = client.post("/v1/audio/speech", json={"input": "Bună"})
        assert first.content == wav_bytes(12)
        # Billed characters: four code points, although "ă" takes two bytes.
        assert first.headers["x-usage-characters"] == "4"
        second = client.post("/v1/audio/speech", json={"input": "Bună", "voice": "doina", "response_format": "pcm"})
        assert second.status_code == 200 and second.content == bytes([11, 0]) * 100
        assert second.headers["x-audio-sample-rate"] == "22050"
        assert second.headers["x-usage-characters"] == "4"
        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(client.post, "/v1/audio/speech", json={"input": "Bună", "voice": name})
                for name in ["Doina", "Tudor"]
            ]
            assert [f.result().content for f in futures] == [wav_bytes(11), wav_bytes(12)]
        assert child_pids(process.pid) == children
        assert not list(work.glob("*.wav"))
        assert (work / "stats.jsonl").resolve() == Path(os.devnull)
    assert not work.exists()


@pytest.mark.parametrize("options,expected", [
    ([], ["4", "4", "auto", "cpu"]),
    (["--threads", "2"], ["2", "2", "auto", "cpu"]),
    (["--threads", "3", "--codec-threads", "1", "--cpu-kernels", "reference"], ["3", "1", "reference", "cpu"]),
    (["--threads", "2", "--codec-threads", "0", "--cpu-kernels", "optimized"], ["2", "2", "optimized", "cpu"]),
    (["--device", "cpu"], ["4", "4", "auto", "cpu"]),
])
def test_compute_options_reach_worker(bundle, tmp_path, options, expected):
    with native_server(bundle, tmp_path, *options) as (_, process):
        child, = child_pids(process.pid)
        args = Path(f"/proc/{child}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        assert [arg.decode() for arg in args[-4:]] == expected
        environment = Path(f"/proc/{child}/environ").read_bytes().split(b"\0")
        assert f"OMP_NUM_THREADS={expected[0]}".encode() in environment
        assert b"CUDA_VISIBLE_DEVICES=" in environment  # the CPU worker sees no GPU


@pytest.mark.parametrize("inherited,device,visible", [
    (None, "3", "3"), ("5,7", "1", "7"), ("7", "0", "7"), (" 5 , 7", "1", "7"), ("GPU-a,GPU-b", "1", "GPU-b"),
    (None, "00", "0"),
])
def test_gpu_device_reaches_worker_as_its_only_card(bundle, tmp_path, inherited, device, visible):
    # --device N is an ordinal among the server's own visible devices, as on the LLM server.
    env = {"EXPECT_CUDA_VISIBLE_DEVICES": visible, "CUDA_VISIBLE_DEVICES": inherited}
    with native_server(bundle, tmp_path, "--device", device, env=env) as (client, process):
        child, = child_pids(process.pid)
        args = Path(f"/proc/{child}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        assert args[-1] == b"cuda"
        environment = Path(f"/proc/{child}/environ").read_bytes().split(b"\0")
        assert f"CUDA_VISIBLE_DEVICES={visible}".encode() in environment
        assert client.get("/health").json()["device"] == f"cuda:{int(device)}"


@pytest.mark.parametrize("device,inherited,message", [
    ("gpu", None, "must be cpu or a CUDA device index"),
    ("-1", None, "must be cpu or a CUDA device index"),
    ("1x", None, "must be cpu or a CUDA device index"),
    ("2", "5,7", "is not among CUDA_VISIBLE_DEVICES=5,7"),
    ("0", "", "is not among CUDA_VISIBLE_DEVICES="),  # set but empty hides every card
    ("0", "5,,7", "has an empty entry"),
])
def test_invalid_devices_are_refused_at_startup(bundle, tmp_path, device, inherited, message):
    binary = serve._resolve_binary("tts")
    if not binary:
        pytest.skip("make serve-tts-build first")
    env = {**os.environ, "SUROGATE_TTS_WORKER_BIN": str(bundle.root / "bin/synthesize")}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if inherited is not None:
        env["CUDA_VISIBLE_DEVICES"] = inherited
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    result = subprocess.run([binary, str(bundle.root), "--port", str(port), "--device", device],
                            capture_output=True, text=True, timeout=30, env=env)
    assert result.returncode != 0
    assert message in result.stderr, result.stderr


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"input": " "},
        {"input": 7},
        {"input": "x" * 4097},
        {"input": "Bună", "model": "wrong"},
        {"input": "Bună", "voice": "missing"},
        {"input": "Bună", "response_format": "mp3"},
        {"input": "Bună", "speed": 0.5},
        {"input": "Bună", "speed": True},
        {"input": "Bună", "seed": True},
        {"input": "Bună", "seed": -1},
        {"input": "Bună", "seed": 2**31},
        {"input": "Bună", "seed": 2**64},
        {"input": "Bună", "instructions": "whisper"},
        {"input": "Bună", "voice": None},
    ],
)
def test_native_http_rejects_invalid_fields(bundle, tmp_path, body):
    with native_server(bundle, tmp_path) as (client, _):
        r = client.post("/v1/audio/speech", json=body)
        assert r.status_code == 400 and "error" in r.json()
        assert "x-usage-characters" not in r.headers  # an error response bills nothing


def test_native_http_body_limits_and_worker_recovery(bundle, tmp_path):
    with native_server(bundle, tmp_path, "--max-pending-requests", "0", "--request-timeout", "0.6") as (
        client,
        process,
    ):
        def unbilled(response, status):
            # Error responses carry no billable character count.
            assert response.status_code == status and "x-usage-characters" not in response.headers
            return True

        assert unbilled(client.post("/v1/audio/speech", content="{}", headers={"Content-Type": "text/plain"}), 415)
        assert unbilled(client.post("/v1/audio/speech", content="{", headers={"Content-Type": "application/json"}), 400)
        assert unbilled(
            client.post("/v1/audio/speech", content=" " * 65537, headers={"Content-Type": "application/json"}), 413
        )
        original = child_pids(process.pid)
        with ThreadPoolExecutor(1) as pool:
            slow = pool.submit(client.post, "/v1/audio/speech", json={"input": "Bună", "seed": 102})
            time.sleep(0.1)
            assert unbilled(client.post("/v1/audio/speech", json={"input": "Bună"}), 429)
            assert unbilled(slow.result(), 504)
        assert client.get("/health").status_code == 503
        assert not child_pids(process.pid)
        recovered = client.post("/v1/audio/speech", json={"input": "Bună", "voice": "Tudor"})
        assert recovered.content == wav_bytes(12) and recovered.headers["x-usage-characters"] == "4"
        assert child_pids(process.pid) != original
        assert unbilled(client.post("/v1/audio/speech", json={"input": "Bună", "seed": 101}), 503)
        assert client.post("/v1/audio/speech", json={"input": "Bună"}).content == wav_bytes(11)


def test_native_http_shutdown_cancels_active_inference(bundle, tmp_path):
    with native_server(bundle, tmp_path) as (client, process):
        with ThreadPoolExecutor(1) as pool:
            active = pool.submit(client.post, "/v1/audio/speech", json={"input": "Bună", "seed": 102})
            time.sleep(0.1)
            process.terminate()
            process.wait(timeout=3)
            try:
                assert active.result().status_code == 503
            except httpx.TransportError:
                pass
