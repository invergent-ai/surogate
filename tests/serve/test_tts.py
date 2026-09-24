"""Native TTS asset integrity, Romanian token parity and HTTP lifecycle."""

import base64
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
        + """import os, select, sys, time, wave
from pathlib import Path
stream = sys.argv[4] == '-'
assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ.get('EXPECT_CUDA_VISIBLE_DEVICES', '')
out = sys.stdout.buffer
pending = b''
def lines():
    global pending
    while True:
        while b'\\n' not in pending:
            data = os.read(0, 65536)
            if not data:
                return
            pending += data
        line, _, pending = pending.partition(b'\\n')
        yield line.decode()
def cancelled(wait):
    # The server's cancel line has arrived (within `wait` seconds).
    return b'\\n' in pending or bool(select.select([0], [], [], wait)[0])
source = lines() if stream else open(sys.argv[3])
for line in source:
    if stream and line == 'cancel':
        continue  # came after its request had finished
    identity, voice, seed, temperature, cfg, chunks = line.rstrip().split('\\t')
    if seed == '101':
        sys.exit(3)
    if seed == '102':
        time.sleep(5)
    pcm = bytes([int(voice), 0]) * 100
    if not stream:
        with wave.open(str(Path(sys.argv[4]) / (identity + '.wav')), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(pcm)
        print(identity + ' completed', flush=True)
        continue
    # Seed 103: three chunks, 0.4 s apart, as a long reply produces them. Seed 104: a chunk, then
    # a failure. Seed 105: 48 MiB at once, more than the sockets between server and client hold
    # (up to 16 MiB each way on Linux).
    # Seed 106: a chunk, then another 3 s later, a slow step during which it misses a cancel.
    parts = {'103': [pcm] * 3, '105': [pcm * 5243] * 48, '106': [pcm] * 2}.get(seed, [pcm])
    stopped = False
    for index, part in enumerate(parts):
        if index and seed != '105':
            if seed == '106':
                time.sleep(3)
            if cancelled(0 if seed == '106' else 0.4):
                stopped = True
                break
        out.write(f'{identity} pcm {len(part)}\\n'.encode() + part)
        out.flush()
    if stopped:
        assert next(source) == 'cancel'
        out.write(f'{identity} cancelled\\n'.encode())
        out.flush()
        continue
    if seed == '104':
        time.sleep(0.4)
        sys.exit(3)
    out.write(f'{identity} done 22050 0.01 0.01 1.0\\n'.encode())
    out.flush()
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

    # A CPU-only package is refused for a GPU before anything starts; the CPU keeps working.
    assert assets.prepare_bundle(str(bundle.root), device="cpu") == bundle
    with pytest.raises(ValueError, match="only the CPU runtime"):
        assets.prepare_bundle(str(bundle.root), device="0")
    # A GPU variant carries the CUDA runtime in lib/.
    gpu_variant_of(bundle)
    assert assets.prepare_bundle(str(bundle.root), device="0").root == bundle.root
    # An invalid device is refused before anything is downloaded.
    monkeypatch.setattr("huggingface_hub.snapshot_download", Mock(side_effect=AssertionError("no download")))
    for device in ("gpu", "", "CPU", "cuda:0", "-1", "1000"):
        with pytest.raises(ValueError, match="--device must be cpu or a CUDA device index"):
            assets.prepare_bundle(assets.MODEL_ID, device=device)


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


def gpu_variant_of(bundle):
    """The fixture package with a CUDA runtime in lib/, as a GPU variant has."""
    cuda = bundle.root / "lib/libggml-cuda.so.0"
    cuda.write_bytes(b"fixture")
    profile = json.loads((bundle.root / "voices.json").read_text())
    profile["files"]["lib/libggml-cuda.so.0"] = sha256(cuda)
    (bundle.root / "voices.json").write_text(json.dumps(profile))
    return validate_bundle(bundle.root)


def published_download(bundle, variant):
    """A fake snapshot_download of the model repository: the CPU package and the GPU variant."""
    from surogate.serve.tts import assets

    def download(model, *, revision, allow_patterns, local_dir, force_download):
        assert model == assets.MODEL_ID
        prefix, source = {assets.REVISION: (assets.PREFIX, bundle), assets.GPU_REVISION: ("gpu", variant)}[revision]
        assert allow_patterns == [prefix + "/*"]
        for name in ["voices.json", "README.md", *source.profile["files"]]:
            target = local_dir / prefix / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.root / name if name != "README.md" else source.root / "voices.json", target)

    return Mock(side_effect=download)


def test_hf_gpu_variant_download_is_pinned_and_cached(bundle, tmp_path_factory, monkeypatch):
    """--device N fetches the published GPU variant (gpu/ at its pinned revision), not the CPU package."""
    from surogate.serve.tts import assets

    tmp_path = tmp_path_factory.mktemp("published")  # outside the fixture package
    cpu_root = tmp_path / "cpu-package"
    shutil.copytree(bundle.root, cpu_root)
    cpu = validate_bundle(cpu_root)
    variant = gpu_variant_of(bundle)
    monkeypatch.setenv("SUROGATE_SERVE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(assets, "PROFILE_SHA256", sha256(cpu.root / "voices.json"))
    monkeypatch.setattr(assets, "GPU_PROFILE_SHA256", sha256(variant.root / "voices.json"))
    download = published_download(cpu, variant)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    first = assets.prepare_bundle(assets.MODEL_ID, device="0")
    assert first.root == tmp_path / "cache" / f"tts-{assets.GPU_REVISION}" / "gpu"
    assert os.access(first.root / "bin/synthesize", os.X_OK)
    assert download.call_args.kwargs["revision"] == assets.GPU_REVISION
    assert not download.call_args.kwargs["force_download"]
    assert assets.prepare_bundle(assets.MODEL_ID, device="1") == first and download.call_count == 1
    # The CPU package goes to a cache of its own, and neither replaces the other.
    on_cpu = assets.prepare_bundle(assets.MODEL_ID, device="cpu")
    assert on_cpu.root == tmp_path / "cache" / f"tts-{assets.REVISION}" / assets.PREFIX
    assert "lib/libggml-cuda.so.0" not in on_cpu.profile["files"] and download.call_count == 2
    assert assets.prepare_bundle(assets.MODEL_ID, device="0") == first and download.call_count == 2
    # --no-cache downloads the variant afresh.
    assert assets.prepare_bundle(assets.MODEL_ID, device="0", reuse_cache=False) == first
    assert download.call_count == 3 and download.call_args.kwargs["force_download"]


def test_a_repository_copy_serves_each_device_its_package(bundle, tmp_path_factory):
    from surogate.serve.tts import assets

    tmp_path = tmp_path_factory.mktemp("repository")  # outside the fixture package
    cpu_root = tmp_path / "cpu-package"
    shutil.copytree(bundle.root, cpu_root)
    cpu = validate_bundle(cpu_root)
    variant = gpu_variant_of(bundle)
    repository = tmp_path / "surogate-ro-tts"
    for folder, source in ((assets.PREFIX, cpu), ("gpu", variant)):
        shutil.copytree(source.root, repository / folder)
    assert assets.prepare_bundle(str(repository), device="0").root == repository / "gpu"
    assert assets.prepare_bundle(str(repository), device="cpu").root == repository / assets.PREFIX
    # Only the GPU variant downloaded: a clear error on the CPU, not a traceback.
    shutil.rmtree(repository / assets.PREFIX)
    with pytest.raises(ValueError, match="No voices.json"):
        assets.prepare_bundle(str(repository), device="cpu")


def test_an_interrupted_download_is_completed(bundle, tmp_path_factory, monkeypatch):
    from surogate.serve.tts import assets

    tmp_path = tmp_path_factory.mktemp("interrupted")  # outside the fixture package
    variant = gpu_variant_of(bundle)
    monkeypatch.setenv("SUROGATE_SERVE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(assets, "GPU_PROFILE_SHA256", sha256(variant.root / "voices.json"))
    download = published_download(bundle, variant)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    first = assets.prepare_bundle(assets.MODEL_ID, device="0")
    (first.root / "model.gguf").unlink()  # as a download interrupted during the model leaves it
    assert assets.prepare_bundle(assets.MODEL_ID, device="0") == first
    assert download.call_count == 2 and not download.call_args.kwargs["force_download"]
    # A pin the published files do not match is refused, with the way out named.
    monkeypatch.setattr(assets, "GPU_PROFILE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="checksum mismatch.*--no-cache"):
        assets.prepare_bundle(assets.MODEL_ID, device="0")


def test_the_gpu_variant_pins_are_well_formed():
    from surogate.serve.tts import assets

    assert re.fullmatch(r"[0-9a-f]{40}", assets.GPU_REVISION) and assets.GPU_REVISION != assets.REVISION
    assert re.fullmatch(r"[0-9a-f]{64}", assets.GPU_PROFILE_SHA256)
    assert assets.GPU_PREFIX == "gpu"


def test_the_pinned_gpu_profile_is_the_published_one(tmp_path):
    """Opt in with SUROGATE_TTS_TEST_MODEL=surogate/surogate-ro-tts (reads one small file from HF)."""
    from surogate.serve.tts import assets

    if os.environ.get("SUROGATE_TTS_TEST_MODEL") != assets.MODEL_ID:
        pytest.skip("set SUROGATE_TTS_TEST_MODEL=surogate/surogate-ro-tts to check the published pins")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(assets.MODEL_ID, "gpu/voices.json", revision=assets.GPU_REVISION, local_dir=tmp_path)
    assert sha256(path) == assets.GPU_PROFILE_SHA256
    assert "lib/libggml-cuda.so.0" in json.loads(Path(path).read_text())["files"]


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
        assert args[args.index(b"/dev/stdin") + 1] == b"-"  # stream mode: audio on stdout, no files
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


def test_streamed_audio_arrives_as_it_is_synthesized(bundle, tmp_path):
    """stream_format "audio": the first chunk reaches the client long before the last is made."""
    with native_server(bundle, tmp_path) as (client, _):
        whole = client.post("/v1/audio/speech", json={"input": "Bună", "seed": 103, "voice": "Doina"})
        for fmt in ("wav", "pcm"):
            start = time.monotonic()
            arrivals, body = [], b""
            with client.stream("POST", "/v1/audio/speech", json={"input": "Bună", "seed": 103, "voice": "Doina",
                                                                 "response_format": fmt, "stream_format": "audio"}) as r:
                assert r.status_code == 200
                assert r.headers["x-usage-characters"] == "4" and r.headers["x-audio-sample-rate"] == "22050"
                assert r.headers["content-type"] == ("audio/wav" if fmt == "wav" else "audio/pcm")
                for part in r.iter_raw():
                    arrivals.append(time.monotonic() - start)
                    body += part
            assert arrivals[-1] - arrivals[0] > 0.6, arrivals  # three chunks, 0.4 s apart
            pcm = bytes([11, 0]) * 300
            if fmt == "pcm":
                assert body == pcm
            else:  # an open-ended WAV header, then the same samples the whole file carries
                assert body[:4] == b"RIFF" and body[40:44] == b"\xff\xff\xff\xff" and body[44:] == pcm
                assert whole.content[44:] == pcm


def test_streamed_audio_as_server_sent_events(bundle, tmp_path):
    with native_server(bundle, tmp_path) as (client, _):
        with client.stream("POST", "/v1/audio/speech", json={"input": "Bună", "seed": 103, "voice": "Doina",
                                                             "response_format": "pcm", "stream_format": "sse"}) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
        assert [e["type"] for e in events] == ["speech.audio.delta"] * 3 + ["speech.audio.done"]
        audio = b"".join(base64.b64decode(e["audio"]) for e in events[:3])
        assert audio == bytes([11, 0]) * 300
        assert events[-1]["usage"]["input_characters"] == 4


def test_a_stream_that_fails_partway_is_not_completed(bundle, tmp_path):
    with native_server(bundle, tmp_path) as (client, _):
        # Raw audio: the connection ends without the final chunk, which a client sees as an error.
        with pytest.raises(httpx.RemoteProtocolError):
            with client.stream("POST", "/v1/audio/speech",
                               json={"input": "Bună", "seed": 104, "stream_format": "audio"}) as r:
                assert r.status_code == 200
                for _ in r.iter_raw():
                    pass
        # SSE: an error event instead of speech.audio.done.
        with client.stream("POST", "/v1/audio/speech",
                           json={"input": "Bună", "seed": 104, "stream_format": "sse"}) as r:
            events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
        assert [e["type"] for e in events] == ["speech.audio.delta", "error"]
        # Neither is billed, and both count as failed requests.
        metrics = client.get("/metrics").text
        assert 'surogate_requests_total{model="test",endpoint="speech",outcome="error"} 2' in metrics
        assert 'surogate_characters_total{model="test"} 0' in metrics
        # The worker is replaced and the next request succeeds.
        assert client.post("/v1/audio/speech", json={"input": "Bună"}).content == wav_bytes(11)
        with client.stream("POST", "/v1/audio/speech", json={"input": "Bună", "stream_format": "audio"}) as r:
            r.read()
        metrics = client.get("/metrics").text
        assert 'surogate_requests_total{model="test",endpoint="speech",outcome="ok"} 2' in metrics
        assert 'surogate_characters_total{model="test"} 8' in metrics


@pytest.mark.parametrize("device", ["cpu", "0"])
def test_a_client_that_leaves_mid_stream(bundle, tmp_path, device):
    """The worker stops the request at its next chunk and is kept: the next request pays no new load."""
    env = {"CUDA_VISIBLE_DEVICES": None, "EXPECT_CUDA_VISIBLE_DEVICES": "0" if device == "0" else ""}
    with native_server(bundle, tmp_path, "--device", device, env=env) as (client, process):
        worker = child_pids(process.pid)
        with client.stream("POST", "/v1/audio/speech",
                           json={"input": "Bună", "seed": 103, "stream_format": "audio"}) as r:
            next(r.iter_raw())  # the first chunk, then hang up
        started = time.monotonic()
        assert client.post("/v1/audio/speech", json={"input": "Bună"}).content == wav_bytes(11)
        assert time.monotonic() - started < 0.6  # not after the rest of the request (0.8 s)
        assert child_pids(process.pid) == worker


def test_a_streamed_request_still_gets_queue_errors_as_http_statuses(bundle, tmp_path):
    with native_server(bundle, tmp_path, "--max-pending-requests", "0", "--request-timeout", "0.6") as (client, _):
        with ThreadPoolExecutor(1) as pool:
            slow = pool.submit(client.post, "/v1/audio/speech", json={"input": "Bună", "seed": 102})
            time.sleep(0.1)
            queued = client.post("/v1/audio/speech", json={"input": "Bună", "stream_format": "audio"})
            assert queued.status_code == 429 and "x-usage-characters" not in queued.headers
            slow.result()


def test_a_streamed_request_that_times_out_in_the_queue_gets_504(bundle, tmp_path):
    # On a GPU a client that leaves keeps its worker until the rest of its request is drained
    # (seed 106: 3 s), which no deadline cuts short; a stream queued behind it times out.
    env = {"CUDA_VISIBLE_DEVICES": None, "EXPECT_CUDA_VISIBLE_DEVICES": "0"}
    with native_server(bundle, tmp_path, "--device", "0", "--request-timeout", "1", env=env) as (client, _):
        with client.stream("POST", "/v1/audio/speech",
                           json={"input": "Bună", "seed": 106, "stream_format": "audio"}) as r:
            next(r.iter_raw())
        started = time.monotonic()
        queued = client.post("/v1/audio/speech", json={"input": "Bună", "stream_format": "sse"})
        assert queued.status_code == 504 and "x-usage-characters" not in queued.headers
        assert 0.8 < time.monotonic() - started < 2.5


def test_a_slow_client_does_not_hold_the_worker(bundle, tmp_path):
    """The worker is handed back when synthesis ends, while the audio is still being delivered."""
    with native_server(bundle, tmp_path) as (client, _):
        expected = bytes([11, 0]) * 100 * 5243 * 48
        with client.stream("POST", "/v1/audio/speech",
                           json={"input": "Bună", "seed": 105, "response_format": "pcm",
                                 "stream_format": "audio"}) as r:
            parts = r.iter_raw()
            received = [next(parts)]
            # The client stops reading for 2 s, far more than the socket buffers hold is still
            # undelivered; the next request is served meanwhile, not after the slow one.
            started = time.monotonic()
            other = client.post("/v1/audio/speech", json={"input": "Bună", "voice": "Tudor"})
            assert other.content == wav_bytes(12) and time.monotonic() - started < 1.5
            time.sleep(max(0.0, 2 - (time.monotonic() - started)))
            received.extend(parts)
        assert b"".join(received) == expected


def test_listeners_that_read_slowly_keep_their_place_in_the_queue(bundle, tmp_path):
    """A stream still being delivered counts against the queue: more are refused, /health answers."""
    with native_server(bundle, tmp_path, "--max-pending-requests", "2") as (client, _):
        statuses, readers = [], []  # an iterator dropped would close its stream
        with contextlib.ExitStack() as streams:
            started = time.monotonic()
            for _ in range(6):
                listener = streams.enter_context(httpx.Client(base_url=client.base_url, headers=client.headers,
                                                              timeout=10))
                r = streams.enter_context(listener.stream(
                    "POST", "/v1/audio/speech",
                    json={"input": "Bună", "seed": 105, "response_format": "pcm", "stream_format": "audio"}))
                statuses.append(r.status_code)
                if r.status_code == 200:
                    readers.append(r.iter_raw())
                    next(readers[-1])  # then read nothing more for now
            assert statuses.count(200) == 3 and statuses.count(429) == 3, statuses
            # The refusals come at once, not when an idle connection lets a thread go.
            assert time.monotonic() - started < 2
            started = time.monotonic()
            assert client.get("/health", timeout=2).status_code == 200
            assert client.get("/metrics", timeout=2).status_code == 200
            assert time.monotonic() - started < 1
            # On new connections too, as a supervisor or a gateway's fresh connection comes: the
            # refused listeners' idle connections must not hold the threads that would serve them.
            with httpx.Client(base_url=client.base_url, headers=client.headers, timeout=5) as fresh:
                started = time.monotonic()
                assert fresh.get("/health").status_code == 200
                assert time.monotonic() - started < 1.5
            with httpx.Client(base_url=client.base_url, headers=client.headers, timeout=5) as fresh:
                started = time.monotonic()
                assert fresh.post("/v1/audio/speech", json={"input": "Bună", "seed": 105}).status_code == 429
                assert time.monotonic() - started < 1.5


def test_whole_recordings_keep_their_place_until_delivered(bundle, tmp_path):
    """A whole recording being written to a client that reads slowly counts against the queue too."""
    with native_server(bundle, tmp_path, "--max-pending-requests", "0") as (client, _):
        with httpx.Client(base_url=client.base_url, headers=client.headers, timeout=10) as listener:
            with listener.stream("POST", "/v1/audio/speech", json={"input": "Bună", "seed": 105}) as r:
                assert r.status_code == 200
                reader = r.iter_raw()
                next(reader)  # then read nothing more for now
                started = time.monotonic()
                assert client.post("/v1/audio/speech", json={"input": "Bună"}).status_code == 429
                with httpx.Client(base_url=client.base_url, headers=client.headers, timeout=5) as fresh:
                    assert fresh.get("/health").status_code == 200
                assert time.monotonic() - started < 1.5
                received = sum(len(part) for part in reader)  # all of it, then the place is free
                assert received > 40 << 20
        assert client.post("/v1/audio/speech", json={"input": "Bună"}).status_code == 200


def test_a_range_request_is_refused_and_not_billed(bundle, tmp_path):
    with native_server(bundle, tmp_path) as (client, _):
        for extra in ({}, {"stream_format": "audio"}):
            r = client.post("/v1/audio/speech", json={"input": "Bună", **extra}, headers={"Range": "bytes=0-10"})
            assert r.status_code == 416 and "x-usage-characters" not in r.headers


def test_shutdown_during_a_gpu_stream_does_not_wait_for_it(bundle, tmp_path):
    """A request cancelled by a shutdown kills its worker instead of draining it."""
    env = {"CUDA_VISIBLE_DEVICES": None, "EXPECT_CUDA_VISIBLE_DEVICES": "0"}
    with native_server(bundle, tmp_path, "--device", "0", env=env) as (client, process):
        with client.stream("POST", "/v1/audio/speech",
                           json={"input": "Bună", "seed": 106, "stream_format": "audio"}) as r:
            next(r.iter_raw())  # the second chunk is 3 s away
            started = time.monotonic()
            process.terminate()
            assert process.wait(timeout=2) == 0
            assert time.monotonic() - started < 1.5


def test_longer_inputs_with_a_higher_limit(bundle, tmp_path):
    sentence = "Aceasta este o propoziție de test, citită de server. "  # 53 characters
    text = sentence * 150  # 7,950 characters, 150 sentences
    with native_server(bundle, tmp_path) as (client, _):
        refused = client.post("/v1/audio/speech", json={"input": text})
        assert refused.status_code == 400 and "4096" in refused.json()["error"]["message"]
        # Up to the limit in ordinary sentences: 77 of them.
        assert client.post("/v1/audio/speech", json={"input": sentence * 77}).status_code == 200
    with native_server(bundle, tmp_path, "--max-input-characters", "8192") as (client, _):
        with client.stream("POST", "/v1/audio/speech", json={"input": text, "stream_format": "audio"}) as r:
            assert r.status_code == 200 and r.headers["x-usage-characters"] == str(len(text))
            r.read()
        assert client.post("/v1/audio/speech", json={"input": text * 2}).status_code == 400


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
