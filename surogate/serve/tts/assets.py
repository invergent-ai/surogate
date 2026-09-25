"""Resolve and verify the published native packages (CPU, and its GPU variant) or a local voice export."""

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "surogate/amami-357m-ro"
RENAMED_FROM = "surogate/surogate-ro-tts"
REVISION = "5393e9bf69ec08ef2c89bf387bf3791c446fb26a"
PREFIX = "cpu"
PROFILE_SHA256 = "958dfcda804ebacb3e190963008433c3121cecc9b433646f595cd03b858f5581"
# The GPU variant: the same model, codec, tokenizer and voices, with lib/ holding the runtime built with
# CUDA (surogate.serve.tools.tts.gpu_variant), published as built, in the same revision.
GPU_REVISION = REVISION
GPU_PREFIX = "gpu"
GPU_PROFILE_SHA256 = "66c6757589e1b7592e073fafffa18eecbcbaeca5ffad69c348eb64dc330152cb"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Bundle:
    root: Path
    profile: dict


def _policy(value):
    expected = {"temperature", "cfg_scale", "max_steps", "topk", "longform_mode"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Unexpected native TTS decoding fields")
    if (
        type(value["max_steps"]) is not int
        or value["max_steps"] != 900
        or type(value["topk"]) is not int
        or value["topk"] != 80
        or value["longform_mode"] != "auto"
    ):
        raise ValueError("Native TTS requires max_steps=900, topk=80 and longform_mode=auto")
    for key in ("temperature", "cfg_scale"):
        number = value[key]
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"Invalid voice setting: {key}")


def validate_bundle(root, *, expected_profile_sha256=None):
    root = Path(root).resolve()
    path = root / "voices.json"
    if expected_profile_sha256 and sha256(path) != expected_profile_sha256:
        raise ValueError("Published TTS voice profile checksum mismatch")
    profile = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(profile, dict)
        or profile.get("schema") != 1
        or profile.get("method") != "experimental_magpie_native_cpu"
    ):
        raise ValueError("Expected an exported native CPU TTS voice package")
    if profile.get("tokenizer") != {"profile": "v2607", "offset": 96, "eos": 3358}:
        raise ValueError("Unsupported TTS tokenizer profile")
    _policy(profile.get("decoding", {}))
    voices = profile.get("voices")
    if not isinstance(voices, dict) or not voices:
        raise ValueError("The TTS package has no named voices")
    names = set()
    for name, voice in voices.items():
        if not isinstance(name, str) or not name.strip() or name != name.strip() or name.casefold() in names:
            raise ValueError("Voice names must be nonempty and distinct ignoring case")
        names.add(name.casefold())
        if not isinstance(voice, dict) or set(voice) - {"id", "decoding"}:
            raise ValueError(f"Invalid voice profile: {name}")
        if type(voice.get("id")) is not int or voice["id"] < 0:
            raise ValueError(f"Invalid speaker index for {name}")
        if not isinstance(voice.get("decoding", {}), dict):
            raise ValueError(f"Invalid voice decoding settings for {name}")
        _policy({**profile["decoding"], **voice.get("decoding", {})})
    files = profile.get("files")
    model_file = profile.get("model", "model.gguf")
    if not isinstance(model_file, str) or Path(model_file).name != model_file or not model_file.endswith(".gguf"):
        raise ValueError("The TTS profile names an invalid model file")
    required = {
        "bin/synthesize",
        model_file,
        "codec.gguf",
        "lib/libnemo_speech_tts.so.1",
        "lib/libggml.so.0",
        "lib/libggml-base.so.0",
        "lib/libggml-cpu.so.0",
        "tokenizer/config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/special_tokens_map.json",
    }
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError("The TTS package is missing required file hashes")
    for library in (root / "lib").glob("*.so*"):
        if str(library.relative_to(root)) not in files:
            raise ValueError(f"Native library is missing its checksum: {library.name}")
    for name, digest in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError("TTS asset paths must stay inside the package")
        asset = (root / relative).resolve()
        if not asset.is_relative_to(root):
            raise ValueError(f"TTS asset escapes the package: {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid TTS checksum: {name}")
        if not asset.is_file() or sha256(asset) != digest:
            raise ValueError(f"TTS asset missing or checksum mismatch: {name}")
    binary = root / "bin/synthesize"
    if not os.access(binary, os.X_OK):
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return Bundle(root, profile)


def _for_device(bundle, device):
    """A GPU needs the package's GPU variant: the same runtime built with CUDA (lib/libggml-cuda)."""
    if device != "cpu" and "lib/libggml-cuda.so.0" not in bundle.profile["files"]:
        raise ValueError(
            "This TTS package has only the CPU runtime. Serving on a GPU needs its GPU variant, whose "
            "lib/ holds the same runtime built with CUDA; see docs/inference/tts.md"
        )
    return bundle


def prepare_bundle(model, *, reuse_cache=True, echo=print, device="cpu"):
    # The server's own rule, checked before anything is downloaded for a GPU.
    if device != "cpu" and not re.fullmatch(r"\d{1,3}", device):
        raise ValueError("--device must be cpu or a CUDA device index")
    gpu = device != "cpu"
    path = Path(model).expanduser()
    if path.is_file() and path.name == "voices.json":
        return _for_device(validate_bundle(path.parent), device)
    if path.is_dir():
        if not (path / "voices.json").is_file():
            # A copy of the model repository: the GPU variant for a GPU, the CPU package otherwise.
            package = path / (GPU_PREFIX if gpu else PREFIX)
            if not (package / "voices.json").is_file():
                raise ValueError(f"No voices.json in {path} or {package}")
            path = package
        return _for_device(validate_bundle(path), device)
    if model == RENAMED_FROM:
        raise ValueError(f"{RENAMED_FROM} was renamed to {MODEL_ID}")
    if model != MODEL_ID:
        raise ValueError(f"Use {MODEL_ID} or a local native TTS package containing voices.json")
    from filelock import FileLock

    # A GPU gets the pinned GPU variant, the CPU the pinned CPU package, each in its own cache.
    if gpu:
        revision, prefix, profile, kind = GPU_REVISION, GPU_PREFIX, GPU_PROFILE_SHA256, "GPU"
    else:
        revision, prefix, profile, kind = REVISION, PREFIX, PROFILE_SHA256, "CPU"
    cache = Path(os.environ.get("SUROGATE_SERVE_CACHE", Path.home() / ".cache/surogate/serve"))
    cache.mkdir(parents=True, exist_ok=True)
    snapshot = cache / f"tts-{revision}"
    root = snapshot / prefix

    def download(force):
        from huggingface_hub import snapshot_download

        echo(f"surogate serve: downloading the native {kind} TTS package from {MODEL_ID}")
        snapshot_download(
            MODEL_ID,
            revision=revision,
            allow_patterns=[f"{prefix}/*"],
            local_dir=snapshot,
            force_download=force,
        )

    with FileLock(str(snapshot) + ".lock"):
        if not reuse_cache or not (root / "voices.json").is_file():
            download(force=not reuse_cache)
        echo(f"surogate serve: verifying native {kind} TTS assets")
        try:
            bundle = validate_bundle(root, expected_profile_sha256=profile)
        except ValueError as first:
            # An interrupted download leaves the profile and some of the files: fetch what is
            # missing, once.
            try:
                download(force=False)
                echo(f"surogate serve: verifying native {kind} TTS assets")
                bundle = validate_bundle(root, expected_profile_sha256=profile)
            except Exception as error:
                raise ValueError(f"{first}. Run again with --no-cache to download the package afresh") from error
        return _for_device(bundle, device)
