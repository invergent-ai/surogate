"""Resolve and verify the published native CPU package or a local voice export."""

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "surogate/surogate-ro-tts"
REVISION = "2bf175b4edc7b3ca7261d80e4d4ad85117c4f0a4"
PREFIX = "releases/2026-09-18/cpu-voices/native"
PROFILE_SHA256 = "6c52964585162b516ab3ac6573ee5def535c80067300eb0ad06d0bb310642c9c"


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
    required = {
        "bin/synthesize",
        "model.gguf",
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
    path = Path(model).expanduser()
    if path.is_file() and path.name == "voices.json":
        return _for_device(validate_bundle(path.parent), device)
    if path.is_dir():
        if not (path / "voices.json").is_file():
            path = path / PREFIX
        return _for_device(validate_bundle(path), device)
    if model == MODEL_ID and device != "cpu":
        raise ValueError(
            f"The published {MODEL_ID} package has only the CPU runtime so far. To serve on a GPU, "
            "pass a local GPU variant of the package; see docs/inference/tts.md"
        )
    if model != MODEL_ID:
        raise ValueError(f"Use {MODEL_ID} or a local native CPU package containing voices.json")
    from filelock import FileLock

    cache = Path(os.environ.get("SUROGATE_SERVE_CACHE", Path.home() / ".cache/surogate/serve"))
    cache.mkdir(parents=True, exist_ok=True)
    snapshot = cache / f"tts-{REVISION}"
    root = snapshot / PREFIX
    with FileLock(str(snapshot) + ".lock"):
        if not reuse_cache or not (root / "voices.json").is_file():
            from huggingface_hub import snapshot_download

            echo(f"surogate serve: downloading the native CPU TTS package from {MODEL_ID}")
            snapshot_download(
                MODEL_ID,
                revision=REVISION,
                allow_patterns=[f"{PREFIX}/*"],
                local_dir=snapshot,
                force_download=not reuse_cache,
            )
        echo("surogate serve: verifying native CPU TTS assets")
        return validate_bundle(root, expected_profile_sha256=PROFILE_SHA256)
