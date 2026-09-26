"""Prepare NeMo FastConformer/TDT/CTC archives for native speech serving."""

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from filelock import FileLock

# repository: (checkpoint, 4-gram LM, pinned revision)
_MODELS = {
    "surogate/jackrabbit-110m-ro": (
        "jackrabbit-110m-ro.nemo", "lm-4gram-ro.nemo", "a3cce77ded84db5aa936494040d29e6d631c8af4"),
    "surogate/jackrabbit-110m-ro-streaming": (
        "jackrabbit-110m-ro-streaming.nemo", "lm-4gram-ro.nemo", "663337372128b936b65a8cc2ad0ce99b5b3e6a03"),
}
_RENAMED = {
    "surogate/surogate-ro-110m-tdt-ctc": "surogate/jackrabbit-110m-ro",
    "surogate/surogate-ro-110m-streaming": "surogate/jackrabbit-110m-ro-streaming",
}
_VERSION = 1


def _archive(path: Path):
    """Read only the named regular members; never extract archive paths."""
    import torch
    import yaml

    with tarfile.open(path) as archive:
        members = {}
        for member in archive:
            name = member.name.removeprefix("./")
            if member.isfile() and "/" not in name:
                if name in members:
                    raise ValueError(f"duplicate NeMo member: {name}")
                members[name] = member

        def read(name):
            if name not in members:
                raise ValueError(f"NeMo archive lacks {name}")
            return archive.extractfile(members[name])

        config = yaml.safe_load(read("model_config.yaml"))
        weights = torch.load(read("model_weights.ckpt"), map_location="cpu", weights_only=True)
        tokenizer = config.get("tokenizer", {}).get("model_path", "").removeprefix("nemo:")
        return config, weights, read(tokenizer).read() if tokenizer else None


def _validate(config):
    expected = {
        "subsampling": "dw_striding",
        "subsampling_factor": 8,
        "self_attention_model": "rel_pos",
        "conv_norm_type": "batch_norm",
        "xscaling": False,
        "feat_in": 80,
        "subsampling_conv_channels": 256,
    }
    encoder = config["encoder"]
    for key, value in expected.items():
        if encoder.get(key) != value:
            raise ValueError(f"unsupported speech encoder {key}: {encoder.get(key)!r}")
    if encoder.get("reduction") is not None or encoder.get("feat_out", -1) != -1:
        raise ValueError("speech encoder reduction/output projection is unsupported")
    streaming = encoder.get("att_context_style") == "chunked_limited"
    if streaming:
        left, right = encoder["att_context_size"]
        if left <= 0 or right < 0 or left % (right + 1):
            raise ValueError("speech attention context must contain whole streaming chunks")
        if encoder.get("causal_downsampling") is not True or encoder.get("conv_context_size") != "causal":
            raise ValueError("streaming speech requires causal downsampling and convolution")
        if config["preprocessor"].get("normalize") not in (None, "NA", False):
            raise ValueError("streaming speech requires unnormalized features")
    else:
        if (
            encoder.get("att_context_style") != "regular"
            or encoder.get("att_context_size") != [-1, -1]
            or encoder.get("causal_downsampling") is not False
            or encoder.get("conv_context_size") is not None
        ):
            raise ValueError("offline speech requires full attention and symmetric convolution")
        if config["preprocessor"].get("normalize") != "per_feature":
            raise ValueError("offline speech requires per-feature normalization")
    if config["sample_rate"] != 16000 or config["decoder"]["prednet"]["pred_rnn_layers"] != 1:
        raise ValueError("speech serving requires 16 kHz audio and a one-layer TDT predictor")
    preprocessor = config["preprocessor"]
    for key, value in {
        "n_fft": 512,
        "features": 80,
        "window_size": 0.025,
        "window_stride": 0.01,
        "frame_splicing": 1,
        "pad_to": 0,
    }.items():
        if preprocessor.get(key) != value:
            raise ValueError(f"unsupported speech preprocessor {key}: {preprocessor.get(key)!r}")


def ensure_speech_weights(model, *, lm=None, reuse_cache=True, echo=print):
    """Resolve the acoustic model, matching LM and VAD, then convert atomically."""
    path = Path(model).expanduser()
    if path.is_dir() and (path / "speech.json").is_file():
        return path
    if not path.exists():
        from huggingface_hub import hf_hub_download

        if model in _RENAMED:
            raise ValueError(f"{model} was renamed to {_RENAMED[model]}")
        if model not in _MODELS:
            raise ValueError(f"unsupported speech repository {model!r}; provide a local .nemo checkpoint")
        checkpoint, language_model, revision = _MODELS[model]
        path = Path(hf_hub_download(model, checkpoint, revision=revision))
        lm = lm or hf_hub_download(model, language_model, revision=revision)
    if path.is_dir():
        candidates = [path / checkpoint for checkpoint, _, _ in _MODELS.values() if (path / checkpoint).is_file()]
        if len(candidates) != 1:
            raise ValueError(
                "model directory must contain exactly one supported speech checkpoint; otherwise provide its file path"
            )
        path = candidates[0]
    if lm is None:
        candidates = [path.parent / name for name in ("lm-4gram-ro.nemo", "lm-4gram-ro.arpa")]
        candidates = [p for p in candidates if p.is_file()]
        if not candidates:
            raise ValueError("provide the matching Romanian 4-gram archive with --lm PATH")
        lm = candidates[0]
    lm = Path(lm).expanduser().resolve(strict=True)
    path = path.resolve(strict=True)
    # HF snapshots resolve to content-addressed blobs without file extensions.
    # Recognize the actual format, including when --lm names such a blob.
    with lm.open("rb") as source:
        arpa = source.read(1024).lstrip().startswith(b"\\data\\")
    from importlib.util import find_spec

    # Read the small configuration before deciding whether VAD is needed.
    with tarfile.open(path) as archive:
        import yaml

        member = next((m for m in archive if m.isfile() and m.name.removeprefix("./") == "model_config.yaml"), None)
        if member is None:
            raise ValueError("NeMo archive lacks model_config.yaml")
        config = yaml.safe_load(archive.extractfile(member))
    _validate(config)
    streaming = config["encoder"]["att_context_style"] == "chunked_limited"
    vad = None
    if streaming:
        spec = find_spec("silero_vad")
        if spec is None:
            raise ValueError("streaming speech preparation needs silero-vad: pip install silero-vad==6.2.1")
        vad = Path(spec.origin).parent / "data" / "silero_vad.jit"
    identity = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in (path, lm)]
    if vad:
        identity.append(("vad", hashlib.sha256(vad.read_bytes()).hexdigest()))
    key = hashlib.sha256(json.dumps([_VERSION, identity]).encode()).hexdigest()[:24]
    root = Path(os.environ.get("SUROGATE_SERVE_CACHE", Path.home() / ".cache/surogate/serve"))
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"speech-{key}"
    with FileLock(str(target) + ".lock"):
        if reuse_cache and (target / "speech.json").is_file():
            return target
        with tempfile.TemporaryDirectory(prefix="speech-", dir=root) as temporary:
            output = Path(temporary)
            from safetensors.torch import save_file

            config, weights, tokenizer = _archive(path)
            _validate(config)
            if tokenizer is None:
                raise ValueError("speech checkpoint lacks a SentencePiece tokenizer")
            echo("surogate serve: preparing speech weights and Romanian language model")
            save_file({k: v.contiguous() for k, v in weights.items()}, output / "acoustic.safetensors")
            del weights
            (output / "tokenizer.model").write_bytes(tokenizer)
            if arpa:
                from surogate.cli.serve import _resolve_binary

                binary = _resolve_binary("stt")
                converter = Path(binary).with_name("surogate-stt-lm") if binary else None
                if converter is None or not converter.is_file():
                    raise ValueError(
                        "ARPA preparation needs surogate-stt-lm; rebuild with make serve-build or make serve-stt-build"
                    )
                subprocess.run([str(converter), str(lm), str(output), str(config["decoder"]["vocab_size"])], check=True)
                lm_config = json.loads((output / "lm.json").read_text())
                (output / "lm.json").unlink()
            else:
                lm_config, weights, _ = _archive(lm)
                if lm_config["vocab_size"] != config["decoder"]["vocab_size"]:
                    raise ValueError("speech language-model vocabulary does not match the acoustic model")
                keep = (
                    "arcs_weights",
                    "backoff_weights",
                    "final_weights",
                    "to_states",
                    "ilabels",
                    "backoff_to_states",
                    "start_end_arcs",
                )
                save_file({k: weights[k].contiguous() for k in keep}, output / "lm.safetensors")
            if vad:
                shutil.copyfile(vad, output / "vad.jit")
            serving_config = {
                key: config[key] for key in ("sample_rate", "encoder", "decoder", "preprocessor", "model_defaults")
            }
            (output / "speech.json").write_text(
                json.dumps({"version": _VERSION, "model": serving_config, "lm": lm_config}, allow_nan=False)
            )
            if target.exists():
                shutil.rmtree(target)
            os.rename(output, target)
    return target
