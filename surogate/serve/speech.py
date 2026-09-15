"""Prepare NeMo FastConformer/TDT/CTC archives for native speech serving."""

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from filelock import FileLock

_MODEL = "surogate/surogate-ro-110m-streaming"
_CHECKPOINT = "Is_ctc_final_20260915.nemo"
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
        "causal_downsampling": True,
        "self_attention_model": "rel_pos",
        "att_context_style": "chunked_limited",
        "conv_context_size": "causal",
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
    left, right = encoder["att_context_size"]
    if left <= 0 or right < 0 or left % (right + 1):
        raise ValueError("speech attention context must contain whole streaming chunks")
    if config["preprocessor"].get("normalize") not in (None, "NA", False):
        raise ValueError("speech serving requires unnormalized streaming features")
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

        if model != _MODEL:
            raise ValueError(f"unsupported speech repository {model!r}; provide a local .nemo checkpoint")
        path = Path(hf_hub_download(model, _CHECKPOINT))
        lm = lm or hf_hub_download(model, "ro_4gram.nemo")
    if path.is_dir():
        lm = lm or str(path / "ro_4gram.nemo")
        path = path / _CHECKPOINT
    if lm is None:
        candidate = path.parent / "ro_4gram.nemo"
        if not candidate.is_file():
            raise ValueError("provide the matching Romanian 4-gram archive with --lm PATH")
        lm = candidate
    lm = Path(lm).expanduser().resolve(strict=True)
    path = path.resolve(strict=True)
    from importlib.util import find_spec

    spec = find_spec("silero_vad")
    if spec is None:
        raise ValueError("speech preparation needs silero-vad: pip install silero-vad==6.2.1")
    vad = Path(spec.origin).parent / "data" / "silero_vad.jit"
    identity = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in (path, lm)]
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
