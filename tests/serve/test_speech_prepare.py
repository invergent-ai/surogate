"""Speech format validation and launch behavior, without loading CUDA or NeMo."""

import copy
import io
import json
import sys
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from surogate.cli import serve
from surogate.serve.speech import _archive, _validate


def test_speech_lm_is_preparation_only(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["surogate", "serve", "--stt", "model.nemo", "--lm", "ro.nemo", "--device", "cpu"])
    monkeypatch.setattr(serve, "_resolve_binary", lambda _: "/stt")
    prepare = Mock(return_value=Path("/prepared"))
    monkeypatch.setattr("surogate.serve.speech.ensure_speech_weights", prepare)
    execute = Mock()
    monkeypatch.setattr(serve.os, "execv", execute)
    serve.maybe_exec_serve()
    assert prepare.call_args.kwargs["lm"] == "ro.nemo"
    execute.assert_called_once_with(
        "/stt", ["/stt", "/prepared", "--device", "cpu", "--served-model-name", "model.nemo"]
    )


@pytest.mark.parametrize(
    "args",
    [
        ["--stt", "--embed", "model"],
        ["--stt", "--generate", "model"],
        ["--stt", "model", "--mmproj", "vision"],
        ["model", "--lm", "lm"],
    ],
)
def test_incompatible_modes_and_resources(args):
    with pytest.raises(ValueError):
        serve._parse_invocation(args)


def test_archive_does_not_extract_paths(tmp_path):
    path = tmp_path / "bad.nemo"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("../escaped")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="model_config.yaml"):
        _archive(path)
    assert not (tmp_path.parent / "escaped").exists()


def test_duplicate_archive_member_is_rejected(tmp_path):
    path = tmp_path / "bad.nemo"
    with tarfile.open(path, "w") as archive:
        for name in ("model_config.yaml", "./model_config.yaml"):
            info = tarfile.TarInfo(name)
            info.size = 2
            archive.addfile(info, io.BytesIO(b"{}"))
    with pytest.raises(ValueError, match="duplicate"):
        _archive(path)


def test_incompatible_encoder_is_rejected():
    with pytest.raises(ValueError, match="subsampling"):
        _validate({"encoder": {"subsampling": "striding"}})


def _offline_config():
    return {
        "encoder": {
            "subsampling": "dw_striding",
            "subsampling_factor": 8,
            "causal_downsampling": False,
            "self_attention_model": "rel_pos",
            "att_context_style": "regular",
            "att_context_size": [-1, -1],
            "conv_context_size": None,
            "conv_norm_type": "batch_norm",
            "xscaling": False,
            "feat_in": 80,
            "subsampling_conv_channels": 256,
        },
        "sample_rate": 16000,
        "decoder": {"prednet": {"pred_rnn_layers": 1}},
        "preprocessor": {
            "normalize": "per_feature",
            "n_fft": 512,
            "features": 80,
            "window_size": 0.025,
            "window_stride": 0.01,
            "frame_splicing": 1,
            "pad_to": 0,
        },
    }


def test_offline_architecture_accepted():
    _validate(_offline_config())


@pytest.mark.parametrize(
    "field,value", [("causal_downsampling", True), ("att_context_size", [70, 13]), ("conv_context_size", "causal")]
)
def test_offline_architecture_mismatches_are_rejected(field, value):
    config = copy.deepcopy(_offline_config())
    config["encoder"][field] = value
    with pytest.raises(ValueError, match="offline speech"):
        _validate(config)


def test_offline_preparation_handles_hf_arpa_blob_without_vad(tmp_path, monkeypatch):
    import torch
    import yaml

    import surogate.serve.speech as speech

    config = _offline_config()
    config["decoder"]["vocab_size"] = 5
    config["model_defaults"] = {"tdt_durations": [0, 1, 2, 3, 4]}
    checkpoint = tmp_path / "jackrabbit-110m-ro.nemo"
    data = yaml.safe_dump(config).encode()
    with tarfile.open(checkpoint, "w") as archive:
        info = tarfile.TarInfo("model_config.yaml")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    blob = tmp_path / "123456abcdef"
    blob.write_text("\\data\\\nngram 1=5\n")
    lm = tmp_path / "lm-4gram-ro.arpa"
    lm.symlink_to(blob)
    binary = tmp_path / "surogate-stt"
    binary.with_name("surogate-stt-lm").touch()
    monkeypatch.setattr(serve, "_resolve_binary", lambda _: str(binary))
    monkeypatch.setattr("importlib.util.find_spec", lambda _: None)
    monkeypatch.setattr(speech, "_archive", lambda _: (config, {"weight": torch.ones(1)}, b"tokenizer"))
    monkeypatch.setenv("SUROGATE_SERVE_CACHE", str(tmp_path / "cache"))

    def convert(args, *, check):
        assert check and Path(args[1]) == blob
        output = Path(args[2])
        (output / "lm.json").write_text(json.dumps({"vocab_size": 5, "max_order": 3}))
        (output / "lm.safetensors").write_bytes(b"converted")

    runner = Mock(side_effect=convert)
    monkeypatch.setattr(speech.subprocess, "run", runner)
    result = speech.ensure_speech_weights(str(checkpoint), lm=lm, echo=lambda _: None)
    runner.assert_called_once()
    assert (result / "lm.safetensors").read_bytes() == b"converted"
    assert not (result / "vad.jit").exists()
    assert speech.ensure_speech_weights(str(checkpoint), lm=lm) == result
    runner.assert_called_once()
