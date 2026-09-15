"""Speech format validation and launch behavior, without loading CUDA or NeMo."""

import io
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
