"""The launcher must preserve native arguments before it prepares any weights."""

import re
import sys
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from surogate.cli import serve


@pytest.mark.parametrize("mode", serve._MODES)
def test_native_option_inventory(mode):
    root = Path(__file__).resolve().parents[2] / "csrc/src/serve"
    path = root / {
        "server": "serve/serve_options.cpp",
        "generate": "cli/options.cpp",
        "embed": "encoder/options.h",
    }[mode]
    native = set(re.findall(r'arg == "(-[^\"]+)"', path.read_text())) - {"--help", "-h"}
    assert native == serve._VALUE_OPTIONS[mode] | serve._SWITCH_OPTIONS[mode]


@pytest.mark.parametrize("mode", serve._MODES)
def test_native_help_describes_every_option(mode):
    binary = serve._resolve_binary(mode)
    if binary is None:
        pytest.skip("native serving binaries are not built")
    result = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    for flag in serve._VALUE_OPTIONS[mode] | serve._SWITCH_OPTIONS[mode]:
        assert flag in result.stdout, flag


@pytest.mark.parametrize("mode", serve._MODES)
def test_every_value_option_preserves_flag_like_value(mode):
    selector = [] if mode == "server" else [f"--{mode}"]
    for flag in serve._VALUE_OPTIONS[mode]:
        for value in ("--embed", "--no-cache", "--engine-help", "-1", "a=b"):
            parsed = serve._parse_invocation([*selector, flag, value, "model"])
            assert parsed == (mode, "model", [flag, value], True, None)
            assert serve._parse_invocation([*selector, f"{flag}={value}", "model"]) == parsed


@pytest.mark.parametrize("mode", serve._MODES)
def test_every_switch_can_precede_model(mode):
    selector = [] if mode == "server" else [f"--{mode}"]
    for flag in serve._SWITCH_OPTIONS[mode]:
        assert serve._parse_invocation([*selector, flag, "model"]) == (
            mode, "model", [flag], True, None,
        )


@pytest.mark.parametrize("mode", serve._MODES)
@pytest.mark.parametrize("help_flag", ["--engine-help", "--help", "-h"])
def test_help_does_not_prepare_model(mode, help_flag, monkeypatch):
    selector = [] if mode == "server" else [f"--{mode}"]
    monkeypatch.setattr(sys, "argv", ["surogate", "serve", *selector, "missing/model", help_flag])
    monkeypatch.setattr(serve, "_resolve_binary", lambda mode: "/engine")
    execute = Mock()
    monkeypatch.setattr(serve.os, "execv", execute)
    # No model preparation module is imported on the help path.
    monkeypatch.setitem(sys.modules, "surogate.serve.ingest", None)
    serve.maybe_exec_serve()
    execute.assert_called_once_with("/engine", ["/engine", "--help"])


@pytest.mark.parametrize("mode", serve._MODES)
def test_execution_puts_resolved_model_first(mode, monkeypatch):
    selector = [] if mode == "server" else [f"--{mode}"]
    monkeypatch.setattr(sys, "argv", ["surogate", "serve", *selector, "--device=1", "model", "--no-cache"])
    monkeypatch.setattr(serve, "_resolve_binary", lambda mode: "/engine")
    ingest = Mock()
    ingest.ensure_engine_weights.return_value = ingest.ensure_encoder_weights.return_value = Path("/prepared.sinfer")
    monkeypatch.setitem(sys.modules, "surogate.serve.ingest", ingest)
    execute = Mock()
    monkeypatch.setattr(serve.os, "execv", execute)
    serve.maybe_exec_serve()
    execute.assert_called_once_with("/engine", ["/engine", "/prepared.sinfer", "--device", "1"])
    prepare = ingest.ensure_encoder_weights if mode == "embed" else ingest.ensure_engine_weights
    assert prepare.call_args.args == ("model",)
    assert prepare.call_args.kwargs["reuse_cache"] is False


def test_frontend_before_model():
    assert serve._parse_invocation(["--embed", "--frontend=tokenizer", "--device", "cpu", "model"]) == (
        "embed", "model", ["--device", "cpu"], True, "tokenizer",
    )


@pytest.mark.parametrize("args", [
    [], ["--embed", "--generate", "model"], ["model", "extra"],
    ["--device"], ["model", "--unknown"], ["model", "--cors=yes"],
    ["--generate", "model", "--port", "8080"], ["model", "--frontend", "tokenizer"],
    ["--embed", "model", "--frontend="],
])
def test_bad_invocations_fail_before_ingest(args):
    with pytest.raises(ValueError):
        serve._parse_invocation(args)


@pytest.mark.parametrize("selector,flags,expected", [
    ([], ["--device", "2"], "cuda:2"),
    ([], ["--devices", "2,1", "--device", "0"], "cuda:2"),
    (["--generate"], ["--device", "1", "--prompt", "--devices"], "cuda:1"),
    (["--embed"], ["--device", "cpu"], "cpu"),
])
def test_preparation_uses_runtime_device(selector, flags, expected, monkeypatch):
    monkeypatch.delenv("SUROGATE_CONVERT_DEVICE", raising=False)
    monkeypatch.setattr(sys, "argv", ["surogate", "serve", *selector, "model", *flags])
    monkeypatch.setattr(serve, "_resolve_binary", lambda mode: "/engine")
    ingest = Mock()

    def prepare(*args, **kwargs):
        assert serve.os.environ["SUROGATE_CONVERT_DEVICE"] == expected
        return Path("/prepared.sinfer")

    ingest.ensure_engine_weights.side_effect = ingest.ensure_encoder_weights.side_effect = prepare
    monkeypatch.setitem(sys.modules, "surogate.serve.ingest", ingest)
    monkeypatch.setattr(serve.os, "execv", Mock())
    serve.maybe_exec_serve()
    assert "SUROGATE_CONVERT_DEVICE" not in serve.os.environ


@pytest.mark.parametrize("selector", [[], ["--generate"]])
def test_projector_is_consumed_before_native_execution(selector, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["surogate", "serve", *selector,
                                    "--mmproj=projector.gguf", "model.gguf", "--vision"])
    monkeypatch.setattr(serve, "_resolve_binary", lambda mode: "/engine")
    ingest = Mock()
    ingest.ensure_engine_weights.return_value = Path("/prepared.sinfer")
    monkeypatch.setitem(sys.modules, "surogate.serve.ingest", ingest)
    execute = Mock()
    monkeypatch.setattr(serve.os, "execv", execute)
    serve.maybe_exec_serve()
    assert ingest.ensure_engine_weights.call_args.kwargs["mmproj"] == "projector.gguf"
    execute.assert_called_once_with("/engine", ["/engine", "/prepared.sinfer", "--vision"])


@pytest.mark.parametrize("args", [["model.gguf", "--mmproj="], ["--embed", "model.gguf", "--mmproj", "p.gguf"]])
def test_invalid_projector_invocation(args):
    with pytest.raises(ValueError):
        serve._parse_invocation(args)
