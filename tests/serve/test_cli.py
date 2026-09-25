"""The launcher must preserve native arguments before it prepares any weights."""

import re
import subprocess
import sys
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
        "stt": "speech/server.cpp",
        "tts": "tts/server.cpp",
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


@pytest.mark.parametrize("mode,flag,values", [
    ("stt", "--threads", ["0", "-1", "257", "four", "4oops"]),
    ("tts", "--threads", ["0", "-1", "257", "four", "4oops"]),
    ("tts", "--codec-threads", ["-1", "257", "four", "4oops"]),
    ("tts", "--max-input-characters", ["0", "-1", "16385", "many"]),
    ("stt", "--cpu-kernels", ["missing"]),
    ("tts", "--cpu-kernels", ["missing"]),
])
def test_invalid_cpu_compute_options_fail_before_model_load(mode, flag, values):
    binary = serve._resolve_binary(mode)
    if binary is None:
        pytest.skip("native serving binaries are not built")
    for value in values:
        result = subprocess.run(
            [binary, "/nonexistent-model", "--device", "cpu", flag, value],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0
        assert "cannot read speech weights" not in result.stderr
        assert "voices.json" not in result.stderr


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
    speech = Mock()
    speech.ensure_speech_weights.return_value = Path("/prepared.sinfer")
    monkeypatch.setitem(sys.modules, "surogate.serve.speech", speech)
    tts = Mock()
    tts.prepare_bundle.return_value.root = Path("/prepared.sinfer")
    monkeypatch.setitem(sys.modules, "surogate.serve.tts.assets", tts)
    execute = Mock()
    monkeypatch.setattr(serve.os, "execv", execute)
    serve.maybe_exec_serve()
    identity = ["--served-model-name", "model"] if mode != "generate" else []
    execute.assert_called_once_with("/engine", ["/engine", "/prepared.sinfer", "--device", "1", *identity])
    prepare = tts.prepare_bundle if mode == "tts" else speech.ensure_speech_weights if mode == "stt" else ingest.ensure_encoder_weights if mode == "embed" else ingest.ensure_engine_weights
    assert prepare.call_args.args == ("model",)
    assert prepare.call_args.kwargs["reuse_cache"] is False
    if mode == "tts":  # the TTS package is checked for the device it will serve on
        assert prepare.call_args.kwargs["device"] == "1"


def test_tts_package_check_uses_the_last_device_and_refuses_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["surogate", "serve", "--tts", "--device", "cpu", "--device", "0", "model"])
    monkeypatch.setattr(serve, "_resolve_binary", lambda mode: "/engine")
    tts = Mock()
    tts.prepare_bundle.side_effect = ValueError("This TTS package has only the CPU runtime")
    monkeypatch.setitem(sys.modules, "surogate.serve.tts.assets", tts)
    monkeypatch.setattr(serve.os, "execv", Mock())
    with pytest.raises(SystemExit) as exit_info:
        serve.maybe_exec_serve()
    assert exit_info.value.code == 2
    assert tts.prepare_bundle.call_args.kwargs["device"] == "0"
    assert "surogate serve: This TTS package has only the CPU runtime" in capsys.readouterr().err


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
    identity = ["--served-model-name", "model.gguf"] if not selector else []
    execute.assert_called_once_with("/engine", ["/engine", "/prepared.sinfer", "--vision", *identity])


@pytest.mark.parametrize("selector", [[], ["--embed"]])
@pytest.mark.parametrize("model", ["google/embeddinggemma-300m", "./models/my model.gguf", "/models/checkpoint"])
@pytest.mark.parametrize("alias", [None, "deployment"])
def test_public_model_id_preserves_original_argument(selector, model, alias, monkeypatch):
    flags = [] if alias is None else ["--served-model-name", alias]
    monkeypatch.setattr(sys, "argv", ["surogate", "serve", *selector, model, *flags])
    monkeypatch.setattr(serve, "_resolve_binary", lambda mode: "/engine")
    ingest = Mock()
    ingest.ensure_engine_weights.return_value = ingest.ensure_encoder_weights.return_value = Path("/cache/prepared.sinfer")
    monkeypatch.setitem(sys.modules, "surogate.serve.ingest", ingest)
    execute = Mock()
    monkeypatch.setattr(serve.os, "execv", execute)
    serve.maybe_exec_serve()
    execute.assert_called_once_with("/engine", ["/engine", "/cache/prepared.sinfer",
                                              "--served-model-name", alias or model])


@pytest.mark.parametrize("args", [["model.gguf", "--mmproj="], ["--embed", "model.gguf", "--mmproj", "p.gguf"]])
def test_invalid_projector_invocation(args):
    with pytest.raises(ValueError):
        serve._parse_invocation(args)


@pytest.mark.parametrize("selector", [[], ["--generate"]])
@pytest.mark.parametrize("explicit", [False, True])
def test_muse_drafter_is_preparation_only(selector, explicit, monkeypatch):
    extra = ["--dflash-model=draft.gguf"] if explicit else []
    monkeypatch.setattr(sys, "argv", ["surogate", "serve", *selector, "model.gguf",
        "--spec", "dflash", *extra])
    monkeypatch.setattr(serve, "_resolve_binary", lambda mode: "/engine")
    ingest = Mock()
    ingest.ensure_engine_weights.return_value = Path("/prepared.sinfer")
    monkeypatch.setitem(sys.modules, "surogate.serve.ingest", ingest)
    execute = Mock()
    monkeypatch.setattr(serve.os, "execv", execute)
    serve.maybe_exec_serve()
    assert ingest.ensure_engine_weights.call_args.kwargs["dflash_model"] == ("draft.gguf" if explicit else "auto")
    assert "--dflash-model" not in execute.call_args.args[1]
    assert execute.call_args.args[1][2:4] == ["--spec", "dflash"]


@pytest.mark.parametrize("literal", ["--dflash-model", "--spec"])
def test_drafter_flag_like_prompt_is_not_consumed(literal, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["surogate", "serve", "--generate", "model.gguf", "--prompt", literal])
    monkeypatch.setattr(serve, "_resolve_binary", lambda mode: "/engine")
    ingest = Mock()
    ingest.ensure_engine_weights.return_value = Path("/prepared.sinfer")
    monkeypatch.setitem(sys.modules, "surogate.serve.ingest", ingest)
    execute = Mock()
    monkeypatch.setattr(serve.os, "execv", execute)
    serve.maybe_exec_serve()
    assert "dflash_model" not in ingest.ensure_engine_weights.call_args.kwargs
    execute.assert_called_once_with("/engine", ["/engine", "/prepared.sinfer", "--prompt", literal])


def test_decision_temperature_is_a_server_option():
    """The decisions calibration reaches the server engine as typed; other modes refuse it."""
    expected = ("server", "model", ["--decision-temperature", "2.5"], True, None)
    assert serve._parse_invocation(["model", "--decision-temperature", "2.5"]) == expected
    assert serve._parse_invocation(["--decision-temperature=2.5", "model"]) == expected
    with pytest.raises(ValueError, match="needs a value"):
        serve._parse_invocation(["model", "--decision-temperature"])
    for mode in ("generate", "embed", "stt", "tts"):
        with pytest.raises(ValueError, match="not supported"):
            serve._parse_invocation([f"--{mode}", "model", "--decision-temperature", "2.5"])


def test_decision_attempts_is_a_server_option():
    """The decisions retry bound reaches the server engine as typed; other modes refuse it."""
    expected = ("server", "model", ["--decision-attempts", "5"], True, None)
    assert serve._parse_invocation(["model", "--decision-attempts", "5"]) == expected
    assert serve._parse_invocation(["--decision-attempts=5", "model"]) == expected
    with pytest.raises(ValueError, match="needs a value"):
        serve._parse_invocation(["model", "--decision-attempts"])
    for mode in ("generate", "embed", "stt", "tts"):
        with pytest.raises(ValueError, match="not supported"):
            serve._parse_invocation([f"--{mode}", "model", "--decision-attempts", "5"])


def test_decision_temperature_is_passed_to_the_engine(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["surogate", "serve", "model", "--decision-temperature", "2.5"])
    monkeypatch.setattr(serve, "_resolve_binary", lambda mode: "/engine")
    ingest = Mock()
    ingest.ensure_engine_weights.return_value = Path("/prepared.sinfer")
    monkeypatch.setitem(sys.modules, "surogate.serve.ingest", ingest)
    execute = Mock()
    monkeypatch.setattr(serve.os, "execv", execute)
    serve.maybe_exec_serve()
    execute.assert_called_once_with("/engine", ["/engine", "/prepared.sinfer", "--decision-temperature", "2.5",
                                                "--served-model-name", "model"])


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "1e999", "abc", ""])
def test_invalid_decision_temperature_fails_before_model_load(value):
    # The native parser owns validation and runs before any CUDA call; no GPU is made visible.
    binary = serve._resolve_binary("server")
    if binary is None:
        pytest.skip("native serving binaries are not built")
    result = subprocess.run([binary, "/nonexistent-model", "--decision-temperature", value],
                            capture_output=True, text=True, timeout=20,
                            env={**serve.os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode != 0
    assert "invalid decision-temperature" in result.stderr
