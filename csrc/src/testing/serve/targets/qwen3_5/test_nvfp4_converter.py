import importlib
import pytest


@pytest.mark.parametrize("name", ["all", "uniform", "mixed_bf16", "mlp_only"])
@pytest.mark.parametrize("dual", [False, True])
def test_export_wrapper_forwards_checkpoint_paths_and_optional_flags(tmp_path, monkeypatch, name, dual):
    module = importlib.import_module("surogate.serve.convert.qwen3_5.exports.convert_nvfp4_" + name)
    calls = []
    monkeypatch.setattr(module.quantized, "convert", lambda *args, **kwargs: calls.append((args, kwargs)))
    base, quant, output = (tmp_path / n for n in ("renamed-base", "renamed-quant", "arbitrary-name.sinfer"))
    args = (base, quant, output) if dual else (base, output)
    module.convert(*args, device="cpu", mtp=False, vision=False)
    positional, options = calls[0]
    assert positional == (base, output)
    assert options["quantized_model_dir"] == (quant if dual else None)
    assert options["mtp"] is False and options["vision"] is False
