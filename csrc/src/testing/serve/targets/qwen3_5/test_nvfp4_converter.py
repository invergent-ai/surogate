from __future__ import annotations

import pytest

from surogate.serve.convert.qwen3_5.exports import convert_nvfp4_mlp_only as convert_nvfp4


def test_converter_rejects_wrong_basename_before_reading_sources(tmp_path) -> None:
    output = tmp_path / "qwen3_8_27b.sinfer"
    with pytest.raises(ValueError, match="output basename"):
        convert_nvfp4.convert(
            tmp_path / "missing-official",
            tmp_path / "missing-quantized",
            output,
            device="cpu",
        )
    assert not output.exists()
