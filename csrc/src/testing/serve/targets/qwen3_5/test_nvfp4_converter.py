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


def test_exports_resolve_the_draft_head_ranking() -> None:
    # The exports live one directory below the family converter and copied its relative
    # path to `serve/tools`; the copy pointed at `convert/tools` and every 27B export
    # failed at preflight on a missing ranking file.
    from surogate.serve.convert.qwen3_5 import convert as family_convert
    from surogate.serve.convert.qwen3_5 import draft_head
    from surogate.serve.convert.qwen3_5.exports import (
        convert_nvfp4_all,
        convert_nvfp4_mixed_bf16,
    )

    expected = family_convert._tools_root()
    for module in (convert_nvfp4, convert_nvfp4_mixed_bf16, convert_nvfp4_all):
        assert module._tools_root() == expected
    assert (expected / draft_head.DEFAULT_RANKING).is_file()


def test_geometry_block_states_the_checkpoint_dimensions() -> None:
    # The engine target is compiled at one size and reads the rest from the artifact's
    # geometry member; an export that writes none binds a 27B against the 2B's dimensions.
    from surogate.serve.convert.qwen3_5 import convert as family_convert

    text = {
        "hidden_size": 5120, "num_hidden_layers": 64, "intermediate_size": 17408,
        "vocab_size": 248320, "num_attention_heads": 24, "num_key_value_heads": 4,
        "head_dim": 256, "linear_num_key_heads": 16, "linear_key_head_dim": 128,
        "linear_num_value_heads": 48, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "mtp_num_hidden_layers": 1, "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1.0e7},
    }
    block = family_convert.geometry_block({"text_config": text})
    assert block["output_rows"] == 248320 and block["hidden"] == 5120
    assert block["layers"] == 64 and block["gdn_value_heads"] == 48


def test_vision_geometry_block_states_the_tower() -> None:
    # The family target compiles one tower (the 2B's 24 x 1024); a 27B artifact that does not
    # declare its 27 x 1152 fails to bind its first vision object.
    from surogate.serve.convert.qwen3_5 import convert as family_convert

    vision = {
        "depth": 27, "hidden_size": 1152, "intermediate_size": 4304, "num_heads": 16,
        "patch_size": 16, "temporal_patch_size": 2, "in_channels": 3,
        "num_position_embeddings": 2304, "spatial_merge_size": 2, "out_hidden_size": 5120,
    }
    block = family_convert.vision_geometry_block({"vision_config": vision})
    assert block == {
        "layers": 27, "hidden": 1152, "intermediate": 4304, "heads": 16, "patch_dim": 1536,
        "merge": 2, "position_embeddings": 2304, "rotary_dim": 72, "output_hidden": 5120,
    }
    assert family_convert.vision_geometry_block({"text_config": {}}) is None
