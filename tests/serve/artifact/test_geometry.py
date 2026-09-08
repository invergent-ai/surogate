from pathlib import Path
from types import SimpleNamespace

import pytest

from surogate.serve.artifact.container import ArtifactError, parse_geometry
from surogate.serve.artifact.geometry import cpp_fields, cpp_required_fields, validate_geometry


def test_cpp_geometry_fields_are_generated_from_the_writer_schema():
    root = Path(__file__).resolve().parents[3]
    for name, vision in (("text", False), ("vision", True)):
        generated = root / f"csrc/src/serve/api/family/{name}_geometry_fields.inc"
        assert generated.read_text() == cpp_fields(vision=vision)
    assert (root / "csrc/src/serve/api/family/dflash_geometry_fields.inc").read_text() == cpp_fields(dflash=True)
    required = root / "csrc/src/serve/api/family/required_text_geometry_fields.inc"
    assert required.read_text() == cpp_required_fields()


@pytest.mark.parametrize(
    "values, message",
    [
        ({"vocab": 4096}, "unknown geometry field"),
        ({"conv_kernel": 5}, "unknown geometry field"),
        ({"hidden": True}, "finite number"),
        ({"hidden": 16.5}, "int32"),
        ({"layers": -1}, "int32"),
        ({"layers": 2**31}, "int32"),
        ({"layers": 10**1000}, "int32"),
        ({"rope_theta": float("inf")}, "finite number"),
        ({"rms_epsilon": float("nan")}, "finite number"),
        ({"rope_theta": 1e100}, "float32"),
    ],
)
def test_rejects_unrecognized_or_invalid_checkpoint_fields(values, message):
    with pytest.raises(ValueError, match=message):
        validate_geometry(values)


def test_artifact_reader_rejects_misspelled_geometry():
    with pytest.raises(ArtifactError, match="unknown geometry field"):
        parse_geometry(b'{"geometry":{"vocab":4096}}')


def test_vision_and_text_have_distinct_field_contracts():
    assert validate_geometry({"heads": 7, "hidden": 448}, vision=True) == {
        "heads": 7,
        "hidden": 448,
    }
    with pytest.raises(ValueError, match="unknown geometry field"):
        validate_geometry({"heads": 7})


def test_lfm2_metadata_preserves_checkpoint_vocabulary_and_convolution():
    from surogate.serve.convert.lfm2.convert import _geometry_block

    source = SimpleNamespace(hidden=384, layers=7, query_heads=6, kv_heads=2,
                             head_dim=64, intermediate=896, vocab=8192, conv_kernel=5)
    source.declared = SimpleNamespace(hf_config={
        "rms_norm_eps": 1e-5, "rope_theta": 123456.0, "max_position_embeddings": 4096,
    })
    geometry = validate_geometry(_geometry_block(source, token_domain=7680))
    assert geometry["output_rows"] == 8192
    assert geometry["token_domain"] == 7680
    assert geometry["gdn_conv_kernel"] == 5


def test_moe_metadata_preserves_checkpoint_experts_and_top_k():
    from surogate.serve.convert.qwen3_moe.convert import _geometry_block

    source = SimpleNamespace(hidden=384, layers=7, query_heads=6, kv_heads=2,
                             head_dim=64, intermediate=896, vocab=8192, experts=32,
                             experts_per_token=4)
    source.declared = SimpleNamespace(hf_config={
        "rms_norm_eps": 1e-5, "rope_theta": 123456.0, "max_position_embeddings": 4096,
    })
    geometry = validate_geometry(_geometry_block(source, token_domain=7680))
    assert geometry["output_rows"] == 8192
    assert geometry["experts"] == 32
    assert geometry["experts_per_token"] == 4
