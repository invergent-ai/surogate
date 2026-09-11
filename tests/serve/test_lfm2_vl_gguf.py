"""LFM2-VL projector pairing, quantization and RGB patch layout."""

import numpy as np
import pytest
import torch

from surogate.serve.artifact.container import Artifact
from surogate.serve.convert.lfm2_vl import gguf as converter
from tests.serve.test_lfm2_gguf import checkpoint, gguf, quantize


def projector(path, *, hidden=256):
    writer = gguf.GGUFWriter(str(path), "clip")
    writer.add_string("clip.projector_type", "lfm2")
    for key, value in {
        "embedding_length": 64,
        "feed_forward_length": 128,
        "block_count": 1,
        "attention.head_count": 1,
        "patch_size": 16,
        "image_size": 256,
        "projection_dim": hidden,
    }.items():
        writer.add_uint32("clip.vision." + key, value)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    writer.add_array("clip.vision.image_mean", [0.5] * 3)
    writer.add_array("clip.vision.image_std", [0.5] * 3)
    rng = np.random.default_rng(31)
    patch = rng.normal(0, 0.05, (64, 3, 16, 16)).astype(np.float32)

    def add(name, shape):
        value = patch if name == "v.patch_embd.weight" else rng.normal(0, 0.05, shape).astype(np.float32)
        fmt = (
            gguf.GGMLQuantizationType.Q8_0
            if len(shape) == 2 and "position" not in name
            else gguf.GGMLQuantizationType.F32
        )
        writer.add_tensor(name, quantize(value, fmt), raw_dtype=fmt)

    add("v.patch_embd.weight", patch.shape)
    add("v.patch_embd.bias", (64,))
    add("v.position_embd.weight", (256, 64))
    for name in ("attn_q", "attn_k", "attn_v", "attn_out"):
        add(f"v.blk.0.{name}.weight", (64, 64))
        add(f"v.blk.0.{name}.bias", (64,))
    for name, rows, columns in (("ffn_up", 128, 64), ("ffn_down", 64, 128)):
        add(f"v.blk.0.{name}.weight", (rows, columns))
        add(f"v.blk.0.{name}.bias", (rows,))
    for name in ("v.blk.0.ln1", "v.blk.0.ln2", "v.post_ln"):
        for suffix in ("weight", "bias"):
            add(name + "." + suffix, (64,))
    for name, rows, columns in (("mm.1", 128, 256), ("mm.2", hidden, 128)):
        add(name + ".weight", (rows, columns))
        add(name + ".bias", (rows,))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return patch


def test_projector_pairing_and_quantized_conversion(tmp_path):
    text, vision = tmp_path / "model.gguf", tmp_path / "mmproj-Q8.gguf"
    checkpoint(text, image_token=True)
    patch = projector(vision)
    assert converter.find_projector(text) == vision
    output = tmp_path / "model.sinfer"
    converter.convert(text, vision, output)
    with Artifact(output) as artifact:
        assert artifact.vision_geometry["siglip2"] == 1
        assert artifact.vision_geometry["merge"] == 2
        assert artifact.find("vision/layers/0/attention/qkv").format == "Q8_0"
        assert artifact.find("vision/merger/fc1").format == "Q8_0"
        actual = torch.frombuffer(bytearray(artifact.payload("vision/patch_embedding")), dtype=torch.bfloat16)
        expected = torch.from_numpy(patch).permute(0, 2, 3, 1).contiguous().bfloat16().reshape(-1)
        assert torch.equal(actual, expected)
    projector(tmp_path / "mmproj-other.gguf")
    with pytest.raises(ValueError, match="2 compatible"):
        converter.find_projector(text)


def test_wrong_decoder_width_is_rejected(tmp_path):
    text, vision = tmp_path / "model.gguf", tmp_path / "mmproj.gguf"
    checkpoint(text, image_token=True)
    projector(vision, hidden=512)
    with pytest.raises(ValueError, match="hidden size"):
        converter.find_projector(text, vision)
