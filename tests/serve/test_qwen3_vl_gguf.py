"""Native VL GGUF conversion, projector selection, and layout regression checks."""

from pathlib import Path

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import dequantize, quantize

from surogate.serve import ingest
from surogate.serve.artifact.container import Artifact
from surogate.serve.convert.common.recipe import expression_sources
from surogate.serve.convert.qwen3_vl import inventory
from surogate.serve.convert.qwen3_vl.gguf import build_recipes, convert, find_projector
from surogate.serve.gguf.bridge import gguf_target_key
from tests.serve.test_qwen3_vl import config_for, moe_config_for


def checkpoint_pair(root, *, moe=False, vision_overrides=None, text_overrides=None, quantized_vision=False):
    root.mkdir(parents=True, exist_ok=True)
    config = moe_config_for() if moe else config_for()
    g = inventory.geometry_from_config(config)
    arch = "qwen3vlmoe" if moe else "qwen3vl"
    text_path, vision_path = root / "text.gguf", root / "mmproj.gguf"
    text, vision = GGUFWriter(str(text_path), arch), GGUFWriter(str(vision_path), "clip")
    text.add_block_count(g.layers)
    text.add_embedding_length(g.hidden)
    text.add_feed_forward_length(config["text_config"]["intermediate_size"])
    text.add_head_count(g.query_heads)
    text.add_head_count_kv(g.kv_heads)
    text.add_key_length(g.head_dim)
    text.add_context_length(4096)
    text.add_rope_freq_base(5000000.0)
    text.add_layer_norm_rms_eps(1e-6)
    text.add_array(f"{arch}.rope.dimension_sections", [12, 10, 10, 0])
    text.add_uint32(f"{arch}.n_deepstack_layers", 2)
    if moe:
        text.add_expert_count(g.experts)
        text.add_expert_used_count(g.experts_per_token)
        text.add_expert_feed_forward_length(g.intermediate)
    for key, value in (text_overrides or {}).items():
        if isinstance(value, bool):
            text.add_bool(f"{arch}.{key}", value)
        elif isinstance(value, str):
            text.add_string(f"{arch}.{key}", value)
        elif isinstance(value, float):
            text.add_float32(f"{arch}.{key}", value)
        else:
            text.add_uint32(f"{arch}.{key}", value)
    tokens = ["<bos>", "<eos>", "a", "b", "ab"] + [f"t{i}" for i in range(5, g.vocab)]
    text.add_tokenizer_model("gpt2")
    text.add_tokenizer_pre("qwen2")
    text.add_token_list(tokens)
    text.add_token_types([3, 3, 1, 1, 1] + [1] * (g.vocab - 5))
    text.add_token_merges(["a b"])
    text.add_bos_token_id(0)
    text.add_eos_token_id(1)
    text.add_chat_template("{% for m in messages %}{{ m.content }}{% endfor %}")
    fields = {
        "clip.projector_type": "qwen3vl_merger",
        "clip.vision.block_count": 4,
        "clip.vision.embedding_length": 128,
        "clip.vision.feed_forward_length": 256,
        "clip.vision.attention.head_count": 2,
        "clip.vision.image_size": 64,
        "clip.vision.patch_size": 16,
        "clip.vision.spatial_merge_size": 2,
        "clip.vision.projection_dim": g.hidden,
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.vision.is_deepstack_layers": [True, False, True, False],
        "clip.vision.image_mean": [0.5] * 3,
        "clip.vision.image_std": [0.5] * 3,
    }
    fields.update(vision_overrides or {})
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, list):
            vision.add_array(key, value)
        elif isinstance(value, str):
            vision.add_string(key, value)
        elif isinstance(value, float):
            vision.add_float32(key, value)
        else:
            vision.add_uint32(key, value)
    rng = np.random.default_rng(81)
    sources = {s.name: s for r in build_recipes(g, True) for s in expression_sources(r.expression)}
    stored = {}
    for name, source in sources.items():
        visual = name.startswith(("v.", "mm."))
        data = rng.normal(0, 0.05, source.shape).astype(np.float32)
        if name == "v.patch_embd.weight":
            data.fill(0.125)
        if name == "v.patch_embd.weight.1":
            data.fill(0.25)
        kind = (
            GGMLQuantizationType.Q8_0
            if len(source.shape) > 1 and ((not visual and "gate_inp" not in name) or
                (quantized_vision and visual and len(source.shape) == 2 and name != "v.position_embd.weight"))
            else GGMLQuantizationType.F32
        )
        encoded = quantize(data, kind)
        (vision if visual else text).add_tensor(name, encoded, raw_dtype=kind)
        stored[name] = (encoded, kind, data)
    for writer in (text, vision):
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    return text_path, vision_path, stored


@pytest.mark.parametrize("moe", [False, True])
def test_native_text_and_split_patch_layout(tmp_path, moe):
    text, vision, stored = checkpoint_pair(tmp_path, moe=moe)
    assert gguf_target_key(text) == "qwen3_vl"
    assert find_projector(text) == vision
    out = tmp_path / "converted.sinfer"
    convert(text, vision, out)
    with Artifact(out) as a:
        assert a.identity.architecture == ("qwen3_vl_moe" if moe else "qwen3_vl")
        assert a.vision_geometry["deepstack_layers"] == 2
        assert a.geometry["mrope_height"] == 10
        patch = torch.frombuffer(bytearray(a.payload("vision/patch_embedding")), dtype=torch.bfloat16)
        patch = patch.reshape(128, 3, 2, 16, 16)
        assert torch.all(patch[:, :, 0] == 0.125)
        assert torch.all(patch[:, :, 1] == 0.25)
        obj = a.find("text/layers/0/moe/routed_gate_up" if moe else "text/layers/0/attention/query_key_value")
        assert obj.runs and obj.format == "Q8_0"
        raw = bytearray()
        for source, offset, size in obj.runs:
            with open(a.external[source - 1][0], "rb") as stream:
                stream.seek(offset)
                raw.extend(stream.read(size))
        actual = dequantize(np.frombuffer(raw, dtype=np.uint8).reshape(obj.shape[0], -1), GGMLQuantizationType.Q8_0)
        if moe:
            expected = np.concatenate(
                [
                    dequantize(stored["blk.0.ffn_" + half + "_exps.weight"][0], GGMLQuantizationType.Q8_0)
                    for half in ("gate", "up")
                ],
                axis=1,
            ).reshape(obj.shape)
        else:
            expected = np.concatenate(
                [
                    dequantize(stored["blk.0.attn_" + axis + ".weight"][0], GGMLQuantizationType.Q8_0)
                    for axis in ("q", "k", "v")
                ],
                axis=0,
            )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"clip.projector_type": "qwen2vl_merger"}, "Qwen3-VL vision projector"),
        ({"clip.vision.projection_dim": 512}, "decoder hidden size"),
        ({"clip.vision.is_deepstack_layers": [True, False, False, False]}, "deepstack feature count"),
        ({"clip.vision.is_deepstack_layers": None}, "deepstack layer schedule"),
    ],
)
def test_incompatible_projector_is_rejected(tmp_path, override, match):
    text, vision, _ = checkpoint_pair(tmp_path, vision_overrides=override)
    with pytest.raises(ValueError, match=match):
        find_projector(text, vision)
    with pytest.raises(ValueError):
        convert(text, vision, tmp_path / "invalid.sinfer")
    assert not (tmp_path / "invalid.sinfer").exists()


def test_projector_selection_and_cache_identity(tmp_path, monkeypatch):
    text, vision, _ = checkpoint_pair(tmp_path)
    second = tmp_path / "mmproj-second.gguf"
    second.write_bytes(vision.read_bytes())
    with pytest.raises(ValueError, match="2 compatible projectors"):
        find_projector(text)
    monkeypatch.setattr(ingest, "cache_dir", lambda: tmp_path / "cache")
    first = ingest.ensure_engine_weights(str(text), mmproj=str(vision), echo=lambda _: None)
    assert ingest.ensure_engine_weights(str(text), mmproj=str(vision), echo=lambda _: None) == first
    second_artifact = ingest.ensure_engine_weights(str(text), mmproj=str(second), echo=lambda _: None)
    assert second_artifact != first
    vision.unlink()
    second.unlink()
    with pytest.raises(ValueError, match="0 compatible projectors"):
        find_projector(text)


@pytest.mark.parametrize(
    "metadata",
    [
        {"rope.scaling.type": "yarn"},
        {"expert_gating_func": 2},
        {"expert_weights_scale": 2.0},
        {"expert_weights_norm": False},
    ],
)
def test_unsupported_text_settings_are_not_silently_ignored(tmp_path, metadata):
    text, vision, _ = checkpoint_pair(tmp_path, moe=True, text_overrides=metadata)
    with pytest.raises(ValueError, match="requires"):
        convert(text, vision, tmp_path / "invalid.sinfer")
    assert not (tmp_path / "invalid.sinfer").exists()


def test_quantized_vision_weights_remain_quantized(tmp_path):
    text, vision, stored = checkpoint_pair(tmp_path, quantized_vision=True)
    output = tmp_path / "native-vision.sinfer"
    convert(text, vision, output)
    with Artifact(output) as artifact:
        for name in ("vision/layers/0/attention/qkv", "vision/layers/0/mlp/fc1", "vision/merger/fc2",
                     "vision/layers/2/deepstack/fc1"):
            obj = artifact.find(name)
            assert obj.format == "Q8_0" and obj.runs, name
        assert artifact.find("vision/position_embedding").format == "BF16"
