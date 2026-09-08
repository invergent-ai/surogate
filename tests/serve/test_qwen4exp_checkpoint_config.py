"""Hyper-connected hybrid objects and GGUF mappings use checkpoint dimensions."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from tests.serve.test_qwen3_5_moe_checkpoint_config import config_for as moe_config
from surogate.serve.convert.qwen4exp import inventory as inv, recipe
from surogate.serve.convert.common.qwen4exp import geometry_block
from surogate.serve.convert.common.recipe import expression_shape


def config_for(*, hidden=128, streams=2, ple=True, vision=False, mtp=False, tied=False):
    c = moe_config(hidden=hidden, nested=vision, vision=vision, tied=tied)
    c.update(architectures=["Qwen4ExpForConditionalGeneration" if vision else "Qwen4ExpForCausalLM"],
             model_type="qwen4_exp", image_token_id=498)
    t = c.get("text_config", c)
    t.update(hc_count=streams, hc_lowrank=32, indexer_n_heads=4, indexer_head_dim=128,
             indexer_budget=64, indexer_compress_ratio=4, indexer_kv_heads=1,
             output_gate_type="sigmoid", ple_layer_ids=[2] if ple else [],
             ngram_size=2, heads_per_ngram=2, ple_embed_dim=64, ple_conv_kernel_size=3,
             eos_token_id=499, mtp_num_hidden_layers=int(mtp), full_attention_interval=2,
             layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"])
    return c


@pytest.mark.parametrize("hidden,streams,ple,mtp,tied", [
    (128, 2, True, False, False), (384, 3, False, True, True),
])
def test_inventory_and_recipe_geometry(hidden, streams, ple, mtp, tied):
    c = config_for(hidden=hidden, streams=streams, ple=ple, mtp=mtp, tied=tied)
    g = inv.geometry_from_config(c, ple_table_rows=128 if ple else 0, token_domain=500)
    inv.validate_inventory(g)
    recipe.validate_recipe_coverage(g)
    tensors, objects = inv.active_specs(geometry=g)
    by_name = {s.name: s for s in tensors}
    assert by_name["text/layers/0/hc_attn/down"].shape == (32, hidden * streams)
    assert by_name["text/layers/1/attention/query_norm"].shape == (g.head_dim,)
    assert by_name["text/layers/0/mlp/shared_down"].shape == (hidden, 128)
    assert ("text/ple/head_offsets" in by_name) == ple
    assert ("mtp/input_projection" in by_name) == mtp
    for r in recipe.build_recipes(g):
        assert expression_shape(r.expression) == by_name[r.object_name].shape
    metadata = geometry_block(g)
    assert metadata["residual"] == hidden * streams
    assert metadata["draft_vocab"] == 0
    assert metadata["token_domain"] == 500
    assert metadata["ple_table_rows"] == (128 if ple else 0)
    renamed = deepcopy(c)
    renamed["_name_or_path"] = "renamed-with-no-size"
    assert inv.active_specs(geometry=inv.geometry_from_config(
        renamed, ple_table_rows=128 if ple else 0, token_domain=500)) == (tensors, objects)


@pytest.mark.parametrize("field", ["hc_count", "hc_lowrank", "indexer_n_heads", "indexer_head_dim",
                                   "indexer_budget", "indexer_compress_ratio", "ple_layer_ids"])
def test_missing_required_metadata_rejects(field):
    c = config_for()
    c.pop(field)
    with pytest.raises(ValueError):
        inv.geometry_from_config(c, ple_table_rows=128)


def test_ple_table_rows_are_not_guessed_from_vocab_base():
    c = config_for()
    c["ngram_vocab_size_base"] = 20000000
    with pytest.raises(ValueError, match="ple_table_rows"):
        inv.geometry_from_config(c)


def test_vision_uses_its_own_config():
    g = inv.geometry_from_config(config_for(vision=True), ple_table_rows=128)
    with_tower, _ = inv.active_specs(geometry=g)
    text, _ = inv.active_specs(geometry=g, vision=False)
    assert len(with_tower) - len(text) == len(inv.build_vision_specs(g)) > 0
    assert all(not s.name.startswith("vision/") for s in text)


def gguf_metadata(g):
    values = {
        "block_count": g.layers, "context_length": g.max_context, "embedding_length": g.hidden,
        "attention.head_count": g.query_heads, "attention.head_count_kv": g.kv_heads,
        "attention.key_length": g.head_dim, "attention.value_length": g.head_dim,
        "rope.dimension_sections": [11, 11, 10, 0], "rope.freq_base": g.rope_theta,
        "rope.dimension_count": 64, "attention.layer_norm_rms_epsilon": g.rms_epsilon,
        "expert_count": g.experts, "expert_used_count": g.experts_per_token,
        "expert_feed_forward_length": g.intermediate,
        "expert_shared_feed_forward_length": g.shared_intermediate,
        "ssm.conv_kernel": g.gdn_conv_kernel, "ssm.state_size": g.gdn_key_head_dim,
        "ssm.group_count": g.gdn_key_heads, "ssm.time_step_rank": g.gdn_value_heads,
        "ssm.inner_size": g.value_dim, "full_attention_interval": 2,
        "hyper_connection.count": g.hc_streams, "hyper_connection.low_rank": g.hc_low_rank,
        "attention.indexer.head_count": g.indexer_heads, "attention.indexer.key_length": g.indexer_head_dim,
        "attention.indexer.top_k": g.indexer_top_k, "attention.compress_ratios": [0, 4, 0, 4],
    }
    if g.ple_ngram:
        values.update({"ple.layers": [g.ple_layer], "ple.ngram_size": g.ple_ngram,
                       "ple.heads_per_ngram": g.ple_heads_per_ngram,
                       "ple.conv_kernel": g.ple_conv_kernel, "ple.eos_token_id": g.ple_eos_token,
                       "ple.image_token_id": g.ple_image_token,
                       "embedding_length_per_layer_input": g.ple_head_dim,
                       "ple.layer_multipliers": [100003, 200003],
                       "ple.head_offsets": [0, 64], "ple.head_vocab_sizes": [61, 61]})
    return {"qwen4exp." + k: v for k, v in values.items()}


def test_cpu_gguf_conversion_preserves_resolved_geometry(tmp_path):
    from gguf import GGUFWriter, GGMLQuantizationType
    from gguf.quants import quantize
    from surogate.serve.convert.qwen4exp.convert import convert
    from surogate.serve.artifact.container import Artifact
    g = inv.geometry_from_config(config_for(), ple_table_rows=128)
    src = tmp_path / "renamed.gguf"
    writer = GGUFWriter(str(src), "qwen4exp")
    for name, value in gguf_metadata(g).items():
        if isinstance(value, list): writer.add_array(name, value)
        elif isinstance(value, float): writer.add_float32(name, value)
        else: writer.add_uint32(name, value)
    for name, required in recipe.source_requirements(g).items():
        shape = required.shape
        if name == recipe.PLE_TABLE_SOURCE:
            writer.add_tensor(name, np.zeros((128, 18), dtype=np.uint8), raw_dtype=GGMLQuantizationType.IQ4_NL)
        else:
            value = np.ones(shape, dtype=np.float32) * .01
            if len(shape) >= 2 and shape[-1] % 32 == 0:
                writer.add_tensor(name, quantize(value, GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0)
            else:
                writer.add_tensor(name, value)
    for i in g.gdn_layers:
        writer.add_tensor(f"blk.{i}.ssm_a", -np.ones((g.gdn_value_heads,), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ssm_dt.bias", np.zeros((g.gdn_value_heads,), dtype=np.float32))
    for i in g.full_attention_layers:
        for suffix in ("q", "k"):
            writer.add_tensor(f"blk.{i}.attn_{suffix}_norm.weight", np.ones((g.head_dim,), dtype=np.float32))
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    front = tmp_path / "frontend"
    front.mkdir()
    resources = {"config.json": config_for(), "tokenizer.json": {
        "model": {"type": "BPE", "vocab": {str(i): i for i in range(500)}}}, "tokenizer_config.json": {}}
    for name, value in resources.items():
        (front / name).write_text(json.dumps(value))
    (front / "chat_template.jinja").write_text("{{ messages }}")
    out = convert(src, front, tmp_path / "model.sinfer", device="cpu")
    with Artifact(out) as artifact:
        assert artifact.identity.architecture == "qwen4exp"
        assert artifact.geometry["hidden"] == g.hidden
        assert artifact.geometry["residual"] == g.residual
        assert artifact.geometry["gdn_value_head_dim"] == g.gdn_value_head_dim
        assert artifact.geometry["token_domain"] == 500
        assert artifact.geometry["ple_table_rows"] == 128
        assert tuple(artifact.layer_types) == g.layer_types
