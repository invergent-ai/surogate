"""Gemma 4 mixture with a DFlash drafter: config spellings and the converted artifact."""

import json

import pytest
import torch
from safetensors.torch import save_file

from surogate.serve.convert.common import dflash
from surogate.serve.convert.common.inventory import RESOURCE_SPECS
from surogate.serve.convert.common.recipe import source_requirements
from surogate.serve.convert.gemma_vl import convert, inventory
from tests.serve.test_gemma_vl import config_for


def drafter_config(text, **extra):
    """A z-lab-style Gemma 4 drafter config: `rope_theta` and `block_size` at the top level."""
    config = {
        "architectures": ["DFlashDraftModel"], "model_type": "qwen3",
        "hidden_size": text["hidden_size"], "num_hidden_layers": 2, "intermediate_size": 512,
        "head_dim": 64, "num_attention_heads": 4, "num_key_value_heads": 2, "sliding_window": 128,
        "max_position_embeddings": 4096, "vocab_size": text["vocab_size"],
        "num_target_layers": text["num_hidden_layers"], "rms_norm_eps": 1e-6, "hidden_act": "silu",
        "attention_bias": False, "layer_types": ["sliding_attention", "full_attention"],
        "rope_theta": 1000000, "rope_scaling": None, "block_size": 16,
        "final_logit_softcapping": 30.0,
        "dflash_config": {"mask_token_id": 4, "target_layer_ids": [0, 2]},
    }
    config.update(extra)
    return config


def test_top_level_rope_and_block_size_are_read():
    target = inventory.geometry_from_config(config_for("gemma4_moe"))
    g = dflash.geometry_from_config(drafter_config(target.config["text_config"]), target.text)
    assert g.rope_theta == 1000000 and g.block_size == 16 and g.local_layers == 1
    assert g.feature_rows == 2 * target.text.hidden
    with pytest.raises(ValueError, match="default RoPE"):
        dflash.geometry_from_config(
            drafter_config(target.config["text_config"], rope_scaling={"type": "yarn"}), target.text)
    with pytest.raises(ValueError, match="input embedding"):
        dflash.geometry_from_config(
            drafter_config(target.config["text_config"], input_embedding_scale=2.0), target.text)


def test_gemma_vl_writes_the_drafter_beside_the_target(tmp_path, monkeypatch):
    from surogate.serve.artifact.container import Artifact

    config = config_for("gemma4_moe")
    g = inventory.geometry_from_config(config)
    _, text_recipes = inventory.text_specs_and_recipes(g)
    _, vision_recipes = inventory.vision_recipes(g)
    torch.manual_seed(3)
    tensors = {}
    for recipe in (*text_recipes, *vision_recipes):
        for name, source in source_requirements((recipe,)).items():
            dtype = torch.float32 if "per_expert_scale" in name else torch.bfloat16
            tensors[name] = (torch.randn(source.shape) * 0.02).to(dtype)
    model = tmp_path / "model"
    model.mkdir()
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(model / "model.safetensors"))
    (model / "config.json").write_text(json.dumps(config))

    draft_config = drafter_config(config["text_config"])
    dg = dflash.geometry_from_config(draft_config, g.text)
    _, draft_recipes = dflash.conversion_plan(dg, g.text)
    draft = {}
    for name, source in source_requirements(draft_recipes).items():
        draft[name] = (torch.randn(source.shape) * 0.02).to(torch.bfloat16)
    drafter = tmp_path / "drafter"
    drafter.mkdir()
    save_file({k: v.contiguous() for k, v in draft.items()}, str(drafter / "model.safetensors"))
    (drafter / "config.json").write_text(json.dumps(draft_config))

    monkeypatch.setattr(convert, "resources_for", lambda m, geometry: {s.name: b"{}" for s in RESOURCE_SPECS})
    monkeypatch.setattr(convert, "tokenizer_domain", lambda m: 512)
    out = tmp_path / "out.sinfer"
    convert.convert(model, out, device="cpu", dflash_model=drafter)
    with Artifact(out) as artifact:
        assert artifact.dflash_target_layers == [0, 2]
        assert artifact.dflash_geometry["block_size"] == 16
        assert artifact.dflash_geometry["rope_theta"] == 1000000
        fc = artifact.find("dflash/feature_projection")
        assert tuple(fc.shape) == (g.text.hidden, 2 * g.text.hidden)
        assert artifact.find("dflash/layers/1/mlp/gate_up").shape[0] == 1024
        assert artifact.find("text/layers/0/attention/query") is not None
    report = json.loads((tmp_path / "out.sinfer.conversion.json").read_text())
    assert report["dflash"]["target_layers"] == [0, 2]
