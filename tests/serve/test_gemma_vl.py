"""Gemma vision configuration, source layouts and GGUF artifact composition."""

from types import SimpleNamespace

import pytest
import torch

from surogate.serve import ingest
from surogate.serve.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, ResourceSpec, TensorSpec
from surogate.serve.convert.common.recipe import materialize_recipe, validate_recipe_coverage
from surogate.serve.convert.gemma_vl import gguf, inventory
from tests.serve.test_gemma3_convert import config as gemma3_text
from tests.serve.test_gemma4_checkpoint_config import checkpoint as gemma4_text


def config_for(target):
    if target == "gemma3":
        tc = gemma3_text(
            hidden_size=256,
            intermediate_size=512,
            head_dim=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            vocab_size=512,
        )
    else:
        tc = gemma4_text(target if target != "unified" else "gemma4")
    vc = dict(
        model_type="siglip_vision_model" if target == "gemma3" else "gemma4_vision",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=1,
        head_dim=64,
        hidden_act="gelu_pytorch_tanh",
        patch_size=14 if target == "gemma3" else 16,
        image_size=56,
        layer_norm_eps=1e-6,
        pooling_kernel_size=3,
        position_embedding_size=32,
    )
    if target == "unified":
        vc = dict(
            model_type="gemma4_unified_vision",
            mm_embed_dim=64,
            mm_posemb_size=32,
            model_patch_size=48,
            output_proj_dims=64,
        )
    return dict(
        model_type="gemma3" if target == "gemma3" else "gemma4_unified" if target == "unified" else "gemma4",
        architectures=["Gemma3ForConditionalGeneration" if target == "gemma3" else "Gemma4ForConditionalGeneration"],
        text_config=tc,
        vision_config=vc,
        mm_tokens_per_image=4,
        image_token_id=500,
    )


@pytest.mark.parametrize("target", ["gemma3", "gemma4", "gemma4_e", "gemma4_moe", "unified"])
def test_checkpoint_dispatch_and_vision_inventory(target):
    config = config_for(target)
    g = inventory.geometry_from_config(config)
    assert g.target == ("gemma4" if target == "unified" else target)
    assert (
        ingest.converter_for_config(ingest._flatten_text_config(config)).module
        == "surogate.serve.convert.gemma_vl.convert"
    )
    specs, recipes = inventory.vision_recipes(g)
    validate_recipe_coverage(recipes, specs)
    gguf_specs, gguf_recipes = gguf.build_recipes(g)
    validate_recipe_coverage(gguf_recipes, gguf_specs)
    assert specs == gguf_specs
    metadata = inventory.geometry_block(g, token_domain=512)
    assert metadata["sliding_window"] > 0
    assert g.vision["merge"] == (2 if target == "gemma3" else 1 if target == "unified" else 3)
    assert g.vision["encoder_free"] == int(target == "unified")


@pytest.mark.parametrize("target", ["gemma4", "unified"])
def test_gguf_patch_permutation_matches_hwc(target):
    g = inventory.geometry_from_config(config_for(target))
    _, recipes = gguf.build_recipes(g)
    patch = 48 if target == "unified" else 16
    h = g.vision["hidden"]
    values = torch.arange(h * 3 * patch * patch, dtype=torch.float32).reshape(h, 3, patch, patch)
    stored = values.reshape(h, -1) if target == "unified" else values
    recipe = next(r for r in recipes if r.object_name == "vision/patch_embedding")
    actual = materialize_recipe(recipe, SimpleNamespace(get=lambda name: stored))
    assert torch.equal(actual, values.permute(0, 2, 3, 1).reshape(h, -1))
    if target == "unified":
        recipe = next(r for r in recipes if r.object_name == "vision/patch_norm1/weight")
        values = torch.arange(3 * patch * patch, dtype=torch.float32)
        actual = materialize_recipe(recipe, SimpleNamespace(get=lambda name: values))
        assert torch.equal(actual, values.reshape(3, patch, patch).permute(1, 2, 0).reshape(-1))


def test_image_attention_and_invalid_encoder_configuration():
    c = config_for("gemma4")
    c["text_config"]["use_bidirectional_attention"] = "vision"
    assert inventory.geometry_from_config(c).vision["attention_mode"] == 2
    c["text_config"]["use_bidirectional_attention"] = True
    with pytest.raises(ValueError, match="causal text"):
        inventory.geometry_from_config(c)
    c = config_for("gemma3")
    assert inventory.geometry_from_config(c).vision["attention_mode"] == 1
    c["vision_config"]["hidden_act"] = "relu"
    with pytest.raises(ValueError, match="tanh GELU"):
        inventory.geometry_from_config(c)


def test_combine_preserves_external_runs_and_replaces_frontend(tmp_path):
    identity = ArtifactIdentity("gemma4", "groupwise-int", architecture="gemma4")
    external = tmp_path / "source.gguf"
    external.write_bytes(b"x" * (64 * 34))
    paths = [tmp_path / "text.sinfer", tmp_path / "vision.sinfer"]
    for index, path in enumerate(paths):
        name = "text/token_embedding" if index == 0 else "vision/projection"
        specs = [
            TensorSpec(name, (64, 32), "Q8_0", "ggml-blocks-v1", runs=((1, 0, 64 * 34),)),
            ResourceSpec("frontend/tokenizer.json", "raw-bytes-v1", 3),
        ]
        with ArtifactWriter(path, identity, specs, external=((str(external), 64 * 34),)) as writer:
            writer.write("frontend/tokenizer.json", b"old" if index == 0 else b"new")
    output = tmp_path / "combined.sinfer"
    gguf.combine(*paths, output)
    with Artifact(output) as artifact:
        assert artifact.find("text/token_embedding").runs == ((1, 0, 64 * 34),)
        assert artifact.find("vision/projection").runs == ((2, 0, 64 * 34),)
        assert bytes(artifact.payload("frontend/tokenizer.json")) == b"new"
