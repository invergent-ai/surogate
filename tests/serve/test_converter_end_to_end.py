"""Exercise converter entry points with complete, small checkpoint directories."""

import importlib
import json

import pytest
import torch
from safetensors.torch import save_file

from surogate.serve.artifact.container import Artifact
from surogate.serve.convert.common.recipe import source_requirements
from tests.serve.test_checkpoint_inventory import config_for


@pytest.mark.parametrize("family", ["qwen3", "qwen3_moe"])
def test_complete_checkpoint_conversion(tmp_path, family):
    converter = importlib.import_module(f"surogate.serve.convert.{family}.convert")
    recipe = importlib.import_module(f"surogate.serve.convert.{family}.recipe")
    config = {
        **config_for("qwen3", layers=2, hidden=128, head_dim=32),
        "vocab_size": 256, "tie_word_embeddings": False, "hidden_act": "silu",
        "attention_bias": False, "rope_scaling": None, "sliding_window": None,
        "use_sliding_window": False,
    }
    if family == "qwen3_moe":
        config.update(architectures=["Qwen3MoeForCausalLM"], model_type="qwen3_moe",
                      moe_intermediate_size=128, num_experts=4, num_experts_per_tok=2)
    geometry = recipe.geometry_from_config(config)
    recipes = recipe.build_recipes(geometry)
    tensors = {
        name: torch.full(source.shape, 0.125, dtype=torch.bfloat16)
        for name, source in source_requirements(recipes).items()
    }
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "model": {"vocab": {"a": 0, "b": 1}}, "added_tokens": [],
    }))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    output = tmp_path / "converted.sinfer"
    converter.convert(tmp_path, output, device="cpu")
    with Artifact(output) as artifact:
        assert artifact.identity.architecture == family
        assert artifact.geometry["hidden"] == 128
        assert artifact.geometry["layers"] == 2
        assert artifact.geometry["token_domain"] == 2
        assert artifact.geometry["output_rows"] == 256
        assert artifact.find("text/token_embedding").shape == (256, 128)
        if family == "qwen3_moe":
            assert artifact.geometry["experts"] == 4


def test_encoder_uses_the_serialized_sentencepiece_vocabulary():
    from sentencepiece import sentencepiece_model_pb2
    from surogate.serve.convert.gemma_embedding.convert import sentencepiece_domain

    model = sentencepiece_model_pb2.ModelProto()
    for token, kind in [("<unk>", 2), ("a", 1), ("b", 1)]:
        piece = model.pieces.add()
        piece.piece, piece.type, piece.score = token, kind, 0.0
    # tokenizer.json can carry additional HF-only tokens; the encoder stores tokenizer.model.
    resources = {"frontend/tokenizer.model": model.SerializeToString(),
                 "frontend/tokenizer.json": b'{"added_tokens":[{"id":999}]}' }
    assert sentencepiece_domain(resources) == 3
