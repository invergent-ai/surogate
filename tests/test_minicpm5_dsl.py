"""MiniCPM5 uses the published Llama architecture and explicit attention width."""

import json

import pytest

from surogate.dsl.py_compiler import compile_model_for_hf


@pytest.mark.parametrize("hidden,layers,ffn", [(1536, 24, 4608), (2048, 42, 6144)])
def test_minicpm5_compiles_through_llama_with_checkpoint_dimensions(hidden, layers, ffn):
    cfg = dict(
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        hidden_act="silu",
        hidden_size=hidden,
        num_hidden_layers=layers,
        intermediate_size=ffn,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=130560,
        max_position_embeddings=131072,
        rms_norm_eps=1e-6,
        rope_theta=5000000,
        tie_word_embeddings=False,
    )
    result = json.loads(compile_model_for_hf("LlamaForCausalLM", cfg))
    assert result["success"], result.get("errors")
    model = result["modules"][0]
    assert model["config"]["head_size"] == 128
    assert model["config"]["d_model"] == hidden
    assert model["config"]["n_layers"] == layers
    assert model["hf_mapping"]["embedding"] == "model.embed_tokens.weight"
    assert model["hf_mapping"]["lm_head"] == "lm_head.weight"
    ops = model["forward"]["operations"]
    assert len([op for op in ops if op["kernel_type"] == "flash_attention"]) == layers
