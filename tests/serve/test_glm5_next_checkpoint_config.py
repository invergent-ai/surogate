"""GGUF settings determine GLM's declared objects and serving execution metadata."""

import pytest

from surogate.serve.convert.glm5_next import inventory, recipe
from surogate.serve.convert.glm5_next.convert import _geometry_block


def metadata(*, hidden=256, qk_dim=64, experts=8, nextn=1):
    blocks = 4 + nextn
    values = {
        "block_count": blocks, "nextn_predict_layers": nextn,
        "context_length": 4096, "attention.head_count_kv": [0, 1, 0, 1] + [1] * nextn,
        "leading_dense_block_count": 2, "rope.dimension_count": 0,
        "embedding_length": hidden, "attention.head_count": 4,
        "feed_forward_length": 2 * hidden, "expert_feed_forward_length": 128,
        "expert_shared_feed_forward_length": 128, "expert_shared_count": 1,
        "vocab_size": 512, "expert_count": experts, "expert_used_count": 2,
        "expert_weights_scale": 1.5, "swiglu_clamp_exp": [7.0] * blocks,
        "swiglu_clamp_shexp": [7.0] * blocks, "hyper_connection.count": 2,
        "hyper_connection.sinkhorn_iterations": 12, "hyper_connection.epsilon": 2e-5,
        "kda.head_dim": 32, "ssm.conv_kernel": 5, "kda.gate_lower_bound": -3.0,
        "attention.q_lora_rank": 64, "attention.kv_lora_rank": 64,
        "attention.key_length_mla": qk_dim, "attention.value_length_mla": qk_dim,
        "attention.layer_norm_rms_epsilon": 1e-4,
        "attention.indexer.top_k": 128, "attention.indexer.kpool": 8,
    }
    return {"glm5next." + key: value for key, value in values.items()}


@pytest.mark.parametrize("hidden,qk_dim,experts,nextn", [(256, 64, 8, 1), (384, 128, 16, 0)])
def test_shapes_and_runtime_settings_follow_gguf(hidden, qk_dim, experts, nextn):
    values = metadata(hidden=hidden, qk_dim=qk_dim, experts=experts, nextn=nextn)
    geometry = inventory.geometry_from_gguf(values.get)
    specs = inventory.build_tensor_specs(geometry)
    recipes = recipe.build_recipes(geometry)
    materialized = set(recipe.materialized_objects(geometry))
    recipe_specs = tuple(spec for spec in specs if spec.name not in materialized)
    ordered = tuple(recipes[spec.name] for spec in recipe_specs)
    recipe.validate_recipe_coverage(ordered, recipe_specs)
    assert set(recipes) | materialized == {spec.name for spec in specs}
    by_name = {spec.name: spec for spec in specs}
    assert by_name["text/token_embedding"].shape == (512, hidden)
    assert by_name["text/layers/1/mla/query_b"].shape == (4 * qk_dim, 64)
    assert by_name["text/layers/3/moe/routed_gate_up"].shape == (experts * 256, hidden)
    assert ("mtp/input_projection" in by_name) == bool(nextn)
    runtime = _geometry_block(geometry, token_domain=geometry.vocab - 1)
    assert runtime["qk_head_dim"] == qk_dim and runtime["v_head_dim"] == qk_dim
    assert runtime["experts"] == experts and runtime["experts_per_token"] == 2
    assert runtime["max_context"] == 135
    assert runtime["hc_streams"] == 2 and runtime["residual"] == 2 * hidden
    assert runtime["hc_sinkhorn_iterations"] == 12 and runtime["hc_epsilon"] == 2e-5
    assert runtime["routed_scale"] == 1.5 and runtime["swiglu_limit"] == 7.0
    assert runtime["kda_gate_bound"] == 3.0 and runtime["gdn_conv_kernel"] == 5
    assert geometry.layer_types == ("linear_attention", "full_attention") * 2
    assert geometry.declared.hf_config["text_config"]["max_position_embeddings"] == 4096
    renamed = inventory.geometry_from_gguf({**values, "general.name": "renamed-checkpoint"}.get)
    assert inventory.build_tensor_specs(renamed) == specs


@pytest.mark.parametrize("key", ["context_length", "attention.indexer.kpool", "attention.key_length_mla",
                                 "hyper_connection.sinkhorn_iterations", "expert_count"])
def test_missing_gguf_metadata_does_not_select_defaults(key):
    values = metadata()
    del values["glm5next." + key]
    with pytest.raises(ValueError, match=key):
        inventory.geometry_from_gguf(values.get)


def test_nonuniform_clamps_fail_explicitly():
    values = metadata()
    values["glm5next.swiglu_clamp_exp"][2] = 8.0
    with pytest.raises(ValueError, match="one SwiGLU clamp"):
        inventory.geometry_from_gguf(values.get)
