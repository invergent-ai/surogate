"""Spark's inventory is derived from checkpoint metadata, including non-published shapes."""

import pytest

from surogate.serve.convert.spark2_5 import inventory, recipe
from surogate.serve.convert.spark2_5.convert import geometry_block
from surogate.serve.convert.common.recipe import expression_sources, validate_recipe_coverage
from surogate.serve.ingest import converter_for_config


@pytest.fixture
def config():
    return {
        "architectures": ["Spark2_5ForCausalLM"], "model_type": "spark2_5",
        "hidden_act": "gelu", "headwise_attn_output_gate": True, "gate_attn_act_mode": "sigmoid",
        "attention_bias": False, "mlp_bias": False, "tie_word_embeddings": True,
        "hidden_size": 768, "intermediate_size": 1792, "vocab_size": 4096,
        "num_hidden_layers": 3, "num_attention_heads": 8, "num_key_value_heads": 2,
        "head_dim": 128, "max_position_embeddings": 16384, "rms_norm_eps": 1e-5,
        "sliding_window": 192, "layer_types": ["full_attention", "sliding_attention", "full_attention"],
        "rope_parameters": {
            "full_attention": {"rope_theta": 123456, "partial_rotary_factor": .25},
            "sliding_attention": {"rope_theta": 23456, "partial_rotary_factor": .5},
        },
    }


def test_dimensions_schedule_rotations_and_sources_come_from_config(config):
    g = recipe.geometry_from_config(config)
    assert g.hidden == 768 and g.query_size == 1024 and g.intermediate == 1792
    assert g.layer_types == tuple(config["layer_types"])
    meta = geometry_block(g, token_domain=4000)
    assert meta["rotary_dim"] == 32 and meta["sliding_rotary_dim"] == 64
    assert meta["rope_theta"] == 123456 and meta["sliding_rope_theta"] == 23456
    assert meta["sliding_window"] == 192 and meta["max_context"] == 16384
    assert meta["residual_fp32"] == 1 and meta["residual"] == 768
    recipes = recipe.build_recipes(g)
    specs = inventory.build_tensor_specs(g)
    validate_recipe_coverage(recipes, specs)
    shapes = {s.name: s.shape for s in specs}
    assert shapes["text/layers/0/attention/output_gate"] == (8, 768)
    assert shapes["text/layers/0/attention/output"] == (768, 1024)
    by_name = {r.object_name: r.expression for r in recipes}
    assert by_name["text/output_head"].name == "model.embedding.weight"
    assert by_name["text/layers/0/attention/query_key_value"].name == "model.layers.0.self_attn.q_k_v_proj.weight"
    parts = expression_sources(by_name["text/layers/0/mlp/gate_up"])
    assert [p.name for p in parts] == ["model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight"]
    assert converter_for_config(config).module == "surogate.serve.convert.spark2_5.convert"


@pytest.mark.parametrize('hidden,heads,kv,ffn,layers',[(2048,8,2,6656,28),(2560,16,4,10240,36)])
def test_both_published_geometries(config,hidden,heads,kv,ffn,layers):
    config.update(hidden_size=hidden,num_attention_heads=heads,num_key_value_heads=kv,
                  intermediate_size=ffn,num_hidden_layers=layers,head_dim=256,
                  layer_types=["sliding_attention"] * layers)
    g=recipe.geometry_from_config(config)
    assert (g.hidden,g.query_size,g.kv_size,g.layers)==(hidden,heads*256,kv*256,layers)
    validate_recipe_coverage(recipe.build_recipes(g),inventory.build_tensor_specs(g))


@pytest.mark.parametrize('key', ['hidden_size','num_hidden_layers','intermediate_size','vocab_size',
                                 'num_attention_heads','num_key_value_heads','head_dim',
                                 'max_position_embeddings','sliding_window','layer_types',
                                 'rms_norm_eps','rope_parameters','tie_word_embeddings'])
def test_missing_metadata_never_selects_a_size(config,key):
    config.pop(key)
    with pytest.raises(ValueError):
        recipe.geometry_from_config(config)


@pytest.mark.parametrize('key,value', [
    ('hidden_size',True),('num_key_value_heads',3),('layer_types',['full_attention']),
    ('hidden_act','silu'),('gate_attn_act_mode','silu'),('attention_bias',True),
    ('headwise_attn_output_gate',False),('rms_norm_eps',float('nan')),
    ('rope_scaling',{'rope_type':'yarn'}),
])
def test_unsupported_or_inconsistent_metadata_is_rejected(config,key,value):
    config[key]=value
    with pytest.raises(ValueError): recipe.geometry_from_config(config)


@pytest.mark.parametrize('factor',[0,1.5,.1,True,float('inf')])
def test_rotary_prefix_must_be_valid(config,factor):
    config['rope_parameters']['full_attention']['partial_rotary_factor']=factor
    with pytest.raises(ValueError): recipe.geometry_from_config(config)


def test_untied_head_is_read_from_its_own_tensor(config):
    config['tie_word_embeddings']=False
    expressions={r.object_name:r.expression for r in recipe.build_recipes(recipe.geometry_from_config(config))}
    assert expressions['text/output_head'].name=='lm_head.weight'


def test_standalone_template_has_hf_precedence(tmp_path):
    import json
    from surogate.serve.convert.spark2_5.convert import load_resources
    (tmp_path / "tokenizer.json").write_text('{}')
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "old"}))
    (tmp_path / "chat_template.jinja").write_text('standalone')
    (tmp_path / "generation_config.json").write_text('{}')
    resources={r.name:r.data for r in load_resources(tmp_path)}
    assert resources['frontend/chat_template.jinja']==b'standalone'
    assert json.loads(resources['frontend/tokenizer_config.json'])['chat_template']=='standalone'
    assert json.loads((tmp_path / "tokenizer_config.json").read_text())['chat_template']=='old'
