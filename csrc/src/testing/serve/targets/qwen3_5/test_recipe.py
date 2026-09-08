import torch
from tests.serve.test_qwen3_5_checkpoint_config import config_for
from surogate.serve.convert.qwen3_5 import inventory, recipe

GEOMETRY = inventory.geometry_from_config(config_for(), token_domain=500)
RECIPES = {r.object_name: r for r in recipe.build_recipes(GEOMETRY)}


class TensorReader:
    def __init__(self, tensors):
        self.tensors = tensors
    def get(self, name):
        return self.tensors[name]
    def has(self, name):
        return name in self.tensors


def test_recipe_exactly_covers_checkpoint_inventory():
    tensors = inventory.build_tensor_specs(GEOMETRY)
    assert tuple(RECIPES) == tuple(s.name for s in tensors)
    assert all(recipe.expression_shape(RECIPES[s.name].expression) == s.shape for s in tensors)
    requirements = recipe.source_requirements(tuple(RECIPES.values()))
    assert requirements["model.embed_tokens.weight"].shape == (512, 128)
    assert requirements["mtp.layers.0.self_attn.q_proj.weight"].shape == (512, 128)


def test_attention_materializes_query_key_gate_value_order():
    prefix = "mtp.layers.0.self_attn."
    q = torch.arange(512).view(-1, 1).expand(-1, 128)
    k = torch.full((128, 128), -1)
    v = torch.full((128, 128), -2)
    fused = recipe.materialize_recipe(RECIPES["mtp/layer/attention/query_key_gate_value"],
        TensorReader({prefix + "q_proj.weight": q, prefix + "k_proj.weight": k, prefix + "v_proj.weight": v}))
    expected = torch.cat((q[:128], q[256:384], k, q[128:256], q[384:], v))
    assert torch.equal(fused, expected)


def test_gdn_materializes_projection_and_convolution_order():
    prefix = "model.layers.1.linear_attn."
    qkv = torch.arange(384).view(-1, 1).expand(-1, 128)
    z = torch.full((256, 128), -1)
    fused = recipe.materialize_recipe(RECIPES["text/layers/1/gdn/query_key_value_z"],
        TensorReader({prefix + "in_proj_qkv.weight": qkv, prefix + "in_proj_z.weight": z}))
    assert torch.equal(fused, torch.cat((qkv, z)))
    conv = torch.arange(384 * 4).reshape(384, 1, 4).to(torch.bfloat16)
    result = recipe.materialize_recipe(RECIPES["text/layers/1/gdn/convolution"],
        TensorReader({prefix + "conv1d.weight": conv}))
    assert torch.equal(result, conv[:, 0].T.contiguous())
    values = torch.arange(4).to(torch.bfloat16)
    result = recipe.materialize_recipe(RECIPES["text/layers/1/gdn/a_log"], TensorReader({prefix + "A_log": values}))
    assert torch.equal(result, values.float())
