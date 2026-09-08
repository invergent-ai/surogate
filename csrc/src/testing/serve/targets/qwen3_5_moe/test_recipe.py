import torch
from safetensors.torch import save_file
from surogate.serve.convert.common.safetensors import ShardReader
from tests.serve.test_qwen3_5_moe_checkpoint_config import config_for, draft_config
from surogate.serve.convert.qwen3_5_moe import inventory, recipe
from surogate.serve.convert.common import dflash

G = inventory.geometry_from_config(config_for(), token_domain=500)
D = dflash.geometry_from_config(draft_config(G), G)
RECIPES = {r.object_name: r for r in recipe.build_recipes(G, dflash=D)}


class TensorReader:
    def __init__(self, tensors):
        self.tensors = tensors
    def get(self, name):
        return self.tensors[name]
    def has(self, name):
        return name in self.tensors


def test_attention_materializes_query_key_gate_value_order():
    p = "model.layers.0.self_attn."
    q = torch.arange(512).view(-1, 1).expand(-1, 128)
    k, v = torch.full((128, 128), -1), torch.full((128, 128), -2)
    got = recipe.materialize_recipe(RECIPES["text/layers/0/attention/query_key_gate_value"],
        TensorReader({p + "q_proj.weight": q, p + "k_proj.weight": k, p + "v_proj.weight": v}))
    assert torch.equal(got, torch.cat((q[:128], q[256:384], k, q[128:256], q[384:], v)))


def test_moe_keeps_expert_major_half_split_rows():
    p = "model.layers.0.mlp.experts."
    gate_up = torch.arange(4 * 128).reshape(4, 128, 1).expand(-1, -1, 128)
    down = torch.arange(4 * 128).reshape(4, 128, 1).expand(-1, -1, 64)
    reader = TensorReader({p + "gate_up_proj": gate_up, p + "down_proj": down})
    got = recipe.materialize_recipe(RECIPES["text/layers/0/moe/routed_gate_up"], reader)
    assert torch.equal(got, gate_up.reshape(512, 128))
    got = recipe.materialize_recipe(RECIPES["text/layers/0/moe/routed_down"], reader)
    assert torch.equal(got, down.reshape(512, 64))


def test_gdn_keeps_control_halves_and_projection_order():
    p = "model.layers.1.linear_attn."
    a, b = torch.ones((4, 128)), torch.full((4, 128), 2.)
    qkv, z = torch.arange(384).view(-1, 1).expand(-1, 128), torch.full((256, 128), -1)
    reader = TensorReader({p + "in_proj_a.weight": a, p + "in_proj_b.weight": b,
                           p + "in_proj_qkv.weight": qkv, p + "in_proj_z.weight": z})
    got = recipe.materialize_recipe(RECIPES["text/layers/1/gdn/a_b_projection"], reader)
    assert torch.equal(got, torch.cat((a, b)))
    got = recipe.materialize_recipe(RECIPES["text/layers/1/gdn/query_key_value_z"], reader)
    assert torch.equal(got, torch.cat((qkv, z)))


def test_dflash_uses_its_own_projection_widths():
    p = "layers.0."
    q, k, v = torch.ones((256, 128)), torch.full((128, 128), 2.), torch.full((128, 128), 3.)
    gate, up = torch.full((192, 128), 4.), torch.full((192, 128), 5.)
    reader = TensorReader({p + "self_attn.q_proj.weight": q, p + "self_attn.k_proj.weight": k,
                           p + "self_attn.v_proj.weight": v, p + "mlp.gate_proj.weight": gate,
                           p + "mlp.up_proj.weight": up})
    got = recipe.materialize_recipe(RECIPES["dflash/layers/0/attention/query_key_value"], reader)
    assert torch.equal(got, torch.cat((q, k, v)))
    got = recipe.materialize_recipe(RECIPES["dflash/layers/0/mlp/gate_up"], reader)
    assert torch.equal(got, torch.cat((gate, up)))


def test_single_file_reader_is_explicit_and_lazy(tmp_path) -> None:
    path = tmp_path / "dflash.safetensors"
    save_file({"weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)}, path)
    with ShardReader.from_file(path) as reader:
        assert reader.names == ("weight",)
        metadata = reader.metadata(("weight",))["weight"]
        assert metadata.shape == (2, 4)
        assert metadata.dtype == "BF16"
        assert metadata.shard == path.name
        assert torch.equal(
            reader.get("weight"),
            torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        )
