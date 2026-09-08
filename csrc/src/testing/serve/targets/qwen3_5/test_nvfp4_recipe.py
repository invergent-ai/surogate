import torch
from tests.serve.test_qwen3_5_checkpoint_config import config_for
from surogate.serve.convert.qwen3_5 import inventory
from surogate.serve.convert.qwen3_5.exports import recipe_nvfp4_uniform as recipe


def test_attention_row_program_preserves_head_interleaving():
    g = inventory.geometry_from_config(config_for())
    entry = next(r for r in recipe.build(g, "model.").nvfp4_weights
                 if r.object_name == "text/layers/0/attention/query_key_gate_value")
    q, key, gate, value = entry.parts
    rows = torch.arange(512).view(-1, 1)
    assert torch.equal(recipe._select_rows(rows, q).flatten(), torch.cat((torch.arange(128), torch.arange(256, 384))))
    assert torch.equal(recipe._select_rows(rows, gate).flatten(), torch.cat((torch.arange(128, 256), torch.arange(384, 512))))
    assert key.output_rows == value.output_rows == 128
