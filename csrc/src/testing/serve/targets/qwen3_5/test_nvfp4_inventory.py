import pytest
from tests.serve.test_qwen3_5_checkpoint_config import config_for
from surogate.serve.convert.qwen3_5 import inventory
from surogate.serve.convert.qwen3_5.exports import recipe_nvfp4_uniform as recipe


@pytest.mark.parametrize("hidden", [128, 384])
def test_matrix_programs_use_config_shapes_and_schedule(hidden):
    g = inventory.geometry_from_config(config_for(hidden=hidden))
    plan = recipe.build(g, "model.")
    entries = {r.object_name: r for r in plan.nvfp4_weights}
    assert entries["text/layers/0/attention/query_key_gate_value"].shape == (768, hidden)
    assert entries["text/layers/1/gdn/query_key_value_z"].shape == (640, hidden)
    assert entries["text/layers/3/mlp/gate_up"].shape == (512, hidden)
    for entry in entries.values():
        assert sum(p.output_rows for p in entry.parts) == entry.shape[0]
        assert all(p.source.shape[1] == entry.shape[1] for p in entry.parts)
