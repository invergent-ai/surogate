import pytest
from tests.serve.test_qwen3_5_checkpoint_config import config_for
from surogate.serve.convert.qwen3_5 import inventory


@pytest.mark.parametrize("hidden,mtp,vision", [(128, 0, False), (384, 1, True)])
def test_inventory_and_logical_views_follow_config(hidden, mtp, vision):
    config = config_for(hidden=hidden, vision=vision)
    config["mtp_num_hidden_layers"] = mtp
    g = inventory.geometry_from_config(config, token_domain=500)
    export = inventory.export_inventory(inventory.GROUPWISE_INT, g)
    tensors = {s.name: s for s in export.TENSOR_SPECS}
    assert export.MODEL_ID == export.TARGET_KEY == "qwen3_5"
    assert tensors["text/token_embedding"].shape == (512, hidden)
    assert tensors["text/draft_head"].shape == (500, hidden)
    assert tensors["text/layers/1/gdn/convolution"].shape == (4, 384)
    assert tensors["text/layers/0/attention/query_key_gate_value"].shape == (768, hidden)
    assert len(tensors) == len(export.TENSOR_SPECS)
    assert any(n.startswith("mtp/") for n in tensors) == bool(mtp)
    assert any(n.startswith("vision/") for n in tensors) == vision
    for view in export.LOGICAL_ROW_VIEW_SPECS:
        for layer in view.layers if view.layers is not None else (None,):
            parent = tensors[view.parent_pattern if layer is None else view.parent_pattern.format(l=layer)]
            assert 0 <= view.row_begin < view.row_end <= parent.shape[0]
            assert view.shape == (view.row_count, hidden)
    assert any(a.role_pattern.startswith("mtp/") for a in export.ALIAS_SPECS) == bool(mtp)


def test_inventory_requires_explicit_geometry():
    with pytest.raises(TypeError):
        inventory.build_tensor_specs()
