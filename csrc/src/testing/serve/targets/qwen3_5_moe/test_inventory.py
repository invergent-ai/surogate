import pytest
from tests.serve.test_qwen3_5_moe_checkpoint_config import config_for
from surogate.serve.convert.qwen3_5_moe import inventory


@pytest.mark.parametrize("hidden,experts,width", [(128, 4, 64), (384, 8, 192)])
def test_expert_shapes_and_identity_follow_configuration(hidden, experts, width):
    c = config_for(hidden=hidden, vision=True)
    c.update(num_experts=experts, moe_intermediate_size=width)
    g = inventory.geometry_from_config(c, token_domain=500)
    tensors = {s.name: s for s in inventory.build_tensor_specs(g)}
    assert inventory.MODEL_ID == inventory.TARGET_KEY == "qwen3_5_moe"
    assert tensors["text/layers/1/gdn/a_b_projection"].shape == (8, hidden)
    assert tensors["text/layers/0/moe/routed_gate_up"].shape == (experts * 2 * width, hidden)
    assert tensors["mtp/layer/moe/routed_down"].shape == (experts * hidden, width)
    assert tensors["vision/merger/fc2"].shape == (hidden, 384)
    assert not any(n.startswith("dflash/") for n in tensors)
