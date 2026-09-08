import pytest
import torch
import torch.nn.functional as F

from tools.reference.qwen3_5_moe.bindings import (
    ExpertBank,
    MoeBinding,
)
from tools.reference.qwen3_5_moe.config import ModelConfig
from tools.reference.qwen3_5_moe.moe import forward


@pytest.mark.parametrize(
    "hidden,experts,width,shared,selected,scale",
    [
        (64, 7, 16, 32, (6, 0, 3), 1.0),
        (128, 11, 32, 16, (10, 5), 0.75),
    ],
)
def test_selected_expert_rows_and_high_precision_moe_formula(hidden, experts, width, shared, selected, scale) -> None:
    cfg = ModelConfig(
        hidden=hidden,
        experts=experts,
        experts_per_token=len(selected),
        expert_intermediate=width,
        shared_intermediate=shared,
        routed_scale=scale,
    )
    x = torch.zeros((1, cfg.hidden), dtype=torch.bfloat16)
    x[0, 0] = 1

    router_shared = torch.full(
        (cfg.experts + 1, cfg.hidden),
        -20.0,
        dtype=torch.bfloat16,
    )
    for rank, expert in enumerate(selected):
        router_shared[expert, 0] = 8.0 - rank * 0.5
    router_shared[cfg.experts, 0] = 0.0
    shared_gate_weight = 0.2513
    shared_up_weight = 3.0061
    shared_down_weight = 2.0097
    expert_gate_weight = 0.5031
    expert_up_weight = 2.0063
    shared_gate_up = torch.zeros((2 * cfg.shared_intermediate, cfg.hidden), dtype=torch.float32)
    shared_gate_up[0, 0] = shared_gate_weight
    shared_gate_up[cfg.shared_intermediate, 0] = shared_up_weight
    shared_down = torch.zeros(
        (cfg.hidden, cfg.shared_intermediate),
        dtype=torch.float32,
    )
    shared_down[0, 0] = shared_down_weight

    class FakeModel:
        def __init__(self) -> None:
            self.config = cfg
            self.blocks = {
                "router": router_shared,
                "shared_gate_up": shared_gate_up,
                "shared_down": shared_down,
            }
            self.reads: list[tuple[str, int, int]] = []

        def block_weight(self, block: str, *, small_t: bool) -> torch.Tensor:
            assert small_t
            return self.blocks[block]

        def rows(
            self,
            block: str,
            rows: torch.Tensor,
            *,
            small_t: bool,
        ) -> torch.Tensor:
            assert small_t
            indices = rows.cpu()
            begin, end = int(indices[0]), int(indices[-1])
            self.reads.append((block, begin, end))
            if block == "routed_gate_up":
                assert begin // (2 * cfg.expert_intermediate) == end // (2 * cfg.expert_intermediate)
                assert end - begin + 1 == (2 * cfg.expert_intermediate)
                weight = torch.zeros(((2 * cfg.expert_intermediate), cfg.hidden), dtype=torch.float32)
                weight[0, 0] = expert_gate_weight
                weight[cfg.expert_intermediate, 0] = expert_up_weight
                return weight
            assert begin // cfg.hidden == end // cfg.hidden
            assert end - begin + 1 == cfg.hidden
            expert = begin // cfg.hidden
            weight = torch.zeros(
                (cfg.hidden, cfg.expert_intermediate),
                dtype=torch.float32,
            )
            weight[0, 0] = expert + 1.0031
            return weight

    binding = MoeBinding(
        router_shared_gate="router",
        router=None,
        shared_gate=None,
        routed_gate_up=ExpertBank(
            "routed_gate_up", cfg.experts, (2 * cfg.expert_intermediate), cfg.expert_intermediate
        ),
        routed_down=ExpertBank("routed_down", cfg.experts, cfg.hidden, None),
        shared_gate_up="shared_gate_up",
        shared_expert_gate=None,
        shared_up=None,
        shared_down="shared_down",
    )
    model = FakeModel()
    actual = forward(model, binding, x, small_t=True)

    router_logits = router_shared[: cfg.experts, 0].unsqueeze(0)
    expected_ids = torch.argsort(
        router_logits.float(),
        dim=-1,
        descending=True,
        stable=True,
    )[:, : cfg.experts_per_token]
    expected_weights = torch.softmax(
        torch.gather(router_logits.float(), -1, expected_ids),
        dim=-1,
    )
    assert torch.equal(actual.expert_ids, expected_ids)
    assert torch.equal(actual.route_weights, expected_weights)
    assert set(actual.expert_ids[0].tolist()) == set(selected)

    read_experts = {
        begin // ((2 * cfg.expert_intermediate) if block == "routed_gate_up" else cfg.hidden)
        for block, begin, _ in model.reads
    }
    assert read_experts == set(selected)
    assert len(model.reads) == 2 * cfg.experts_per_token

    gate = torch.tensor(expert_gate_weight, dtype=torch.float32)
    up = torch.tensor(expert_up_weight, dtype=torch.float32)
    expert_hidden = F.silu(gate) * up
    routed_terms = []
    for expert_id, route_weight in zip(
        expected_ids[0].tolist(),
        expected_weights[0],
        strict=True,
    ):
        expert_down = expert_hidden * float(expert_id + 1.0031)
        routed_terms.append(route_weight.float() * expert_down)
    routed = torch.stack(routed_terms).sum()

    shared_hidden = F.silu(torch.tensor(shared_gate_weight)) * torch.tensor(shared_up_weight)
    shared = shared_hidden * shared_down_weight
    shared_scale = torch.sigmoid(torch.tensor(0.0))
    expected = (cfg.routed_scale * routed + shared_scale * shared).to(torch.bfloat16)
    assert torch.equal(actual.output[0, 0], expected)
    assert torch.count_nonzero(actual.output[0, 1:]) == 0
