"""Unit tests for the dispatch-PP planner (pure Python, GPU-free)."""

from surogate.train.dispatch_pp.types import BlockProfile, StageRange, StagePlan
from surogate.train.dispatch_pp.planner import (
    pack_stages,
    candidate_budgets,
    plan_stages,
    token_threshold,
    microbatch_warning,
    envelope_warning,
    assign_numa,
)


def _uniform_profiles(n, fwd, bwd, wbytes=1, abytes=0, needs_grad=True):
    return [BlockProfile(fwd, bwd, wbytes, abytes, needs_grad) for _ in range(n)]


def test_stage_range_to_dict_roundtrips():
    s = StageRange(lo=0, hi=2, weight_bytes=100, est_time=1.5, needs_grad=True, numa_node=1)
    d = s.to_dict()
    assert d == {
        "lo": 0, "hi": 2, "weight_bytes": 100,
        "est_time": 1.5, "needs_grad": True, "numa_node": 1,
    }


def test_stage_plan_to_dict_is_json_serializable():
    import json
    plan = StagePlan(
        fwd_stages=[StageRange(0, 1, 10, 0.5, True, None)],
        fused_tail=StageRange(2, 3, 20, 1.0, True, None),
        bwd_stages=[StageRange(0, 1, 10, 0.7, True, None)],
        num_blocks=4,
        warnings=["hello"],
    )
    # Must serialize without custom encoders (it is the cross-language contract).
    json.dumps(plan.to_dict())
    assert plan.to_dict()["num_blocks"] == 4
    assert plan.to_dict()["fused_tail"]["lo"] == 2


def test_pack_splits_on_workload_ceiling():
    # costs 1+1=2 fits budget 2; adding a third (=3) exceeds -> new stage.
    stages = pack_stages(costs=[1, 1, 1, 1], sizes=[0, 0, 0, 0],
                         max_workload=2.0, mem_ceiling=10**18)
    assert stages == [(0, 1), (2, 3)]


def test_pack_splits_on_memory_ceiling():
    # workload budget is huge; memory forces a split after 2 blocks of size 5 (10 == ceiling, third overflows).
    stages = pack_stages(costs=[0, 0, 0, 0], sizes=[5, 5, 5, 5],
                         max_workload=10**18, mem_ceiling=10)
    assert stages == [(0, 1), (2, 3)]


def test_pack_single_oversized_block_is_alone():
    # a block larger than the ceiling cannot be split; it occupies its own stage.
    stages = pack_stages(costs=[1, 1], sizes=[100, 1],
                         max_workload=10**18, mem_ceiling=10)
    assert stages == [(0, 0), (1, 1)]


def test_pack_all_in_one_stage_when_budgets_large():
    stages = pack_stages(costs=[1, 1, 1], sizes=[1, 1, 1],
                         max_workload=10**18, mem_ceiling=10**18)
    assert stages == [(0, 2)]


def test_pack_is_asymmetric_forward_holds_more_blocks_than_backward():
    # The core asymmetry: at the SAME time budget, forward (cost 1/block) packs
    # more blocks per stage than backward (cost 3/block incl. recompute).
    fwd = pack_stages([1] * 12, [0] * 12, max_workload=6.0, mem_ceiling=10**18)
    bwd = pack_stages([3] * 12, [0] * 12, max_workload=6.0, mem_ceiling=10**18)
    fwd_width = max(hi - lo + 1 for lo, hi in fwd)   # 6 blocks/stage
    bwd_width = max(hi - lo + 1 for lo, hi in bwd)   # 2 blocks/stage
    assert fwd_width > bwd_width


def test_candidates_are_prefix_sums_between_maxblock_and_threshold():
    # max single block = 3. upper_threshold 1.1 -> ceiling 3.3.
    # prefix sums per start that land in [3.0, 3.3]:
    #   [3,1,2,2]: start0 ->3(ok); start1 1,1+2=3(ok),3+2=5 stop; start2 2,2+2=4 stop; start3 2 stop
    # candidates = {3.0}
    cands = candidate_budgets([3, 1, 2, 2], upper_threshold=1.1)
    assert cands == [3.0]


def test_candidates_include_multiblock_sums_within_threshold():
    # max block = 2. threshold 1.6 -> ceiling 3.2. prefix sums in [2.0, 3.2]:
    #   start0: 2(ok), 2+1=3(ok), 3+1=4 stop
    #   start1: 1, 1+1=2(ok); start2: 1
    # candidates = {2.0, 3.0}
    cands = candidate_budgets([2, 1, 1], upper_threshold=1.6)
    assert cands == [2.0, 3.0]


def test_candidates_single_block_returns_that_block():
    assert candidate_budgets([5], upper_threshold=1.1) == [5.0]


def test_plan_covers_every_block_exactly_once():
    profiles = _uniform_profiles(8, fwd=1.0, bwd=3.0)
    plan = plan_stages(profiles, min_stages=4, upper_threshold=1.1,
                       vram_budget_bytes=10**18)
    # forward path (fwd_stages + fused_tail) covers all blocks ascending:
    fwd_cover = []
    for s in plan.fwd_stages:
        fwd_cover.extend(range(s.lo, s.hi + 1))
    fwd_cover.extend(range(plan.fused_tail.lo, plan.fused_tail.hi + 1))
    assert fwd_cover == list(range(8))
    # backward path (fused_tail + bwd_stages) also covers all blocks:
    bwd_cover = list(range(plan.fused_tail.lo, plan.fused_tail.hi + 1))
    for s in plan.bwd_stages:
        bwd_cover.extend(range(s.lo, s.hi + 1))
    assert sorted(bwd_cover) == list(range(8))


def test_plan_bwd_stages_are_descending():
    profiles = _uniform_profiles(8, fwd=1.0, bwd=3.0)
    plan = plan_stages(profiles, min_stages=4, upper_threshold=1.1,
                       vram_budget_bytes=10**18)
    los = [s.lo for s in plan.bwd_stages]
    assert los == sorted(los, reverse=True)


def test_plan_under_memory_pressure_packs_multiblock_stages():
    # When memory binds, stages hold >1 block (the regime where asymmetry shows).
    # 6 blocks, weight 8 bytes; backward stage memory = weight+grad = 16/block, so
    # vram_budget 96 -> ceiling 48 -> up to 3 blocks per backward/fused stage.
    profiles = _uniform_profiles(6, fwd=1.0, bwd=1.0, wbytes=8)
    plan = plan_stages(profiles, min_stages=2, upper_threshold=3.0,
                       vram_budget_bytes=96)
    widths = [s.hi - s.lo + 1 for s in plan.fwd_stages] + [
        plan.fused_tail.hi - plan.fused_tail.lo + 1
    ]
    assert max(widths) >= 2  # at least one multi-block stage


def test_plan_single_block_is_all_fused_tail():
    profiles = _uniform_profiles(1, fwd=1.0, bwd=3.0)
    plan = plan_stages(profiles, min_stages=1, upper_threshold=1.1,
                       vram_budget_bytes=10**18)
    assert plan.fwd_stages == []
    assert plan.bwd_stages == []
    assert (plan.fused_tail.lo, plan.fused_tail.hi) == (0, 0)


def test_plan_fused_tail_sized_by_backward_not_forward():
    # The fused tail runs fwd+loss+bwd, so it must be budgeted/sized by backward
    # cost. Its est_time therefore reflects bwd_time (3.0/block), never fwd_time
    # (1.0/block). Sizing it by forward time would over-pack and make it the
    # makespan straggler.
    profiles = _uniform_profiles(6, fwd=1.0, bwd=3.0)
    plan = plan_stages(profiles, min_stages=1, upper_threshold=1.1,
                       vram_budget_bytes=10**18)
    tail = plan.fused_tail
    n_tail = tail.hi - tail.lo + 1
    assert tail.est_time == 3.0 * n_tail   # bwd_time per block, not fwd (=1.0)


def test_plan_warns_when_block_exceeds_half_vram_budget():
    # one block needs 30 bytes; vram_budget 40 -> ceiling (budget/2)=20 -> block overflows.
    profiles = [BlockProfile(1.0, 3.0, weight_bytes=30, act_bytes=0) for _ in range(3)]
    plan = plan_stages(profiles, min_stages=2, upper_threshold=1.1,
                       vram_budget_bytes=40)
    assert any("exceeds" in w and "VRAM" in w for w in plan.warnings)


def test_plan_no_oversize_warning_when_blocks_fit():
    profiles = [BlockProfile(1.0, 3.0, weight_bytes=5, act_bytes=0) for _ in range(3)]
    plan = plan_stages(profiles, min_stages=2, upper_threshold=1.1,
                       vram_budget_bytes=40)
    assert not any("VRAM" in w for w in plan.warnings)


def test_token_threshold_matches_formula():
    # threshold = 0.75 * flops / pcie_bw
    thr = token_threshold(gpu_flops=80e12, pcie_bw=20e9)
    assert abs(thr - 0.75 * 80e12 / 20e9) < 1e-6  # == 3000.0


def test_microbatch_warning_dense_threshold_is_8():
    assert microbatch_warning(num_microbatches=4, is_moe=False) is not None
    assert microbatch_warning(num_microbatches=8, is_moe=False) is None


def test_microbatch_warning_moe_threshold_is_80():
    assert microbatch_warning(num_microbatches=16, is_moe=True) is not None
    assert microbatch_warning(num_microbatches=80, is_moe=True) is None


def test_envelope_warning_fires_below_token_threshold():
    w = envelope_warning(tokens_per_step=500, gpu_flops=80e12, pcie_bw=20e9)
    assert w is not None and "transfer-bound" in w


def test_envelope_warning_silent_above_token_threshold():
    w = envelope_warning(tokens_per_step=8192, gpu_flops=80e12, pcie_bw=20e9)
    assert w is None


def test_frozen_lora_base_stage_marked_no_grad():
    # all base blocks frozen -> every produced stage has needs_grad False.
    profiles = [BlockProfile(1.0, 3.0, 1, 0, needs_grad=False) for _ in range(6)]
    plan = plan_stages(profiles, min_stages=2, upper_threshold=1.1,
                       vram_budget_bytes=10**18)
    all_stages = plan.fwd_stages + [plan.fused_tail] + plan.bwd_stages
    assert all(s.needs_grad is False for s in all_stages)


def test_assign_numa_round_robin_by_socket():
    plan = plan_stages(_uniform_profiles(8, 1.0, 3.0), min_stages=2,
                       upper_threshold=1.1, vram_budget_bytes=10**18)
    placed = assign_numa(plan, num_numa_nodes=2)
    nodes = [s.numa_node for s in placed.fwd_stages] + [placed.fused_tail.numa_node]
    assert set(n for n in nodes if n is not None) <= {0, 1}
    assert all(n is not None for n in nodes)


def test_assign_numa_single_socket_is_none():
    plan = plan_stages(_uniform_profiles(4, 1.0, 3.0), min_stages=1,
                       upper_threshold=1.1, vram_budget_bytes=10**18)
    placed = assign_numa(plan, num_numa_nodes=1)
    assert placed.fused_tail.numa_node is None
