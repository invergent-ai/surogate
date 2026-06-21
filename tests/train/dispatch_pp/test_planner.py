"""Unit tests for the dispatch-PP planner (pure Python, GPU-free)."""

from surogate.train.dispatch_pp.types import BlockProfile, StageRange, StagePlan
from surogate.train.dispatch_pp.planner import pack_stages, candidate_budgets


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
