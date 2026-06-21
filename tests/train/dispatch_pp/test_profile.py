"""Planner -> real-model integration tests (GPU-free; needs a cached model).

Reads per-block weight bytes from the safetensors header and config dims from
config.json, then exercises build_block_profiles / plan_for_model end-to-end
against a real checkpoint. No CUDA, no model load.
"""

import pytest

from surogate.train.dispatch_pp import (
    block_weight_bytes,
    build_block_profiles,
    plan_for_model,
    resolve_vram_budget_bytes,
)
from surogate.train.dispatch_pp.types import BlockProfile

# Reuse the onboarding helper to locate a cached model snapshot (no GPU needed).
from surogate.utils.hf import get_model_weights_path
from tests.test_onboarding_qwen3 import resolve_model_path


@pytest.fixture(scope="module")
def model_paths():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3 weights not found. Set QWEN3_MODEL_PATH or cache Qwen/Qwen3-0.6B.")
    return str(snapshot), get_model_weights_path(str(snapshot))


def _all_blocks(plan):
    blocks = []
    for s in plan.fwd_stages:
        blocks.extend(range(s.lo, s.hi + 1))
    blocks.extend(range(plan.fused_tail.lo, plan.fused_tail.hi + 1))
    return blocks


def test_block_weight_bytes_contiguous_and_positive(model_paths):
    _, weights = model_paths
    per_block = block_weight_bytes(weights)
    assert per_block, "expected at least one transformer block"
    n = max(per_block) + 1
    assert sorted(per_block) == list(range(n))  # contiguous block indices
    assert all(b > 0 for b in per_block.values())


def test_build_block_profiles_one_per_block(model_paths):
    model_dir, weights = model_paths
    profiles = build_block_profiles(model_dir, weights, seq_len=2048, micro_batch=1)
    n = len(block_weight_bytes(weights))
    assert len(profiles) == n
    assert all(isinstance(p, BlockProfile) for p in profiles)
    p = profiles[0]
    assert p.weight_bytes > 0 and p.act_bytes > 0
    assert p.bwd_time == pytest.approx(3.0 * p.fwd_time)  # backward incl. recompute ~3x
    assert p.needs_grad is True


def test_act_bytes_scale_with_seq_and_microbatch(model_paths):
    model_dir, weights = model_paths
    small = build_block_profiles(model_dir, weights, seq_len=512, micro_batch=1)[0]
    big = build_block_profiles(model_dir, weights, seq_len=2048, micro_batch=2)[0]
    assert big.act_bytes == small.act_bytes * (2048 * 2) // (512 * 1)
    assert big.weight_bytes == small.weight_bytes  # weights are shape-independent


def test_plan_for_model_covers_every_block(model_paths):
    model_dir, weights = model_paths
    n = len(block_weight_bytes(weights))
    plan = plan_for_model(
        model_dir, weights, min_stages=4, seq_len=2048, micro_batch=1, vram_budget_bytes=10**18
    )
    assert plan.num_blocks == n
    assert _all_blocks(plan) == list(range(n))
    bwd_blocks = list(range(plan.fused_tail.lo, plan.fused_tail.hi + 1))
    for s in plan.bwd_stages:
        bwd_blocks.extend(range(s.lo, s.hi + 1))
    assert sorted(bwd_blocks) == list(range(n))


def test_plan_for_model_numa_round_robin(model_paths):
    model_dir, weights = model_paths
    plan = plan_for_model(
        model_dir, weights, min_stages=2, seq_len=2048, micro_batch=1,
        vram_budget_bytes=10**18, num_numa_nodes=2,
    )
    nodes = [s.numa_node for s in plan.fwd_stages] + [plan.fused_tail.numa_node]
    assert all(n is not None for n in nodes)
    assert set(nodes) <= {0, 1}


def test_plan_for_model_emits_microbatch_warning(model_paths):
    model_dir, weights = model_paths
    plan = plan_for_model(
        model_dir, weights, min_stages=4, seq_len=2048, micro_batch=1,
        vram_budget_bytes=10**18, num_microbatches=4, is_moe=False,
    )
    assert any("roofline" in w for w in plan.warnings)


def test_resolve_vram_budget_bytes_explicit_and_auto():
    assert resolve_vram_budget_bytes(24.0, free_vram_bytes=999) == int(24.0 * 1024**3)
    assert resolve_vram_budget_bytes(None, free_vram_bytes=20 * 1024**3) == int(20 * 1024**3 * 0.9)
