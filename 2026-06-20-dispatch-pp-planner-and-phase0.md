# Dispatch-PP — Planner, Config & Phase-0 Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure-Python `DispatchPlanner` (asymmetric stage partitioning + cost search + envelope warnings), wire the `parallelism: dispatch_pp` config + validation, and run the Phase-0 C++ feasibility gate that proves `GraphExecutor` can execute a contiguous block sub-range.

**Architecture:** Python orchestrates, C++ runs the hot path (matches surogate's existing trainer split). This plan delivers (a) the planner as a self-contained, GPU-free, unit-tested module emitting a serializable `StagePlan`; (b) additive config plumbing in `SFTConfig`; (c) a Phase-0 spike that either proves sub-range execution works or surfaces the exact blocker before any runtime is built. Phases 1–3 (DispatchScheduler, stage streaming, async optimizer) are authored as separate plans after Phase 0.

**Tech Stack:** Python 3.12, `dataclasses`, `pytest`; surogate C++ engine (`csrc/`, pybind/nanobind via `_surogate`), DSL IR.

**Spec:** `2026-06-20-dispatch-pp-design.md` (§3 planner, §5 config/contract, §7 Phase 0).

## Implementation Progress

Updated before each commit. Status: ☐ not started · ◐ in progress · ☑ done.

- ☑ Task 1 — Scaffold package + data types
- ☑ Task 2 — Stage packing under workload + memory ceilings
- ◐ Task 3 — Candidate budget enumeration
- ☐ Task 4 — Cost search + plan assembly (fwd / fused tail / bwd)
- ☐ Task 5 — Oversized-block warning
- ☐ Task 6 — PCIe token-threshold / microbatch envelope warnings
- ☐ Task 7 — LoRA `needs_grad` propagation + NUMA placement
- ☐ Task 8 — Config plumbing + validation in `SFTConfig`
- ☐ Task 9 — Phase-0 `GraphExecutor` sub-range execution gate
- ☐ Task 10 — Full-suite run + record Phase-0 verdict

## Review Update — 2026-06-21

This plan was reviewed against the current repo layout before implementation. Changes from the first
draft:
- Corrected the design-spec path to the repo-root `2026-06-20-dispatch-pp-design.md`.
- Kept the pure-Python planner/config work before the C++ gate because it is self-contained and
  low-risk, but made Task 9 a hard stop before any scheduler/stage-streaming work.
- Tightened Task 8 around `SFTConfig`'s existing defaults: `zero_level=1` is allowed, while
  `zero_level>1`, `shard_weights`, `shard_gradients`, EP, non-BF16 recipes, CUDA graphs, and the
  separate `cpu_training` mode are handled explicitly.
- Updated Phase 0 to match current engine surfaces: `CompiledGraph` already exposes
  `layer_start_indices`, `layer_end_indices`, `phase_tree`, and `instruction_stream`; the minimal
  spike should expose narrow compiled-executor range runners rather than inventing a separate
  execution path.
- Made the Phase-0 weight claim honest: this plan proves resident-weight contiguous sub-range
  execution and CPU-boundary activation handoff. Externally streamed stage weights remain the first
  gate of the later stage-streaming plan unless Phase 0 discovers an existing external-weight hook.

---

## File Structure

| Path | Responsibility | Status |
|---|---|---|
| `surogate/train/dispatch_pp/__init__.py` | Package marker; re-export public API | Create |
| `surogate/train/dispatch_pp/types.py` | `BlockProfile`, `StageRange`, `StagePlan` dataclasses + `to_dict()` | Create |
| `surogate/train/dispatch_pp/planner.py` | Packing, candidate search, cost search, envelope warnings, `plan_stages()` | Create |
| `tests/train/dispatch_pp/test_planner.py` | Unit tests for all planner logic (no GPU) | Create |
| `tests/train/dispatch_pp/test_config.py` | Config parsing + validation tests | Create |
| `surogate/core/config/sft_config.py` | Add `parallelism`, `dispatch_pp` sub-block, validations | Modify |
| `csrc/src/runtime/executor/compiled_ops.h` | Add debug-only compiled range runner declarations if Phase 0 needs them | Modify |
| `csrc/src/runtime/executor/compiled_ops_execute_forward.cpp` | Add/route debug-only forward range execution over compiled block op ranges | Modify |
| `csrc/src/runtime/executor/compiled_ops_execute_backward.cpp` | Add/route debug-only backward range execution over compiled block op ranges | Modify |
| `csrc/src/runtime/executor/graph_executor.h` / `csrc/src/runtime/executor/graph_executor.cpp` | Add debug-only GraphExecutor range wrappers if private executor state is needed | Modify |
| `csrc/src/runtime/executor/dispatch_pp_phase0.h` | Phase-0 debug declarations and findings comment | Create |
| `csrc/src/runtime/executor/dispatch_pp_phase0.cpp` | Phase-0 debug harness over existing compiled executor machinery | Create |
| `csrc/CMakeLists.txt` | Add `dispatch_pp_phase0.cpp` to `surogate-common` sources | Modify |
| `csrc/src/binding/py_train.h` / `csrc/src/binding/py_train.cpp` / `csrc/src/binding/binding.cpp` | Debug-only Phase-0 Python hooks | Modify |
| `tests/train/dispatch_pp/test_phase0_subrange.py` | Numerical parity test for sub-range vs whole-graph, or xfail with precise obstruction | Create |

All planner units are pure functions over plain data — testable in isolation, no CUDA, no engine import.

---

## Task 1: Scaffold package and data types

**Files:**
- Create: `surogate/train/dispatch_pp/__init__.py`
- Create: `surogate/train/dispatch_pp/types.py`
- Create: `tests/train/dispatch_pp/__init__.py`
- Test: `tests/train/dispatch_pp/test_planner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/train/dispatch_pp/test_planner.py
from surogate.train.dispatch_pp.types import BlockProfile, StageRange, StagePlan


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/train/dispatch_pp/test_planner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surogate.train.dispatch_pp'`

- [ ] **Step 3: Write minimal implementation**

```python
# surogate/train/dispatch_pp/__init__.py
"""Dispatch pipeline-parallelism planner (pure Python, GPU-free)."""
from .types import BlockProfile, StageRange, StagePlan

__all__ = ["BlockProfile", "StageRange", "StagePlan"]
```

```python
# surogate/train/dispatch_pp/types.py
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class BlockProfile:
    """Per-block profile fed to the planner. One entry per DSL transformer block."""
    fwd_time: float      # forward wall-time estimate (seconds)
    bwd_time: float      # backward incl. recompute wall-time estimate (seconds)
    weight_bytes: int    # work-weight size (bf16)
    act_bytes: int       # activation/recompute working-set at current seq_len x micro_batch
    needs_grad: bool = True   # False for frozen LoRA base blocks


@dataclass(frozen=True)
class StageRange:
    """A contiguous, inclusive block range [lo, hi] forming one scheduling unit."""
    lo: int
    hi: int
    weight_bytes: int
    est_time: float
    needs_grad: bool
    numa_node: int | None   # preferred NUMA placement of this stage's CPU weights; None = single-socket

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StagePlan:
    """The cross-language contract handed to the C++ scheduler once per run."""
    fwd_stages: list[StageRange]   # pure-forward stages, ascending
    fused_tail: StageRange             # trailing range run as fwd+loss+bwd
    bwd_stages: list[StageRange]  # descending execution order
    num_blocks: int
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "fwd_stages": [s.to_dict() for s in self.fwd_stages],
            "fused_tail": self.fused_tail.to_dict(),
            "bwd_stages": [s.to_dict() for s in self.bwd_stages],
            "num_blocks": self.num_blocks,
            "warnings": list(self.warnings),
        }
```

Also create an empty `tests/train/dispatch_pp/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/train/dispatch_pp/test_planner.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add surogate/train/dispatch_pp/ tests/train/dispatch_pp/
git commit -m "feat(dispatch-pp): scaffold planner package + StagePlan data types"
```

---

## Task 2: Stage packing under workload + memory ceilings

**Files:**
- Create: `surogate/train/dispatch_pp/planner.py`
- Test: `tests/train/dispatch_pp/test_planner.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/train/dispatch_pp/test_planner.py
from surogate.train.dispatch_pp.planner import pack_stages


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/train/dispatch_pp/test_planner.py -k pack -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'surogate.train.dispatch_pp.planner'`

- [ ] **Step 3: Write minimal implementation**

```python
# surogate/train/dispatch_pp/planner.py
from __future__ import annotations

from typing import Sequence

from .types import BlockProfile, StageRange, StagePlan


def pack_stages(
    costs: Sequence[float],
    sizes: Sequence[int],
    max_workload: float,
    mem_ceiling: int,
) -> list[tuple[int, int]]:
    """Greedily pack consecutive blocks into contiguous stages.

    A new stage starts when adding the next block would exceed EITHER the
    per-stage time budget (max_workload) OR the per-stage memory ceiling
    (mem_ceiling). A single block that exceeds the ceiling still occupies its
    own stage (blocks are indivisible in v1).

    Returns inclusive (lo, hi) ranges covering [0, len(costs)) exactly once.
    """
    n = len(costs)
    if n == 0:
        return []
    stages: list[tuple[int, int]] = []
    start = 0
    acc_cost = 0.0
    acc_size = 0
    for i in range(n):
        if i > start and (
            acc_cost + costs[i] > max_workload or acc_size + sizes[i] > mem_ceiling
        ):
            stages.append((start, i - 1))
            start = i
            acc_cost = 0.0
            acc_size = 0
        acc_cost += costs[i]
        acc_size += sizes[i]
    stages.append((start, n - 1))
    return stages

```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/train/dispatch_pp/test_planner.py -k pack -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add surogate/train/dispatch_pp/planner.py tests/train/dispatch_pp/test_planner.py
git commit -m "feat(dispatch-pp): greedy stage packing under workload+memory ceilings"
```

---

## Task 3: Candidate budget enumeration

**Files:**
- Modify: `surogate/train/dispatch_pp/planner.py`
- Test: `tests/train/dispatch_pp/test_planner.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/train/dispatch_pp/test_planner.py
from surogate.train.dispatch_pp.planner import candidate_budgets


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/train/dispatch_pp/test_planner.py -k candidate -v`
Expected: FAIL with `ImportError: cannot import name 'candidate_budgets'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to surogate/train/dispatch_pp/planner.py (above plan_stages)
def candidate_budgets(costs: Sequence[float], upper_threshold: float) -> list[float]:
    """Enumerate candidate per-stage time budgets.

    A candidate is any sum of consecutive block costs that lands in
    [max_single_block, max_single_block * upper_threshold]. A stage can never be
    smaller than the slowest single block, and never larger than the slack
    threshold above it. Returned sorted and de-duplicated.
    """
    if not costs:
        return []
    max_block = max(costs)
    ceiling = max_block * upper_threshold
    cands: set[float] = set()
    n = len(costs)
    for start in range(n):
        s = 0.0
        for end in range(start, n):
            s += costs[end]
            if s > ceiling:
                break
            if s >= max_block:
                cands.add(round(s, 9))
    return sorted(cands)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/train/dispatch_pp/test_planner.py -k candidate -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add surogate/train/dispatch_pp/planner.py tests/train/dispatch_pp/test_planner.py
git commit -m "feat(dispatch-pp): candidate stage-budget enumeration"
```

---

## Task 4: Cost search + plan assembly (forward / fused tail / backward)

**Files:**
- Modify: `surogate/train/dispatch_pp/planner.py`
- Test: `tests/train/dispatch_pp/test_planner.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/train/dispatch_pp/test_planner.py
from surogate.train.dispatch_pp.types import BlockProfile
from surogate.train.dispatch_pp.planner import plan_stages


def _uniform_profiles(n, fwd, bwd, wbytes=1, abytes=0, needs_grad=True):
    return [BlockProfile(fwd, bwd, wbytes, abytes, needs_grad) for _ in range(n)]


def test_plan_covers_every_block_exactly_once():
    profiles = _uniform_profiles(8, fwd=1.0, bwd=3.0)
    plan = plan_stages(profiles, min_stages=4, upper_threshold=1.1,
                       vram_budget_bytes=10**18)
    covered = []
    for s in plan.fwd_stages:
        covered.extend(range(s.lo, s.hi + 1))
    covered.extend(range(plan.fused_tail.lo, plan.fused_tail.hi + 1))
    for s in plan.bwd_stages:
        covered.extend(range(s.lo, s.hi + 1))
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/train/dispatch_pp/test_planner.py -k plan -v`
Expected: FAIL with `ImportError: cannot import name 'plan_stages'`

- [ ] **Step 3: Write minimal implementation**

```python
# append plan assembly to surogate/train/dispatch_pp/planner.py
def _stage_range(
    lo: int, hi: int, profiles: Sequence[BlockProfile], time_attr: str
) -> StageRange:
    weight_bytes = sum(profiles[i].weight_bytes for i in range(lo, hi + 1))
    est_time = sum(getattr(profiles[i], time_attr) for i in range(lo, hi + 1))
    needs_grad = any(profiles[i].needs_grad for i in range(lo, hi + 1))
    return StageRange(lo, hi, weight_bytes, est_time, needs_grad, numa_node=None)


def _fused_tail_lo(
    bwd: Sequence[float], bwd_sizes: Sequence[int], budget: float, mem_ceiling: int
) -> int:
    """First block of the fused tail.

    The fused tail runs fwd+loss+bwd in one shot, so its binding constraints are
    the BACKWARD cost (≈3x forward incl. recompute) and backward memory (weight +
    activations + grads), not the forward ones. Reverse-pack from the last block,
    absorbing trailing blocks while both budgets allow. The last block is always
    included (a stage is never empty), matching pack_stages' single-oversized-
    block rule.

    Sizing the tail by fwd_time would over-pack it ~3x and make it the makespan
    straggler, since it actually executes the backward too.
    """
    n = len(bwd)
    lo = n - 1
    acc_time = bwd[n - 1]
    acc_size = bwd_sizes[n - 1]
    for i in range(n - 2, -1, -1):
        if acc_time + bwd[i] > budget or acc_size + bwd_sizes[i] > mem_ceiling:
            break
        acc_time += bwd[i]
        acc_size += bwd_sizes[i]
        lo = i
    return lo


def _assemble(profiles, fwd, bwd, fwd_sizes, bwd_sizes, budget, mem_ceiling):
    """Return (fwd_stages, fused_tail_range, bwd_stages, total_count).

    Forward stages pack under forward memory (weight + activations); backward and
    fused-tail stages pack under backward memory (weight + activations + grads).
    The fused tail is reverse-packed by backward cost; the forward and backward
    prefixes [0, fused_lo) are then packed independently (asymmetric partitioning:
    forward packs more blocks/stage than backward at one budget).
    """
    fused_lo = _fused_tail_lo(bwd, bwd_sizes, budget, mem_ceiling)
    fused_hi = len(fwd) - 1
    fused = _stage_range(fused_lo, fused_hi, profiles, "bwd_time")
    if fused_lo == 0:
        forward: list[StageRange] = []
        backward: list[StageRange] = []
    else:
        fwd_pack = pack_stages(fwd[:fused_lo], fwd_sizes[:fused_lo], budget, mem_ceiling)
        forward = [_stage_range(lo, hi, profiles, "fwd_time") for (lo, hi) in fwd_pack]
        bwd_pack = pack_stages(bwd[:fused_lo], bwd_sizes[:fused_lo], budget, mem_ceiling)
        backward = [
            _stage_range(lo, hi, profiles, "bwd_time")
            for (lo, hi) in reversed(bwd_pack)
        ]
    total = len(forward) + 1 + len(backward)
    return forward, fused, backward, total


def plan_stages(
    profiles: Sequence[BlockProfile],
    min_stages: int,
    upper_threshold: float,
    vram_budget_bytes: int,
) -> StagePlan:
    """Pick the minimum-makespan stage partition via 1-D budget search.

    min_stages floors the stage count in the cost model (default = num local
    GPUs; raising it forces more, smaller stages to relieve memory pressure).
    """
    n = len(profiles)
    assert n > 0, "plan_stages requires at least one block"
    fwd = [p.fwd_time for p in profiles]
    bwd = [p.bwd_time for p in profiles]
    # Forward stage memory = weight + activation working set. Backward/fused
    # stage memory adds the gradient buffer, but only for trainable blocks
    # (frozen LoRA base blocks carry no grad -> needs_grad=False). Mirrors
    # RoundPipe's layer_fwd_size (param-only) vs layer_bwd_size (param + grad).
    # Deriving grad bytes from weight_bytes is *exact* for v1: full-FT blocks
    # have a bf16 grad == weight_bytes, and LoRA base is needs_grad=False
    # (adapters ride a separate path). No grad_bytes field until some mode has
    # trainable blocks whose on-GPU grad differs from weight_bytes.
    fwd_sizes = [p.weight_bytes + p.act_bytes for p in profiles]
    bwd_sizes = [
        s + (p.weight_bytes if p.needs_grad else 0)
        for s, p in zip(fwd_sizes, profiles)
    ]
    mem_ceiling = vram_budget_bytes // 2  # reserve half for double-buffering

    cands = sorted(
        set(candidate_budgets(fwd, upper_threshold))
        | set(candidate_budgets(bwd, upper_threshold))
    )
    if not cands:  # degenerate (e.g. all-zero times); use the largest single cost
        cands = [max(max(fwd), max(bwd), 1e-9)]

    best = None
    best_cost = float("inf")
    for budget in cands:
        forward, fused, backward, total = _assemble(
            profiles, fwd, bwd, fwd_sizes, bwd_sizes, budget, mem_ceiling
        )
        cost = max(total, min_stages) * budget
        if cost < best_cost:
            best_cost = cost
            best = (forward, fused, backward)

    forward, fused, backward = best
    return StagePlan(
        fwd_stages=forward,
        fused_tail=fused,
        bwd_stages=backward,
        num_blocks=n,
        warnings=[],
    )
```

Then promote `plan_stages()` to the package public API:

```python
# surogate/train/dispatch_pp/__init__.py
"""Dispatch pipeline-parallelism planner (pure Python, GPU-free)."""
from .planner import plan_stages
from .types import BlockProfile, StageRange, StagePlan

__all__ = ["BlockProfile", "StageRange", "StagePlan", "plan_stages"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/train/dispatch_pp/test_planner.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add surogate/train/dispatch_pp/planner.py tests/train/dispatch_pp/test_planner.py
git commit -m "feat(dispatch-pp): cost-search plan assembly (fwd/fused-tail/bwd)"
```

---

## Task 5: Oversized-block warning

**Files:**
- Modify: `surogate/train/dispatch_pp/planner.py`
- Test: `tests/train/dispatch_pp/test_planner.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/train/dispatch_pp/test_planner.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/train/dispatch_pp/test_planner.py -k oversize -v`
Expected: FAIL (no warning emitted)

- [ ] **Step 3: Write minimal implementation**

```python
# in plan_stages, after computing `bwd_sizes` and `mem_ceiling`, before the search:
    warnings: list[str] = []
    max_block_bytes = max(bwd_sizes)  # backward is the binding (largest) footprint
    if max_block_bytes > mem_ceiling:
        warnings.append(
            f"Largest block needs {max_block_bytes} bytes (weight+act+grad) but the "
            f"per-stage VRAM ceiling (vram_budget/2) is {mem_ceiling}; a single block "
            f"exceeds it. Reduce seq_len/micro_batch or raise vram_budget_gb, else OOM."
        )
```

```python
# and change the final return to thread warnings through:
    return StagePlan(
        fwd_stages=forward,
        fused_tail=fused,
        bwd_stages=backward,
        num_blocks=n,
        warnings=warnings,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/train/dispatch_pp/test_planner.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add surogate/train/dispatch_pp/planner.py tests/train/dispatch_pp/test_planner.py
git commit -m "feat(dispatch-pp): warn when a single block exceeds the per-stage VRAM ceiling"
```

---

## Task 6: PCIe token-threshold (operating-envelope) warning

**Files:**
- Modify: `surogate/train/dispatch_pp/planner.py`
- Test: `tests/train/dispatch_pp/test_planner.py`

Two complementary envelope checks (spec §6):
- **Primary — RoundPipe roofline (paper §3.3/Appendix C):** PCIe is hidden once microbatch count
  `B >= 8` (dense) / `B >= 80` (MoE). This is the authoritative, unit-clean guidance — warn directly on it.
- **Cross-check — hardware token-threshold:** full-FT crosses ~6 bytes/param/step (2 fwd-up + 2 bwd-up
  + 2 grad-down, bf16); to stay compute-bound `tokens_per_step >= (6/8) * gpu_flops / pcie_bw`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/train/dispatch_pp/test_planner.py
from surogate.train.dispatch_pp.planner import (
    token_threshold, microbatch_warning, envelope_warning,
)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/train/dispatch_pp/test_planner.py -k "threshold or envelope or microbatch" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# add to surogate/train/dispatch_pp/planner.py
def token_threshold(gpu_flops: float, pcie_bw: float) -> float:
    """Min tokens-per-optimizer-step to stay compute-bound (full-FT, ~6B/param/step)."""
    return 0.75 * gpu_flops / pcie_bw


def microbatch_warning(num_microbatches: int, is_moe: bool) -> str | None:
    """Primary roofline check (RoundPipe paper): PCIe hidden once B>=8 dense / B>=80 MoE."""
    floor = 80 if is_moe else 8
    if num_microbatches < floor:
        kind = "MoE" if is_moe else "dense"
        return (
            f"microbatches={num_microbatches} is below the RoundPipe roofline "
            f"(B>={floor} for {kind}); dispatch-PP will be transfer-bound. "
            f"Increase gradient_accumulation_steps."
        )
    return None


def envelope_warning(
    tokens_per_step: int, gpu_flops: float, pcie_bw: float
) -> str | None:
    """Hardware cross-check: warn if tokens-per-step can't amortize weight uploads."""
    thr = token_threshold(gpu_flops, pcie_bw)
    if tokens_per_step < thr:
        return (
            f"tokens_per_step={tokens_per_step} is below the ~{thr:.0f}-token "
            f"compute-bound threshold for this GPU/PCIe; dispatch-PP will be "
            f"transfer-bound. Increase gradient_accumulation_steps or batch/seq_len."
        )
    return None
```

These are standalone helpers; the train-mode glue (future plans) calls `microbatch_warning` (primary)
and `envelope_warning` (cross-check) with measured values and appends to `StagePlan.warnings`. They
live in the planner module so they are unit-tested without hardware.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/train/dispatch_pp/test_planner.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add surogate/train/dispatch_pp/planner.py tests/train/dispatch_pp/test_planner.py
git commit -m "feat(dispatch-pp): PCIe token-threshold operating-envelope warning"
```

---

## Task 7: LoRA `needs_grad` propagation + NUMA placement assignment

**Files:**
- Modify: `surogate/train/dispatch_pp/planner.py`
- Test: `tests/train/dispatch_pp/test_planner.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/train/dispatch_pp/test_planner.py
from surogate.train.dispatch_pp.planner import assign_numa


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/train/dispatch_pp/test_planner.py -k "numa or lora" -v`
Expected: FAIL — `needs_grad` test passes already (propagated in Task 4) but `assign_numa` import fails.

- [ ] **Step 3: Write minimal implementation**

```python
# add to surogate/train/dispatch_pp/planner.py
import dataclasses


def assign_numa(plan: StagePlan, num_numa_nodes: int) -> StagePlan:
    """Assign each stage a preferred NUMA node so the scheduler can place pinned
    weights and bias dispatch. Single-socket (num_numa_nodes<=1) -> None.

    Forward-path stages are spread round-robin by block order so consecutive
    stages land on alternating sockets, balancing host-memory bandwidth; the
    matching backward stage inherits its block range's node.
    """
    if num_numa_nodes <= 1:
        return plan  # numa_node already None on every StageRange

    fwd_path = list(plan.fwd_stages) + [plan.fused_tail]
    node_of_block: dict[int, int] = {}
    placed_fwd = []
    for idx, s in enumerate(fwd_path):
        node = idx % num_numa_nodes
        placed_fwd.append(dataclasses.replace(s, numa_node=node))
        for b in range(s.lo, s.hi + 1):
            node_of_block[b] = node

    placed_back = [
        dataclasses.replace(s, numa_node=node_of_block.get(s.lo, 0))
        for s in plan.bwd_stages
    ]
    return dataclasses.replace(
        plan,
        fwd_stages=placed_fwd[:-1],
        fused_tail=placed_fwd[-1],
        bwd_stages=placed_back,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/train/dispatch_pp/test_planner.py -v`
Expected: PASS (all). Export `assign_numa` from `__init__.py` (`from .planner import plan_stages, assign_numa`; add to `__all__`).

- [ ] **Step 5: Commit**

```bash
git add surogate/train/dispatch_pp/ tests/train/dispatch_pp/test_planner.py
git commit -m "feat(dispatch-pp): NUMA placement assignment + LoRA needs_grad propagation"
```

---

## Task 8: Config plumbing + validation in `SFTConfig`

**Files:**
- Modify: `surogate/core/config/sft_config.py`
- Test: `tests/train/dispatch_pp/test_config.py`

First read the current shape so the additions follow existing patterns:
Run: `rg -n "parallelism|self\.zero_level|self\.shard_weights|def __post_init__|raise ValueError|self\.ep_size|use_cuda_graphs|recipe|cpu_training" surogate/core/config/sft_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/train/dispatch_pp/test_config.py
import pytest
from surogate.core.config.sft_config import SFTConfig
from surogate.utils.dict import DictDefault

MODEL = "ro/train/out_opmix_wordform_s50_a010_eval"


def _cfg(data):
    cfg = SFTConfig(DictDefault(data))
    # SFT normally reaches __post_init__ through TokenizeDatasets; config tests
    # call it directly so validation/runtime_config creation is exercised.
    cfg.__post_init__()
    return cfg


def _base(tmp_path, **over):
    cfg = {
        "model": MODEL,
        "output_dir": str(tmp_path / "out"),
        "gpus": 1,
        "parallelism": "dispatch_pp",
        "dispatch_pp": {"min_stages": 4, "upper_threshold": 1.1},
    }
    cfg.update(over)
    return cfg


def test_dispatch_pp_parses_subblock(tmp_path):
    cfg = _cfg(_base(tmp_path))
    assert cfg.parallelism == "dispatch_pp"
    assert cfg.dispatch_pp["min_stages"] == 4


def test_default_parallelism_is_unset_and_path_unchanged(tmp_path):
    cfg = _cfg({"model": MODEL, "output_dir": str(tmp_path / "out"), "gpus": 1})
    assert cfg.parallelism in (None, "ddp")


def test_dispatch_pp_rejects_zero_sharding(tmp_path):
    with pytest.raises(ValueError, match="dispatch_pp.*ZeRO|zero_level"):
        _cfg(_base(tmp_path, zero_level=3, shard_weights=True))


def test_dispatch_pp_rejects_cpu_training_mode(tmp_path):
    with pytest.raises(ValueError, match="dispatch_pp.*cpu_training"):
        _cfg(_base(tmp_path, cpu_training=True))


def test_dispatch_pp_rejects_moe_ep(tmp_path):
    with pytest.raises(ValueError, match="dispatch_pp.*MoE|ep_size"):
        _cfg(_base(tmp_path, ep_size=2))


def test_dispatch_pp_rejects_fp8_recipe_in_v1(tmp_path):
    with pytest.raises(ValueError, match="dispatch_pp.*BF16|recipe"):
        _cfg(_base(tmp_path, recipe="fp8_hybrid"))


def test_dispatch_pp_disables_cuda_graphs(tmp_path):
    cfg = _cfg(_base(tmp_path, use_cuda_graphs=True))
    assert cfg.use_cuda_graphs is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/train/dispatch_pp/test_config.py -v`
Expected: FAIL — `parallelism`/`dispatch_pp` attributes don't exist.

- [ ] **Step 3: Write minimal implementation**

Add fields near the other parallelism flags (after `zero_level` / `ep_size`):

```python
    # Dispatch pipeline parallelism (opt-in, single-node model-parallel mode).
    # Unset / "ddp" -> existing data-parallel path, unchanged.
    parallelism: str | None = None
    dispatch_pp: dict | None = None
```

In `__init__`, parse the new fields immediately after `self.memcpy_send_recv = ...` so config files can
set them:

```python
        self.parallelism = cfg.get("parallelism", self.parallelism)
        self.dispatch_pp = dict(cfg.get("dispatch_pp", self.dispatch_pp) or {})
```

In `__post_init__`, add this block **before** `self._validate_ep_config()` so dispatch-PP's explicit
`ep_size>1` rejection wins over the generic "EP requires MoE" validation:

```python
        if self.parallelism in (None, "", "ddp"):
            self.parallelism = None
            self.dispatch_pp = {}
        elif self.parallelism == "dispatch_pp":
            self.dispatch_pp = dict(self.dispatch_pp or {})
            self.dispatch_pp.setdefault("min_stages", None)        # planner default = num_gpus
            self.dispatch_pp.setdefault("upper_threshold", 1.1)
            self.dispatch_pp.setdefault("vram_budget_gb", None)    # auto = free_vram * 0.9
            self.dispatch_pp.setdefault("recompute_grain", "stage")

            if self.cpu_training:
                raise ValueError(
                    "parallelism=dispatch_pp is a separate model-parallel training mode and "
                    "cannot be combined with cpu_training=true in v1."
                )
            if (self.zero_level and self.zero_level > 1) or self.shard_weights or self.shard_gradients:
                raise ValueError(
                    "parallelism=dispatch_pp is mutually exclusive with ZeRO sharding "
                    "(zero_level>1 / shard_weights / shard_gradients) in v1."
                )
            if getattr(self, "ep_size", 1) and self.ep_size > 1:
                raise ValueError(
                    "parallelism=dispatch_pp does not support MoE expert parallelism "
                    "(ep_size>1) in v1; this is a deferred phase."
                )
            if self.recipe in ("fp8-hybrid", "fp8_hybrid", "nvfp4", "nvfp4_quartet", "fp4"):
                raise ValueError(
                    "parallelism=dispatch_pp is BF16/LoRA-only in v1; FP8/NVFP4 stage "
                    f"streaming is deferred (got recipe={self.recipe!r})."
                )
            if self.use_cuda_graphs:
                logger.warning(
                    "[dispatch_pp]: disabling CUDA graphs (per-stage weight streaming "
                    "is incompatible with graph capture)."
                )
                self.use_cuda_graphs = False
        else:
            raise ValueError(f"Unknown parallelism={self.parallelism!r}; expected unset, 'ddp', or 'dispatch_pp'.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/train/dispatch_pp/test_config.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add surogate/core/config/sft_config.py tests/train/dispatch_pp/test_config.py
git commit -m "feat(dispatch-pp): add parallelism=dispatch_pp config + v1 validations"
```

---

## Task 9: Phase-0 feasibility gate — `GraphExecutor` sub-range execution

**This is the GATE.** It either proves the engine can run a contiguous block sub-range with
resident weights and CPU-boundary activations (matching whole-graph numerically), or surfaces the
exact blocker before any runtime is built. The path is scoped to debug-only entry points; production
scheduler/runtime work stays out of this plan.

This plan does **not** claim to prove streamed external stage weights unless the investigation finds an
existing hook that makes that cheap. Stage-owned external weights are the first gate of the later
stage-streaming plan. Phase 0 is still mandatory because the scheduler and streaming work are pointless
if the compiled executor cannot stop and restart cleanly at block boundaries.

**Files:**
- Read first: `csrc/src/runtime/executor/graph_executor.h`, `csrc/src/runtime/executor/graph_executor.cpp`, `csrc/src/runtime/executor/compiled_ops.h`, `csrc/src/runtime/executor/compiled_ops_execute_forward.cpp`, `csrc/src/runtime/executor/compiled_ops_execute_backward.cpp`, `csrc/src/runtime/dsl/graph_compiler.h`, `csrc/src/runtime/dsl/dsl_model.h`, `csrc/src/runtime/dsl/ir.h`, `csrc/src/binding/py_train.h`, `csrc/src/binding/binding.cpp`
- Modify if needed: `csrc/src/runtime/executor/compiled_ops.h`
- Modify if needed: `csrc/src/runtime/executor/compiled_ops_execute_forward.cpp`
- Modify if needed: `csrc/src/runtime/executor/compiled_ops_execute_backward.cpp`
- Modify if needed: `csrc/src/runtime/executor/graph_executor.h`
- Modify if needed: `csrc/src/runtime/executor/graph_executor.cpp`
- Create: `csrc/src/runtime/executor/dispatch_pp_phase0.h`
- Create: `csrc/src/runtime/executor/dispatch_pp_phase0.cpp`
- Modify: `csrc/CMakeLists.txt`
- Modify: `csrc/src/binding/py_train.h`
- Modify: `csrc/src/binding/py_train.cpp`
- Modify: `csrc/src/binding/binding.cpp`
- Test: `tests/train/dispatch_pp/test_phase0_subrange.py`

- [ ] **Step 1: Investigate the executor boundary (no code yet)**

Run:
```bash
sed -n '1,240p' csrc/src/runtime/executor/graph_executor.h
sed -n '520,840p' csrc/src/runtime/dsl/graph_compiler.h
rg -n "layer_start_indices|layer_end_indices|phase_tree|instruction_stream|execute_forward|execute_backward|replay_layer_forward" csrc/src/runtime/executor csrc/src/runtime/dsl
sed -n '1080,1460p' csrc/src/binding/binding.cpp
```
Write findings as a short comment block at the top of `csrc/src/runtime/executor/dispatch_pp_phase0.h`:
- Confirm the current executor evidence: `CompiledGraph` exposes `layer_start_indices`,
  `layer_end_indices`, `phase_tree`, and `instruction_stream`; `CompiledExecutor::execute_forward`
  already consumes the instruction stream when not capturing; backward has explicit layer boundary
  handling and recompute replay.
- Identify the narrowest additive implementation: prefer debug-only range methods on
  `CompiledExecutor` and/or `GraphExecutor` that dispatch existing compiled op ranges and reuse the
  existing layer boundary handlers, stack checkpoint/restore, saves, recompute, and runtime bindings.
- State whether weights for blocks `[i..j]` can be supplied from an external buffer today. If not,
  record "resident weights only in Phase 0; external stage weights deferred to stage-streaming gate."
- Confirm whether block boundaries are clean (no residual/norm op fused across the chosen boundary,
  and no cross-layer tensor lifetime assumption that requires whole-graph execution).

**Decision gate:** if block boundaries are NOT cleanly separable, STOP and report — record the precise obstruction (which op spans the boundary) in the test file as an xfail with the reason, and surface it to the spec's §7 open questions. Do not fabricate a runtime around an unconfirmed capability.

- [ ] **Step 2: Write the failing parity test (single GPU, no scheduler)**

```python
# tests/train/dispatch_pp/test_phase0_subrange.py
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sg = pytest.importorskip("surogate._surogate")

from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.utils.hf import get_model_weights_path
from tests.test_onboarding_qwen3 import (
    BATCH,
    NUM_LAYERS,
    SEQ_LEN,
    make_inputs,
    prepare_mini_model,
    resolve_model_path,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Phase-0 sub-range parity needs a GPU"
)


def _build_tiny_trainer():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3 weights not found. Set QWEN3_MODEL_PATH or cache Qwen/Qwen3-0.6B.")
    model_dir = prepare_mini_model(snapshot)

    cfg = sg.PretrainedConfig.from_pretrained(str(model_dir), "bf16")
    opts = sg.RuntimeOptions(
        offload_residual=False,
        use_cuda_graphs=False,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        shard_gradients=False,
        use_zero_copy=False,
    )
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))

    trainer = sg.SurogateTrainer(
        ngpu=1,
        config=cfg,
        options=opts,
        batch_size=BATCH,
        seq_len=SEQ_LEN,
        grad_accum=1,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
        lora_config=None,
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model_dir)))
    return trainer, model_dir


def test_subrange_forward_matches_whole_graph():
    """Running blocks [0..L) as two sub-ranges [0..k) then [k..L) must produce
    the same hidden state as a single whole-graph forward, within tolerance."""
    trainer, model_dir = _build_tiny_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    inputs = make_inputs(vocab_size)["inputs"]
    split_after = (NUM_LAYERS // 2) - 1

    whole = trainer.dispatch_pp_debug_forward_hidden(inputs)
    part = trainer.dispatch_pp_debug_forward_subranges(inputs, split_after)
    np.testing.assert_allclose(whole, part, rtol=1e-2, atol=1e-2)


def test_subrange_backward_grad_matches_whole_graph():
    """Sub-range backward must accumulate the same per-block weight grads as a
    single whole-graph backward, within tolerance."""
    trainer, model_dir = _build_tiny_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    batch = make_inputs(vocab_size)
    split_after = (NUM_LAYERS // 2) - 1

    g_whole = trainer.dispatch_pp_debug_grad_norms_whole(batch["inputs"], batch["targets"])
    g_part = trainer.dispatch_pp_debug_grad_norms_subranges(batch["inputs"], batch["targets"], split_after)
    np.testing.assert_allclose(g_whole, g_part, rtol=2e-2, atol=2e-2)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/train/dispatch_pp/test_phase0_subrange.py -v`
Expected: FAIL with `AttributeError: 'SurogateTrainer' object has no attribute 'dispatch_pp_debug_forward_hidden'`
or the equivalent missing debug binding.

- [ ] **Step 4: Implement the minimal sub-range execution path + debug hooks**

Implement in `csrc/src/runtime/executor/dispatch_pp_phase0.{h,cpp}` and expose thin bindings through
`MultiGPUPyTrainer`:

```cpp
// csrc/src/runtime/executor/dispatch_pp_phase0.h
#pragma once

#include <cstdint>
#include <vector>

class MultiGPUPyTrainer;

namespace dsl::dispatch_pp_phase0 {
std::vector<float> forward_hidden_whole(MultiGPUPyTrainer& trainer,
                                        const std::int32_t* inputs);
std::vector<float> forward_hidden_subranges(MultiGPUPyTrainer& trainer,
                                            const std::int32_t* inputs,
                                            int split_after_block);
std::vector<float> grad_norms_whole(MultiGPUPyTrainer& trainer,
                                    const std::int32_t* inputs,
                                    const std::int32_t* targets);
std::vector<float> grad_norms_subranges(MultiGPUPyTrainer& trainer,
                                        const std::int32_t* inputs,
                                        const std::int32_t* targets,
                                        int split_after_block);
}  // namespace dsl::dispatch_pp_phase0
```

Add matching `MultiGPUPyTrainer` methods in `csrc/src/binding/py_train.h/.cpp`:

```cpp
std::vector<float> dispatch_pp_debug_forward_hidden(const std::int32_t* inputs);
std::vector<float> dispatch_pp_debug_forward_subranges(const std::int32_t* inputs, int split_after_block);
std::vector<float> dispatch_pp_debug_grad_norms_whole(const std::int32_t* inputs,
                                                      const std::int32_t* targets);
std::vector<float> dispatch_pp_debug_grad_norms_subranges(const std::int32_t* inputs,
                                                         const std::int32_t* targets,
                                                         int split_after_block);
```

Expose those four methods in `csrc/src/binding/binding.cpp` with nanobind `TokenArray` lambdas
(matching the existing `step`/`validate` wrappers, including shape checks) rather than binding raw
pointer signatures directly. Return owned NumPy float32 arrays from the returned `std::vector<float>`.
Add `src/runtime/executor/dispatch_pp_phase0.cpp` to the `surogate-common` `target_sources` list in
`csrc/CMakeLists.txt`.

The implementation requirements are:
- Add the smallest debug-only executor primitive needed to reuse existing compiled execution:
  an inclusive block range `[i..j]` maps to the op span
  `[layer_start_indices[i], layer_end_indices[j])` — **start-inclusive, end-exclusive** in op-index
  space (per graph_compiler.h: `ops[layer_end_indices[L]-1]` is the last op of layer L), so mind the
  off-by-one at the boundary. The phase `instruction_stream` is the preferred source of layer-boundary
  enter/exit semantics when it is available. The range runner must reuse the current layer-boundary
  handlers, stack
  checkpoint/restore, save-list persistence, runtime bindings, recompute, and last-use pruning.
- Do **not** manually dispatch kernels from `dispatch_pp_phase0.cpp`; it should call the existing
  compiled executor / graph executor machinery. If a private member blocks this, expose a narrow
  debug wrapper in `GraphExecutor` instead of making production state public.
- `forward_hidden_whole` runs the current whole-graph forward and returns the final hidden-state buffer
  flattened as `float32`.
- `forward_hidden_subranges` runs `[0..split_after_block]` then `[split_after_block+1..last_block]`
  using the new sub-range executor path and returns the same hidden-state buffer flattened as `float32`.
  At the split, force the boundary hidden state through a CPU round-trip (D2H into pinned or ordinary
  host memory, then H2D into the second range input) so the test proves the dispatch-PP rendezvous
  shape instead of merely proving "two calls on one GPU."
- `grad_norms_whole` runs the current whole-graph backward and returns deterministic per-block grad
  norms in block order.
- `grad_norms_subranges` runs the sub-range backward path and returns grad norms in the same order.
  It should execute the higher block range first, CPU-round-trip the boundary grad, then execute the
  lower block range, matching the dependency direction a future scheduler will enforce.

Keep it single-GPU, weights resident — this isolates "can the executor run a sub-range correctly" from
"can we stream weights" (the later stage-streaming plan). Reuse existing op-dispatch; do not rewrite
kernels.

- [ ] **Step 5: Build and run the test to verify it passes**

Run:
```bash
make build && pytest tests/train/dispatch_pp/test_phase0_subrange.py -v
```
Expected: PASS (forward within rtol 1e-2; grad norms within rtol 2e-2).

If it cannot be made to pass cleanly, record the exact obstruction and STOP — the gate has failed and the runtime design (Phases 1–3) must be revisited per spec §7 risk #1.

- [ ] **Step 6: Commit**

```bash
git add csrc/src/runtime/executor/dispatch_pp_phase0.h \
        csrc/src/runtime/executor/dispatch_pp_phase0.cpp \
        csrc/src/runtime/executor/compiled_ops.h \
        csrc/src/runtime/executor/compiled_ops_execute_forward.cpp \
        csrc/src/runtime/executor/compiled_ops_execute_backward.cpp \
        csrc/src/runtime/executor/graph_executor.h csrc/src/runtime/executor/graph_executor.cpp \
        csrc/CMakeLists.txt csrc/src/binding/py_train.h csrc/src/binding/py_train.cpp \
        csrc/src/binding/binding.cpp tests/train/dispatch_pp/test_phase0_subrange.py
git commit -m "feat(dispatch-pp): Phase-0 GraphExecutor sub-range execution + parity gate"
```

---

## Task 10: Full-suite run + record Phase-0 outcome

**Files:**
- Modify: `2026-06-20-dispatch-pp-design.md` (update §7 open question with the Phase-0 finding)

- [ ] **Step 1: Run the planner + config suites**

Run: `pytest tests/train/dispatch_pp/test_planner.py tests/train/dispatch_pp/test_config.py -v`
Expected: PASS (all)

- [ ] **Step 2: Run the existing engine regression to prove additivity**

Run: `pytest tests/ -k "config or trainer" -q`
Expected: PASS — existing behavior unchanged with `parallelism` unset.

- [ ] **Step 3: Record the Phase-0 verdict in the spec**

Edit the spec's §7 open question "Does the IR cleanly expose contiguous block sub-ranges…" with the concrete finding (works / works-with-caveat / blocked + reason). This decides whether the Phase 1–3 plans can be authored.

- [ ] **Step 4: Commit**

```bash
git add 2026-06-20-dispatch-pp-design.md
git commit -m "docs(dispatch-pp): record Phase-0 sub-range feasibility verdict"
```

---

## Out of scope (subsequent plans, authored after Phase 0 lands)

- **Plan 2 — DispatchScheduler (C++):** stateless GPU worker pool, round-robin + NUMA-biased dispatch, stage dependency edges + CUDA-event handoff, cross-stream buffer lifetime (spec §4).
- **Plan 3 — Stage streaming (C++):** lift `DslWeightManager` per-layer gather/release to scheduler-owned stage ranges; chunked gap-filling uploads (spec §2 component 5).
- **Plan 4 — Async 1-step-stale optimizer (C++):** overlapped optimizer thread on the CPU-master path, `param_copied`/`grad_copied` fences, staleness-isolated correctness test (spec §4, §7 Phase 3).

Each depends on Phase 0's verdict and on reading the engine internals confirmed in Task 9.
