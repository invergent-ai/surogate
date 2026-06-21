"""Dispatch-PP stage planner.

Pure logic over plain data — no CUDA, no engine import, fully unit-testable. The
planner turns per-block profiles into a :class:`StagePlan`: an asymmetric
forward/backward partition (forward packs more blocks per stage than backward,
which costs ~3x more incl. recompute) chosen to minimize a pipeline-makespan
proxy under a per-stage VRAM ceiling.
"""

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
    per-stage time budget (``max_workload``) OR the per-stage memory ceiling
    (``mem_ceiling``). A single block that exceeds the ceiling still occupies its
    own stage (blocks are indivisible in v1).

    Returns inclusive ``(lo, hi)`` ranges covering ``[0, len(costs))`` exactly once.
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


def candidate_budgets(costs: Sequence[float], upper_threshold: float) -> list[float]:
    """Enumerate candidate per-stage time budgets.

    A candidate is any sum of consecutive block costs that lands in
    ``[max_single_block, max_single_block * upper_threshold]``. A stage can never
    be smaller than the slowest single block, and never larger than the slack
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


def token_threshold(gpu_flops: float, pcie_bw: float) -> float:
    """Min tokens-per-optimizer-step to stay compute-bound (full-FT, ~6 B/param/step).

    Hardware cross-check derived from the transfer/compute roofline: full-FT moves
    ~6 bytes/param/step over PCIe (2 fwd-up + 2 bwd-up + 2 grad-down, bf16) while
    computing ~8 FLOPs/param/token, so transfer is hidden once
    ``tokens_per_step >= (6/8) * gpu_flops / pcie_bw``.
    """
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
    the BACKWARD cost (~3x forward incl. recompute) and backward memory (weight +
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


def _assemble(
    profiles: Sequence[BlockProfile],
    fwd: Sequence[float],
    bwd: Sequence[float],
    fwd_sizes: Sequence[int],
    bwd_sizes: Sequence[int],
    budget: float,
    mem_ceiling: int,
) -> tuple[list[StageRange], StageRange, list[StageRange], int]:
    """Return ``(fwd_stages, fused_tail, bwd_stages, total_stage_count)``.

    Forward stages pack under forward memory (weight + activations); backward and
    fused-tail stages pack under backward memory (weight + activations + grads).
    The fused tail is reverse-packed by backward cost; the forward and backward
    prefixes ``[0, fused_lo)`` are then packed independently (asymmetric
    partitioning: forward packs more blocks/stage than backward at one budget).
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

    ``min_stages`` floors the stage count in the cost model (default = num local
    GPUs; raising it forces more, smaller stages to relieve memory pressure).
    """
    n = len(profiles)
    if n == 0:
        raise ValueError("plan_stages requires at least one block")
    fwd = [p.fwd_time for p in profiles]
    bwd = [p.bwd_time for p in profiles]
    # Forward stage memory = weight + activation working set. Backward/fused
    # stage memory adds the gradient buffer, but only for trainable blocks
    # (frozen LoRA base blocks carry no grad -> needs_grad=False). Mirrors
    # RoundPipe's layer_fwd_size (param-only) vs layer_bwd_size (param + grad).
    # Deriving grad bytes from weight_bytes is exact for v1: full-FT blocks have
    # a bf16 grad == weight_bytes, and LoRA base is needs_grad=False (adapters
    # ride a separate path). No grad_bytes field until some mode has trainable
    # blocks whose on-GPU grad differs from weight_bytes.
    fwd_sizes = [p.weight_bytes + p.act_bytes for p in profiles]
    bwd_sizes = [
        s + (p.weight_bytes if p.needs_grad else 0)
        for s, p in zip(fwd_sizes, profiles)
    ]
    mem_ceiling = vram_budget_bytes // 2  # reserve half for double-buffering

    warnings: list[str] = []
    max_block_bytes = max(bwd_sizes)  # backward is the binding (largest) footprint
    if max_block_bytes > mem_ceiling:
        warnings.append(
            f"Largest block needs {max_block_bytes} bytes (weight+act+grad) but the "
            f"per-stage VRAM ceiling (vram_budget/2) is {mem_ceiling}; a single block "
            f"exceeds it. Reduce seq_len/micro_batch or raise vram_budget_gb, else OOM."
        )

    cands = sorted(
        set(candidate_budgets(fwd, upper_threshold))
        | set(candidate_budgets(bwd, upper_threshold))
    )
    if not cands:  # degenerate (e.g. all-zero times); use the largest single cost
        cands = [max(max(fwd), max(bwd), 1e-9)]

    best: tuple[list[StageRange], StageRange, list[StageRange]] | None = None
    best_cost = float("inf")
    for budget in cands:
        forward, fused, backward, total = _assemble(
            profiles, fwd, bwd, fwd_sizes, bwd_sizes, budget, mem_ceiling
        )
        cost = max(total, min_stages) * budget
        if cost < best_cost:
            best_cost = cost
            best = (forward, fused, backward)

    assert best is not None  # cands is always non-empty, so the loop selects a plan
    forward, fused, backward = best
    return StagePlan(
        fwd_stages=forward,
        fused_tail=fused,
        bwd_stages=backward,
        num_blocks=n,
        warnings=warnings,
    )
