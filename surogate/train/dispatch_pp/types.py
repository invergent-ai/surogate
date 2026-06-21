"""Data types for the dispatch-PP planner.

These are plain, frozen dataclasses over primitive data. ``StagePlan`` is the
single cross-language contract handed Python -> C++ once per training run, so its
``to_dict()`` must stay JSON-serializable without custom encoders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BlockProfile:
    """Per-block profile fed to the planner. One entry per DSL transformer block."""

    fwd_time: float          # forward wall-time estimate (seconds)
    bwd_time: float          # backward incl. recompute wall-time estimate (seconds)
    weight_bytes: int        # work-weight size (bf16)
    act_bytes: int           # activation/recompute working-set at current seq_len x micro_batch
    needs_grad: bool = True  # False for frozen LoRA base blocks


@dataclass(frozen=True)
class StageRange:
    """A contiguous, inclusive block range [lo, hi] forming one scheduling unit."""

    lo: int
    hi: int
    weight_bytes: int
    est_time: float
    needs_grad: bool
    numa_node: int | None    # preferred NUMA placement of this stage's CPU weights; None = single-socket

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StagePlan:
    """The cross-language contract handed to the C++ scheduler once per run."""

    fwd_stages: list[StageRange]   # pure-forward stages, ascending
    fused_tail: StageRange         # trailing range run as fwd+loss+bwd
    bwd_stages: list[StageRange]   # descending execution order
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
