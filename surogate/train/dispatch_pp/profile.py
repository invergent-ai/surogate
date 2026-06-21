"""Derive dispatch-PP planner inputs from a real model (pure Python, GPU-free).

Bridges the abstract planner (``plan_stages``) to an actual checkpoint: per-block
work-weight bytes are read straight from the safetensors header(s) (no tensor load),
the activation working-set is estimated from the model dims and the runtime shape,
and per-block fwd/bwd times use the size-proportional fallback the design specifies
for the first iterations (recalibrated from measured timings later).
"""

from __future__ import annotations

import json
import re
import struct
from collections import defaultdict
from pathlib import Path

from .planner import assign_numa, envelope_warning, microbatch_warning, plan_stages
from .types import BlockProfile, StagePlan

# safetensors dtype -> bytes per element.
_DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}

# Matches a transformer-block parameter name, capturing the block index, e.g.
# "model.layers.7.mlp.up_proj.weight" or "layers.7.self_attn.q_proj.weight".
_BLOCK_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _read_safetensors_header(path: Path) -> dict:
    """Return the JSON header of a .safetensors file without loading tensors."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(header_len))


def _tensor_bytes(meta: dict) -> int:
    n = 1
    for s in meta["shape"]:
        n *= s
    return n * _DTYPE_BYTES.get(meta["dtype"], 2)


def block_weight_bytes(weights_path: str | Path) -> dict[int, int]:
    """Map transformer-block index -> total work-weight bytes, from safetensors.

    Accepts either a single ``*.safetensors`` file or a ``*.safetensors.index.json``
    shard index (sums across shards). Non-block params (embeddings, lm_head, final
    norm) are excluded — the planner partitions transformer blocks.
    """
    weights_path = Path(weights_path)
    if weights_path.suffix == ".json":  # sharded: *.safetensors.index.json
        index = json.loads(weights_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
        headers = [_read_safetensors_header(weights_path.parent / s) for s in shards]
    else:
        headers = [_read_safetensors_header(weights_path)]

    per_block: dict[int, int] = defaultdict(int)
    for header in headers:
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            m = _BLOCK_RE.search(name)
            if m:
                per_block[int(m.group(1))] += _tensor_bytes(meta)
    return dict(per_block)


def activation_bytes(
    hidden_size: int, intermediate_size: int, seq_len: int, micro_batch: int, dtype_bytes: int = 2
) -> int:
    """Estimate a block's activation working-set (bytes) at the given runtime shape.

    Heuristic over-estimate of what is resident in VRAM while a block computes,
    beyond its weights: the residual stream plus attention temporaries (~4*hidden)
    and the MLP intermediate (SwiGLU, ~intermediate_size). Used for the per-stage
    memory ceiling and the oversized-block warning, so erring high is safe.
    """
    per_token = 4 * hidden_size + intermediate_size
    return micro_batch * seq_len * per_token * dtype_bytes


def build_block_profiles(
    model_dir: str | Path,
    weights_path: str | Path,
    *,
    seq_len: int,
    micro_batch: int,
    needs_grad: bool = True,
    bwd_fwd_ratio: float = 3.0,
    dtype_bytes: int = 2,
) -> list[BlockProfile]:
    """Build one ``BlockProfile`` per transformer block from a real checkpoint.

    Weight bytes come from the safetensors header; activation bytes from the model
    dims + runtime shape; fwd time is size-proportional (relative units — only
    ratios matter to the planner) and bwd time is ``bwd_fwd_ratio``x fwd (backward
    incl. recompute), giving the asymmetric forward/backward partition. Replace the
    size-proportional times with measured per-block timings once available.
    """
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    hidden_size = int(cfg["hidden_size"])
    intermediate_size = int(cfg.get("intermediate_size", 4 * hidden_size))

    per_block = block_weight_bytes(weights_path)
    if not per_block:
        raise ValueError(f"no transformer-block weights found in {weights_path}")
    n_blocks = max(per_block) + 1
    if sorted(per_block) != list(range(n_blocks)):
        raise ValueError(f"non-contiguous block indices in {weights_path}: {sorted(per_block)}")

    act = activation_bytes(hidden_size, intermediate_size, seq_len, micro_batch, dtype_bytes)
    profiles: list[BlockProfile] = []
    for i in range(n_blocks):
        wbytes = per_block[i]
        fwd_time = wbytes / 1.0e6  # relative units (MB-proportional); ratios are what matter
        profiles.append(
            BlockProfile(
                fwd_time=fwd_time,
                bwd_time=fwd_time * bwd_fwd_ratio,
                weight_bytes=wbytes,
                act_bytes=act,
                needs_grad=needs_grad,
            )
        )
    return profiles


def plan_for_model(
    model_dir: str | Path,
    weights_path: str | Path,
    *,
    min_stages: int,
    seq_len: int,
    micro_batch: int,
    vram_budget_bytes: int,
    upper_threshold: float = 1.1,
    num_numa_nodes: int = 1,
    num_microbatches: int | None = None,
    is_moe: bool = False,
    gpu_flops: float | None = None,
    pcie_bw: float | None = None,
    needs_grad: bool = True,
) -> StagePlan:
    """Produce a NUMA-placed ``StagePlan`` for a real model + runtime configuration.

    Builds per-block profiles, runs the cost-search partition, assigns NUMA nodes,
    and appends operating-envelope warnings (microbatch roofline + optional PCIe
    token-threshold cross-check) to ``StagePlan.warnings``.
    """
    profiles = build_block_profiles(
        model_dir, weights_path, seq_len=seq_len, micro_batch=micro_batch, needs_grad=needs_grad
    )
    plan = plan_stages(profiles, min_stages, upper_threshold, vram_budget_bytes)
    plan = assign_numa(plan, num_numa_nodes)

    warnings = list(plan.warnings)
    if num_microbatches is not None:
        w = microbatch_warning(num_microbatches, is_moe)
        if w:
            warnings.append(w)
    if gpu_flops is not None and pcie_bw is not None:
        tokens_per_step = micro_batch * seq_len * (num_microbatches or 1)
        w = envelope_warning(tokens_per_step, gpu_flops, pcie_bw)
        if w:
            warnings.append(w)

    if warnings != list(plan.warnings):
        import dataclasses

        plan = dataclasses.replace(plan, warnings=warnings)
    return plan


def resolve_vram_budget_bytes(vram_budget_gb: float | None, free_vram_bytes: int) -> int:
    """Resolve the planner VRAM budget: explicit ``vram_budget_gb`` or auto = 90% of free.

    ``free_vram_bytes`` is measured at trainer startup (e.g. ``GPUUtilInfo.mem_free``).
    """
    if vram_budget_gb is not None:
        return int(vram_budget_gb * (1024**3))
    return int(free_vram_bytes * 0.9)
