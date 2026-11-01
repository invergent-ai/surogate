"""Shared dispatch-PP stage planning.

Both the SFT trainer and the GRPO trainer need the same layer->stage partition,
so it lives here rather than as a method on either one.
"""

from __future__ import annotations

import os


def resolve_num_layers(model_config) -> int:
    """Number of transformer blocks, looking through a text_config wrapper."""
    n_layers = getattr(model_config, "num_hidden_layers", 0) or getattr(
        getattr(model_config, "text_config", None), "num_hidden_layers", 0
    )
    if not n_layers:
        raise RuntimeError("dispatch_pp: could not determine num_hidden_layers from model config")
    return int(n_layers)


def plan_stages(model_config, gpus: int) -> tuple[list[int], list[int], int, int, int, int]:
    """Partition layers into SMALL stages for dispatch-PP.

    Unlike a one-stage-per-GPU split (big stages that can't be held resident),
    aim for ~SUROGATE_DISPATCH_STAGE_BLOCKS layers per stage (default 4) so 2x a
    stage fits the VRAM budget and a stage stays cached across its microbatches.
    Always make at least `gpus` stages so every device is used.

    The activation arenas are colored cyclically (layer L -> section L % sb), so
    no stage may hold more than sb layers: then the layers of one stage get
    distinct sections. Uniform stages are aligned blocks of sb; the last may be
    shorter.

    SUROGATE_DISPATCH_STAGE_PLAN (comma-separated stage sizes, e.g. "6,5,5,5,5,5,5,4")
    gives the stages explicitly, for instance a shorter last stage beside the
    lm-head; sb is then its largest stage. The native side reads the same
    variable for stage-local LoRA.

    Returns (los, his, n_layers, num_stages, max_stage_blocks, stage_blocks).
    """
    n_layers = resolve_num_layers(model_config)
    plan = os.environ.get("SUROGATE_DISPATCH_STAGE_PLAN", "").strip()
    if plan:
        sizes = [int(x) for x in plan.split(",") if x.strip()]
        if not sizes or min(sizes) <= 0 or sum(sizes) != n_layers:
            raise ValueError(
                f"SUROGATE_DISPATCH_STAGE_PLAN={plan!r}: expected positive stage sizes summing to {n_layers} layers"
            )
        if len(sizes) < gpus:
            raise ValueError(f"SUROGATE_DISPATCH_STAGE_PLAN={plan!r}: {len(sizes)} stages leave GPUs idle ({gpus} GPUs)")
        los, his, lo = [], [], 0
        for size in sizes:
            los.append(lo)
            his.append(lo + size - 1)
            lo += size
        return los, his, n_layers, len(sizes), max(sizes), max(sizes)
    sb = max(1, int(os.environ.get("SUROGATE_DISPATCH_STAGE_BLOCKS", "4")))
    # Shrink sb if needed so there are at least `gpus` stages (every device used).
    if (n_layers + sb - 1) // sb < gpus:
        sb = max(1, n_layers // gpus)

    los: list[int] = []
    his: list[int] = []
    lo = 0
    while lo < n_layers:
        hi = min(lo + sb, n_layers) - 1
        los.append(lo)
        his.append(hi)
        lo = hi + 1

    nst = len(los)
    max_stage = max(h - l + 1 for l, h in zip(los, his))
    return los, his, n_layers, nst, max_stage, sb


def layers_per_gpu(los: list[int], his: list[int], gpus: int) -> int:
    """Most layers any GPU runs (stage s runs on GPU s % gpus): the resident weight window has to hold all of a GPU's
    stages, which with more stages than GPUs alternate on it unit by unit."""
    per_gpu = [0] * max(1, min(gpus, len(los)))
    for s, (lo, hi) in enumerate(zip(los, his)):
        per_gpu[s % len(per_gpu)] += hi - lo + 1
    return max(per_gpu)
