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

    Stages must be ALIGNED uniform blocks of `sb` ([0..sb-1], [sb..2sb-1], ...):
    the recompute:false activation arena is colored cyclically (layer L ->
    section L % sb), which is only correct if every stage starts at a multiple of
    sb (so L % sb == L - lo within a stage). The last stage may be shorter.

    Returns (los, his, n_layers, num_stages, max_stage_blocks, stage_blocks).
    """
    n_layers = resolve_num_layers(model_config)
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
