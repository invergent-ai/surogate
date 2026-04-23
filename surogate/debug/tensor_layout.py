"""`surogate debug tensor-layout` — per-tid layout + region assignment dump.

Emits one TENSOR_LAYOUT record per (graph, tid) pair, covering both the
compiled forward and backward graphs. Fields mirror `TensorMeta` + the
computed offset/bytes produced by `compute_layout()`:

    {tag: TENSOR_LAYOUT, graph, tid, name, kind, region, block_layer_idx,
     offset, bytes, offset_assigned, retain_through_forward, base_param_tid,
     base_producer_tid, base_grad_tid, is_blocks, is_d_blocks,
     is_cross_layer, is_moe_offsets, is_moe_gather}

Grep/filter examples::

    jq 'select(.region=="FwdStack")'                debug_tensor-layout_*.jsonl
    jq 'select(.block_layer_idx==3 and .graph=="forward")'  debug_*.jsonl
    jq 'select(.retain_through_forward==true)'      debug_*.jsonl

This is the structured form of the `SUROGATE_DEBUG_TID_TABLE` /
`SUROGATE_DEBUG_LAYOUT` / `SUROGATE_DEBUG_REGIONS` stderr dumps.
"""

from __future__ import annotations

from typing import Any

from surogate.utils.logger import get_logger

from ._shared import DebugResolveError, resolve_model_and_ir
from .schema import Severity, Tag
from .tensor_debug_shared import build_introspection_trainer, write_run_and_model_records
from .writer import DebugJsonlWriter, default_output_path, make_run_id

logger = get_logger()


def run_tensor_layout(
    config_path: str,
    *,
    output: str | None = None,
    hub_token: str | None = None,
) -> int:
    try:
        resolved = resolve_model_and_ir(config_path, hub_token=hub_token)
    except DebugResolveError as e:
        logger.error(f"tensor-layout: {e}")
        return 2

    run_id = make_run_id()
    out_path = output or default_output_path("tensor-layout", resolved.model_id)
    header = {
        "subcommand": "tensor-layout",
        "config_path": config_path,
        "model_id": resolved.model_id,
        "architecture": resolved.architecture,
    }

    with DebugJsonlWriter(out_path, run_id, header) as w:
        write_run_and_model_records(w, resolved, "tensor-layout")
        try:
            trainer = build_introspection_trainer(resolved)
        except Exception as e:
            logger.error(f"tensor-layout: failed to build introspection trainer: {type(e).__name__}: {e}")
            w.write(Tag.ERROR, severity=Severity.ERROR, stage="build_trainer", error=str(e))
            return 1

        try:
            entries = trainer.get_debug_tensor_layout()
        except Exception as e:
            logger.error(f"tensor-layout: get_debug_tensor_layout failed: {type(e).__name__}: {e}")
            w.write(Tag.ERROR, severity=Severity.ERROR, stage="get_layout", error=str(e))
            return 1

        fwd_count = 0
        bwd_count = 0
        unassigned_count = 0
        retain_count = 0
        for e in entries:
            w.write(Tag.TENSOR_LAYOUT, **e)
            if e["graph"] == "forward":
                fwd_count += 1
            elif e["graph"] == "backward":
                bwd_count += 1
            if not e["offset_assigned"]:
                unassigned_count += 1
            if e["retain_through_forward"]:
                retain_count += 1

        w.summary(
            tensors_total=len(entries),
            forward_tids=fwd_count,
            backward_tids=bwd_count,
            unassigned_offsets=unassigned_count,
            retain_through_forward_tids=retain_count,
        )
    logger.info(f"tensor-layout wrote {len(entries)} records to {out_path}")
    return 0
