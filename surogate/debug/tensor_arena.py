"""`surogate debug tensor-arena` — arena-size + region-coverage dump.

Emits:
  - One TENSOR_ARENA record per graph (forward/backward) with region peaks,
    layout hash, tid coverage, op-operand coverage.
  - One TENSOR_REGION record per (graph, region) pair with tid count + byte total.
  - One SUMMARY record aggregating arena totals and the `PhaseArenas` cudaMalloc'd
    sizes (what the runtime actually holds).

This is the structured form of `SUROGATE_DEBUG_ARENA_COVERAGE` +
`SUROGATE_DEBUG_OPERAND_COVERAGE` + `SUROGATE_DEBUG_LAYOUT` stderr dumps.
"""

from __future__ import annotations

from typing import Any

from surogate.utils.logger import get_logger

from ._shared import DebugResolveError, resolve_model_and_ir
from .schema import Severity, Tag
from .tensor_debug_shared import build_introspection_trainer, write_run_and_model_records
from .writer import DebugJsonlWriter, default_output_path, make_run_id

logger = get_logger()


def _emit_graph_records(w: Any, graph_dict: dict) -> None:
    regions = graph_dict.pop("regions", [])
    w.write(Tag.TENSOR_ARENA, **graph_dict)
    for r in regions:
        w.write(
            Tag.TENSOR_REGION,
            graph=graph_dict["graph"],
            region=r["region"],
            tid_count=r["tid_count"],
            tid_bytes=r["tid_bytes"],
        )


def run_tensor_arena(
    config_path: str,
    *,
    output: str | None = None,
    hub_token: str | None = None,
) -> int:
    try:
        resolved = resolve_model_and_ir(config_path, hub_token=hub_token)
    except DebugResolveError as e:
        logger.error(f"tensor-arena: {e}")
        return 2

    run_id = make_run_id()
    out_path = output or default_output_path("tensor-arena", resolved.model_id)
    header = {
        "subcommand": "tensor-arena",
        "config_path": config_path,
        "model_id": resolved.model_id,
        "architecture": resolved.architecture,
    }

    with DebugJsonlWriter(out_path, run_id, header) as w:
        write_run_and_model_records(w, resolved, "tensor-arena")
        try:
            trainer = build_introspection_trainer(resolved)
        except Exception as e:
            logger.error(f"tensor-arena: failed to build introspection trainer: {type(e).__name__}: {e}")
            w.write(Tag.ERROR, severity=Severity.ERROR, stage="build_trainer", error=str(e))
            return 1

        try:
            summary = trainer.get_debug_arena_summary()
        except Exception as e:
            logger.error(f"tensor-arena: get_debug_arena_summary failed: {type(e).__name__}: {e}")
            w.write(Tag.ERROR, severity=Severity.ERROR, stage="get_arena", error=str(e))
            return 1

        _emit_graph_records(w, dict(summary["forward"]))
        _emit_graph_records(w, dict(summary["backward"]))

        arena_totals = {
            "arenas_allocated": summary["arenas_allocated"],
            "arena_persistent_bytes": summary["arena_persistent_bytes"],
            "arena_persistent_activation_bytes": summary["arena_persistent_activation_bytes"],
            "arena_accumulator_bytes": summary["arena_accumulator_bytes"],
            "arena_fwd_stack_bytes": summary["arena_fwd_stack_bytes"],
            "arena_bwd_stack_bytes": summary["arena_bwd_stack_bytes"],
            "arena_save_for_bwd_bytes": summary["arena_save_for_bwd_bytes"],
            "arena_unified_stack_bytes": summary["arena_unified_stack_bytes"],
            "arena_bwd_cross_layer_bytes": summary["arena_bwd_cross_layer_bytes"],
            "arena_moe_saved_bytes": summary["arena_moe_saved_bytes"],
            "arena_save_for_bwd_block_bases": list(summary["arena_save_for_bwd_block_bases"]),
        }
        w.summary(**arena_totals)
    logger.info(f"tensor-arena wrote {out_path}")
    return 0
