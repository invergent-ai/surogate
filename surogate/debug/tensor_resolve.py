"""`surogate debug tensor-resolve` — single-tensor provenance lookup.

Given a tensor name (or --tid), dumps the full provenance record:
  - graph (forward / backward)
  - TensorMeta fields (region, block_layer_idx, offset, bytes, kind, etc.)
  - Phase path (root phase → deepest enclosing phase)
  - first_write_op / last_use_op (within that graph's ops[])
  - CompiledGraph::describe_tensor_id() human string

If neither --graph forward nor --graph backward is specified, the tool
resolves in both graphs and emits one TENSOR_RESOLVE record per hit.
"""

from __future__ import annotations

from typing import Any

from surogate.utils.logger import get_logger

from ._shared import DebugResolveError, resolve_model_and_ir
from .schema import Severity, Tag
from .tensor_debug_shared import build_introspection_trainer, write_run_and_model_records
from .writer import DebugJsonlWriter, default_output_path, make_run_id

logger = get_logger()


def _emit_resolution(w: Any, res: dict) -> None:
    if not res["found"]:
        w.write(
            Tag.TENSOR_RESOLVE,
            severity=Severity.WARN,
            found=False,
            graph=res["graph"],
        )
        return
    entry = res["entry"]
    record = {
        "found": True,
        "graph": res["graph"],
        "description": res["description"],
        "first_write_op": res["first_write_op"],
        "last_use_op": res["last_use_op"],
        "phase_path": list(res["phase_path"]),
        **entry,
    }
    w.write(Tag.TENSOR_RESOLVE, **record)


def run_tensor_resolve(
    config_path: str,
    *,
    output: str | None = None,
    hub_token: str | None = None,
    name: str | None = None,
    tid: int = -1,
    graph: str | None = None,
) -> int:
    if (name is None or not name.strip()) and tid < 0:
        logger.error("tensor-resolve: must specify --name or --tid")
        return 2
    if graph not in (None, "forward", "backward"):
        logger.error(f"tensor-resolve: invalid --graph {graph!r} (expected forward/backward)")
        return 2

    try:
        resolved = resolve_model_and_ir(config_path, hub_token=hub_token)
    except DebugResolveError as e:
        logger.error(f"tensor-resolve: {e}")
        return 2

    run_id = make_run_id()
    out_path = output or default_output_path("tensor-resolve", resolved.model_id)
    header = {
        "subcommand": "tensor-resolve",
        "config_path": config_path,
        "model_id": resolved.model_id,
        "architecture": resolved.architecture,
        "query_name": name or "",
        "query_tid": tid,
        "query_graph": graph or "both",
    }

    with DebugJsonlWriter(out_path, run_id, header) as w:
        write_run_and_model_records(w, resolved, "tensor-resolve")
        try:
            trainer = build_introspection_trainer(resolved)
        except Exception as e:
            logger.error(f"tensor-resolve: failed to build trainer: {type(e).__name__}: {e}")
            w.write(Tag.ERROR, severity=Severity.ERROR, stage="build_trainer", error=str(e))
            return 1

        graphs_to_check: list[bool] = [False, True] if graph is None else [graph == "backward"]
        hits = 0
        for is_backward in graphs_to_check:
            try:
                res = trainer.get_debug_tensor_resolution(name=name or "", tid=int(tid), is_backward=is_backward)
            except Exception as e:
                logger.error(f"tensor-resolve: lookup failed: {type(e).__name__}: {e}")
                w.write(Tag.ERROR, severity=Severity.ERROR, stage="resolve", error=str(e))
                return 1
            _emit_resolution(w, res)
            if res["found"]:
                hits += 1
        w.summary(query_name=name or "", query_tid=tid, hits=hits)
    logger.info(f"tensor-resolve: {hits} hit(s), output at {out_path}")
    return 0 if hits > 0 else 3
