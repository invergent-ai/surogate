"""`surogate debug tensor-aliasing` — tensor-aliasing audit.

Two modes:

  --mode static (default)
      Compile-only scan for pairs of tids whose (region, block_layer_idx,
      offset, bytes) ranges overlap in the same coloring frame. Under
      correct compilation this is empty (modulo intentional `alias_of`
      declarations). Emits one TENSOR_ALIAS_STATIC record per overlap.

  --mode runtime
      Runs one forward+backward step with `SUROGATE_CHECK_OP_IO_ALIASING=log`
      set. The C++ side checks every op's read-after-write byte ranges and
      emits `[op-io-alias]` lines to stderr. This subcommand captures those
      lines via fd-level redirection, parses them, and emits one
      TENSOR_ALIAS_RUNTIME record per event.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from surogate.utils.logger import get_logger

from ._shared import DebugResolveError, resolve_model_and_ir
from .schema import Severity, Tag
from .tensor_debug_shared import build_introspection_trainer, write_run_and_model_records
from .writer import DebugJsonlWriter, default_output_path, make_run_id

logger = get_logger()


# Format from compiled_ops_save.cpp:1572-1585. Example:
#   [op-io-alias] <forward> op_idx=42 type=Matmul input='blocks[3].q'#15 ptr=0x7f... bytes=32768 vs output='blocks[3].att_out'#18 ptr=0x7f... bytes=32768
_ALIAS_RE = re.compile(
    r"\[op-io-alias\]\s+<(?P<phase>[^>]+)>\s+op_idx=(?P<op>\d+)\s+type=(?P<type>\S+)\s+"
    r"input='(?P<input>[^']*)'#(?P<input_tid>-?\d+)\s+ptr=(?P<input_ptr>0x[0-9a-fA-F]+)\s+bytes=(?P<input_bytes>\d+)\s+"
    r"vs\s+output='(?P<output>[^']*)'#(?P<output_tid>-?\d+)\s+ptr=(?P<output_ptr>0x[0-9a-fA-F]+)\s+bytes=(?P<output_bytes>\d+)"
)


def _run_static_mode(writer: Any, trainer: Any) -> int:
    try:
        pairs = trainer.get_debug_static_aliasing()
    except Exception as e:
        logger.error(f"tensor-aliasing static: {type(e).__name__}: {e}")
        writer.write(Tag.ERROR, severity=Severity.ERROR, stage="get_static_aliasing", error=str(e))
        return 1

    # Categorize the overlap. An exact (offset_a, bytes_a) == (offset_b, bytes_b)
    # match with overlapping lifetimes is almost always a legitimate view alias
    # (e.g., `blocks[L].ln1` vs `blocks[L].x_flat`). A partial overlap is more
    # suspicious — either an unintended alias or a slab layout bug.
    exact = 0
    partial = 0
    for p in pairs:
        exact_match = p["offset_a"] == p["offset_b"] and p["bytes_a"] == p["bytes_b"]
        overlap_kind = "exact" if exact_match else "partial"
        severity = Severity.INFO if exact_match else Severity.WARN
        writer.write(Tag.TENSOR_ALIAS_STATIC, severity=severity, overlap_kind=overlap_kind, **p)
        if exact_match:
            exact += 1
        else:
            partial += 1

    writer.summary(
        mode="static",
        overlap_pairs=len(pairs),
        exact_byte_matches=exact,
        partial_overlaps=partial,
        note=(
            "exact_byte_matches are almost always legitimate view aliases "
            "(e.g., `blocks[L].ln1` vs `blocks[L].x_flat`). "
            "partial_overlaps with lifetime overlap are more suspicious."
        ),
    )
    logger.info(
        f"tensor-aliasing static: {len(pairs)} pair(s) — {exact} exact (likely view alias), {partial} partial (inspect)"
    )
    return 0


def _run_runtime_mode(writer: Any, resolved: Any) -> int:
    # Ensure the env var is set BEFORE the first op dispatch. The check fires
    # on the per-op hot path; it reads the env var at each call but picks it
    # up once and caches it inside the executor.
    os.environ["SUROGATE_CHECK_OP_IO_ALIASING"] = "log"

    # Disable CUDA-graph capture on the runtime config BEFORE trainer
    # construction. Capture allocates device memory on first step, which
    # conflicts with cudaMalloc in compiled_ops.cpp:492 under concurrent
    # stream capture. The aliasing check is the goal, not perf.
    config = resolved.config
    if getattr(config, "runtime_config", None) is not None:
        config.runtime_config.use_cuda_graphs = False
    config.use_cuda_graphs = False

    try:
        trainer = build_introspection_trainer(resolved)
    except Exception as e:
        logger.error(f"tensor-aliasing runtime: build trainer failed: {type(e).__name__}: {e}")
        writer.write(Tag.ERROR, severity=Severity.ERROR, stage="build_trainer", error=str(e))
        return 1

    ngpu = int(trainer.world_size)
    B = int(trainer.batch_size)
    T = int(trainer.seq_length)
    total_rows = ngpu * B

    # Init random weights + dummy inputs. Full step so backward aliasing
    # checks also fire.
    try:
        trainer.init_weights()
    except Exception as e:
        logger.error(f"tensor-aliasing runtime: init_weights failed: {type(e).__name__}: {e}")
        writer.write(Tag.ERROR, severity=Severity.ERROR, stage="init_weights", error=str(e))
        return 1

    inputs = np.zeros((total_rows, T), dtype=np.int32)
    targets = np.zeros((total_rows, T), dtype=np.int32)

    # Redirect fd 2 to a tempfile so we capture std::cerr from C++.
    captured_path = Path(tempfile.mkdtemp(prefix="surogate_alias_")) / "stderr.log"
    saved_fd = os.dup(2)
    try:
        with open(captured_path, "w") as log_fp:
            os.dup2(log_fp.fileno(), 2)
            try:
                trainer.step(inputs, targets)
                from surogate import _surogate

                opt = _surogate.OptimizerConfig(optimizer="adamw", learning_rate=1e-4, weight_decay=0.0)
                trainer.update_with_config(opt, 1)
            except Exception as e:
                # Preserve the exception but make sure stderr is restored.
                os.dup2(saved_fd, 2)
                logger.error(f"tensor-aliasing runtime: step failed: {type(e).__name__}: {e}")
                writer.write(Tag.ERROR, severity=Severity.ERROR, stage="step", error=str(e))
                return 1
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)

    # Parse the captured stderr.
    count = 0
    other_lines = 0
    with open(captured_path) as fp:
        for line in fp:
            m = _ALIAS_RE.search(line)
            if not m:
                other_lines += 1
                continue
            count += 1
            writer.write(
                Tag.TENSOR_ALIAS_RUNTIME,
                severity=Severity.WARN,
                phase=m.group("phase"),
                op_idx=int(m.group("op")),
                op_type=m.group("type"),
                input_name=m.group("input"),
                input_tid=int(m.group("input_tid")),
                input_ptr=m.group("input_ptr"),
                input_bytes=int(m.group("input_bytes")),
                output_name=m.group("output"),
                output_tid=int(m.group("output_tid")),
                output_ptr=m.group("output_ptr"),
                output_bytes=int(m.group("output_bytes")),
            )
    writer.summary(
        mode="runtime",
        runtime_aliasing_events=count,
        unmatched_stderr_lines=other_lines,
        captured_stderr=str(captured_path),
    )
    logger.info(f"tensor-aliasing runtime: {count} event(s); raw capture at {captured_path}")
    return 0


def run_tensor_aliasing(
    config_path: str,
    *,
    output: str | None = None,
    hub_token: str | None = None,
    mode: str = "static",
) -> int:
    if mode not in {"static", "runtime"}:
        logger.error(f"tensor-aliasing: invalid mode {mode!r} (expected 'static' or 'runtime')")
        return 2

    try:
        resolved = resolve_model_and_ir(config_path, hub_token=hub_token)
    except DebugResolveError as e:
        logger.error(f"tensor-aliasing: {e}")
        return 2

    run_id = make_run_id()
    out_path = output or default_output_path(f"tensor-aliasing-{mode}", resolved.model_id)
    header = {
        "subcommand": "tensor-aliasing",
        "mode": mode,
        "config_path": config_path,
        "model_id": resolved.model_id,
        "architecture": resolved.architecture,
    }

    with DebugJsonlWriter(out_path, run_id, header) as w:
        write_run_and_model_records(w, resolved, f"tensor-aliasing-{mode}")
        if mode == "static":
            try:
                trainer = build_introspection_trainer(resolved)
            except Exception as e:
                logger.error(f"tensor-aliasing static: build trainer failed: {type(e).__name__}: {e}")
                w.write(Tag.ERROR, severity=Severity.ERROR, stage="build_trainer", error=str(e))
                return 1
            return _run_static_mode(w, trainer)
        # Runtime mode does its own trainer construction so it can tune the
        # runtime config (disable CUDA graphs) before build.
        return _run_runtime_mode(w, resolved)
