"""Per-step + per-param + per-layer gradient tracing.

Runs N training steps, capturing at each step:

* **Param gradients** via ``trainer.get_lora_gradients(0)`` (LoRA) or
  ``trainer.get_gradients(0)`` (FFT). Stats computed on GPU via torch with
  a single ``.cpu()`` sync per tensor.
* **Intermediate backward gradients** (``d_blocks[N].<base>`` tensors) via
  the existing ``SUROGATE_DEBUG_DUMP_*`` infra, dump dir rotated per step.

Between ``trainer.step()`` (fwd+bwd) and ``trainer.update_with_config()``
(optimizer step) we snapshot param gradients so they survive the zeroing
the optimizer performs. The optimizer step still runs so weights evolve
and multi-step degradation (``--steps 10`` for "NaN appears after step 5")
is observable.

Args
----
config_path : str
    Path to a training config YAML (same format as ``surogate sft``).
output : str | None
    Output JSONL path. Defaults to
    ``./debug/debug_gradients_<model>_<ts>.jsonl``. A sibling
    ``.header.json`` carries invocation metadata.
hub_token : str | None
    HuggingFace Hub token for private models.
steps : int
    Number of ``(fwd+bwd+optimizer)`` cycles. Default ``1``. Use ``>1`` to
    diagnose "training degrades after N steps" — every step produces its
    own per-layer dumps (rotated into ``<tmpdir>/step_NNNN/`` and
    ``rmtree``'d after parse, so disk peak stays ~1 step).

Returns
-------
int
    ``0`` on success (inspect JSONL for severity=error). ``1`` on setup
    failure (model unresolvable, tokenization failure, trainer ctor error).

Output records (``tag`` field, one per line)
-------------------------------------------
RUN
    ``subcommand="gradients"``, ``model_id``, ``model_dir``,
    ``architecture``, ``n_layers``, ``steps``, ``lora``.
MODEL
    Compiled-IR summary plus ``block_grad_slot_count``,
    ``global_grad_slot_count``, ``dumped_slots_per_step``,
    ``global_grad_slots`` (list of ``{name, gradient_of, shape, dtype}`` for
    global-scope gradients not capturable by the layer hook).
GRAD — scope=param
    One per (step × parameter). Fields: ``step``, ``scope="param"``,
    ``name`` (e.g. ``base_model...lora_A.weight``), ``shape``, ``dtype``,
    stats (``size``, ``finite_count``, ``nan_count``, ``inf_count``,
    ``nan_pct``, ``inf_pct``, ``zero_count``, ``zero_pct``, ``min``,
    ``max``, ``mean``, ``std``, ``abs_mean``, ``norm``).
    severity=info when clean; =warn when all-zero (disconnected from
    graph); =error on NaN/Inf.
GRAD — scope=intermediate
    One per (step × layer × block-scope gradient slot). Fields: ``step``,
    ``scope="intermediate"``, ``layer``, ``op`` (short slot name, e.g.
    ``d_qkv``), ``slot`` (runtime name ``d_blocks[N].<base>``),
    ``grad_of`` (forward tensor it gradients), ``declared_shape``,
    ``declared_dtype``, ``dumped_shape``, ``dumped_dtype``, ``status``,
    ``error``, plus the same stats as scope=param.
    severity=warn when dump missing (some small-scalar backward tensors
    like ``*_rstd`` aren't bound via ``bind_tensor`` so can't be captured
    by the hook); =error on NaN/Inf.
ERROR
    severity=error. Two phases:
    ``phase="gradient_scan"`` records the first NaN/Inf site across all
    steps and both scopes (``step``, ``scope``, ``name``, ``layer``, ``op``);
    ``phase="trainer"`` records a formatted traceback if a stage
    (trainer.step, optimizer_update, advance_batch) raised — partial
    per-step records are still emitted before the loop breaks.
SUMMARY
    Terminal. ``counts_by_tag``, ``counts_by_severity``, ``status`` ∈
    {``completed``, ``completed_with_trainer_error``}, ``steps_requested``,
    ``steps_completed``, ``first_failure_step``, ``first_failure_scope``,
    ``first_failure_name``.

Grep recipes
------------
::

    grep '"severity":"error"' debug_gradients_*.jsonl           # punch list
    grep '"step":0,"scope":"param"' debug_gradients_*.jsonl     # step 0 params
    jq 'select(.tag=="GRAD" and .nan_pct>0) | {step,layer,op,nan_pct}' debug_*.jsonl
    # evolution of one param across steps:
    jq 'select(.name=="<param>") | {step,norm,nan_pct,zero_pct}' debug_*.jsonl
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np  # noqa: F401  (used by shared helpers via _shared)
import torch

from surogate.utils.logger import get_logger

from ._shared import (
    DebugResolveError,
    allocate_token_buffers,
    build_optimizer_config,
    capture_exception,
    configure_for_single_step,
    make_dumps_root,
    parallel_stat,
    resolve_model_and_ir,
    rmtree_quiet,
    tokenize_and_get_train_files,
)
from .schema import DumpStatus, Severity, Tag
from .writer import DebugJsonlWriter, default_output_path, make_run_id

logger = get_logger()


class _FirstFailure(NamedTuple):
    step: int
    scope: str  # "param" | "intermediate"
    name: str  # param name or runtime tensor name
    layer: int  # -1 for param scope
    op: str  # slot short name or param name


def _bwd_tensor_name(layer_idx: int, slot: dict) -> str:
    """Build the runtime backward-tensor name, e.g. ``d_blocks[5].qkv``.

    The runtime stores backward tensors with the ``d_`` prefix on the whole
    path (``d_blocks[N].<base>``), not appended to the base name. Prefer the
    IR slot's ``gradient_of`` field; fall back to stripping a ``d_`` prefix.
    """
    name = slot["name"]
    base = slot.get("gradient_of") or (name[2:] if name.startswith("d_") else name)
    assert base, f"cannot derive grad base from slot {slot!r}"
    return f"d_blocks[{layer_idx}].{base}"


def run_gradient_trace(
    config_path: str,
    output: str | None = None,
    hub_token: str | None = None,
    steps: int = 1,
) -> int:
    """Run the gradient trace over ``steps`` full fwd+bwd+optimizer cycles.

    Returns 0 on success (even if the trace found NaN/Inf — check the JSONL
    for severity=error rows). Returns 1 only if setup fails before any step.
    """
    dumps_root = make_dumps_root("grads")
    try:
        return _run_with_dumps_root(config_path, output, hub_token, steps, dumps_root)
    finally:
        rmtree_quiet(dumps_root)


def _run_with_dumps_root(
    config_path: str,
    output: str | None,
    hub_token: str | None,
    steps: int,
    dumps_root: Path,
) -> int:
    try:
        resolved = resolve_model_and_ir(config_path, hub_token=hub_token)
    except DebugResolveError as e:
        logger.error(str(e))
        return 1

    module = resolved.module
    dsl_cfg = module.get("config", {})
    n_layers = int(dsl_cfg.get("n_layers", 0))
    layout = module.get("activation_layout", {})
    grad_slots = layout.get("gradient_slots", [])
    block_grad_slots = [g for g in grad_slots if g.get("scope") == "gradient"]
    global_grad_slots = [g for g in grad_slots if g.get("scope") == "global_gradient"]

    dump_names = [_bwd_tensor_name(i, g) for i in range(n_layers) for g in block_grad_slots]

    run_id = make_run_id()
    model_name = os.path.basename(resolved.model_dir.rstrip("/"))
    out_path = output or default_output_path("gradients", model_name)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Tensor list set once; SUROGATE_DEBUG_DUMP_DIR rotates per step below.
    os.environ["SUROGATE_DEBUG_DUMP_TENSORS"] = ",".join(dump_names)

    steps = max(1, int(steps))
    configure_for_single_step(resolved.config, steps=steps)
    config = resolved.config

    try:
        train_files = tokenize_and_get_train_files(config)
    except DebugResolveError as e:
        logger.error(str(e))
        return 1
    if not train_files:
        logger.error(
            f"no tokenized train-*.bin files in {config.output_dir}; run `surogate tokenize {config_path}` first"
        )
        return 1

    from surogate.train.trainer import SurogateTrainerWrapper

    try:
        wrapper = SurogateTrainerWrapper(config=config, train_files=train_files, eval_files=None)
    except Exception as e:
        logger.error(f"failed to construct trainer: {type(e).__name__}: {e}")
        return 1

    in_tokens, out_tokens, pos_ids = allocate_token_buffers(config)
    wrapper.train_loader.load_batch(in_tokens, out_tokens, pos_ids)

    grad_fn = wrapper.trainer.get_lora_gradients if config.lora else wrapper.trainer.get_gradients

    header = {
        "subcommand": "gradients",
        "config_path": os.path.abspath(config_path),
        "model_id": resolved.model_id,
        "model_dir": resolved.model_dir,
        "architecture": resolved.architecture,
        "n_layers": n_layers,
        "steps": steps,
        "lora": bool(config.lora),
        "dumps_root": str(dumps_root),
        "block_grad_slot_count": len(block_grad_slots),
        "global_grad_slot_count": len(global_grad_slots),
        "dumped_slots_per_step": len(dump_names),
    }

    first_failure: _FirstFailure | None = None
    steps_completed = 0
    trainer_error: str | None = None

    with DebugJsonlWriter(out_path, run_id=run_id, header=header) as w:
        _emit_run_and_model(
            w,
            resolved,
            module,
            dsl_cfg,
            n_layers,
            steps,
            bool(config.lora),
            block_grad_slots,
            global_grad_slots,
            len(dump_names),
        )

        for step in range(steps):
            step_start = time.time()
            step_dir = dumps_root / f"step_{step:04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            os.environ["SUROGATE_DEBUG_DUMP_DIR"] = str(step_dir)

            ok, err = capture_exception(lambda: wrapper.trainer.step(in_tokens, out_tokens, pos_ids))
            if not ok:
                trainer_error = err
                logger.warning(f"step {step} raised in trainer.step; emitting partial: {err.splitlines()[0]}")
                break

            first_failure = _emit_param_grad_records(w, grad_fn, step, first_failure)
            dump_files = set(os.listdir(step_dir)) if step_dir.exists() else set()
            first_failure = _emit_intermediate_grad_records(
                w, step_dir, dump_files, n_layers, block_grad_slots, step, first_failure
            )

            ok, err = capture_exception(
                lambda: wrapper.trainer.update_with_config(
                    build_optimizer_config(config, _lr_for_step(wrapper, config, step)), step + 1
                ),
            )
            if not ok:
                trainer_error = err
                logger.warning(f"step {step} raised in optimizer_update; ending: {err.splitlines()[0]}")
                break

            rmtree_quiet(step_dir)

            ok, err = capture_exception(lambda: _advance_batch(wrapper, in_tokens, out_tokens, pos_ids))
            if not ok:
                trainer_error = err
                logger.warning(f"step {step} raised in advance_batch; ending: {err.splitlines()[0]}")
                break

            steps_completed = step + 1
            logger.info(f"step {step} done in {time.time() - step_start:.1f}s")

        if first_failure is not None:
            w.write(
                Tag.ERROR,
                severity=Severity.ERROR,
                phase="gradient_scan",
                step=first_failure.step,
                scope=first_failure.scope,
                name=first_failure.name,
                layer=first_failure.layer,
                op=first_failure.op,
                error=(
                    f"first NaN/Inf gradient at step={first_failure.step} "
                    f"scope={first_failure.scope} name={first_failure.name}"
                ),
            )

        if trainer_error:
            w.write(Tag.ERROR, severity=Severity.ERROR, phase="trainer", error=trainer_error)

        w.summary(
            status="completed" if trainer_error is None else "completed_with_trainer_error",
            steps_requested=steps,
            steps_completed=steps_completed,
            first_failure_step=first_failure.step if first_failure else None,
            first_failure_scope=first_failure.scope if first_failure else None,
            first_failure_name=first_failure.name if first_failure else None,
        )

    errors = w.counts_by_severity.get(Severity.ERROR.value, 0)
    warns = w.counts_by_severity.get(Severity.WARN.value, 0)
    if first_failure is not None:
        logger.error(
            f"first NaN/Inf gradient at step={first_failure.step} "
            f"scope={first_failure.scope} name={first_failure.name!r} "
            f"({errors} error rows, {warns} warnings) — see {out_path}"
        )
    else:
        logger.info(f"wrote {out_path} ({errors} errors, {warns} warnings, {steps_completed}/{steps} steps)")
    return 0


def _emit_run_and_model(
    w: DebugJsonlWriter,
    resolved: Any,
    module: dict,
    dsl_cfg: dict,
    n_layers: int,
    steps: int,
    lora: bool,
    block_grad_slots: list[dict],
    global_grad_slots: list[dict],
    dumped_slots_per_step: int,
) -> None:
    w.write(
        Tag.RUN,
        subcommand="gradients",
        model_id=resolved.model_id,
        model_dir=resolved.model_dir,
        architecture=resolved.architecture,
        n_layers=n_layers,
        steps=steps,
        lora=lora,
    )
    w.write(
        Tag.MODEL,
        name=module.get("name"),
        kind=module.get("kind"),
        dsl_config=dsl_cfg,
        block_grad_slot_count=len(block_grad_slots),
        global_grad_slot_count=len(global_grad_slots),
        dumped_slots_per_step=dumped_slots_per_step,
        global_grad_slots=[
            {
                "name": s["name"],
                "gradient_of": s.get("gradient_of"),
                "shape": s.get("shape"),
                "dtype": s.get("dtype", "bf16"),
            }
            for s in global_grad_slots
        ],
    )


def _lr_for_step(wrapper: Any, config: Any, step: int) -> float:
    return wrapper.lr_schedule.get_lr(step) if hasattr(wrapper, "lr_schedule") else config.learning_rate


def _advance_batch(wrapper: Any, in_tokens: np.ndarray, out_tokens: np.ndarray, pos_ids: np.ndarray) -> None:
    if not wrapper.train_loader.has_next():
        wrapper.train_loader.advance_epoch()
    wrapper.train_loader.load_batch(in_tokens, out_tokens, pos_ids)


def _emit_param_grad_records(
    w: DebugJsonlWriter,
    grad_fn: Any,
    step: int,
    first_failure: _FirstFailure | None,
) -> _FirstFailure | None:
    try:
        grads = grad_fn(0)
    except Exception as e:
        w.write(
            Tag.ERROR,
            severity=Severity.ERROR,
            phase="get_gradients",
            step=step,
            error=f"{type(e).__name__}: {e}",
        )
        return first_failure

    for name, arr in grads.items():
        try:
            t = torch.utils.dlpack.from_dlpack(arr)
            stats = _torch_param_grad_stats(t)
            shape = list(t.shape)
            dtype = str(t.dtype).removeprefix("torch.")
        except Exception as e:
            w.write(
                Tag.GRAD,
                severity=Severity.ERROR,
                step=step,
                scope="param",
                name=name,
                error=f"{type(e).__name__}: {e}",
            )
            continue

        severity = _severity_from_stats(stats)
        w.write(
            Tag.GRAD,
            severity=severity,
            step=step,
            scope="param",
            name=name,
            shape=shape,
            dtype=dtype,
            **stats,
        )
        if severity == Severity.ERROR and first_failure is None:
            first_failure = _FirstFailure(step=step, scope="param", name=name, layer=-1, op=name)
    return first_failure


def _emit_intermediate_grad_records(
    w: DebugJsonlWriter,
    step_dir: Path,
    dump_files: set[str],
    n_layers: int,
    block_grad_slots: list[dict],
    step: int,
    first_failure: _FirstFailure | None,
) -> _FirstFailure | None:
    tasks = [
        ((layer_idx, slot["name"]), _bwd_tensor_name(layer_idx, slot))
        for layer_idx in range(n_layers)
        for slot in block_grad_slots
    ]
    results = parallel_stat(step_dir, dump_files, tasks)

    for layer_idx in range(n_layers):
        for slot in block_grad_slots:
            slot_name = slot["name"]
            tensor_name = _bwd_tensor_name(layer_idx, slot)
            result = results[(layer_idx, slot_name)]

            severity = Severity.INFO
            if result.status == DumpStatus.MISSING:
                severity = Severity.WARN
            elif result.status == DumpStatus.LOADED:
                severity = _severity_from_stats(result.stats or {})
            elif result.status == DumpStatus.READ_FAILED:
                severity = Severity.ERROR

            w.write(
                Tag.GRAD,
                severity=severity,
                step=step,
                scope="intermediate",
                layer=layer_idx,
                op=slot_name,
                slot=tensor_name,
                grad_of=slot.get("gradient_of"),
                declared_shape=slot.get("shape"),
                declared_dtype=slot.get("dtype", "bf16"),
                dumped_shape=(result.meta or {}).get("shape"),
                dumped_dtype=(result.meta or {}).get("dtype"),
                status=result.status,
                error=result.error,
                **(result.stats or {}),
            )
            if severity == Severity.ERROR and first_failure is None:
                first_failure = _FirstFailure(
                    step=step, scope="intermediate", name=tensor_name, layer=layer_idx, op=slot_name
                )
    return first_failure


def _torch_param_grad_stats(t: torch.Tensor) -> dict:
    """On-GPU stats for a param-grad tensor, compacted to one ``.cpu()`` sync.

    Naïve implementation uses ~10 ``.item()`` calls per tensor → ~400 tensors ×
    10 syncs per step = 4k GPU syncs. Stacking the scalar reductions into one
    1-D tensor and transferring it in a single call reduces that by ~10×.
    """
    if t.numel() == 0:
        return {"size": 0}
    t_f = t.float() if t.dtype != torch.float32 else t
    total = int(t_f.numel())

    isnan = torch.isnan(t_f)
    isinf = torch.isinf(t_f)
    finite_mask = ~(isnan | isinf)

    nan_count_t = isnan.sum()
    inf_count_t = isinf.sum()
    finite_count_t = finite_mask.sum()

    if int(finite_count_t.item()) == 0:
        # All-nonfinite: only the counts are meaningful.
        counts = torch.stack([nan_count_t, inf_count_t, finite_count_t]).cpu().numpy()
        nan_count = int(counts[0])
        inf_count = int(counts[1])
        return {
            "size": total,
            "finite_count": 0,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "nan_pct": 100.0 * nan_count / total,
            "inf_pct": 100.0 * inf_count / total,
            "zero_count": 0,
            "zero_pct": 0.0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "abs_mean": None,
            "norm": None,
        }

    finite = t_f[finite_mask]
    # Pack every scalar we need into one 1-D tensor for a single D2H transfer.
    std_t = finite.std() if finite.numel() > 1 else torch.zeros((), device=finite.device)
    zero_count_t = (finite == 0).sum()
    packed = (
        torch.stack(
            [
                nan_count_t.float(),
                inf_count_t.float(),
                finite_count_t.float(),
                zero_count_t.float(),
                finite.min().float(),
                finite.max().float(),
                finite.mean(),
                std_t,
                finite.abs().mean(),
                finite.norm(),
            ]
        )
        .cpu()
        .numpy()
    )

    nan_count = int(packed[0])
    inf_count = int(packed[1])
    finite_count = int(packed[2])
    zero_count = int(packed[3])
    return {
        "size": total,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nan_pct": 100.0 * nan_count / total,
        "inf_pct": 100.0 * inf_count / total,
        "zero_count": zero_count,
        "zero_pct": 100.0 * zero_count / total,
        "min": float(packed[4]),
        "max": float(packed[5]),
        "mean": float(packed[6]),
        "std": float(packed[7]),
        "abs_mean": float(packed[8]),
        "norm": float(packed[9]),
    }


def _severity_from_stats(stats: dict) -> str:
    """Map a stats dict to INFO/WARN/ERROR.

    ERROR: any NaN or Inf. WARN: all-zero gradient (param disconnected from
    graph) or all-nonfinite. INFO: anything else.
    """
    if not stats:
        return Severity.INFO
    if stats.get("nan_count", 0) > 0 or stats.get("inf_count", 0) > 0:
        return Severity.ERROR
    size = stats.get("size", 0)
    if size > 0 and stats.get("zero_count", 0) == size:
        return Severity.WARN
    return Severity.INFO
