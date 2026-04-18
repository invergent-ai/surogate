"""Layer-by-layer numerical diff vs HuggingFace transformers.

Runs the same checkpoint twice — once through ``AutoModelForCausalLM`` as the
reference, once through our DSL runtime — with identical synthetic token
input, and compares per-layer hidden-state outputs. The first layer whose
``max_abs_diff`` exceeds ``atol + rtol * max(|hf|)`` is the divergence site —
the killer tool for silent DSL-vs-HF bugs.

The two forwards run **sequentially on the same GPU** (HF first, freed, then
DSL). Same device avoids CPU/GPU bf16-rounding artifacts that would otherwise
look like divergence. HF loads with ``torch_dtype=bfloat16`` so precision
matches the DSL runtime; the diff arithmetic is in fp32 on CPU.

Args
----
config_path : str
    Training config YAML (same format as ``surogate sft``).
output : str | None
    Output JSONL path. Defaults to ``./debug/debug_diff_<model>_<ts>.jsonl``.
hub_token : str | None
    HF Hub token for private-repo model downloads.
reference : str | None
    Optional HF model repo/path for the reference. Defaults to ``config.model``.
max_tokens : int
    Sequence length for the synthetic forward input. Capped at
    ``config.sequence_len``. Default ``64``.
seed : int
    RNG seed for synthetic token ids. Default ``0`` (deterministic across runs).
rtol, atol : float
    numpy-style tolerance; severity=error when ``max_abs_diff > atol + rtol * max(|hf|)``.
    Defaults ``rtol=1e-2`` (≈ bf16's 7-bit mantissa precision of ~0.8%) and
    ``atol=1e-3``. Pass tighter values for fp32 comparisons.
ref_device_map : str
    Passed to ``AutoModelForCausalLM.from_pretrained(device_map=...)``. Default
    ``"auto"`` shards the HF reference across every visible GPU (requires the
    ``accelerate`` package — standard HF dep). Use ``"cuda"`` to force a
    single-GPU load, or a specific device string like ``"cuda:0"``. Set
    ``CUDA_VISIBLE_DEVICES`` upstream to control which GPUs are eligible; the
    DSL runs on the first visible GPU after the HF model is freed.

Returns
-------
int
    ``0`` on success (inspect JSONL for severity=error). ``1`` on setup failure.

Output records (``tag`` field, one per line)
-------------------------------------------
RUN
    Invocation context: ``model_id``, ``reference``, ``architecture``, ``n_layers``,
    ``seq_len``, ``seed``, ``rtol``, ``atol``.
MODEL
    Compiled-IR summary + ``arch_map`` (the HF-layers-attr → DSL-slot mapping used).
REFERENCE
    HF reference context: ``path``, ``hf_class``, ``dtype="bf16"``, ``device``.
DIFF
    One per layer. Fields: ``layer``, ``op`` (DSL slot name, usually ``res_ffn``),
    ``slot`` (full ``blocks[N].<slot>``), ``hf_shape``, ``dsl_shape``, ``status``,
    plus diff stats: ``max_abs_diff``, ``mean_abs_diff``, ``cos_sim``, ``rel_err``,
    ``hf_max_abs``, ``dsl_max_abs``, ``hf_norm``, ``dsl_norm``.
    severity=info when within tolerance; =warn when dumps missing or either
    side has non-finite values; =error on divergence > tolerance.
ERROR
    severity=error. ``phase`` ∈ {``hf_reference``, ``dsl_step``, ``diff_scan``}.
    ``diff_scan`` records the first diverging layer.
SUMMARY
    Terminal. ``counts_by_tag``, ``counts_by_severity``, ``status``,
    ``first_diverging_layer``, ``layers_compared``, ``rtol``, ``atol``.

Grep recipes
------------
::

    grep '"severity":"error"' debug_diff_*.jsonl                 # divergence sites
    jq 'select(.tag=="DIFF") | {layer,max_abs_diff,cos_sim}' debug_*.jsonl
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

from surogate.utils.logger import get_logger

from ._shared import (
    DebugResolveError,
    allocate_token_buffers,
    capture_exception,
    configure_for_single_step,
    disable_cuda_graphs,
    load_dump_tensor,
    make_dumps_root,
    resolve_model_and_ir,
    rmtree_quiet,
    tokenize_and_get_train_files,
)
from .schema import DiffStatus, DumpStatus, Severity, Tag
from .writer import DebugJsonlWriter, default_output_path, make_run_id

logger = get_logger()


# Per-architecture mapping: where HF stores the block list, and which DSL
# activation slot represents a block's output residual stream. Extend as new
# architectures land. Unknown architectures fall through to the defaults.
# ``layer_offset`` handles fused-residual architectures where the DSL slot
# ``blocks[N].res_ffn`` is the residual stream *arriving at* block N (output
# of block N-1 + embedding), so HF ``layer[N].output`` aligns with DSL
# ``blocks[N+1].res_ffn``. The final HF layer's output has no paired DSL
# block slot — it's a global slot the layer hook can't dump.
#
# ``layers_attr`` lists dotted paths to try in order. ``AutoModelForCausalLM``
# often strips a multimodal wrapper (e.g. ``Qwen3_5ForConditionalGeneration``
# → ``Qwen3_5ForCausalLM``), so ``model.language_model.layers`` may not exist
# on the loaded instance — fall back to ``model.layers``.
_COMMON_LAYER_PATHS = ["model.language_model.layers", "model.layers"]
_ARCH_MAPS: dict[str, dict[str, Any]] = {
    "Qwen3ForCausalLM": {"layers_attr": _COMMON_LAYER_PATHS, "dsl_slot": "res_ffn", "layer_offset": 1},
    "Qwen3_5ForCausalLM": {"layers_attr": _COMMON_LAYER_PATHS, "dsl_slot": "res_ffn", "layer_offset": 1},
    "Qwen3_5ForConditionalGeneration": {
        "layers_attr": _COMMON_LAYER_PATHS,
        "dsl_slot": "res_ffn",
        "layer_offset": 1,
    },
    "Qwen3MoeForCausalLM": {"layers_attr": _COMMON_LAYER_PATHS, "dsl_slot": "res_ffn", "layer_offset": 1},
    "Qwen3_5MoeForCausalLM": {"layers_attr": _COMMON_LAYER_PATHS, "dsl_slot": "res_ffn", "layer_offset": 1},
    "Qwen3_5MoeForConditionalGeneration": {
        "layers_attr": _COMMON_LAYER_PATHS,
        "dsl_slot": "res_ffn",
        "layer_offset": 1,
    },
    "LlamaForCausalLM": {"layers_attr": _COMMON_LAYER_PATHS, "dsl_slot": "res_ffn", "layer_offset": 1},
    "MistralForCausalLM": {"layers_attr": _COMMON_LAYER_PATHS, "dsl_slot": "res_ffn", "layer_offset": 1},
    "Qwen2ForCausalLM": {"layers_attr": _COMMON_LAYER_PATHS, "dsl_slot": "res_ffn", "layer_offset": 1},
    "Gemma4ForCausalLM": {"layers_attr": _COMMON_LAYER_PATHS, "dsl_slot": "res_ffn", "layer_offset": 1},
    "Gemma4ForConditionalGeneration": {
        "layers_attr": _COMMON_LAYER_PATHS,
        "dsl_slot": "res_ffn",
        "layer_offset": 1,
    },
}
# Conservative default: offset=0. Unknown architectures should surface a shape
# mismatch at layer N rather than silently compare off-by-one. Registered
# architectures above declare their offset explicitly.
_DEFAULT_ARCH_MAP = {"layers_attr": _COMMON_LAYER_PATHS, "dsl_slot": "res_ffn", "layer_offset": 0}


_NO_PAIRED_SLOT_NOTE = (
    "HF layer {hf} output has no DSL block slot at index {dsl} "
    "(layer_offset={offset}); pre-final-norm residual is a global slot "
    "the layer dump hook cannot capture in this architecture."
)


class _FirstDiff(NamedTuple):
    layer: int
    max_abs_diff: float


def run_reference_diff(
    config_path: str,
    output: str | None = None,
    hub_token: str | None = None,
    reference: str | None = None,
    max_tokens: int = 64,
    seed: int = 0,
    rtol: float = 1e-2,
    atol: float = 1e-3,
    ref_device_map: str = "auto",
) -> int:
    dumps_root = make_dumps_root("diff")
    try:
        return _run_with_dumps_root(
            config_path,
            output,
            hub_token,
            reference,
            max_tokens,
            seed,
            rtol,
            atol,
            ref_device_map,
            dumps_root,
        )
    finally:
        rmtree_quiet(dumps_root)


def _run_with_dumps_root(
    config_path: str,
    output: str | None,
    hub_token: str | None,
    reference: str | None,
    max_tokens: int,
    seed: int,
    rtol: float,
    atol: float,
    ref_device_map: str,
    dumps_root: Path,
) -> int:
    try:
        resolved = resolve_model_and_ir(config_path, hub_token=hub_token)
    except DebugResolveError as e:
        logger.error(str(e))
        return 1

    arch = resolved.architecture
    arch_map = _ARCH_MAPS.get(arch) or _DEFAULT_ARCH_MAP
    if arch not in _ARCH_MAPS:
        logger.warning(f"no architecture map for {arch!r}; using defaults {arch_map}")

    config = resolved.config
    module = resolved.module
    dsl_cfg = module.get("config", {})
    n_layers = int(dsl_cfg.get("n_layers", 0))
    dsl_slot = arch_map["dsl_slot"]
    layer_offset = int(arch_map.get("layer_offset", 0))

    total_rows = config.gpus * config.per_device_train_batch_size
    seq_len = min(int(max_tokens), int(config.sequence_len))
    if seq_len <= 0:
        logger.error(f"invalid max_tokens={max_tokens}")
        return 1

    vocab_size = int(resolved.hf_config.get("vocab_size") or dsl_cfg.get("vocab_size") or 0)
    if vocab_size <= 0:
        logger.error(f"could not determine vocab_size for {arch}")
        return 1

    rng = np.random.default_rng(int(seed))
    token_ids = rng.integers(0, vocab_size, size=(total_rows, seq_len), dtype=np.int32)

    ref_path = reference or resolved.model_dir
    run_id = make_run_id()
    model_name = os.path.basename(resolved.model_dir.rstrip("/"))
    out_path = output or default_output_path("diff", model_name)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    dump_dir = dumps_root / run_id
    dump_dir.mkdir(parents=True, exist_ok=True)

    hf_layer_outputs: dict[int, np.ndarray] = {}
    hf_class_name: str | None = None

    def _do_hf() -> None:
        nonlocal hf_layer_outputs, hf_class_name
        hf_layer_outputs, hf_class_name = _run_hf_reference(ref_path, token_ids, arch_map, device_map=ref_device_map)

    ok, _hf_err = capture_exception(_do_hf)
    hf_error: str | None = None if ok else _hf_err
    if hf_error:
        logger.error(f"HF reference failed: {hf_error.splitlines()[0]}")

    os.environ["SUROGATE_DEBUG_DUMP_DIR"] = str(dump_dir)
    os.environ["SUROGATE_DEBUG_DUMP_TENSORS"] = ",".join(f"blocks[{i}].{dsl_slot}" for i in range(n_layers))
    configure_for_single_step(config, steps=1)
    disable_cuda_graphs(config)

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

    dsl_error = _run_dsl_with_tokens(config, train_files, token_ids, seq_len, total_rows)
    if dsl_error:
        logger.error(f"DSL step failed: {dsl_error.splitlines()[0]}")
    header = {
        "subcommand": "diff",
        "config_path": os.path.abspath(config_path),
        "model_id": resolved.model_id,
        "model_dir": resolved.model_dir,
        "reference": ref_path,
        "architecture": arch,
        "hf_class": hf_class_name,
        "arch_map": arch_map,
        "n_layers": n_layers,
        "seq_len": seq_len,
        "batch": total_rows,
        "seed": int(seed),
        "rtol": float(rtol),
        "atol": float(atol),
        "hf_error": hf_error,
        "dsl_error": dsl_error,
    }

    first_diff: _FirstDiff | None = None
    layers_compared = 0
    layers_missing = 0

    with DebugJsonlWriter(out_path, run_id=run_id, header=header) as w:
        w.write(
            Tag.RUN,
            subcommand="diff",
            model_id=resolved.model_id,
            reference=ref_path,
            architecture=arch,
            n_layers=n_layers,
            seq_len=seq_len,
            seed=int(seed),
            rtol=float(rtol),
            atol=float(atol),
        )
        w.write(
            Tag.MODEL,
            name=module.get("name"),
            kind=module.get("kind"),
            dsl_config=dsl_cfg,
            arch_map=arch_map,
        )
        w.write(
            Tag.REFERENCE,
            path=ref_path,
            hf_class=hf_class_name,
            dtype="bf16",
            device="cuda",
        )

        if hf_error:
            w.write(Tag.ERROR, severity=Severity.ERROR, phase="hf_reference", error=hf_error)
        if dsl_error:
            w.write(Tag.ERROR, severity=Severity.ERROR, phase="dsl_step", error=dsl_error)

        dump_files = set(os.listdir(dump_dir)) if dump_dir.exists() else set()
        # ``hf_layer_idx`` indexes HF's transformer block. We pair it with DSL
        # ``blocks[hf_layer_idx + layer_offset].<slot>``. The final HF layer's
        # output has no matched DSL block in fused-residual models (the final
        # residual stream is a global slot the layer-end hook can't capture),
        # so we emit a warn record for it instead of faking a mismatch.
        max_hf = len(hf_layer_outputs)
        for hf_layer_idx in range(max_hf):
            dsl_layer_idx = hf_layer_idx + layer_offset
            tensor_name = f"blocks[{dsl_layer_idx}].{dsl_slot}"
            hf_t = hf_layer_outputs.get(hf_layer_idx)

            if dsl_layer_idx >= n_layers:
                layers_missing += 1
                w.write(
                    Tag.DIFF,
                    severity=Severity.WARN,
                    layer=hf_layer_idx,
                    op=dsl_slot,
                    slot=tensor_name,
                    status=DiffStatus.NO_PAIRED_SLOT,
                    hf_shape=list(hf_t.shape) if hf_t is not None else None,
                    note=_NO_PAIRED_SLOT_NOTE.format(hf=hf_layer_idx, dsl=dsl_layer_idx, offset=layer_offset),
                )
                continue

            dsl_t, dsl_shape, dsl_load_err = _load_dsl_dump(dump_dir, tensor_name, dump_files)
            if dsl_load_err:
                layers_missing += 1
                w.write(
                    Tag.DIFF,
                    severity=Severity.WARN,
                    layer=hf_layer_idx,
                    op=dsl_slot,
                    slot=tensor_name,
                    status=dsl_load_err,
                    hf_shape=list(hf_t.shape) if hf_t is not None else None,
                )
                continue
            if hf_t is None:
                layers_missing += 1
                w.write(
                    Tag.DIFF,
                    severity=Severity.WARN,
                    layer=hf_layer_idx,
                    op=dsl_slot,
                    slot=tensor_name,
                    status=DiffStatus.HF_OUTPUT_MISSING,
                    dsl_shape=dsl_shape,
                )
                continue

            stats, status = _diff_tensors(hf_t, dsl_t, seq_len)
            severity = _severity_from_diff(stats, status, rtol, atol)
            layers_compared += 1

            w.write(
                Tag.DIFF,
                severity=severity,
                layer=hf_layer_idx,
                op=dsl_slot,
                slot=tensor_name,
                dsl_layer=dsl_layer_idx,
                status=status,
                hf_shape=list(hf_t.shape),
                dsl_shape=dsl_shape,
                **stats,
            )
            if severity == Severity.ERROR and first_diff is None:
                first_diff = _FirstDiff(layer=hf_layer_idx, max_abs_diff=stats.get("max_abs_diff", 0.0))

        if first_diff is not None:
            w.write(
                Tag.ERROR,
                severity=Severity.ERROR,
                phase="diff_scan",
                first_diverging_layer=first_diff.layer,
                max_abs_diff=first_diff.max_abs_diff,
                rtol=float(rtol),
                atol=float(atol),
                error=(
                    f"first divergence exceeds tolerance at layer {first_diff.layer} "
                    f"(max_abs_diff={first_diff.max_abs_diff:.3e})"
                ),
            )

        status = "completed"
        if hf_error or dsl_error:
            status = "completed_with_errors"
        w.summary(
            status=status,
            first_diverging_layer=first_diff.layer if first_diff else None,
            layers_compared=layers_compared,
            layers_missing=layers_missing,
            rtol=float(rtol),
            atol=float(atol),
        )

    errors = w.counts_by_severity.get(Severity.ERROR.value, 0)
    warns = w.counts_by_severity.get(Severity.WARN.value, 0)
    if first_diff is not None:
        logger.error(
            f"first divergence at layer={first_diff.layer} "
            f"max_abs_diff={first_diff.max_abs_diff:.3e} "
            f"({errors} errors, {warns} warnings) — see {out_path}"
        )
    else:
        logger.info(
            f"wrote {out_path} ({errors} errors, {warns} warnings, {layers_compared}/{n_layers} layers compared)"
        )
    return 0


# =============================================================================
# HF reference forward
# =============================================================================


def _run_hf_reference(
    ref_path: str,
    token_ids: np.ndarray,
    arch_map: dict[str, Any],
    device_map: str = "auto",
) -> tuple[dict[int, np.ndarray], str]:
    """Load HF reference (bf16), register per-layer forward hooks, run once
    with ``token_ids``, copy captures to CPU fp32. Frees all GPU caches on exit.

    ``device_map`` is passed to ``from_pretrained`` — ``"auto"`` shards across
    every visible GPU via accelerate (required for models larger than one
    GPU), ``"cuda"`` forces single-device.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        ref_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    hf_class_name = type(model).__name__

    layers_spec = arch_map["layers_attr"]
    layer_paths = [layers_spec] if isinstance(layers_spec, str) else list(layers_spec)
    layers = None
    tried: list[str] = []
    for path in layer_paths:
        tried.append(path)
        try:
            candidate = _resolve_nested_attr(model, path)
        except AttributeError:
            continue
        if isinstance(candidate, torch.nn.ModuleList) or hasattr(candidate, "__iter__"):
            layers = candidate
            break
    if layers is None:
        raise RuntimeError(f"no layers attr resolved on {type(model).__name__}; tried paths={tried}")

    captures: dict[int, np.ndarray] = {}
    hooks: list[Any] = []

    def _make_hook(idx: int):
        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            t = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(t):
                return
            captures[idx] = t.detach().to(dtype=torch.float32, device="cpu").numpy()

        return _hook

    try:
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(_make_hook(i)))

        with torch.no_grad():
            # For device_map="auto", HF routes inputs via the embeddings module's
            # device. Using the first parameter's device covers both sharded and
            # single-device loads without special-casing.
            input_device = next(model.parameters()).device
            t_ids = torch.from_numpy(token_ids).to(input_device)
            model(input_ids=t_ids)
    finally:
        for h in hooks:
            h.remove()
        del model
        # Free cached blocks on every visible device — a sharded HF load may
        # have allocated across all of them.
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

    return captures, hf_class_name


def _resolve_nested_attr(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


# =============================================================================
# DSL forward
# =============================================================================


def _run_dsl_with_tokens(
    config: Any,
    train_files: list[str],
    token_ids: np.ndarray,
    used_seq_len: int,
    total_rows: int,
) -> str | None:
    """Construct the trainer, overwrite its token buffer with our synthetic ids,
    run one ``step()``. Returns a formatted error string if anything raised
    during either construction or the step itself.

    Any trailing positions past ``used_seq_len`` are padded to 0 so the batch
    shape matches ``config.sequence_len`` — those positions will be computed
    by the runtime but ignored by the HF comparison.
    """
    wrapper_holder: list[Any] = [None]

    def _construct() -> None:
        from surogate.train.trainer import SurogateTrainerWrapper

        wrapper_holder[0] = SurogateTrainerWrapper(config=config, train_files=train_files, eval_files=None)

    ok, err = capture_exception(_construct)
    if not ok:
        return err

    wrapper = wrapper_holder[0]
    in_tokens, out_tokens, pos_ids = allocate_token_buffers(config)

    # Paste synthetic ids; shift into out_tokens target so cross-entropy
    # doesn't see all-pad (some recipes assert on that).
    w = min(used_seq_len, in_tokens.shape[1])
    in_tokens[:, :w] = token_ids[:total_rows, :w]
    out_tokens[:, : max(0, w - 1)] = in_tokens[:, 1:w]

    ok, err = capture_exception(lambda: wrapper.trainer.step(in_tokens, out_tokens, pos_ids))
    return None if ok else err


# =============================================================================
# Dump loading + diff
# =============================================================================


def _load_dsl_dump(
    dump_dir: Path, tensor_name: str, dump_files: set[str]
) -> tuple[np.ndarray | None, list[int] | None, DiffStatus | None]:
    """Load one DSL dump and reshape by the JSON sidecar. Returns
    (tensor, shape, error_status). ``error_status`` is None on success."""
    data, meta, dump_status, _ = load_dump_tensor(dump_dir, tensor_name, dump_files)
    if dump_status == DumpStatus.MISSING:
        return None, None, DiffStatus.DSL_DUMP_MISSING
    if dump_status == DumpStatus.READ_FAILED:
        return None, None, DiffStatus.DSL_READ_FAILED
    assert data is not None
    shape = (meta or {}).get("shape")
    if shape:
        try:
            data = data.reshape(shape)
        except Exception:
            return data, list(data.shape), DiffStatus.DSL_RESHAPE_FAILED
    return data, list(data.shape), None


_EPS = 1e-12  # guard denominator when a tensor is all-zero


def _diff_tensors(hf: np.ndarray, dsl: np.ndarray, used_seq_len: int) -> tuple[dict, DiffStatus]:
    """Compute diff stats in fp32. Truncates the sequence axis to
    ``used_seq_len`` so padding tail of the DSL batch doesn't dominate stats.

    Stays in fp32 end-to-end — bf16→fp32 inputs have way more precision than
    this compare needs, and fp64 upcast was costing 2× memory per pass.
    """
    hf_c = _crop_seq(hf, used_seq_len)
    dsl_c = _crop_seq(dsl, used_seq_len)

    if hf_c.shape != dsl_c.shape:
        shape_info = {"hf_shape_cropped": list(hf_c.shape), "dsl_shape_cropped": list(dsl_c.shape)}
        if hf_c.size != dsl_c.size:
            shape_info.update({"hf_size": int(hf_c.size), "dsl_size": int(dsl_c.size)})
            return shape_info, DiffStatus.SHAPE_MISMATCH
        try:
            dsl_c = dsl_c.reshape(hf_c.shape)
        except Exception:
            return shape_info, DiffStatus.SHAPE_MISMATCH

    # Collapse the nan/inf detection across both tensors into a single sum
    # of booleans — we only need the scalar count to decide whether to bail.
    hf_isnan = np.isnan(hf_c)
    hf_isinf = np.isinf(hf_c)
    dsl_isnan = np.isnan(dsl_c)
    dsl_isinf = np.isinf(dsl_c)
    hf_nan = int(hf_isnan.sum())
    dsl_nan = int(dsl_isnan.sum())
    hf_inf = int(hf_isinf.sum())
    dsl_inf = int(dsl_isinf.sum())
    if hf_nan + hf_inf + dsl_nan + dsl_inf > 0:
        return (
            {
                "hf_shape_cropped": list(hf_c.shape),
                "dsl_shape_cropped": list(dsl_c.shape),
                "hf_nan": hf_nan,
                "hf_inf": hf_inf,
                "dsl_nan": dsl_nan,
                "dsl_inf": dsl_inf,
            },
            DiffStatus.NONFINITE,
        )

    # Clean path: everything finite. Compute stats in fp32. ``np.sum`` and
    # friends use pairwise summation so accumulator error is sqrt(N)*eps,
    # well within bf16 precision.
    diff = dsl_c - hf_c
    abs_diff = np.abs(diff)
    hf_sq_sum = float((hf_c * hf_c).sum())
    dsl_sq_sum = float((dsl_c * dsl_c).sum())
    dot = float((hf_c * dsl_c).sum())
    hf_norm = hf_sq_sum**0.5
    dsl_norm = dsl_sq_sum**0.5
    denom = hf_norm * dsl_norm
    cos_sim = dot / denom if denom > 0 else None
    hf_max_abs = float(np.abs(hf_c).max())
    dsl_max_abs = float(np.abs(dsl_c).max())
    max_abs_diff = float(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())
    rel_err = max_abs_diff / (hf_max_abs + _EPS)

    return (
        {
            "hf_shape_cropped": list(hf_c.shape),
            "dsl_shape_cropped": list(dsl_c.shape),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "cos_sim": cos_sim,
            "rel_err": rel_err,
            "hf_max_abs": hf_max_abs,
            "dsl_max_abs": dsl_max_abs,
            "hf_norm": hf_norm,
            "dsl_norm": dsl_norm,
        },
        DiffStatus.COMPARED,
    )


def _crop_seq(t: np.ndarray, seq_len: int) -> np.ndarray:
    """Crop the sequence axis (assumed dim 1, BSH layout) to ``seq_len``.

    Returns a view, not a copy. Callers must pass ``[batch, seq, ...]`` tensors;
    other layouts will silently mis-crop.
    """
    if t.ndim < 2 or t.shape[1] <= seq_len:
        return t
    slicer = [slice(None)] * t.ndim
    slicer[1] = slice(0, seq_len)
    return t[tuple(slicer)]


def _severity_from_diff(stats: dict, status: DiffStatus, rtol: float, atol: float) -> str:
    if status == DiffStatus.SHAPE_MISMATCH:
        return Severity.ERROR
    if status == DiffStatus.NONFINITE:
        return Severity.WARN
    max_abs_diff = stats.get("max_abs_diff")
    hf_max_abs = stats.get("hf_max_abs")
    if max_abs_diff is None or hf_max_abs is None:
        return Severity.INFO
    if max_abs_diff > atol + rtol * hf_max_abs:
        return Severity.ERROR
    return Severity.INFO
