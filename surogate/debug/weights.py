"""Static weight-mapping audit.

Cross-references the set of HF safetensors keys in a checkpoint with the set
of DSL parameters expected by the compiled model IR. Emits one JSONL record
per DSL param and per HF key so the full state is greppable. Every issue
(unbound param, unused HF key, shape mismatch) is also emitted as its own
error-severity record so ``grep '"severity":"error"'`` pulls the punch list.

No CUDA, no weight data is read — only safetensors headers and the DSL IR.
Completes in seconds even on 70B-scale sharded checkpoints.

Args
----
config_path : str
    Path to a training config YAML (same format as ``surogate sft``). Must
    contain at least a ``model`` field pointing at an HF repo or local dir.
output : str | None
    Output JSONL path. Defaults to ``./debug/debug_weights_<model>_<ts>.jsonl``.
    A sibling ``.header.json`` is written with invocation metadata (run_id,
    git sha, CLI args, config path).
hub_token : str | None
    HuggingFace Hub token for private-repo model downloads.

Returns
-------
int
    ``0`` if the audit completed (inspect the JSONL for findings). ``1`` if
    the audit could not run (model unresolvable, IR compile failure,
    safetensors unreadable, etc.).

Output records (``tag`` field, one per line)
-------------------------------------------
RUN
    Invocation context: ``model_id``, ``model_dir``, ``architecture``.
    Emitted once at the start.
MODEL
    Compiled-IR summary: ``name``, ``kind``, ``dsl_config`` (all resolved
    @hf_config kwargs), ``dsl_param_count`` (per-layer-expanded),
    ``top_level_param_count``, ``hf_mapping_entries``.
SAFETENSORS
    One per shard file: ``file``, ``path``, ``size_bytes``, ``exists``.
HF_KEY
    One per tensor in the checkpoint (pre-audit snapshot): ``hf_key``,
    ``shape``, ``dtype``, ``file``, ``nbytes``.
DSL_PARAM
    One per expected DSL parameter: ``dsl`` (name), ``shape``, ``dtype``,
    ``quantizable``, ``mapping_kind`` (one of :class:`MappingKind`).
MAPPING
    One per DSL param that resolves to at least one HF key: ``dsl``,
    ``kind``, ``hf`` (list of HF keys), ``dim`` / ``ranges`` / ``fn`` /
    ``pattern`` (kind-specific), ``dsl_shape``, ``dsl_dtype``, ``hf_shape``
    or ``hf_shapes``, ``hf_dtype`` or ``hf_dtypes``, ``status`` ∈
    {``loaded``, ``partial``, ``missing``}. severity=error on ``partial`` /
    ``missing``.
UNBOUND
    severity=error. Emitted when a DSL param cannot be bound to any HF key.
    Fields: ``dsl``, ``kind``, ``hf_key_tried`` or ``missing_hf_keys``,
    ``hf_found``. The punch list for "missing params → illegal memory" bugs.
UNUSED_HF
    severity=warn. An HF safetensors key that no DSL param consumed.
    Fields: ``hf_key``, ``shape``, ``dtype``, ``file``. Often benign
    (vision/audio towers in text-only SFT) but flags misnamed mappings.
SHAPE_MISMATCH
    severity=error. DSL-expected shape differs from HF tensor shape.
    Fields: ``dsl``, ``hf_key``, ``dsl_shape``, ``hf_shape`` (+ ``fused_shape``
    for fuse mappings). Catches head_dim / vocab / hidden mismatches.
DTYPE_NOTE
    severity=info. Informational dtype-mismatch note (cast usually fine).
FALLBACK_DIRECT
    severity=warn. A DSL param had no explicit mapping entry and fell through
    to direct-name lookup. Fields: ``dsl``, ``hf_key_tried``, ``hf_found``.
TIED
    A DSL param is ``tied_to`` another DSL param (no HF key of its own).
    Fields: ``dsl``, ``target``, ``dsl_shape``.
ERROR
    severity=error. Top-level phase failure: ``phase`` ∈ {``ir_compile``,
    ``safetensors_enumerate``, ``mapping_dispatch``}, ``error`` message.
SUMMARY
    Emitted last. Fields: ``counts_by_tag``, ``counts_by_severity``,
    ``dsl_param_count``, ``hf_key_count``, ``hf_keys_consumed``,
    ``hf_keys_unused``, ``output_path``, ``header_path``.

Grep recipes
------------
::

    grep '"severity":"error"' debug_weights_*.jsonl             # punch list
    grep '"tag":"UNBOUND"'    debug_weights_*.jsonl             # missing params
    grep '"tag":"UNUSED_HF"'  debug_weights_*.jsonl             # stale HF keys
    jq 'select(.tag=="MAPPING" and .status!="loaded")' debug_*.jsonl
"""

from __future__ import annotations

import os
from typing import Any

from surogate.utils.logger import get_logger

from ._shared import DebugResolveError, resolve_model_and_ir
from .safetensors_index import HfEntry, enumerate_safetensors, shard_files
from .schema import MappingKind, Severity, Tag
from .writer import DebugJsonlWriter, default_output_path, make_run_id

logger = get_logger()


def run_weight_audit(config_path: str, output: str | None = None, hub_token: str | None = None) -> int:
    """Run the weight audit end-to-end. Returns process-level exit code.

    Returns 0 when the audit completed (even if it found errors — inspect the
    JSONL file for the punch list). Returns 1 only if the audit itself could
    not run (model could not be resolved, IR failed to compile, etc.).
    """
    try:
        resolved = resolve_model_and_ir(config_path, hub_token=hub_token)
    except DebugResolveError as e:
        logger.error(str(e))
        return 1

    run_id = make_run_id()
    model_name = os.path.basename(resolved.model_dir.rstrip("/"))
    out_path = output or default_output_path("weights", model_name)

    header: dict[str, Any] = {
        "subcommand": "weights",
        "config_path": os.path.abspath(config_path),
        "model_id": resolved.model_id,
        "model_dir": resolved.model_dir,
        "architecture": resolved.architecture,
        "hf_config_path": resolved.hf_config_path,
    }

    module = resolved.module
    model_dir = resolved.model_dir

    with DebugJsonlWriter(out_path, run_id=run_id, header=header) as w:
        w.write(
            Tag.RUN,
            subcommand="weights",
            model_id=resolved.model_id,
            model_dir=model_dir,
            architecture=resolved.architecture,
        )
        # The authoritative per-layer-expanded param set lives in forward.params
        # (module.params only holds top-level, non-block params). hf_mapping is
        # already expanded per physical layer for block params.
        forward_params: dict[str, dict] = (module.get("forward") or {}).get("params", {}) or module.get("params", {})
        w.write(
            Tag.MODEL,
            name=module.get("name"),
            kind=module.get("kind"),
            dsl_config=module.get("config", {}),
            dsl_param_count=len(forward_params),
            top_level_param_count=len(module.get("params", {})),
            hf_mapping_entries=len(module.get("hf_mapping", {})),
        )
        shards = shard_files(model_dir)
        for shard in shards:
            w.write(Tag.SAFETENSORS, **shard)

        try:
            hf_entries = enumerate_safetensors(model_dir, shards=shards)
        except Exception as e:
            w.write(Tag.ERROR, severity=Severity.ERROR, phase="safetensors_enumerate", error=str(e))
            w.summary(status="safetensors_read_failed")
            logger.error(f"could not read safetensors in {model_dir}: {e}")
            return 1

        # Emit one row per HF key up-front (pre-audit snapshot).
        for e in hf_entries.values():
            w.write(
                Tag.HF_KEY,
                hf_key=e.key,
                shape=list(e.shape),
                dtype=e.dtype,
                file=e.file,
                nbytes=e.nbytes,
            )

        hf_mapping: dict[str, Any] = module.get("hf_mapping", {})
        consumed: set[str] = set()

        for dsl_name, param_info in forward_params.items():
            _audit_param(
                dsl_name=dsl_name,
                param_info=param_info,
                mapping=hf_mapping.get(dsl_name),
                hf_entries=hf_entries,
                consumed=consumed,
                writer=w,
            )

        for key, entry in hf_entries.items():
            if key not in consumed:
                w.write(
                    Tag.UNUSED_HF,
                    severity=Severity.WARN,
                    hf_key=key,
                    shape=list(entry.shape),
                    dtype=entry.dtype,
                    file=entry.file,
                )

        w.summary(
            status="completed",
            dsl_param_count=len(forward_params),
            hf_key_count=len(hf_entries),
            hf_keys_consumed=len(consumed),
            hf_keys_unused=len(hf_entries) - len(consumed),
        )

    errors = w.counts_by_severity.get(Severity.ERROR.value, 0)
    warns = w.counts_by_severity.get(Severity.WARN.value, 0)
    logger.info(f"wrote {out_path} ({errors} errors, {warns} warnings)")
    return 0


def _audit_param(
    *,
    dsl_name: str,
    param_info: dict,
    mapping: Any,
    hf_entries: dict[str, HfEntry],
    consumed: set[str],
    writer: DebugJsonlWriter,
) -> None:
    """Emit records for one DSL param: DSL_PARAM, MAPPING, and any flags."""
    dsl_shape = list(param_info.get("shape") or [])
    dsl_dtype = param_info.get("dtype")
    kind = _mapping_kind(mapping)
    writer.write(
        Tag.DSL_PARAM,
        dsl=dsl_name,
        shape=dsl_shape,
        dtype=dsl_dtype,
        quantizable=param_info.get("quantizable", True),
        mapping_kind=kind,
    )

    if kind == MappingKind.TIED_TO:
        target = mapping.get("target") if isinstance(mapping, dict) else None
        writer.write(
            Tag.TIED,
            dsl=dsl_name,
            target=target,
            dsl_shape=dsl_shape,
            dsl_dtype=dsl_dtype,
        )
        return

    if kind == MappingKind.DIRECT:
        hf_key = mapping if isinstance(mapping, str) else mapping.get("source")
        _audit_single_key(
            dsl_name=dsl_name,
            hf_key=hf_key,
            dsl_shape=dsl_shape,
            dsl_dtype=dsl_dtype,
            hf_entries=hf_entries,
            consumed=consumed,
            writer=writer,
            mapping_kind=kind,
            extra={},
        )
        return

    if kind == MappingKind.NONE:
        _audit_single_key(
            dsl_name=dsl_name,
            hf_key=dsl_name,
            dsl_shape=dsl_shape,
            dsl_dtype=dsl_dtype,
            hf_entries=hf_entries,
            consumed=consumed,
            writer=writer,
            mapping_kind=MappingKind.DIRECT,
            extra={"fallback": True},
            fallback_record=True,
        )
        return

    if kind == MappingKind.FUSE:
        sources = list(mapping.get("sources", []))
        dim = int(mapping.get("dim", 0))
        present, missing = _resolve_keys(sources, hf_entries)
        src_shapes = [list(hf_entries[k].shape) for k in present]
        src_dtypes = [hf_entries[k].dtype for k in present]

        for k in present:
            consumed.add(k)

        status = "loaded" if not missing else "partial" if present else "missing"
        writer.write(
            Tag.MAPPING,
            severity=Severity.INFO if not missing else Severity.ERROR,
            dsl=dsl_name,
            kind=MappingKind.FUSE,
            hf=sources,
            dim=dim,
            dsl_shape=dsl_shape,
            dsl_dtype=dsl_dtype,
            hf_shapes=src_shapes,
            hf_dtypes=src_dtypes,
            status=status,
        )
        if missing:
            writer.write(
                Tag.UNBOUND,
                severity=Severity.ERROR,
                dsl=dsl_name,
                kind=MappingKind.FUSE,
                missing_hf_keys=missing,
                present_hf_keys=present,
            )
            return

        _check_fuse_shape(
            dsl_name=dsl_name,
            dsl_shape=dsl_shape,
            src_shapes=src_shapes,
            dim=dim,
            sources=sources,
            writer=writer,
        )
        return

    if kind == MappingKind.SPLIT:
        src = mapping.get("source")
        dim = int(mapping.get("dim", 0))
        ranges = mapping.get("ranges", [])
        present, missing = _resolve_keys([src], hf_entries)
        if missing:
            writer.write(
                Tag.UNBOUND,
                severity=Severity.ERROR,
                dsl=dsl_name,
                kind=MappingKind.SPLIT,
                missing_hf_keys=missing,
            )
            return
        hf_shape = list(hf_entries[src].shape)
        consumed.add(src)
        writer.write(
            Tag.MAPPING,
            dsl=dsl_name,
            kind=MappingKind.SPLIT,
            hf=[src],
            dim=dim,
            ranges=ranges,
            dsl_shape=dsl_shape,
            dsl_dtype=dsl_dtype,
            hf_shape=hf_shape,
            hf_dtype=hf_entries[src].dtype,
            status="loaded",
            note="shape validation skipped for split (range-dependent)",
        )
        return

    if kind == MappingKind.TRANSFORM:
        src = mapping.get("source")
        fn = mapping.get("fn")
        present, missing = _resolve_keys([src], hf_entries)
        if missing:
            writer.write(
                Tag.UNBOUND,
                severity=Severity.ERROR,
                dsl=dsl_name,
                kind=MappingKind.TRANSFORM,
                fn=fn,
                missing_hf_keys=missing,
            )
            return
        hf_shape = list(hf_entries[src].shape)
        consumed.add(src)
        writer.write(
            Tag.MAPPING,
            dsl=dsl_name,
            kind=MappingKind.TRANSFORM,
            hf=[src],
            fn=fn,
            dsl_shape=dsl_shape,
            dsl_dtype=dsl_dtype,
            hf_shape=hf_shape,
            hf_dtype=hf_entries[src].dtype,
            status="loaded",
            note=f"shape validation skipped for transform fn={fn!r}",
        )
        return

    if kind == MappingKind.STACK_EXPERTS:
        pattern = mapping.get("pattern", "")
        declared_n = mapping.get("num_experts", 0)
        fuse_gate_up = mapping.get("fuse_gate_up", False)
        expected_n = declared_n or (dsl_shape[0] if dsl_shape else 0)

        resolved = []
        missing = []
        for e in range(int(expected_n) if expected_n else 0):
            k = pattern.replace("{expert}", str(e))
            if k in hf_entries:
                resolved.append(k)
                consumed.add(k)
            else:
                missing.append(k)

        status = "loaded" if not missing and resolved else ("partial" if resolved else "missing")
        writer.write(
            Tag.MAPPING,
            severity=Severity.INFO if status == "loaded" else Severity.ERROR,
            dsl=dsl_name,
            kind=MappingKind.STACK_EXPERTS,
            pattern=pattern,
            num_experts=expected_n,
            fuse_gate_up=fuse_gate_up,
            dsl_shape=dsl_shape,
            dsl_dtype=dsl_dtype,
            hf_keys_present=resolved,
            hf_keys_missing=missing,
            status=status,
        )
        if missing:
            writer.write(
                Tag.UNBOUND,
                severity=Severity.ERROR,
                dsl=dsl_name,
                kind=MappingKind.STACK_EXPERTS,
                missing_hf_keys=missing,
                present_hf_keys=resolved,
            )
        return

    writer.write(
        Tag.ERROR,
        severity=Severity.ERROR,
        phase="mapping_dispatch",
        dsl=dsl_name,
        mapping=mapping,
        error="unrecognized mapping kind",
    )


def _mapping_kind(mapping: Any) -> str:
    if mapping is None:
        return MappingKind.NONE
    if isinstance(mapping, str):
        return MappingKind.DIRECT
    if isinstance(mapping, dict):
        t = mapping.get("type")
        if t in {
            MappingKind.DIRECT,
            MappingKind.FUSE,
            MappingKind.SPLIT,
            MappingKind.TRANSFORM,
            MappingKind.TIED_TO,
            MappingKind.STACK_EXPERTS,
        }:
            return t
    return MappingKind.NONE


def _resolve_keys(keys: list[str], hf_entries: dict[str, HfEntry]) -> tuple[list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    for k in keys:
        if k in hf_entries:
            present.append(k)
        else:
            missing.append(k)
    return present, missing


def _audit_single_key(
    *,
    dsl_name: str,
    hf_key: str | None,
    dsl_shape: list[int],
    dsl_dtype: str | None,
    hf_entries: dict[str, HfEntry],
    consumed: set[str],
    writer: DebugJsonlWriter,
    mapping_kind: str,
    extra: dict,
    fallback_record: bool = False,
) -> None:
    if hf_key is None:
        writer.write(
            Tag.UNBOUND,
            severity=Severity.ERROR,
            dsl=dsl_name,
            kind=mapping_kind,
            error="mapping produced no HF key",
        )
        return

    if hf_key not in hf_entries:
        writer.write(
            Tag.UNBOUND,
            severity=Severity.ERROR,
            dsl=dsl_name,
            kind=mapping_kind,
            hf_key_tried=hf_key,
            hf_found=False,
            **extra,
        )
        if fallback_record:
            writer.write(
                Tag.FALLBACK_DIRECT,
                severity=Severity.WARN,
                dsl=dsl_name,
                hf_key_tried=hf_key,
                hf_found=False,
            )
        return

    entry = hf_entries[hf_key]
    consumed.add(hf_key)
    writer.write(
        Tag.MAPPING,
        dsl=dsl_name,
        kind=mapping_kind,
        hf=[hf_key],
        dsl_shape=dsl_shape,
        dsl_dtype=dsl_dtype,
        hf_shape=list(entry.shape),
        hf_dtype=entry.dtype,
        status="loaded",
        **extra,
    )
    if fallback_record:
        writer.write(
            Tag.FALLBACK_DIRECT,
            severity=Severity.WARN,
            dsl=dsl_name,
            hf_key_tried=hf_key,
            hf_found=True,
            hf_shape=list(entry.shape),
            dsl_shape=dsl_shape,
        )
    _check_shape_match(dsl_name, dsl_shape, list(entry.shape), hf_key, writer)


def _check_shape_match(
    dsl_name: str,
    dsl_shape: list[int],
    hf_shape: list[int],
    hf_key: str,
    writer: DebugJsonlWriter,
) -> None:
    """Emit SHAPE_MISMATCH if a numeric comparison is possible and disagrees.

    Symbolic DSL shapes (strings like ``"d_model"``) are skipped silently —
    they'll be resolved at runtime, not here.
    """
    if not dsl_shape or not hf_shape:
        return
    if not all(isinstance(d, int) for d in dsl_shape):
        return
    if list(dsl_shape) != list(hf_shape):
        writer.write(
            Tag.SHAPE_MISMATCH,
            severity=Severity.ERROR,
            dsl=dsl_name,
            hf_key=hf_key,
            dsl_shape=dsl_shape,
            hf_shape=hf_shape,
        )


def _check_fuse_shape(
    *,
    dsl_name: str,
    dsl_shape: list[int],
    src_shapes: list[list[int]],
    dim: int,
    sources: list[str],
    writer: DebugJsonlWriter,
) -> None:
    if not dsl_shape or not src_shapes:
        return
    if not all(isinstance(d, int) for d in dsl_shape):
        return
    try:
        expected = list(src_shapes[0])
        expected[dim] = sum(int(s[dim]) for s in src_shapes)
    except Exception:
        return
    if list(dsl_shape) != expected:
        writer.write(
            Tag.SHAPE_MISMATCH,
            severity=Severity.ERROR,
            dsl=dsl_name,
            kind=MappingKind.FUSE,
            dim=dim,
            sources=sources,
            dsl_shape=dsl_shape,
            fused_shape=expected,
            src_shapes=src_shapes,
        )
