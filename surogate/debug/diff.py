"""Layer-by-layer numerical diff vs HuggingFace transformers.

Runs the same checkpoint twice — once through ``AutoModelForCausalLM`` as the
reference, once through our DSL runtime — with identical synthetic token
input, and compares per-layer hidden-state outputs. The first layer whose
``max_abs_diff`` exceeds ``atol + rtol * max(|hf|)`` is the divergence site —
the killer tool for silent DSL-vs-HF bugs.

The two forwards run **sequentially on the same GPU** (HF first, freed, then
DSL). Same device avoids CPU/GPU bf16-rounding artifacts that would otherwise
look like divergence. HF loads with ``dtype=bfloat16`` so precision
matches the DSL runtime; the diff arithmetic is in fp32 on CPU.

With ``packed=True`` every row holds two documents (``doc1`` then ``doc2``,
position ids restarting at 0 where ``doc2`` starts, as a packed training row
has them). The HF reference runs each document as its own sequence, and each
document's region of the DSL row is compared with its own reference. ``doc1``
sees nothing before it, so a ``doc2`` that diverges while ``doc1`` matches is
state or attention leaking across the document boundary.

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
ref_device_map : str | None
    Passed to ``AutoModelForCausalLM.from_pretrained(device_map=...)``. Default
    ``None`` places the reference on ``cuda:0`` when its bf16 weights fit there
    (estimated from the safetensors headers against the card's free memory) and
    falls back to ``"auto"`` otherwise, or when the single-card load runs out of
    memory. ``"auto"`` shards the reference across every visible GPU (requires
    ``accelerate``), which makes it depend on the peer-to-peer path between the
    cards: where that path is broken, the layers on the second card come back
    NaN or zero. Any other value (``"cuda"``, ``"cuda:1"``) is passed through.
    Set ``CUDA_VISIBLE_DEVICES`` upstream to control which GPUs are eligible;
    the DSL runs on the first visible GPU after the HF model is freed.
packed : bool
    Feed two packed documents per row instead of one sequence (see above).
pack_split : int | None
    Tokens in ``doc1`` when ``packed``. Default: half the sequence.

Returns
-------
int
    ``0`` on success (inspect JSONL for severity=error). ``1`` on setup failure.

Output records (``tag`` field, one per line)
-------------------------------------------
RUN
    Invocation context: ``model_id``, ``reference``, ``architecture``, ``n_layers``,
    ``seq_len``, ``seed``, ``rtol``, ``atol``, ``packed``, ``documents`` (packed:
    ``[{doc, start, end}]``, token ranges within each row).
MODEL
    Compiled-IR summary + ``arch_map`` (the HF-layers-attr → DSL-slot mapping used).
REFERENCE
    HF reference context: ``path``, ``hf_class``, ``dtype="bf16"``, ``device_map``
    (what was passed to ``from_pretrained``), ``placement`` (why), ``devices``
    (every device a hooked layer ran on).
DIFF
    One per layer (per layer and rank on a multi-GPU config, with a ``rank`` field:
    each rank is compared against the HF rows it consumed; per layer and document
    when packed, with ``doc`` and ``tokens`` fields). Fields: ``layer``,
    ``op`` (DSL slot name, usually ``res_ffn``),
    ``slot`` (full ``blocks[N].<slot>``), ``hf_shape``, ``dsl_shape``, ``status``,
    plus diff stats: ``max_abs_diff``, ``mean_abs_diff``, ``cos_sim``, ``rel_err``,
    ``hf_max_abs``, ``dsl_max_abs``, ``hf_norm``, ``dsl_norm``.
    severity=info when within tolerance; =warn when dumps missing or either
    side has non-finite values; =error on divergence > tolerance. A reference
    layer that is non-finite or all zero (``status`` ``nonfinite`` with
    ``hf_nan``/``hf_inf``, or ``hf_all_zero``) carries ``hf_device`` and a
    ``note``: the reference is broken there, not the DSL.
ERROR
    severity=error. ``phase`` ∈ {``hf_reference``, ``dsl_step``, ``diff_scan``,
    ``doc_boundary``}. ``diff_scan`` records the first diverging layer;
    ``doc_boundary`` (packed) the first layer where doc2 is far further from its
    reference than doc1 is from its own, i.e. a cross-document leak.
SUMMARY
    Terminal. ``counts_by_tag``, ``counts_by_severity``, ``status``,
    ``first_diverging_layer``, ``layers_compared``, ``rtol``, ``atol``,
    ``hf_bad_layers`` (reference layers that were non-finite or zero); packed
    adds ``first_diverging_layer_by_doc`` and ``doc_boundary_leak_layer``.

Grep recipes
------------
::

    grep '"severity":"error"' debug_diff_*.jsonl                 # divergence sites
    jq 'select(.tag=="DIFF") | {layer,max_abs_diff,cos_sim}' debug_*.jsonl
    jq 'select(.tag=="DIFF") | {layer,doc,cos_sim}' debug_*.jsonl   # --packed
"""

from __future__ import annotations

import gc
import math
import os
import tempfile
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
from .safetensors_index import enumerate_safetensors
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


class _Document(NamedTuple):
    """Token range ``[start, end)`` of every row. ``name`` is ``None`` for the one
    unpacked sequence and ``doc1``/``doc2`` when packed."""

    name: str | None
    start: int
    end: int


class _FirstDiff(NamedTuple):
    layer: int
    max_abs_diff: float
    doc: str | None = None


class _Leak(NamedTuple):
    layer: int
    rank: int
    doc1_cos_sim: float
    doc2_cos_sim: float


class _HfReference(NamedTuple):
    outputs: list[dict[int, np.ndarray]]  # per document: {hf_layer: [rows, doc_len, hidden]}
    hf_class: str
    layer_devices: dict[int, str]  # device each hooked layer's output came from
    device_map: str  # what ``from_pretrained`` was given
    placement: str  # why


def run_reference_diff(
    config_path: str,
    output: str | None = None,
    hub_token: str | None = None,
    reference: str | None = None,
    max_tokens: int = 64,
    seed: int = 0,
    rtol: float = 1e-2,
    atol: float = 1e-3,
    ref_device_map: str | None = None,
    packed: bool = False,
    pack_split: int | None = None,
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
            packed,
            pack_split,
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
    ref_device_map: str | None,
    packed: bool,
    pack_split: int | None,
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

    try:
        documents = _documents(seq_len, packed, pack_split)
    except ValueError as e:
        logger.error(str(e))
        return 1
    if pack_split is not None and not packed:
        logger.warning("--pack-split has no effect without --packed")

    rng = np.random.default_rng(int(seed))
    token_ids = rng.integers(0, vocab_size, size=(total_rows, seq_len), dtype=np.int32)
    position_ids = _position_ids(total_rows, documents)

    ref_path = reference or resolved.model_dir
    run_id = make_run_id()
    model_name = os.path.basename(resolved.model_dir.rstrip("/"))
    out_path = output or default_output_path("diff", model_name)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    dump_dir = dumps_root / run_id
    dump_dir.mkdir(parents=True, exist_ok=True)

    hf_ref: _HfReference | None = None

    def _do_hf() -> None:
        nonlocal hf_ref
        hf_ref = _run_hf_reference(
            ref_path,
            [token_ids[:, d.start : d.end] for d in documents],
            arch_map,
            device_map=ref_device_map,
            vocab_size=vocab_size,
        )

    ok, _hf_err = capture_exception(_do_hf)
    hf_error: str | None = None if ok else _hf_err
    if hf_error:
        logger.error(f"HF reference failed: {hf_error.splitlines()[0]}")
    hf_outputs = hf_ref.outputs if hf_ref else [{} for _ in documents]
    hf_class_name = hf_ref.hf_class if hf_ref else None
    layer_devices = hf_ref.layer_devices if hf_ref else {}
    ref_devices = sorted(set(layer_devices.values()))

    os.environ["SUROGATE_DEBUG_DUMP_DIR"] = str(dump_dir)
    os.environ["SUROGATE_DEBUG_DUMP_TENSORS"] = ",".join(f"blocks[{i}].{dsl_slot}" for i in range(n_layers))
    # The diff tool compares a short synthetic sequence by default. Run the DSL
    # side at that exact length as well; allocating the full training
    # ``sequence_len`` makes the compare path much heavier than the HF side and
    # can OOM needlessly on otherwise valid configs.
    config.sequence_len = seq_len
    # Keep debug tokenization/logs isolated from the user's training output dir.
    config.output_dir = tempfile.mkdtemp(prefix="surogate_debug_diff_run_", dir=str(dumps_root))
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
    # The runtime options exist once tokenization has finalised the config.
    runtime = getattr(config, "runtime_config", None)
    if packed and not getattr(runtime, "doc_masking", True):
        logger.warning(
            "doc_masking is off in this config: the runtime treats each row as one sequence, "
            "so doc2 attends to doc1 and is expected to diverge"
        )

    dsl_error = _run_dsl_with_tokens(config, train_files, token_ids, position_ids, seq_len, total_rows)
    if dsl_error:
        logger.error(f"DSL step failed: {dsl_error.splitlines()[0]}")
    multi_doc = len(documents) > 1
    doc_ranges = [{"doc": d.name, "start": d.start, "end": d.end} for d in documents] if multi_doc else None
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
        "packed": bool(packed),
        "documents": doc_ranges,
        "ref_device_map": hf_ref.device_map if hf_ref else ref_device_map,
        "ref_devices": ref_devices,
        "hf_error": hf_error,
        "dsl_error": dsl_error,
    }

    first_diff: _FirstDiff | None = None
    first_by_doc: dict[str, int] = {}
    leak: _Leak | None = None
    hf_bad_layers: dict[int, str | None] = {}
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
            packed=bool(packed),
            documents=doc_ranges,
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
            device_map=hf_ref.device_map if hf_ref else ref_device_map,
            placement=hf_ref.placement if hf_ref else None,
            devices=ref_devices,
        )

        if hf_error:
            w.write(Tag.ERROR, severity=Severity.ERROR, phase="hf_reference", error=hf_error)
        if dsl_error:
            w.write(Tag.ERROR, severity=Severity.ERROR, phase="dsl_step", error=dsl_error)

        # Each rank dumps its own rows: rank 0 into ``dump_dir``, rank r > 0 into
        # ``dump_dir/rank<r>`` (the runtime suffixes the directory by rank). Rank r
        # consumed host rows [row * B, (row + 1) * B) with row = r // ep_size, so it
        # is compared against exactly those rows of the HF reference.
        n_ranks = max(1, int(config.gpus or 1))
        rows_per_rank = int(config.per_device_train_batch_size)
        ep_size = max(1, int(getattr(config, "ep_size", 1) or 1))
        rank_dumps: list[tuple[int, Path, set[str]]] = []
        for rank in range(n_ranks):
            rank_dir = dump_dir if rank == 0 else dump_dir / f"rank{rank}"
            rank_dumps.append((rank, rank_dir, set(os.listdir(rank_dir)) if rank_dir.exists() else set()))
        # ``hf_layer_idx`` indexes HF's transformer block. We pair it with DSL
        # ``blocks[hf_layer_idx + layer_offset].<slot>``. The final HF layer's
        # output has no matched DSL block in fused-residual models (the final
        # residual stream is a global slot the layer-end hook can't capture),
        # so we emit a warn record for it instead of faking a mismatch.
        max_hf = max(len(o) for o in hf_outputs)
        for hf_layer_idx in range(max_hf):
            dsl_layer_idx = hf_layer_idx + layer_offset
            tensor_name = f"blocks[{dsl_layer_idx}].{dsl_slot}"
            hf_docs = [o.get(hf_layer_idx) for o in hf_outputs]
            hf_shape = _row_shape(hf_docs, seq_len)

            if dsl_layer_idx >= n_layers:
                layers_missing += 1
                w.write(
                    Tag.DIFF,
                    severity=Severity.WARN,
                    layer=hf_layer_idx,
                    op=dsl_slot,
                    slot=tensor_name,
                    status=DiffStatus.NO_PAIRED_SLOT,
                    hf_shape=hf_shape,
                    note=_NO_PAIRED_SLOT_NOTE.format(hf=hf_layer_idx, dsl=dsl_layer_idx, offset=layer_offset),
                )
                continue

            if hf_shape is None:
                layers_missing += 1
                w.write(
                    Tag.DIFF,
                    severity=Severity.WARN,
                    layer=hf_layer_idx,
                    op=dsl_slot,
                    slot=tensor_name,
                    status=DiffStatus.HF_OUTPUT_MISSING,
                )
                continue

            layer_complete = True
            for rank, rank_dir, rank_files in rank_dumps:
                rank_fields = {"rank": rank} if n_ranks > 1 else {}
                row0 = (rank // ep_size) * rows_per_rank
                dsl_t, dsl_shape, dsl_load_err = _load_dsl_dump(rank_dir, tensor_name, rank_files)
                if dsl_load_err:
                    layer_complete = False
                    w.write(
                        Tag.DIFF,
                        severity=Severity.WARN,
                        layer=hf_layer_idx,
                        op=dsl_slot,
                        slot=tensor_name,
                        status=dsl_load_err,
                        hf_shape=[rows_per_rank, *hf_shape[1:]] if n_ranks > 1 else hf_shape,
                        **rank_fields,
                    )
                    continue

                doc_cos: dict[str | None, float | None] = {}
                for doc, hf_t in zip(documents, hf_docs, strict=True):
                    hf_rows = hf_t[row0 : row0 + rows_per_rank] if n_ranks > 1 else hf_t
                    doc_fields = {"doc": doc.name, "tokens": [doc.start, doc.end]} if multi_doc else {}
                    if multi_doc:
                        dsl_doc = _doc_region(dsl_t, hf_rows.shape[0], seq_len, doc)
                        if dsl_doc is None:
                            stats, status = {"dsl_size": int(dsl_t.size)}, DiffStatus.SHAPE_MISMATCH
                        else:
                            stats, status = _diff_tensors(hf_rows, dsl_doc, doc.end - doc.start)
                    else:
                        stats, status = _diff_tensors(hf_rows, dsl_t, seq_len)
                    severity = _severity_from_diff(stats, status, rtol, atol)
                    doc_cos[doc.name] = stats.get("cos_sim") if status == DiffStatus.COMPARED else None
                    hf_note: dict[str, Any] = {}
                    if _hf_side_broken(stats, status):
                        device = layer_devices.get(hf_layer_idx)
                        hf_bad_layers.setdefault(hf_layer_idx, device)
                        hf_note = {"hf_device": device, "note": _hf_broken_note(device, ref_devices)}

                    w.write(
                        Tag.DIFF,
                        severity=severity,
                        layer=hf_layer_idx,
                        op=dsl_slot,
                        slot=tensor_name,
                        dsl_layer=dsl_layer_idx,
                        status=status,
                        hf_shape=list(hf_rows.shape),
                        dsl_shape=dsl_shape,
                        **rank_fields,
                        **doc_fields,
                        **stats,
                        **hf_note,
                    )
                    if severity == Severity.ERROR:
                        if first_diff is None:
                            first_diff = _FirstDiff(
                                layer=hf_layer_idx,
                                max_abs_diff=stats.get("max_abs_diff", 0.0),
                                doc=doc.name,
                            )
                        if doc.name is not None:
                            first_by_doc.setdefault(doc.name, hf_layer_idx)
                if leak is None and multi_doc and _crosses_boundary(doc_cos.get("doc1"), doc_cos.get("doc2")):
                    leak = _Leak(hf_layer_idx, rank, doc_cos["doc1"], doc_cos["doc2"])
            if layer_complete:
                layers_compared += 1
            else:
                layers_missing += 1

        hf_broken_msg: str | None = None
        if hf_bad_layers:
            bad = sorted(hf_bad_layers)
            hf_broken_msg = (
                f"HF reference is non-finite or all zero at {len(bad)} layer(s) "
                f"(first: layer {bad[0]} on {hf_bad_layers[bad[0]]}); "
                f"{_hf_broken_note(hf_bad_layers[bad[0]], ref_devices)}"
            )
            w.write(
                Tag.ERROR,
                severity=Severity.ERROR,
                phase="hf_reference",
                layers=bad,
                devices=ref_devices,
                error=hf_broken_msg,
            )

        leak_msg: str | None = None
        if leak is not None:
            leak_msg = (
                f"doc2 leaves its reference at layer {leak.layer} (cos_sim {leak.doc2_cos_sim:.5f}) where doc1 "
                f"stays on it (cos_sim {leak.doc1_cos_sim:.5f}): state or attention crosses the packed "
                "document boundary"
            )
            w.write(
                Tag.ERROR,
                severity=Severity.ERROR,
                phase="doc_boundary",
                layer=leak.layer,
                **({"rank": leak.rank} if n_ranks > 1 else {}),
                doc1_cos_sim=leak.doc1_cos_sim,
                doc2_cos_sim=leak.doc2_cos_sim,
                error=leak_msg,
            )

        if first_diff is not None:
            where = f" in {first_diff.doc}" if first_diff.doc else ""
            w.write(
                Tag.ERROR,
                severity=Severity.ERROR,
                phase="diff_scan",
                first_diverging_layer=first_diff.layer,
                max_abs_diff=first_diff.max_abs_diff,
                rtol=float(rtol),
                atol=float(atol),
                **({"doc": first_diff.doc} if first_diff.doc else {}),
                error=(
                    f"first divergence exceeds tolerance at layer {first_diff.layer}{where} "
                    f"(max_abs_diff={first_diff.max_abs_diff:.3e})"
                ),
            )

        status = "completed"
        if hf_error or dsl_error or hf_bad_layers:
            status = "completed_with_errors"
        w.summary(
            status=status,
            first_diverging_layer=first_diff.layer if first_diff else None,
            **(
                {"first_diverging_layer_by_doc": first_by_doc, "doc_boundary_leak_layer": leak.layer if leak else None}
                if multi_doc
                else {}
            ),
            layers_compared=layers_compared,
            layers_missing=layers_missing,
            hf_bad_layers=sorted(hf_bad_layers),
            rtol=float(rtol),
            atol=float(atol),
        )

    errors = w.counts_by_severity.get(Severity.ERROR.value, 0)
    warns = w.counts_by_severity.get(Severity.WARN.value, 0)
    if hf_broken_msg:
        logger.error(hf_broken_msg)
    if leak_msg:
        logger.error(leak_msg)
    if first_diff is not None:
        where = f" in {first_diff.doc}" if first_diff.doc else ""
        logger.error(
            f"first divergence at layer={first_diff.layer}{where} "
            f"max_abs_diff={first_diff.max_abs_diff:.3e} "
            f"({errors} errors, {warns} warnings) — see {out_path}"
        )
    else:
        logger.info(
            f"wrote {out_path} ({errors} errors, {warns} warnings, {layers_compared}/{n_layers} layers compared)"
        )
    return 0


def _documents(seq_len: int, packed: bool, pack_split: int | None) -> list[_Document]:
    """The row layout: one sequence, or ``doc1`` + ``doc2`` split at ``pack_split``
    (default: half the sequence)."""
    if not packed:
        return [_Document(None, 0, seq_len)]
    split = seq_len // 2 if pack_split is None else int(pack_split)
    if not 0 < split < seq_len:
        raise ValueError(
            f"--pack-split {split} leaves a document empty: it must be between 1 and {seq_len - 1} "
            f"for a {seq_len}-token row (--max-tokens sets the row length)"
        )
    return [_Document("doc1", 0, split), _Document("doc2", split, seq_len)]


def _position_ids(rows: int, documents: list[_Document]) -> np.ndarray:
    """Position ids that restart at 0 at every document: the runtime's packed-document
    boundary (``compute_doc_masking``), as a packed training row carries it."""
    row = np.concatenate([np.arange(d.end - d.start, dtype=np.int32) for d in documents])
    return np.tile(row, (rows, 1))


def _row_shape(hf_docs: list[np.ndarray | None], seq_len: int) -> list[int] | None:
    """Shape of a whole HF row ``[rows, seq_len, ...]`` across its documents, or None
    when any document's output is missing."""
    if not hf_docs or any(t is None for t in hf_docs):
        return None
    first = hf_docs[0]
    return [int(first.shape[0]), int(seq_len), *(int(s) for s in first.shape[2:])]


def _doc_region(dsl: np.ndarray, rows: int, seq_len: int, doc: _Document) -> np.ndarray | None:
    """``doc``'s token range of a DSL dump, viewed as ``[rows, seq_len, ...]``; None
    when the dump does not hold ``rows x seq_len`` tokens."""
    if dsl.ndim < 2 or dsl.shape[0] != rows or dsl.shape[1] != seq_len:
        if rows <= 0 or seq_len <= 0 or dsl.size % (rows * seq_len):
            return None
        dsl = dsl.reshape(rows, seq_len, -1)
    return dsl[:, doc.start : doc.end]


# doc2 sees what doc1 cannot (the tokens before it in the row) when, at the same layer, it is
# this many times further from its reference than doc1 is from its own (cosine distance), and
# clearly off rather than bf16 noise. Tolerance-free: the default rtol already flags plain bf16
# drift on some models (Qwen3.5-0.8B: cos_sim 0.9999 at 2-4% max-abs error), in both documents.
_LEAK_DISTANCE_RATIO = 10.0
_LEAK_DISTANCE_FLOOR = 1e-3


def _crosses_boundary(doc1_cos: float | None, doc2_cos: float | None) -> bool:
    if doc1_cos is None or doc2_cos is None:
        return False
    d1 = max(0.0, 1.0 - doc1_cos)
    d2 = max(0.0, 1.0 - doc2_cos)
    return d2 > _LEAK_DISTANCE_FLOOR and d2 > _LEAK_DISTANCE_RATIO * d1


def _hf_side_broken(stats: dict, status: DiffStatus) -> bool:
    """The reference itself is unusable at this layer: NaN/Inf or exactly zero."""
    if status == DiffStatus.HF_ZERO:
        return True
    return status == DiffStatus.NONFINITE and (stats.get("hf_nan", 0) + stats.get("hf_inf", 0)) > 0


def _hf_broken_note(device: str | None, ref_devices: list[str]) -> str:
    if len(ref_devices) > 1:
        return (
            f"the reference was sharded over {', '.join(ref_devices)} and this layer ran on {device}; "
            "a broken peer-to-peer path between the cards gives exactly this. Rerun with "
            "--ref-device-map cuda:0 to keep the reference on one card"
        )
    return (
        f"the reference ran on {device or 'one device'}, so the HF model itself is broken at this layer "
        "and the comparison there says nothing about the DSL"
    )


# =============================================================================
# HF reference forward
# =============================================================================

# Fixed allowance on top of the weights and logits for the CUDA context, the
# allocator's slack and the attention workspaces of a short forward.
_REF_HEADROOM_BYTES = 1 << 30


def _run_hf_reference(
    ref_path: str,
    documents: list[np.ndarray],
    arch_map: dict[str, Any],
    device_map: str | None = None,
    vocab_size: int = 0,
) -> _HfReference:
    """Run the HF reference once per document (each ``[rows, doc_len]``).

    ``device_map=None`` places it on one card when it fits (see the module
    docstring) and retries sharded (``"auto"``) when that card runs out of
    memory; any other value is passed to ``from_pretrained`` as is.
    """
    if device_map is not None:
        return _hf_forward(ref_path, documents, arch_map, device_map, "requested")

    device_map, placement = _auto_ref_placement(ref_path, documents, vocab_size)
    if device_map == "auto":
        if torch.cuda.device_count() > 1:
            logger.warning(
                f"HF reference sharded over {torch.cuda.device_count()} GPUs ({placement}): it depends on "
                "the peer-to-peer path between them; layers that come back NaN or zero are flagged"
            )
        return _hf_forward(ref_path, documents, arch_map, device_map, placement)

    logger.info(f"HF reference on {device_map} ({placement})")
    try:
        return _hf_forward(ref_path, documents, arch_map, device_map, placement)
    except torch.cuda.OutOfMemoryError:
        pass
    # Outside the except block, so the traceback no longer pins the partial load.
    _free_cuda_caches()
    logger.warning(f"HF reference ran out of memory on {device_map}; sharding it over every visible GPU instead")
    return _hf_forward(ref_path, documents, arch_map, "auto", f"{device_map} ran out of memory")


def _auto_ref_placement(ref_path: str, documents: list[np.ndarray], vocab_size: int) -> tuple[str, str]:
    if not torch.cuda.is_available():
        return "auto", "no CUDA device"
    free_bytes, _total = torch.cuda.mem_get_info(0)
    rows = max(int(d.shape[0]) for d in documents)
    doc_len = max(int(d.shape[1]) for d in documents)
    return _choose_ref_device_map(_reference_weight_bytes(ref_path), int(free_bytes), rows, doc_len, vocab_size)


def _choose_ref_device_map(
    weight_bytes: int | None, free_bytes: int, rows: int, doc_len: int, vocab_size: int
) -> tuple[str, str]:
    """``("cuda:0", why)`` when the reference fits the first card, else ``("auto", why)``.

    The estimate is the bf16 weights plus 5%, the forward's logits in fp32 and a fixed
    headroom. An unknown size tries the card (the caller falls back on OOM)."""
    gib = float(1 << 30)
    if weight_bytes is None:
        return "cuda:0", "one card; weight size unknown, sharded instead if it runs out of memory"
    need = int(weight_bytes * 1.05) + rows * doc_len * max(0, vocab_size) * 4 + _REF_HEADROOM_BYTES
    sizes = f"~{need / gib:.1f} GiB needed, {free_bytes / gib:.1f} GiB free on cuda:0"
    if need <= free_bytes:
        return "cuda:0", f"one card: {sizes}"
    return "auto", f"does not fit one card: {sizes}"


def _reference_weight_bytes(ref_path: str) -> int | None:
    """Bytes the bf16 reference's weights take on the card, from the safetensors
    headers (no weight is read); None when ``ref_path`` is not a local checkpoint."""
    if not os.path.isdir(ref_path):
        return None
    try:
        entries = enumerate_safetensors(ref_path)
    except (OSError, ValueError):
        return None
    total = 0
    for entry in entries.values():
        if entry.dtype.startswith(("F", "BF")):
            total += math.prod(entry.shape) * 2  # floating weights load as bf16
        else:
            total += entry.nbytes * 4  # packed integer quantisation, if dequantised to bf16
    return total


def _free_cuda_caches() -> None:
    """Free cached blocks on every visible device: a sharded HF load may have
    allocated on all of them."""
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()


def _hf_forward(
    ref_path: str,
    documents: list[np.ndarray],
    arch_map: dict[str, Any],
    device_map: str,
    placement: str,
) -> _HfReference:
    """Load the HF reference (bf16), register per-layer forward hooks, run each
    document as its own batch, copy captures to CPU fp32. Frees all GPU caches on exit."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        ref_path,
        dtype=torch.bfloat16,
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

    outputs: list[dict[int, np.ndarray]] = []
    layer_devices: dict[int, str] = {}
    hooks: list[Any] = []

    def _make_hook(idx: int):
        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            t = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(t):
                return
            outputs[-1][idx] = t.detach().to(dtype=torch.float32, device="cpu").numpy()
            layer_devices[idx] = str(t.device)

        return _hook

    try:
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(_make_hook(i)))

        with torch.no_grad():
            # For device_map="auto", HF routes inputs via the embeddings module's
            # device. Using the first parameter's device covers both sharded and
            # single-device loads without special-casing.
            input_device = next(model.parameters()).device
            for doc in documents:
                outputs.append({})
                model(input_ids=torch.from_numpy(np.ascontiguousarray(doc)).to(input_device))
    finally:
        for h in hooks:
            h.remove()
        del model
        _free_cuda_caches()

    return _HfReference(outputs, hf_class_name, layer_devices, device_map, placement)


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
    position_ids: np.ndarray,
    used_seq_len: int,
    total_rows: int,
) -> str | None:
    """Construct the trainer, overwrite its token and position buffers with our
    synthetic ids, run one ``step()``. Returns a formatted error string if
    anything raised during either construction or the step itself.

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
    pos_ids[:, :w] = position_ids[:total_rows, :w]

    def _validate_once() -> None:
        wrapper.trainer.validate(in_tokens, out_tokens, pos_ids)

    ok, err = capture_exception(_validate_once)
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
    # A finite reference that is exactly zero everywhere is broken too (a sharded
    # reference over a dead peer-to-peer path); a real residual stream never is.
    if hf_c.size and not np.any(hf_c):
        return (
            {
                "hf_shape_cropped": list(hf_c.shape),
                "dsl_shape_cropped": list(dsl_c.shape),
                "dsl_max_abs": float(np.abs(dsl_c).max()) if dsl_c.size else 0.0,
            },
            DiffStatus.HF_ZERO,
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
    if status in (DiffStatus.NONFINITE, DiffStatus.HF_ZERO):
        return Severity.WARN
    max_abs_diff = stats.get("max_abs_diff")
    hf_max_abs = stats.get("hf_max_abs")
    if max_abs_diff is None or hf_max_abs is None:
        return Severity.INFO
    if max_abs_diff > atol + rtol * hf_max_abs:
        return Severity.ERROR
    return Severity.INFO
