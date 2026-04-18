"""Record tag vocabulary and severity levels for debug JSONL output.

Every JSONL record written by a debug subcommand has:
  - tag:      one of :class:`Tag`, determines the record's semantics + field set
  - severity: one of :class:`Severity` (info / warn / error)
  - plus tag-specific structured fields (layer, op, name, shape, stats, ...)

The ``run_id`` is recorded once in the sibling ``.header.json`` sidecar, not on
every row, so bulk tags (HF_KEY, DSL_PARAM, ACT, GRAD, ...) stay compact.

Filtering examples::

    grep '"severity":"error"' debug_*.jsonl
    grep '"tag":"UNBOUND"'    debug_weights_*.jsonl
    jq 'select(.layer==5)'    debug_activations_*.jsonl
    jq 'select(.nan_pct>0)'   debug_activations_*.jsonl
"""

from __future__ import annotations

from enum import Enum


class Tag(str, Enum):
    # Run context (emitted once at start of every file)
    RUN = "RUN"
    MODEL = "MODEL"
    SAFETENSORS = "SAFETENSORS"

    # Weight audit
    MAPPING = "MAPPING"
    UNBOUND = "UNBOUND"
    UNUSED_HF = "UNUSED_HF"
    SHAPE_MISMATCH = "SHAPE_MISMATCH"
    DTYPE_NOTE = "DTYPE_NOTE"
    FALLBACK_DIRECT = "FALLBACK_DIRECT"
    TIED = "TIED"
    HF_KEY = "HF_KEY"
    DSL_PARAM = "DSL_PARAM"

    # Activation / gradient tracing
    ACT = "ACT"
    GRAD = "GRAD"

    # Reference diff
    REFERENCE = "REFERENCE"
    DIFF = "DIFF"

    # Terminal
    SUMMARY = "SUMMARY"
    ERROR = "ERROR"


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class DumpStatus(str, Enum):
    """Status of an activation/gradient dump file parse."""

    LOADED = "loaded"
    MISSING = "missing_dump"
    EMPTY = "empty"
    READ_FAILED = "read_failed"


class DiffStatus(str, Enum):
    """Status of one DIFF record emitted by ``surogate debug diff``."""

    COMPARED = "compared"
    SHAPE_MISMATCH = "shape_mismatch"
    NONFINITE = "nonfinite"  # hf or dsl tensor had NaN/Inf; numeric diff skipped
    DSL_DUMP_MISSING = "dsl_dump_missing"
    DSL_READ_FAILED = "dsl_read_failed"
    DSL_RESHAPE_FAILED = "dsl_reshape_failed"
    HF_OUTPUT_MISSING = "hf_output_missing"
    NO_PAIRED_SLOT = "no_paired_dsl_slot"


class MappingKind(str, Enum):
    """Mirrors the ``type`` string produced by ``surogate.dsl.py_compiler._serialize_hf_spec``.

    Values MUST stay in sync with that serializer — if a new mapping kind is
    added to ``surogate/dsl/hf.py``, add it here too.
    """

    DIRECT = "direct"
    FUSE = "fuse"
    SPLIT = "split"
    TRANSFORM = "transform"
    TIED_TO = "tied_to"
    STACK_EXPERTS = "stack_experts"
    NONE = "none"  # no mapping entry at all; will fall through to direct
