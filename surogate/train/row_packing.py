"""Row packing: train the rows of one optimizer step in as few ``sequence_len`` windows as they fit.

A padded shard (``non_overlapping``: one example per window) spends most of its compute on
padding when rows are short -- the SDFT pilot's mean row is 360 tokens in 4,352-token windows,
92% padding. Row packing keeps everything that defines the optimisation and changes only how
the step's rows are laid out on the GPUs:

* **Same rows, same order, same step.** The step loads exactly the rows padded training would
  (``gradient_accumulation_steps`` loader batches), so epochs, the data-loader checkpoint
  position and the batch audit are unchanged.
* **Each row is its own document.** Its position ids restart at 0 where it starts, which is the
  engine's document boundary (``compute_doc_masking``): attention -- sliding-window and global
  layers alike -- never crosses rows, and neither do the token mixers of linear-attention layers
  (Gated DeltaNet: the causal convolution and the delta rule's recurrent state restart at every
  document). Everything else in the model is per token.
* **Only what can influence a loss is kept.** A row is cut after its last supervised input
  position. With causal attention nothing later can reach an earlier position's output, so the
  dropped tail (the padding, and a trailing answer token nobody predicts from) contributes
  exactly nothing in padded training either.
* **The sidecar travels with its tokens.** A row's teacher top-K ids/logprobs are copied with
  its tokens, position for position, so ``targets[i]`` and the sidecar row ``i`` stay aligned.
* **Loss normalisation is untouched.** The engine divides the step's summed loss/gradient by
  the step's global supervised-token count, which is the same set of tokens in either layout.

The step then needs ``waves = ceil(windows / slots)`` micro-steps instead of
``gradient_accumulation_steps`` (``slots`` = GPUs x per-device batch). Rows are spread over all
``waves x slots`` windows (largest first, into the emptiest window that still fits), so no
window is fuller than it has to be; a slot with no rows gets a dummy window without targets.

What is *not* identical to padded training is floating-point summation order: the gradient of a
packed window sums several rows inside one GEMM, where padded training sums them across
micro-steps in the gradient dtype (docs/reference/config.md, "Row packing", has the measured
tolerance). Under ``fp8_hybrid`` the per-tensor quantisation scales also span the whole window,
so a row's arithmetic depends on the other rows unless ``fp8_pow2_scales`` is on (power-of-two
scales only shift exponents; only elements pushed into the FP8 subnormal range by another row's
larger maximum still differ).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

#: This Python package's row packing relies on the engine restarting linear-attention (Gated
#: DeltaNet) layers at every packed document: the causal convolution reads zeros before a
#: document's first token and the gated delta rule starts each document from a zero recurrent
#: state. The compiled extension says whether it does (``_surogate.LINEAR_ATTENTION_DOC_BOUNDARIES``);
#: :func:`linear_attention_doc_boundaries` checks both, and is what deployment scripts should call.
LINEAR_ATTENTION_DOC_BOUNDARIES = True

#: Token-mixing ops, other than attention, that honour document boundaries when the compiled
#: extension reports ``LINEAR_ATTENTION_DOC_BOUNDARIES``.
DOCUMENT_AWARE_MIXERS = frozenset({"chunk_gated_delta_rule", "mamba_conv1d"})

#: Token-mixing ops not verified to honour document boundaries: a packed row would start from the
#: previous row's state. Row packing refuses models that use them (override:
#: SUROGATE_ALLOW_UNVERIFIED_ROW_PACKING=1).
DOCUMENT_UNVERIFIED_MIXERS = frozenset({"mamba_ssm_scan", "glm_causal_conv1d", "chunk_kimi_delta_rule"})


def linear_attention_doc_boundaries() -> bool:
    """True when both this package and the compiled extension restart linear-attention layers at
    every packed document. CPU only (imports the extension, touches no GPU)."""
    if not LINEAR_ATTENTION_DOC_BOUNDARIES:
        return False
    try:
        from surogate import _surogate
    except ImportError:
        return False
    return bool(getattr(_surogate, "LINEAR_ATTENTION_DOC_BOUNDARIES", False))


def _ir_ops(ir_json: str) -> set[str]:
    ops: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("op", "kernel_type", "type") and isinstance(value, str):
                    ops.add(value)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(ir_json))
    return ops


def document_isolation_problem(ir_json: str | None) -> str | None:
    """Why packed rows of this model would not be isolated documents, or None.

    Attention is isolated by document masking; this checks the model's other token mixers."""
    if not ir_json:
        return None
    ops = _ir_ops(ir_json)
    unverified = sorted(ops & DOCUMENT_UNVERIFIED_MIXERS)
    if unverified and os.environ.get("SUROGATE_ALLOW_UNVERIFIED_ROW_PACKING") != "1":
        return (
            f"the model mixes tokens with {unverified}, which are not verified to restart at a packed "
            "document boundary (set SUROGATE_ALLOW_UNVERIFIED_ROW_PACKING=1 to pack anyway)"
        )
    aware = sorted(ops & DOCUMENT_AWARE_MIXERS)
    if aware and not linear_attention_doc_boundaries():
        return (
            f"the model has linear-attention layers ({aware}) and this surogate build carries their "
            "convolution and recurrent state across packed documents (the compiled extension lacks "
            "LINEAR_ATTENTION_DOC_BOUNDARIES); upgrade surogate or train padded"
        )
    return None


def effective_length(targets_row: np.ndarray) -> int:
    """Input positions [0, L) that can influence a loss: through the last supervised one."""
    supervised = np.flatnonzero(targets_row != -100)
    return int(supervised[-1]) + 1 if supervised.size else 0


#: Rows per window at most. Every document in a window costs the attention backward 128 padded
#: rows of fp32 scratch per dQ split (FlashAttention varlen), sized from the stack arena; bounding
#: the count bounds that (64 x 8 MiB for Gemma 4's sliding layers) independently of the row mix.
MAX_ROWS_PER_WINDOW = 64


def plan_windows(lengths: list[int], seq_len: int, slots: int, max_rows: int = MAX_ROWS_PER_WINDOW) -> list[list[int]]:
    """Assign rows (by index) to windows; returns ``waves * slots`` windows, some possibly empty.

    Rows of length 0 are dropped (they supervise nothing). The window count is the smallest
    multiple of ``slots`` that first-fit decreasing can fill; rows are then spread over all of
    those windows, longest first, each into the least-filled window with room (ties to the lower
    index). If that spreading fails to place a row, the first-fit layout is used instead. No
    window holds more than ``max_rows`` rows. Deterministic for a given input.
    """
    if slots < 1 or max_rows < 1:
        raise ValueError("slots and max_rows must be >= 1")
    items = [i for i, n in enumerate(lengths) if n > 0]
    for i in items:
        if lengths[i] > seq_len:
            raise ValueError(f"row {i} needs {lengths[i]} positions, more than sequence_len {seq_len}")
    order = sorted(items, key=lambda i: (-lengths[i], i))

    ffd: list[list[int]] = []
    ffd_fill: list[int] = []
    for i in order:
        for b, used in enumerate(ffd_fill):
            if used + lengths[i] <= seq_len and len(ffd[b]) < max_rows:
                ffd[b].append(i)
                ffd_fill[b] += lengths[i]
                break
        else:
            ffd.append([i])
            ffd_fill.append(lengths[i])

    waves = max(1, -(-len(ffd) // slots))
    count = waves * slots
    spread: list[list[int]] = [[] for _ in range(count)]
    fill = [0] * count
    for i in order:
        best = -1
        for b in range(count):
            if fill[b] + lengths[i] <= seq_len and len(spread[b]) < max_rows and (best < 0 or fill[b] < fill[best]):
                best = b
        if best < 0:
            return ffd + [[] for _ in range(count - len(ffd))]
        spread[best].append(i)
        fill[best] += lengths[i]
    return spread


@dataclass
class _Row:
    tokens: np.ndarray
    targets: np.ndarray
    positions: np.ndarray
    kd_ids: np.ndarray | None
    kd_logprobs: np.ndarray | None


@dataclass
class PackedStep:
    """Materialised micro-steps of one optimizer step: ``waves`` batches of ``slots`` windows."""

    inputs: list[np.ndarray]
    targets: list[np.ndarray]
    positions: list[np.ndarray]
    kd_ids: list[np.ndarray] | None
    kd_logprobs: list[np.ndarray] | None
    windows: list[list[int]]
    rows: int
    packed_rows: int
    real_tokens: int
    stats: dict = field(default_factory=dict)

    @property
    def waves(self) -> int:
        return len(self.inputs)


class RowPacker:
    """Collects one step's padded rows, then lays them out as packed micro-steps.

    ``add_batch`` takes the arrays one ``DataLoader.load_batch`` call filled (``[rows, T]``, plus
    ``[rows, T, K]`` sidecar arrays when KD is on) and keeps only each row's effective prefix.
    ``build`` returns the micro-steps and clears the collector.
    """

    def __init__(self, seq_len: int, slots: int, pad_token_id: int = 0, kd_top_k: int | None = None):
        self.seq_len = int(seq_len)
        self.slots = int(slots)
        self.pad_token_id = int(pad_token_id)
        self.kd_top_k = kd_top_k
        self._rows: list[_Row] = []

    def add_batch(self, inputs, targets, positions, kd_ids=None, kd_logprobs=None) -> None:
        if (kd_ids is None) != (self.kd_top_k is None) or (kd_ids is None) != (kd_logprobs is None):
            raise ValueError("RowPacker: sidecar arrays must be given exactly when kd_top_k is set")
        for r in range(inputs.shape[0]):
            n = effective_length(targets[r])
            pos = positions[r, :n]
            if n and pos[0] != 0:
                # A row whose first position is not 0 could continue its predecessor's positions
                # and silently merge the two documents. Padded shards always start rows at 0.
                raise ValueError(
                    f"row packing needs every row to start at position 0 (got {int(pos[0])}); "
                    "is this a padded (non-overlapping) shard?"
                )
            self._rows.append(
                _Row(
                    tokens=inputs[r, :n].copy(),
                    targets=targets[r, :n].copy(),
                    positions=pos.copy(),
                    kd_ids=None if kd_ids is None else kd_ids[r, :n].copy(),
                    kd_logprobs=None if kd_logprobs is None else kd_logprobs[r, :n].copy(),
                )
            )

    def build(self) -> PackedStep:
        rows, self._rows = self._rows, []
        T, K = self.seq_len, self.kd_top_k
        windows = plan_windows([len(r.tokens) for r in rows], T, self.slots)
        waves = len(windows) // self.slots
        kd = K is not None
        inputs, targets, positions, kd_ids, kd_logprobs = [], [], [], [], []
        fills = []
        for w in range(waves):
            x = np.full((self.slots, T), self.pad_token_id, dtype=np.int32)
            y = np.full((self.slots, T), -100, dtype=np.int32)
            p = np.empty((self.slots, T), dtype=np.int32)
            if kd:
                ids = np.zeros((self.slots, T, K), dtype=np.int32)
                lps = np.zeros((self.slots, T, K), dtype=np.float32)
            for s in range(self.slots):
                o = 0
                for i in windows[w * self.slots + s]:
                    row = rows[i]
                    n = len(row.tokens)
                    x[s, o : o + n] = row.tokens
                    y[s, o : o + n] = row.targets
                    p[s, o : o + n] = row.positions
                    if kd:
                        ids[s, o : o + n] = row.kd_ids
                        lps[s, o : o + n] = row.kd_logprobs
                    o += n
                fills.append(o)
                # The unused tail is one more document of its own (positions restart at 0), so no
                # row's attention span grows by it. A window with no rows at all keeps plain
                # 0..T-1 positions: one dense document, no targets.
                p[s, o:] = np.arange(T - o, dtype=np.int32)
            inputs.append(x)
            targets.append(y)
            positions.append(p)
            if kd:
                kd_ids.append(ids)
                kd_logprobs.append(lps)
        packed = sum(len(w) for w in windows)
        return PackedStep(
            inputs=inputs,
            targets=targets,
            positions=positions,
            kd_ids=kd_ids if kd else None,
            kd_logprobs=kd_logprobs if kd else None,
            windows=windows,
            rows=len(rows),
            packed_rows=packed,
            real_tokens=int(sum(fills)),
            stats={
                "windows_used": sum(1 for w in windows if w),
                "fill_mean": float(np.mean(fills) / T) if fills else 0.0,
                "fill_max": float(max(fills) / T) if fills else 0.0,
            },
        )
