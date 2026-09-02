"""Direct GGUF Q8_0 -> W8G32_F16S artifact repack (surogate vendor patch #14).

GGML ``Q8_0`` and the artifact's ``W8G32_F16S`` are the same numeric format:
signed 8-bit codes with one binary16 scale per 32-value group, ``value =
code * scale``.  ``Q4_0`` (codes = nibble - 8), ``Q5_0`` (codes = q5 - 16)
and ``IQ4_NL`` (codes = an int8 codebook lookup) share exactly that
semantics with narrower code sets, so all four move into W8G32 bit-exactly
by deinterleaving blocks into int8 code / binary16 scale planes -- no
dequantization, no requantization, no GPU.  (``Q4_1``/``Q5_1`` carry a
per-group additive min and K-quants a 6-bit sub-scale product, neither of
which is representable as ``int8 * fp16`` exactly; they stay on the
dequantize path.)

This module evaluates a target's registered ``TensorRecipe`` expressions as
*row algebra*: every W8 object the recipes build from 2D sources uses only
row-level operations (Reshape/Slice/Transpose over row axes, Concat on rows,
GatherRows), and any row rearrangement is exact on quantized planes because
each row's groups are self-contained (``k % 32 == 0``).  The recipe stays the
single source of layout truth; nothing here duplicates fusion knowledge.

The bridge (surogate side) writes a small JSON map next to the bridged
checkpoint dir::

    {"gguf_path": "/path/model.gguf",
     "sources": {"model.layers.0.mlp.gate_proj.weight":
                 {"name": "blk.0.ffn_gate.weight",
                  "rows": 3584, "k": 1024, "offset": 123456}, ...}}

listing exactly the HF-name sources whose GGUF tensor is Q8_0 *and* whose
bridge-side inverse transform is a row identity, with each tensor's logical
shape and absolute payload offset.  Sources in the map are not materialized
as BF16 safetensors; the converter repacks the objects whose expressions
close over mapped sources — reading the Q8_0 payloads straight off a memmap
of the GGUF (no gguf-py metadata parse, which costs ~10s on a 250k-token
vocabulary) — and materializes everything else from the (much smaller)
bridged checkpoint as before.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from surogate.serve.tools.artifact.layouts import encode_row_split, row_split_geometry
from surogate.serve.tools.artifact.numeric import QuantFormat, get_format
from surogate.serve.tools.convert.common.recipe import (
    Concat,
    Expression,
    GatherRows,
    Reshape,
    Slice,
    SourceTensor,
    TensorRecipe,
    Transpose,
)

_REPACK_FORMAT = "W8G32_F16S"
_GROUP = 32

# IQ4_NL codebook (ggml-common.h kvalues_iq4nl): int8 values, so an IQ4_NL
# group IS a W8 group after the lookup.
_IQ4NL_VALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.int8,
)


def _planes_q8_0(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = blocks[:, :, :2].copy().view(np.float16)[..., 0]
    codes = blocks[:, :, 2:].view(np.int8)
    return codes, scales


def _planes_q4_0(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = blocks[:, :, :2].copy().view(np.float16)[..., 0]
    qs = blocks[:, :, 2:]
    low = (qs & 0x0F).astype(np.int8) - 8
    high = (qs >> 4).astype(np.int8) - 8
    return np.concatenate([low, high], axis=-1), scales


def _planes_q5_0(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = blocks[:, :, :2].copy().view(np.float16)[..., 0]
    qh = blocks[:, :, 2:6].copy().view(np.uint32)[..., 0]
    qs = blocks[:, :, 6:]
    j = np.arange(16)
    low = (qs & 0x0F).astype(np.int16) | (((qh[..., None] >> j) & 1) << 4).astype(np.int16)
    high = (qs >> 4).astype(np.int16) | (((qh[..., None] >> (j + 16)) & 1) << 4).astype(
        np.int16
    )
    codes = (np.concatenate([low, high], axis=-1) - 16).astype(np.int8)
    return codes, scales


def _planes_iq4_nl(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = blocks[:, :, :2].copy().view(np.float16)[..., 0]
    qs = blocks[:, :, 2:]
    codes = np.concatenate(
        [_IQ4NL_VALUES[qs & 0x0F], _IQ4NL_VALUES[qs >> 4]], axis=-1
    )
    return codes, scales


# GGML type -> (bytes per 32-value block, plane decoder). Every decoder
# returns (codes int8 [n, groups, 32], scales fp16 [n, groups]) with values
# in logical order; each is pinned bit-exact against gguf-py dequantize in
# tests/serve/test_gguf_repack.py.
REPACKABLE_TYPES = {
    "Q8_0": (34, _planes_q8_0),
    "Q4_0": (18, _planes_q4_0),
    "Q5_0": (22, _planes_q5_0),
    "IQ4_NL": (18, _planes_iq4_nl),
}
_SOURCE_STRIDE = 1 << 40  # global row id = source_index * stride + row


class RepackError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _RowProgram:
    """Row-only evaluation of one recipe expression.

    ``rows`` holds global row ids over ``sources`` (in first-use order); the
    flattened order is the object's logical row order because every supported
    expression keeps the trailing K axis untouched.
    """

    sources: tuple[str, ...]
    rows: np.ndarray  # int64 [N]
    k: int


class GgufRepackSource:
    """Q8_0 plane provider + recipe row-algebra evaluator over one GGUF."""

    def __init__(self, map_path: str | Path):
        raw = json.loads(Path(map_path).read_text())
        self._init(Path(raw["gguf_path"]), dict(raw["sources"]))

    @classmethod
    def from_sources(cls, gguf_path: str | Path, sources: Mapping[str, str]):
        """Planner entry: candidate map held in memory (bridge pre-walk)."""
        self = cls.__new__(cls)
        self._init(Path(gguf_path), dict(sources))
        return self

    def _init(self, gguf_path: Path, sources: dict[str, dict]) -> None:
        self.gguf_path = gguf_path
        self.sources = sources
        if not self.gguf_path.is_file():
            raise RepackError(f"repack map points at a missing GGUF: {self.gguf_path}")
        for hf_name, entry in sources.items():
            if not {"name", "rows", "k", "offset", "type"} <= set(entry):
                raise RepackError(f"{hf_name}: repack map entry is missing fields")
            if entry["type"] not in REPACKABLE_TYPES:
                raise RepackError(
                    f"{hf_name}: GGUF type {entry['type']!r} is not exactly repackable"
                )
        self._file = None
        self._planes: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # -- GGUF payload access (memmap; no metadata parse) --------------------

    def _memmap(self) -> np.ndarray:
        if self._file is None:
            self._file = np.memmap(self.gguf_path, dtype=np.uint8, mode="r")
        return self._file

    def source_shape(self, hf_name: str) -> tuple[int, int]:
        """Logical (rows, k) of a mapped source."""
        entry = self.sources[hf_name]
        return int(entry["rows"]), int(entry["k"])

    def planes(self, hf_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Deinterleaved (codes int8 [n, groups, 32], scales fp16 [n, groups])."""
        cached = self._planes.get(hf_name)
        if cached is not None:
            return cached
        entry = self.sources[hf_name]
        n, k = self.source_shape(hf_name)
        if k % _GROUP != 0:
            raise RepackError(f"{hf_name}: k={k} is not a multiple of {_GROUP}")
        block_bytes, decoder = REPACKABLE_TYPES[entry["type"]]
        offset = int(entry["offset"])
        nbytes = n * (k // _GROUP) * block_bytes
        data = self._memmap()
        if offset < 0 or offset + nbytes > data.shape[0]:
            raise RepackError(f"{hf_name}: quantized payload is outside the GGUF file")
        raw = np.asarray(data[offset : offset + nbytes]).reshape(
            n, k // _GROUP, block_bytes
        )
        codes, scales = decoder(raw)
        codes = np.ascontiguousarray(codes)
        scales = np.ascontiguousarray(scales).reshape(n, k // _GROUP)
        self._planes[hf_name] = (codes, scales)
        return codes, scales

    # -- recipe row algebra ------------------------------------------------

    def _evaluate_rows(
        self,
        expression: Expression,
        token_ids: np.ndarray | None,
    ) -> _RowProgram | None:
        """Evaluate an expression to global row ids, or None if unsupported.

        Values are ndarrays over the row axes only (the trailing K axis of
        every node is asserted equal and dropped); any expression touching K
        or mixing K extents is rejected.
        """

        sources: list[str] = []

        def source_index(name: str) -> int:
            if name not in self.sources:
                raise _Unsupported()
            try:
                return sources.index(name)
            except ValueError:
                sources.append(name)
                return len(sources) - 1

        def walk(node: Expression) -> tuple[np.ndarray, int]:
            if isinstance(node, SourceTensor):
                if len(node.shape) != 2 or node.name not in self.sources:
                    raise _Unsupported()
                n, k = node.shape
                actual = self.source_shape(node.name)
                if actual != (n, k):
                    raise RepackError(
                        f"{node.name}: GGUF shape {actual} != recipe shape {(n, k)}"
                    )
                index = source_index(node.name)
                ids = np.arange(n, dtype=np.int64) + index * _SOURCE_STRIDE
                return ids, k
            if isinstance(node, Reshape):
                ids, k = walk(node.source)
                if not node.shape or node.shape[-1] != k:
                    raise _Unsupported()
                return ids.reshape(node.shape[:-1]), k
            if isinstance(node, Slice):
                ids, k = walk(node.source)
                if node.axis >= ids.ndim:  # slicing the K axis
                    raise _Unsupported()
                return (
                    np.take(ids, np.arange(node.begin, node.end), axis=node.axis),
                    k,
                )
            if isinstance(node, Transpose):
                ids, k = walk(node.source)
                full_rank = ids.ndim + 1
                if tuple(node.axes)[-1] != full_rank - 1:
                    raise _Unsupported()  # moves the K axis
                return ids.transpose(tuple(node.axes)[:-1]), k
            if isinstance(node, Concat):
                parts = [walk(part) for part in node.sources]
                ks = {k for _, k in parts}
                if len(ks) != 1:
                    raise _Unsupported()
                first = parts[0][0]
                if node.axis >= first.ndim:  # concatenating along K
                    raise _Unsupported()
                return (
                    np.concatenate([ids for ids, _ in parts], axis=node.axis),
                    ks.pop(),
                )
            if isinstance(node, GatherRows):
                if token_ids is None:
                    raise _Unsupported()
                ids, k = walk(node.source)
                if ids.ndim != 1:
                    raise _Unsupported()
                gathered = ids[token_ids.astype(np.int64)]
                if gathered.shape[0] != node.rows:
                    raise RepackError(
                        f"GatherRows: {gathered.shape[0]} ids != declared {node.rows}"
                    )
                return gathered, k
            raise _Unsupported()  # Cast, DraftHeadTokenIds, unknown nodes

        try:
            ids, k = walk(expression)
        except _Unsupported:
            return None
        return _RowProgram(tuple(sources), ids.reshape(-1), k)

    # -- public plan/payload API -------------------------------------------

    def plan(
        self,
        recipes_by_name: Mapping[str, TensorRecipe],
        tensor_specs: Sequence,
        *,
        with_token_ids: bool = True,
    ) -> tuple[str, ...]:
        """Object names this source repacks (probe token ids as all-zeros)."""
        probe = np.zeros(1, dtype=np.int64) if with_token_ids else None
        planned: list[str] = []
        for spec in tensor_specs:
            if getattr(spec, "kind", None) != "tensor" or spec.format != _REPACK_FORMAT:
                continue
            recipe = recipes_by_name.get(spec.name)
            if recipe is None:
                continue
            if isinstance(recipe.expression, GatherRows):
                # Plan membership only; real ids arrive in payload_for.
                program = self._evaluate_rows(recipe.expression.source, None)
            else:
                program = self._evaluate_rows(recipe.expression, probe)
            if program is None:
                continue
            geometry = row_split_geometry(get_format(_REPACK_FORMAT), spec.shape)
            if geometry.k_pad != geometry.k or geometry.k != program.k:
                continue  # padding groups would need re-encoding
            planned.append(spec.name)
        return tuple(planned)

    def covered_sources(
        self, recipes_by_name: Mapping[str, TensorRecipe], planned: Sequence[str]
    ) -> frozenset[str]:
        """HF source names consumed exclusively through the repack plan."""
        used: set[str] = set()
        for name in planned:
            _collect_sources(recipes_by_name[name].expression, used)
        return frozenset(used & set(self.sources))

    def payload_for(
        self,
        spec,
        recipe: TensorRecipe,
        token_ids: torch.Tensor | np.ndarray | None,
    ) -> bytes:
        ids = None
        if token_ids is not None:
            ids = (
                token_ids.detach().cpu().numpy()
                if isinstance(token_ids, torch.Tensor)
                else np.asarray(token_ids)
            )
        program = self._evaluate_rows(recipe.expression, ids)
        if program is None:
            raise RepackError(f"{spec.name}: expression is not row-repackable")
        n = int(program.rows.shape[0])
        if (n, program.k) != tuple(spec.shape):
            raise RepackError(
                f"{spec.name}: repacked shape {(n, program.k)} != spec {tuple(spec.shape)}"
            )
        groups = program.k // _GROUP
        codes = np.empty((n, groups, _GROUP), dtype=np.int8)
        scales = np.empty((n, groups), dtype=np.float16)
        source_of = program.rows // _SOURCE_STRIDE
        row_of = program.rows % _SOURCE_STRIDE
        for index, source_name in enumerate(program.sources):
            mask = source_of == index
            if not mask.any():
                continue
            source_codes, source_scales = self.planes(source_name)
            rows = row_of[mask]
            codes[mask] = source_codes[rows]
            scales[mask] = source_scales[rows]
        spec_format = get_format(_REPACK_FORMAT)
        assert isinstance(spec_format, QuantFormat)
        return encode_row_split(
            torch.from_numpy(codes),
            torch.from_numpy(scales),
            spec_format,
            tuple(spec.shape),
        )


class _Unsupported(Exception):
    pass


def _collect_sources(node: Expression, out: set[str]) -> None:
    if isinstance(node, SourceTensor):
        out.add(node.name)
    elif isinstance(node, (Reshape, Slice, Transpose)):
        _collect_sources(node.source, out)
    elif isinstance(node, Concat):
        for part in node.sources:
            _collect_sources(part, out)
    elif isinstance(node, GatherRows):
        _collect_sources(node.source, out)


__all__ = ["GgufRepackSource", "RepackError"]
