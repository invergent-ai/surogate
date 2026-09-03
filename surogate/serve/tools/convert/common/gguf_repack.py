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
import os
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from surogate.serve.tools.convert.common.safetensors import name_spellings
from surogate.serve.tools.convert.common.row_algebra import (
    SOURCE_STRIDE,
    RowProgram,
    RowShapeMismatch,
    collect_sources,
    evaluate_rows,
)

import numpy as np
import torch

from surogate.serve.tools.artifact.layouts import encode_row_split, row_split_geometry
from surogate.serve.tools.artifact.numeric import QuantFormat, get_format
from surogate.serve.tools.convert.common.recipe import (
    Expression,
    GatherRows,
    TensorRecipe,
)

_REPACK_FORMAT = "W8G32_F16S"
# Every artifact format a quantised linear may be spec'd in. The W8 repack targets only
# _REPACK_FORMAT because it produces W8 planes; serving a K-quant natively replaces whichever
# of these the target would otherwise have quantised into, so the native planner considers all.
_QUANT_LINEAR_FORMATS = ("W8G32_F16S", "Q4G64_F16S", "Q5G64_F16S", "Q6G64_F16S")
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
# GGML K-quants are served in their own superblocks (artifact layout ggml-blocks-v1): the
# bytes go into the artifact verbatim, row by row, and the kernels read them as the file holds
# them. Type -> bytes per 256-value superblock.
# Bytes per stored block, and how many values that block holds: 256 for every K-quant, 32 for
# Q8_0, which is not one -- it is the plain 8-bit block a K_M quant leaves the attention, GDN and
# shared-expert projections in.
NATIVE_TYPES = {"Q2_K": 84, "Q3_K": 110, "Q4_K": 144, "Q5_K": 176, "Q6_K": 210, "Q8_0": 34}
NATIVE_BLOCK_VALUES = {"Q8_0": 32}


def native_block_values(gguf_type: str) -> int:
    return NATIVE_BLOCK_VALUES.get(gguf_type, 256)
_NATIVE_LAYOUT = "ggml-blocks-v1"

# What a fused object is called once it is stored as two differently-typed halves. The names
# match the ones the 35B target already binds, and the public split ops take exactly this pair.
HALF_NAMES = {"gdn/query_key_value_z": ("gdn/query_key_value", "gdn/z")}


def half_names(object_name: str) -> tuple[str, str] | None:
    """The two object names a fused parent splits into, or None if it has no registered split."""
    for suffix, halves in HALF_NAMES.items():
        if object_name.endswith(suffix):
            prefix = object_name[: -len(suffix)]
            return (prefix + halves[0], prefix + halves[1])
    return None

REPACKABLE_TYPES = {
    "Q8_0": (34, _planes_q8_0),
    "Q4_0": (18, _planes_q4_0),
    "Q5_0": (22, _planes_q5_0),
    "IQ4_NL": (18, _planes_iq4_nl),
}
# The row algebra lives in row_algebra.py now, shared with the compressed-tensors
# path; these aliases keep this module's internal references unchanged.
_SOURCE_STRIDE = SOURCE_STRIDE
_RowProgram = RowProgram
_collect_sources = collect_sources


class RepackError(ValueError):
    pass


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
        # The recipes and the bridge may spell one tensor differently (a `.weight` suffix, the
        # VL nesting, a router alias); index every spelling so a lookup finds the entry.
        self.sources = dict(sources)
        for stored, entry in list(sources.items()):
            for spelling in name_spellings(stored):
                self.sources.setdefault(spelling, entry)
        if not self.gguf_path.is_file():
            raise RepackError(f"repack map points at a missing GGUF: {self.gguf_path}")
        for hf_name, entry in sources.items():
            if not {"name", "rows", "k", "offset", "type"} <= set(entry):
                raise RepackError(f"{hf_name}: repack map entry is missing fields")
            if entry["type"] not in REPACKABLE_TYPES and entry["type"] not in NATIVE_TYPES:
                raise RepackError(
                    f"{hf_name}: GGUF type {entry['type']!r} is neither exactly repackable "
                    "nor a native K-quant"
                )
        self._file = None
        self._planes: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # -- GGUF payload access (memmap; no metadata parse) --------------------

    def _memmap(self) -> np.ndarray:
        if self._file is None:
            self._file = np.memmap(self.gguf_path, dtype=np.uint8, mode="r")
        return self._file

    def source_shape(self, hf_name: str) -> tuple[int, ...]:
        """The source's stored shape, whatever its rank (the last axis is K)."""
        entry = self.sources[hf_name]
        shape = entry.get("shape")
        if shape is not None:
            return tuple(int(extent) for extent in shape)
        return int(entry["rows"]), int(entry["k"])

    def source_rows_k(self, hf_name: str) -> tuple[int, int]:
        """(rows, K) with every leading axis folded into the row count."""
        shape = self.source_shape(hf_name)
        rows = 1
        for extent in shape[:-1]:
            rows *= extent
        return rows, shape[-1]


    def planes(self, hf_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Deinterleaved (codes int8 [n, groups, 32], scales fp16 [n, groups])."""
        cached = self._planes.get(hf_name)
        if cached is not None:
            return cached
        entry = self.sources[hf_name]
        n, k = self.source_rows_k(hf_name)
        if k % _GROUP != 0:
            raise RepackError(f"{hf_name}: k={k} is not a multiple of {_GROUP}")
        if entry["type"] in NATIVE_TYPES and native_block_values(str(entry["type"])) == 256:
            raise RepackError(f"{hf_name}: {entry['type']} is a superblock format, not planes")
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

        The walker itself is ``row_algebra.evaluate_rows``; this binds it to
        the GGUF's sources and reports a shape disagreement as a RepackError.
        """

        try:
            return evaluate_rows(
                expression,
                {name: self.source_shape(name) for name in self.sources},
                token_ids,
            )
        except RowShapeMismatch as error:
            raise RepackError(str(error)) from error

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
            if any(
                (t := self.native_type_of(name)) is not None and native_block_values(t) == 256
                for name in program.sources
            ):
                # A source held as a *superblock* K-quant never deinterleaves into W8 planes: the
                # object is either served natively (plan_native) or dequantised by the bridge.
                # Q8_0 is not one -- it is the same numbers W8 holds, so it still repacks, which
                # is what an object whose op has no Q8_0 kernel yet falls back to.
                continue
            geometry = row_split_geometry(get_format(_REPACK_FORMAT), spec.shape)
            if geometry.k_pad != geometry.k or geometry.k != program.k:
                continue  # padding groups would need re-encoding
            planned.append(spec.name)
        return tuple(planned)

    # -- native K-quants: bytes verbatim, row by row ------------------------
    def native_type_of(self, hf_name: str) -> str | None:
        """The K-quant type of a mapped source, or None if it is not one."""
        entry = self.sources.get(hf_name)
        if entry is None or entry["type"] not in NATIVE_TYPES:
            return None
        return str(entry["type"])

    def plan_native(
        self,
        recipes_by_name: Mapping[str, TensorRecipe],
        tensor_specs: Sequence,
        *,
        with_token_ids: bool = True,
        exclude_suffixes: Sequence[str] = (),
    ) -> dict[str, str]:
        """Objects served as native K-quants: quantized-linear specs whose row program draws
        every row from mapped sources of one K-quant type at the spec's own k. Returns
        {object name: GGML type}. A parent that mixes types is not planned here."""
        probe = np.zeros(1, dtype=np.int64) if with_token_ids else None
        planned: dict[str, str] = {}
        for spec in tensor_specs:
            if getattr(spec, "kind", None) != "tensor" or spec.format not in _QUANT_LINEAR_FORMATS:
                continue
            recipe = recipes_by_name.get(spec.name)
            if recipe is None:
                continue
            if isinstance(recipe.expression, GatherRows):
                program = self._evaluate_rows(recipe.expression.source, None)
            else:
                program = self._evaluate_rows(recipe.expression, probe)
            if program is None or program.k != int(spec.shape[1]):
                continue
            types = {self.native_type_of(name) for name in program.sources}
            if len(types) != 1 or None in types:
                continue
            if program.k % native_block_values(next(iter(types))):
                continue
            # Objects whose op has no kernel for the stored type yet: the caller names them,
            # because which ops a family routes an object through is the family's knowledge.
            if any(spec.name.endswith(suffix) for suffix in exclude_suffixes):
                continue
            only = os.environ.get("SUROGATE_GGUF_NATIVE_ONLY")  # bisection aid: name substrings
            if only and not any(part and part in spec.name for part in only.split(",")):
                continue
            planned[spec.name] = next(iter(types))
        return planned

    @staticmethod
    def native_half_specs(tensor_specs: Sequence, halves: Mapping[str, tuple]) -> tuple:
        """Each planned fused spec replaced, in place, by its two typed halves."""
        out = []
        for spec in tensor_specs:
            runs = halves.get(getattr(spec, "name", None))
            if runs is None:
                out.append(spec)
                continue
            names = half_names(spec.name)
            for name, (gguf_type, rows) in zip(names, runs):
                out.append(
                    replace(spec, name=name, shape=(rows, int(spec.shape[1])),
                            format=gguf_type, layout=_NATIVE_LAYOUT)
                )
        return tuple(out)

    @staticmethod
    def native_specs(
        tensor_specs: Sequence,
        plan: Mapping[str, str],
        runs: Mapping[str, tuple[tuple[int, int, int], ...]] | None = None,
    ) -> tuple:
        """The specs with each natively planned object's format and layout rewritten, and its
        runs attached when it is served from the file rather than copied into the artifact."""
        out = []
        for spec in tensor_specs:
            name = getattr(spec, "name", None)
            gguf_type = plan.get(name)
            if gguf_type is None:
                out.append(spec)
            else:
                extra = {"runs": runs[name]} if runs is not None and name in runs else {}
                out.append(replace(spec, format=gguf_type, layout=_NATIVE_LAYOUT, **extra))
        return tuple(out)

    def _native_rows(self, hf_name: str) -> np.ndarray:
        """The source's superblock bytes as [rows, bytes per row], zero-copy over the memmap."""
        entry = self.sources[hf_name]
        n, k = self.source_rows_k(hf_name)
        block_bytes = NATIVE_TYPES[entry["type"]]
        values = native_block_values(str(entry["type"]))
        if k % values:
            raise RepackError(f"{hf_name}: k={k} is not a whole number of {values}-value blocks")
        row_bytes = (k // values) * block_bytes
        offset = int(entry["offset"])
        nbytes = n * row_bytes
        data = self._memmap()
        if offset < 0 or offset + nbytes > data.shape[0]:
            raise RepackError(f"{hf_name}: quantized payload is outside the GGUF file")
        return np.asarray(data[offset : offset + nbytes]).reshape(n, row_bytes)

    def payload_for_native(
        self,
        spec,
        recipe: TensorRecipe,
        token_ids: torch.Tensor | np.ndarray | None,
        row_slice: slice | None = None,
    ) -> bytes:
        """The object's superblock bytes: the recipe's rows gathered from the file verbatim.
        `row_slice` takes one half of a parent that splits into differently-typed objects."""
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
        if row_slice is None and (int(program.rows.shape[0]), program.k) != tuple(spec.shape):
            raise RepackError(
                f"{spec.name}: native shape {(int(program.rows.shape[0]), program.k)} "
                f"!= spec {tuple(spec.shape)}"
            )
        if row_slice is None:
            types = {self.native_type_of(name) for name in program.sources}
            if types != {spec.format}:
                raise RepackError(f"{spec.name}: sources are {types}, spec is {spec.format}")
        row_bytes = (program.k // native_block_values(spec.format)) * NATIVE_TYPES[spec.format]
        rows = program.rows if row_slice is None else program.rows[row_slice]
        n = int(rows.shape[0])
        out = np.empty((n, row_bytes), dtype=np.uint8)
        source_of = rows // _SOURCE_STRIDE
        row_of = rows % _SOURCE_STRIDE
        for index, source_name in enumerate(program.sources):
            mask = source_of == index
            if not mask.any():
                continue
            out[mask] = self._native_rows(source_name)[row_of[mask]]
        return out.tobytes()

    def runs_for_native(
        self,
        spec,
        recipe: TensorRecipe,
        token_ids: torch.Tensor | np.ndarray | None,
        source: int = 1,
    ) -> tuple[tuple[int, int, int], ...]:
        """The object's bytes as stretches of the GGUF, rather than a copy of them.

        The same row program `payload_for_native` gathers from, read as extents: a run is a
        maximal stretch of output rows drawn from one source tensor at consecutive source rows,
        which is where the file already holds them side by side. A tensor copied whole is one
        run; a fused routed gate/up is one run per expert per half, because the file keeps gate
        and up as separate tensors and the object interleaves them.
        """
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
        if (int(program.rows.shape[0]), program.k) != tuple(spec.shape):
            raise RepackError(
                f"{spec.name}: native shape {(int(program.rows.shape[0]), program.k)} "
                f"!= spec {tuple(spec.shape)}"
            )
        row_bytes = (program.k // native_block_values(spec.format)) * NATIVE_TYPES[spec.format]
        rows = np.asarray(program.rows)
        source_of = rows // _SOURCE_STRIDE
        row_of = rows % _SOURCE_STRIDE
        base = [int(self.sources[name]["offset"]) for name in program.sources]
        for index, name in enumerate(program.sources):
            n_rows, k = self.source_rows_k(name)
            source_type = str(self.sources[name]["type"])
            if (k // native_block_values(source_type)) * NATIVE_TYPES[source_type] != row_bytes:
                raise RepackError(f"{spec.name}: {name} has a different row width")

        # A new run starts wherever the source changes or the source row is not the next one.
        breaks = np.ones(rows.shape[0], dtype=bool)
        if rows.shape[0] > 1:
            breaks[1:] = (source_of[1:] != source_of[:-1]) | (row_of[1:] != row_of[:-1] + 1)
        starts = np.flatnonzero(breaks)
        ends = np.append(starts[1:], rows.shape[0])
        return tuple(
            (
                int(source),
                base[int(source_of[begin])] + int(row_of[begin]) * row_bytes,
                int(end - begin) * row_bytes,
            )
            for begin, end in zip(starts, ends)
        )

    def plan_repack_in_place(
        self,
        recipes_by_name: Mapping[str, TensorRecipe],
        tensor_specs: Sequence,
        suffixes: Sequence[str],
        *,
        with_token_ids: bool = True,
    ) -> dict[str, tuple[tuple[tuple[int, int, int], ...], str]]:
        """{object: (runs, transform)} for the W8 objects whose rows are all Q8_0.

        Q8_0 and W8G32_F16S hold the same numbers -- signed int8 with one binary16 scale per 32 --
        and differ only in arrangement, so an object whose kernels want the row-split planes can
        still be read from the file: the runs gather its rows as the GGUF holds them and the
        loader rearranges them on the device. Named per family, because which objects run on
        kernels that have no Q8_0 path is the family's knowledge.
        """
        probe = np.zeros(1, dtype=np.int64) if with_token_ids else None
        planned: dict[str, tuple[tuple[tuple[int, int, int], ...], str]] = {}
        for spec in tensor_specs:
            name = getattr(spec, "name", None)
            if getattr(spec, "kind", None) != "tensor" or spec.format != _REPACK_FORMAT:
                continue
            if not any(str(name).endswith(suffix) for suffix in suffixes):
                continue
            recipe = recipes_by_name.get(name)
            if recipe is None or isinstance(recipe.expression, GatherRows):
                continue
            program = self._evaluate_rows(recipe.expression, probe)
            if program is None or program.k != int(spec.shape[1]):
                continue
            types = {self.native_type_of(source) for source in program.sources}
            if types != {"Q8_0"}:
                continue
            row_bytes = (program.k // 32) * NATIVE_TYPES["Q8_0"]
            rows = np.asarray(program.rows)
            source_of = rows // _SOURCE_STRIDE
            row_of = rows % _SOURCE_STRIDE
            base = [int(self.sources[source]["offset"]) for source in program.sources]
            breaks = np.ones(rows.shape[0], dtype=bool)
            if rows.shape[0] > 1:
                breaks[1:] = (source_of[1:] != source_of[:-1]) | (row_of[1:] != row_of[:-1] + 1)
            starts = np.flatnonzero(breaks)
            ends = np.append(starts[1:], rows.shape[0])
            planned[str(name)] = (
                tuple(
                    (
                        1,
                        base[int(source_of[begin])] + int(row_of[begin]) * row_bytes,
                        int(end - begin) * row_bytes,
                    )
                    for begin, end in zip(starts, ends)
                ),
                "q8_0-to-w8g32",
            )
        return planned

    @staticmethod
    def in_place_specs(tensor_specs: Sequence, plan: Mapping[str, tuple]) -> tuple:
        """The specs with each in-place object's runs and transform attached; format and layout
        are unchanged, because the stored form the kernels read is unchanged."""
        out = []
        for spec in tensor_specs:
            entry = plan.get(getattr(spec, "name", None))
            if entry is None:
                out.append(spec)
            else:
                runs, transform = entry
                out.append(replace(spec, runs=runs, transform=transform))
        return tuple(out)

    def plan_native_halves(
        self,
        recipes_by_name: Mapping[str, TensorRecipe],
        tensor_specs: Sequence,
        *,
        with_token_ids: bool = True,
    ) -> dict[str, tuple[tuple[str, int], ...]]:
        """{object: ((ggml type, rows), ...)} for a quantised-linear object whose rows all come
        from native K-quants but of more than one type, in contiguous runs.

        A fused parent cannot hold two formats, and requantising the minority to the majority
        would lose what serving the file natively is for; the object is stored as one per run
        instead and the loader binds the halves. Only two runs are produced today, which is
        what the checkpoints on hand need (a GDN input projection whose qkv half is Q5_K and
        whose z half is Q4_K); anything else is left to the dequantise path.
        """
        # OFF by default, and measured: on Qwen3.5-0.8B-Q4_K_M the split shrinks the artifact
        # 713 -> 652 MB and still loses 6 % of decode (885 -> 828 tok/s), because the fused
        # parent runs one tuned W8 projection-and-convolution kernel per GDN layer while the
        # split runs two K-quant GEMVs and an unfused convolution -- two extra launches a layer,
        # eighteen layers, against a ~1.1 ms step. The bytes are worth having once the K-quant
        # GDN convolution is fused too; until then the dequantised parent is the faster serve.
        if os.environ.get("SUROGATE_GGUF_SPLIT_HALVES", "0") == "0":
            return {}
        probe = np.zeros(1, dtype=np.int64) if with_token_ids else None
        planned: dict[str, tuple[tuple[str, int], ...]] = {}
        for spec in tensor_specs:
            if getattr(spec, "kind", None) != "tensor" or spec.format != _REPACK_FORMAT:
                continue
            recipe = recipes_by_name.get(spec.name)
            if recipe is None or isinstance(recipe.expression, GatherRows):
                continue
            program = self._evaluate_rows(recipe.expression, probe)
            if program is None or program.k != int(spec.shape[1]) or program.k % 256:
                continue
            types = [self.native_type_of(name) for name in program.sources]
            if any(t is None for t in types) or len(set(types)) < 2:
                continue
            per_row = [types[int(index)] for index in (program.rows // _SOURCE_STRIDE)]
            runs: list[list] = []
            for gguf_type in per_row:
                if runs and runs[-1][0] == gguf_type:
                    runs[-1][1] += 1
                else:
                    runs.append([gguf_type, 1])
            if len(runs) != 2 or half_names(spec.name) is None:
                continue
            planned[spec.name] = tuple((str(t), int(rows)) for t, rows in runs)
        return planned

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


__all__ = ["GgufRepackSource", "RepackError"]
