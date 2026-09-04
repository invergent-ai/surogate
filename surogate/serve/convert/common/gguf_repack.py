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

``gguf_path`` may instead be the ordered shards of a split GGUF, in which
case an entry adds ``"source": n`` -- 1-based, the same numbering the
artifact's external file table uses -- to say which shard holds it.

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

from surogate.serve.convert.common.safetensors import name_spellings
from surogate.serve.convert.common.row_algebra import (
    SOURCE_STRIDE,
    RowProgram,
    RowShapeMismatch,
    collect_sources,
    evaluate_rows,
)

import numpy as np
import torch

from surogate.serve.artifact.layouts import encode_row_split, row_split_geometry
from surogate.serve.artifact.numeric import QuantFormat, get_format
from surogate.serve.convert.common.recipe import (
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
# Q4_1 and Q5_1 are `code * scale + min` over 32 values. A quantiser reaches for them when the
# reduction axis is not a multiple of 256, so no K-quant superblock fits a row -- which is why
# they turn up on the MoE down projections of a model whose expert width is not a multiple of
# 256. The additive minimum is what keeps them out of REPACKABLE_TYPES: W8 has a scale and
# nowhere to put a min, so they are served as the file holds them or not at all.
NATIVE_TYPES = {"Q2_K": 84, "Q3_K": 110, "Q4_K": 144, "Q5_K": 176, "Q6_K": 210, "Q8_0": 34,
                "Q4_1": 20, "Q5_1": 24, "IQ4_NL": 18,
                "Q4_0": 18, "Q5_0": 22,
                "IQ2_XXS": 66, "IQ2_XS": 74, "IQ2_S": 82, "IQ3_XXS": 98, "IQ3_S": 110, "IQ1_S": 50, "IQ1_M": 56, "IQ4_XS": 136,
                "TQ1_0": 54, "TQ2_0": 66, "MXFP4": 17, "NVFP4": 36, "Q1_0": 18, "Q2_0": 18}
NATIVE_BLOCK_VALUES = {"Q8_0": 32, "Q4_1": 32, "Q5_1": 32, "IQ4_NL": 32, "Q4_0": 32,
                       "Q5_0": 32, "MXFP4": 32, "NVFP4": 64, "Q1_0": 128, "Q2_0": 64}
# The GGUF spells its microscaling block type "NVFP4", a name the artifact already gives the
# compressed-tensors block-scaled format; as a stored block type it is "NVFP4_GGML". Both
# tables answer to both spellings so a lookup by either side's name lands.
ARTIFACT_FORMAT_FOR_GGML = {"NVFP4": "NVFP4_GGML"}
NATIVE_TYPES["NVFP4_GGML"] = NATIVE_TYPES["NVFP4"]
NATIVE_BLOCK_VALUES["NVFP4_GGML"] = NATIVE_BLOCK_VALUES["NVFP4"]


def artifact_format_for_ggml(gguf_type: str) -> str:
    """The artifact format name a GGUF block type is stored under."""
    return ARTIFACT_FORMAT_FOR_GGML.get(gguf_type, gguf_type)


def native_block_values(gguf_type: str) -> int:
    return NATIVE_BLOCK_VALUES.get(gguf_type, 256)
_NATIVE_LAYOUT = "ggml-blocks-v1"

class NativePlan(dict):
    """{object: GGML type} of the objects served from the file's own blocks, plus, for a parent
    whose rows come in more than one type, its consecutive typed runs ((type, rows), ...)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segments: dict[str, tuple[tuple[str, int], ...]] = {}
        #: {object: k/32 entries} for an object read in the file's column order, whose columns
        #: the runtime rearranges on the activation side.
        self.group_maps: dict[str, tuple[int, ...]] = {}


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
    """Q8_0 plane provider + recipe row-algebra evaluator over one GGUF, split or not."""

    def __init__(self, map_path: str | Path):
        raw = json.loads(Path(map_path).read_text())
        self._init(raw["gguf_path"], dict(raw["sources"]))

    @classmethod
    def from_sources(cls, gguf_path: str | Path | Sequence[str | Path],
                     sources: Mapping[str, str]):
        """Planner entry: candidate map held in memory (bridge pre-walk).

        `gguf_path` may name one file or the ordered shards of a split one; an entry's
        `source` field, 1-based, says which shard holds it and defaults to the first.
        """
        self = cls.__new__(cls)
        self._init(gguf_path, dict(sources))
        return self

    def _init(self, gguf_path, sources: dict[str, dict]) -> None:
        paths = ([gguf_path] if isinstance(gguf_path, (str, Path))
                 else [item for item in gguf_path])
        self.gguf_paths = tuple(Path(item) for item in paths)
        if not self.gguf_paths:
            raise RepackError("a repack source needs at least one GGUF")
        #: The first shard, which is the whole file for every unsplit GGUF. Kept because a
        #: caller that knows its source is one file names it this way.
        self.gguf_path = self.gguf_paths[0]
        # The recipes and the bridge may spell one tensor differently (a `.weight` suffix, the
        # VL nesting, a router alias); index every spelling so a lookup finds the entry.
        self.sources = dict(sources)
        for stored, entry in list(sources.items()):
            for spelling in name_spellings(stored):
                self.sources.setdefault(spelling, entry)
        for path in self.gguf_paths:
            if not path.is_file():
                raise RepackError(f"repack map points at a missing GGUF: {path}")
        for hf_name, entry in sources.items():
            if not {"name", "rows", "k", "offset", "type"} <= set(entry):
                raise RepackError(f"{hf_name}: repack map entry is missing fields")
            if entry["type"] not in REPACKABLE_TYPES and entry["type"] not in NATIVE_TYPES:
                raise RepackError(
                    f"{hf_name}: GGUF type {entry['type']!r} is neither exactly repackable "
                    "nor a native K-quant"
                )
            shard = int(entry.get("source", 1))
            if not 1 <= shard <= len(self.gguf_paths):
                raise RepackError(
                    f"{hf_name}: source {shard} is not one of the {len(self.gguf_paths)} "
                    "declared GGUF shards"
                )
        self._files: dict[int, np.ndarray] = {}
        self._planes: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # -- GGUF payload access (memmap; no metadata parse) --------------------

    def source_index(self, hf_name: str, default: int = 1) -> int:
        """Which shard holds this source, 1-based, as the artifact's external table numbers
        them. A single-file GGUF leaves the field out and every run reads source 1."""
        return int(self.sources[hf_name].get("source", default))

    def _memmap(self, source: int = 1) -> np.ndarray:
        mapped = self._files.get(source)
        if mapped is None:
            mapped = np.memmap(self.gguf_paths[source - 1], dtype=np.uint8, mode="r")
            self._files[source] = mapped
        return mapped

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


    def row_map(self, hf_name: str) -> np.ndarray | None:
        """File row for each of the source's rows, when the two differ.

        llama.cpp reorders the V heads of a GDN projection, so the tensor the recipe names is a
        row permutation of the one the file holds. Expressing that as a map rather than as a
        materialised transform is what lets those weights be read in place: a permuted head is
        128 contiguous rows, so it is one run, not 128.
        """
        entry = self.sources.get(hf_name)
        if entry is None:
            return None
        raw = entry.get("row_perm")
        return None if raw is None else np.asarray(raw, dtype=np.int64)

    def column_group_map(self, hf_name: str) -> np.ndarray | None:
        """Source group for each destination group, when the source's inverse permutes columns.

        Unlike `row_map` this cannot be applied by gathering rows, so the paths that read a
        source's bytes directly refuse it and only the in-place transform carries it.
        """
        entry = self.sources.get(hf_name)
        if entry is None:
            return None
        raw = entry.get("col_groups")
        return None if raw is None else np.asarray(raw, dtype=np.int64)

    def planes(self, hf_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Deinterleaved (codes int8 [n, groups, 32], scales fp16 [n, groups])."""
        cached = self._planes.get(hf_name)
        if cached is not None:
            return cached
        entry = self.sources[hf_name]
        n, k = self.source_rows_k(hf_name)
        if self.column_group_map(hf_name) is not None:
            raise RepackError(f"{hf_name}: its inverse permutes columns; only the in-place "
                              "transform carries that")
        if k % _GROUP != 0:
            raise RepackError(f"{hf_name}: k={k} is not a multiple of {_GROUP}")
        if entry["type"] in NATIVE_TYPES and native_block_values(str(entry["type"])) == 256:
            raise RepackError(f"{hf_name}: {entry['type']} is a superblock format, not planes")
        block_bytes, decoder = REPACKABLE_TYPES[entry["type"]]
        offset = int(entry["offset"])
        nbytes = n * (k // _GROUP) * block_bytes
        data = self._memmap(self.source_index(hf_name))
        if offset < 0 or offset + nbytes > data.shape[0]:
            raise RepackError(f"{hf_name}: quantized payload is outside the GGUF file")
        raw = np.asarray(data[offset : offset + nbytes]).reshape(
            n, k // _GROUP, block_bytes
        )
        codes, scales = decoder(raw)
        row_map = self.row_map(hf_name)
        if row_map is not None:
            codes = codes[row_map]
            scales = scales.reshape(n, k // _GROUP)[row_map]
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
        every row from mapped native sources at the spec's own k. Returns a NativePlan,
        {object name: GGML type}; a parent whose components the file quantised to different
        types -- a UD mixture's q beside its k, gate beside up -- is planned too, as
        consecutive typed row runs the artifact records as `segments` (NativePlan.segments),
        with the first run's type as the object's. The fused ops project each component from
        the segment that holds it, so nothing is requantised."""
        probe = np.zeros(1, dtype=np.int64) if with_token_ids else None
        planned: NativePlan = NativePlan()
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
            types = [self.native_type_of(name) for name in program.sources]
            if not types or any(t is None for t in types):
                continue
            if any(program.k % native_block_values(t) for t in types):
                continue
            # A source whose inverse permutes columns cannot be served from its own bytes as a
            # row map: the native path copies rows. An object that is one whole such tensor
            # (the GDN output projection of a reordered geometry, whose columns follow the V
            # heads) is read as the file holds it and carries the map, which the runtime applies
            # to the activation's column groups instead; anything else with such a source is
            # left to the dequantise path, where the in-place transform carries it.
            column_maps = [self.column_group_map(name) for name in program.sources]
            group_map = None
            if any(m is not None for m in column_maps):
                if len(program.sources) != 1 or len(set(types)) != 1:
                    continue
                n_rows, _ = self.source_rows_k(program.sources[0])
                if n_rows != int(program.rows.shape[0]) or not np.array_equal(
                        program.rows % _SOURCE_STRIDE, np.arange(n_rows)):
                    continue
                group_map = tuple(int(v) for v in column_maps[0])
            # Objects whose op has no kernel for the stored type yet: the caller names them,
            # because which ops a family routes an object through is the family's knowledge.
            if any(spec.name.endswith(suffix) for suffix in exclude_suffixes):
                continue
            only = os.environ.get("SUROGATE_GGUF_NATIVE_ONLY")  # bisection aid: name substrings
            if only and not any(part and part in spec.name for part in only.split(",")):
                continue
            per_row = [types[int(index)] for index in (program.rows // _SOURCE_STRIDE)]
            runs: list[list] = []
            for gguf_type in per_row:
                if runs and runs[-1][0] == gguf_type:
                    runs[-1][1] += 1
                else:
                    runs.append([gguf_type, 1])
            if len(runs) > 1 and os.environ.get("SUROGATE_GGUF_NATIVE_SEGMENTS", "1") == "0":
                continue
            planned[spec.name] = str(runs[0][0])
            if len(runs) > 1:
                planned.segments[spec.name] = tuple((str(t), int(n)) for t, n in runs)
            if group_map is not None:
                planned.group_maps[spec.name] = group_map
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
                            format=artifact_format_for_ggml(gguf_type), layout=_NATIVE_LAYOUT)
                )
        return tuple(out)

    @staticmethod
    def native_specs(
        tensor_specs: Sequence,
        plan: Mapping[str, str],
        runs: Mapping[str, tuple[tuple[int, int, int], ...]] | None = None,
    ) -> tuple:
        """The specs with each natively planned object's format and layout rewritten, its
        runs attached when it is served from the file rather than copied into the artifact,
        and its typed segments when the plan (a NativePlan) split it by format."""
        segments = getattr(plan, "segments", {})
        group_maps = getattr(plan, "group_maps", {})
        out = []
        for spec in tensor_specs:
            name = getattr(spec, "name", None)
            gguf_type = plan.get(name)
            if gguf_type is None:
                out.append(spec)
            else:
                extra = {"runs": runs[name]} if runs is not None and name in runs else {}
                if name in segments:
                    extra["segments"] = tuple(
                        (artifact_format_for_ggml(t), int(n)) for t, n in segments[name])
                if name in group_maps:
                    extra["group_map"] = group_maps[name]
                out.append(replace(spec, format=artifact_format_for_ggml(gguf_type), layout=_NATIVE_LAYOUT, **extra))
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
        data = self._memmap(self.source_index(hf_name))
        if offset < 0 or offset + nbytes > data.shape[0]:
            raise RepackError(f"{hf_name}: quantized payload is outside the GGUF file")
        # A column permutation is not applied here: the object that reads these bytes carries
        # the group map (plan_native) and the runtime rearranges the activation instead.
        rows_out = np.asarray(data[offset : offset + nbytes]).reshape(n, row_bytes)
        row_map = self.row_map(hf_name)
        return rows_out if row_map is None else rows_out[row_map]

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
            if spec.format not in types or (len(types) > 1 and not getattr(spec, "segments", ())):
                raise RepackError(f"{spec.name}: sources are {types}, spec is {spec.format}")
        rows = program.rows if row_slice is None else program.rows[row_slice]
        source_of = rows // _SOURCE_STRIDE
        row_of = rows % _SOURCE_STRIDE
        # Rows are gathered in order; a segmented parent's rows differ in width by source, so
        # the payload is the concatenation of each maximal same-source stretch.
        pieces: list[bytes] = []
        n = int(rows.shape[0])
        begin = 0
        while begin < n:
            index = int(source_of[begin])
            end = begin
            while end < n and int(source_of[end]) == index:
                end += 1
            pieces.append(self._native_rows(program.sources[index])[row_of[begin:end]].tobytes())
            begin = end
        return b"".join(pieces)

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

        Each run names the shard its tensor lives in, so a fused object may draw from more than
        one file of a split GGUF; `source` is the fallback for a map that does not say.
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
        rows = np.asarray(program.rows)
        source_of = rows // _SOURCE_STRIDE
        row_of = rows % _SOURCE_STRIDE
        base = [int(self.sources[name]["offset"]) for name in program.sources]
        shard = [self.source_index(name, source) for name in program.sources]
        # Bytes per row are the source's: a segmented parent draws rows of different widths
        # from differently-typed sources.
        width = []
        for index, name in enumerate(program.sources):
            n_rows, k = self.source_rows_k(name)
            source_type = str(self.sources[name]["type"])
            width.append((k // native_block_values(source_type)) * NATIVE_TYPES[source_type])
            row_map = self.row_map(name)
            if row_map is not None:
                mask = source_of == index
                row_of = np.where(mask, row_map[np.clip(row_of, 0, len(row_map) - 1)], row_of)

        # A new run starts wherever the source changes or the source row is not the next one.
        breaks = np.ones(rows.shape[0], dtype=bool)
        if rows.shape[0] > 1:
            breaks[1:] = (source_of[1:] != source_of[:-1]) | (row_of[1:] != row_of[:-1] + 1)
        starts = np.flatnonzero(breaks)
        ends = np.append(starts[1:], rows.shape[0])
        return tuple(
            (
                shard[int(source_of[begin])],
                base[int(source_of[begin])] + int(row_of[begin]) * width[int(source_of[begin])],
                int(end - begin) * width[int(source_of[begin])],
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
    ) -> dict[str, tuple[tuple[tuple[int, int, int], ...], str, tuple[int, ...]]]:
        """{object: (runs, transform)} for the W8 objects whose rows are all Q8_0.

        Q8_0 and W8G32_F16S hold the same numbers -- signed int8 with one binary16 scale per 32 --
        and differ only in arrangement, so an object whose kernels want the row-split planes can
        still be read from the file: the runs gather its rows as the GGUF holds them and the
        loader rearranges them on the device. Named per family, because which objects run on
        kernels that have no Q8_0 path is the family's knowledge.
        """
        probe = np.zeros(1, dtype=np.int64) if with_token_ids else None
        planned: dict[str, tuple[tuple[tuple[int, int, int], ...], str, tuple[int, ...]]] = {}
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
            # A column permutation travels as a map rather than in the runs, so it must be the
            # same for every source the object draws from.
            maps = {
                None if (m := self.column_group_map(source)) is None else tuple(int(v) for v in m)
                for source in program.sources
            }
            if len(maps) != 1:
                continue
            column_map = next(iter(maps)) or ()
            row_bytes = (program.k // 32) * NATIVE_TYPES["Q8_0"]
            rows = np.asarray(program.rows)
            source_of = rows // _SOURCE_STRIDE
            row_of = rows % _SOURCE_STRIDE
            base = [int(self.sources[source]["offset"]) for source in program.sources]
            shard = [self.source_index(source) for source in program.sources]
            for index, source in enumerate(program.sources):
                row_map = self.row_map(source)
                if row_map is not None:
                    mask = source_of == index
                    row_of = np.where(
                        mask, row_map[np.clip(row_of, 0, len(row_map) - 1)], row_of
                    )
            breaks = np.ones(rows.shape[0], dtype=bool)
            if rows.shape[0] > 1:
                breaks[1:] = (source_of[1:] != source_of[:-1]) | (row_of[1:] != row_of[:-1] + 1)
            starts = np.flatnonzero(breaks)
            ends = np.append(starts[1:], rows.shape[0])
            planned[str(name)] = (
                tuple(
                    (
                        shard[int(source_of[begin])],
                        base[int(source_of[begin])] + int(row_of[begin]) * row_bytes,
                        int(end - begin) * row_bytes,
                    )
                    for begin, end in zip(starts, ends)
                ),
                "q8_0-to-w8g32",
                column_map,
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
                runs, transform, column_map = entry
                out.append(
                    replace(spec, runs=runs, transform=transform, group_map=column_map)
                )
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
            # halves are read from the source's bytes too, so the same refusal applies
            if any(self.column_group_map(name) is not None for name in program.sources):
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
