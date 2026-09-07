"""Reading a split GGUF as a converter's weight source.

Three converters now take a GGUF directly rather than through the HF bridge -- Qwen3.8-Flash-
Next, the Gemma embedding tower and GLM-5.3 -- because gguf-py has no name map for their
architectures, or because the HF checkpoint is a hundred shards nobody wants to fetch. They all
need the same four things: the tensors of every shard by name, their bytes where they lie, a
dequantised view for the objects the artifact materialises, and the candidate map the repack
planners read. This is that, in one place; it used to live inside one of the three converters
and the other two imported it from there.

Nothing here knows any model. What a converter still owns is which objects exist, where their
rows come from, and which of them the row algebra cannot say.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

from surogate.serve.artifact.layouts import encode_direct
from surogate.serve.convert.common.gguf_repack import (
    NATIVE_TYPES,
    REPACKABLE_TYPES,
    native_block_values,
)
from surogate.serve.convert.common.inventory import BF16, FP32, I32
from surogate.serve.gguf.lean import LeanGguf

_GROUP = 32


def stored_bytes(type_name: str, shape: tuple[int, ...]) -> int:
    """How many bytes a GGUF holds for a tensor of this logical shape."""
    direct = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4}.get(type_name)
    count = int(np.prod(shape))
    if direct is not None:
        return count * direct
    if type_name in NATIVE_TYPES:
        return count // native_block_values(type_name) * NATIVE_TYPES[type_name]
    raise ValueError(f"unsupported GGML type {type_name!r}")


@dataclass(frozen=True, slots=True)
class GgufTensor:
    shard: int  # 1-based, the numbering the artifact's external table uses
    shape: tuple[int, ...]  # logical (numpy) shape, i.e. reversed ggml ne
    type_id: int
    type_name: str
    offset: int  # absolute, within its shard
    nbytes: int

    @property
    def tensor_type(self) -> GGMLQuantizationType:
        return GGMLQuantizationType(self.type_id)


class _Fields:
    """gguf-py's `reader.fields` facade over the shards of a split GGUF."""

    def __init__(self, readers):
        self._readers = readers

    def __contains__(self, key: str) -> bool:
        return any(reader.has(key) for reader in self._readers)

    def __getitem__(self, key: str):
        for reader in self._readers:
            field = reader.get_field(key)
            if field is not None:
                return field
        raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class GgufSource:
    """All shards of one split GGUF, tensors by name.

    `LeanGguf` rather than gguf-py's reader: the eager key-value parse costs ~10 s a shard on
    a 250k-token vocabulary, and a converter reads a few key-value arrays and a memmap.
    """

    def __init__(self, first_shard: Path, extra: Sequence[Path] = ()):
        pattern = str(first_shard).replace("-00001-of-", "-*-of-")
        self.shards = [Path(p) for p in sorted(glob.glob(pattern))] or [Path(first_shard)]
        # Files that are not shards of the split but are still read in place: a NextN head
        # ships as its own GGUF. They join the shard list, so they take the next external
        # source numbers and every run points at the file that actually holds the bytes.
        self.shards += [Path(p) for p in extra]
        self.readers = [LeanGguf(path) for path in self.shards]
        self._maps: dict[int, np.ndarray] = {}
        self.tensors: dict[str, GgufTensor] = {}
        for index, reader in enumerate(self.readers, start=1):
            for t in reader.tensors:
                shape = tuple(int(x) for x in reversed(t.shape))
                self.tensors[t.name] = GgufTensor(
                    shard=index,
                    shape=shape,
                    type_id=t.type_id,
                    type_name=t.type_name,
                    offset=t.data_offset,
                    nbytes=stored_bytes(t.type_name, shape),
                )

    @property
    def fields(self) -> "_Fields":
        """The key-values of every shard, under gguf-py's `fields[key].contents()` facade."""
        return _Fields(self.readers)

    def kv(self, key: str, default=None):
        for reader in self.readers:
            value = reader.kv(key, None)
            if value is not None:
                return value
        return default

    def array_field(self, key: str) -> list[int]:
        for reader in self.readers:
            value = reader.kv(key)
            if value is not None:
                return [int(item) for item in value]
        raise KeyError(f"GGUF has no key-value {key!r}")

    def tensor(self, name: str) -> GgufTensor:
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(f"GGUF has no tensor {name!r}") from error

    def _memmap(self, shard: int) -> np.ndarray:
        mapped = self._maps.get(shard)
        if mapped is None:
            mapped = np.memmap(self.shards[shard - 1], dtype=np.uint8, mode="r")
            self._maps[shard] = mapped
        return mapped

    def raw(self, name: str) -> np.ndarray:
        """The tensor's bytes as stored: the leading axis kept, everything after it flat."""
        t = self.tensor(name)
        return self._bytes(t).reshape(t.shape[0], -1)

    def _bytes(self, t: GgufTensor) -> np.ndarray:
        return np.asarray(self._memmap(t.shard)[t.offset : t.offset + t.nbytes])

    def _row_view(self, t: GgufTensor) -> np.ndarray:
        """(rows, bytes per row) -- what every block decoder, ours or gguf-py's, wants."""
        rows = int(np.prod(t.shape[:-1])) if len(t.shape) > 1 else 1
        return self._bytes(t).reshape(rows, t.nbytes // rows)

    def float32(self, name: str) -> np.ndarray:
        """Dequantised logical array (float32), any GGML type."""
        t = self.tensor(name)
        rows = self._row_view(t)
        if t.type_name == "F32":
            return rows.view(np.float32).reshape(t.shape)
        if t.type_name == "F16":
            return rows.view(np.float16).astype(np.float32).reshape(t.shape)
        if t.type_name == "BF16":
            words = rows.view(np.uint16).astype(np.uint32) << 16
            return words.view(np.float32).reshape(t.shape)
        decoded = dequantize(rows, GGMLQuantizationType(t.type_id))
        return np.asarray(decoded, dtype=np.float32).reshape(t.shape)

    def planes_exact(self, name: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """(codes int8 [n, groups, 32], scales fp16 [n, groups], (n, k)) for a repackable type."""
        t = self.tensor(name)
        if t.type_name not in REPACKABLE_TYPES:
            raise ValueError(f"{name}: {t.type_name} is not exactly repackable")
        block_bytes, decoder = REPACKABLE_TYPES[t.type_name]
        k = t.shape[-1]
        n = int(np.prod(t.shape[:-1])) if len(t.shape) > 1 else 1
        if k % _GROUP != 0:
            raise ValueError(f"{name}: k={k} is not a multiple of {_GROUP}")
        codes, scales = decoder(self._row_view(t).reshape(n, k // _GROUP, block_bytes))
        return (np.ascontiguousarray(codes),
                np.ascontiguousarray(scales).reshape(n, k // _GROUP), (n, k))

    def close(self) -> None:
        for reader in self.readers:
            reader.close()


class GgufRecipeReader:
    """`materialize_expression`'s reader protocol over the GGUF shards."""

    def __init__(self, source: GgufSource):
        self._source = source

    def has(self, name: str) -> bool:
        return name in self._source.tensors

    def get(self, name: str) -> torch.Tensor:
        return torch.from_numpy(self._source.float32(name))


def candidate_sources(source: GgufSource,
                      column_groups: dict[str, Sequence[int]] | None = None) -> dict[str, dict]:
    """The repack planner's candidate map, built from the GGUF header alone.

    Every tensor the engine could read where it lies: the K-quants and the 32-value block
    formats, each with the shard it sits in and its absolute offset within that shard. The F32
    and BF16 tensors are deliberately absent -- an object drawing on one falls out of every plan
    and is materialised, which is exactly what its BF16 or FP32 spec needs.

    `column_groups` names the tensors whose stored columns are in another order, by name suffix;
    the map travels with the candidate and the runtime permutes the activation instead.
    """
    candidates: dict[str, dict] = {}
    for name, tensor in source.tensors.items():
        if tensor.type_name not in REPACKABLE_TYPES and tensor.type_name not in NATIVE_TYPES:
            continue
        entry: dict = {
            "name": name,
            "shape": list(tensor.shape),
            "rows": int(np.prod(tensor.shape[:-1])),
            "k": int(tensor.shape[-1]),
            "offset": int(tensor.offset),
            "type": tensor.type_name,
            "source": tensor.shard,
        }
        for suffix, groups in (column_groups or {}).items():
            if name.endswith(suffix):
                entry["col_groups"] = list(groups)
        candidates[name] = entry
    return candidates


def encode_tensor(tensor: torch.Tensor, spec) -> bytes:
    """One materialised tensor in its registered format.

    Only the direct formats appear: a GGUF-native converter reads every quantised object from
    the file, and its "is every quantised object planned" check is what makes that true rather
    than hoped for.
    """
    if spec.format == BF16:
        return encode_direct(tensor.to(torch.bfloat16), BF16)
    if spec.format == FP32:
        return encode_direct(tensor.to(torch.float32), FP32)
    if spec.format == I32:
        return encode_direct(tensor.to(torch.int32), I32)
    raise ValueError(f"{spec.name}: nothing materialises {spec.format}; it must be read in place")


def check_every_quantised_object_is_planned(specs, planned: set[str], quantised_format: str) -> None:
    """No quantised object may reach the quantiser.

    The point of reading a GGUF in place is that nothing is dequantised and re-quantised; an
    object the planners missed would silently reintroduce that, so it is an error rather than a
    fallback.
    """
    stray = [spec.name for spec in specs
             if getattr(spec, "kind", None) == "tensor"
             and spec.format == quantised_format and spec.name not in planned]
    if stray:
        raise RuntimeError(
            f"{len(stray)} quantised objects are read from neither the native nor the "
            f"in-place plan: {stray[:6]}"
        )


__all__ = [
    "GgufRecipeReader",
    "GgufSource",
    "GgufTensor",
    "candidate_sources",
    "check_every_quantised_object_is_planned",
    "encode_tensor",
    "stored_bytes",
]
