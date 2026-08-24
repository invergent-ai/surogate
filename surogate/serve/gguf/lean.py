# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Lean GGUF metadata parser for the serve ingest path.
#
# gguf-py's GGUFReader materializes every KV field eagerly; on a 250k-token
# vocabulary (tokens + token_type + merges arrays) that costs ~10s of pure
# Python before a single tensor byte is touched. The serve path rarely needs
# those arrays: target detection reads a handful of scalars, the repack
# planner needs tensor infos (name/shape/type/offset), and only the one-time
# frontend reconstruction reads the token arrays. This parser walks the
# header once, records value SPANS, and parses individual fields on demand —
# array fields are skipped (fixed-size elements by arithmetic, string
# elements by a length walk) until explicitly requested.
#
# Compatibility contract (pinned by tests/serve/test_gguf_lean.py against
# gguf-py on the same files):
#   - kv(name) returns the same scalar/string/list values as
#     GGUFReader.get_field(name).contents()
#   - tensors() yields (name, shape_ne, type_name, absolute_data_offset)
#     with shape in GGUF ne order (innermost first) and the offset matching
#     gguf-py's ReaderTensor.data_offset
# gguf-py remains the oracle in tests and the dequantizer for non-repacked
# tensors (gguf.quants.dequantize on memmapped payload slices).

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

_MAGIC = b"GGUF"

# GGUF value types.
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STRING, _ARRAY, _U64, _I64, _F64 = range(13)

_SCALAR_FORMATS = {
    _U8: ("<B", 1),
    _I8: ("<b", 1),
    _U16: ("<H", 2),
    _I16: ("<h", 2),
    _U32: ("<I", 4),
    _I32: ("<i", 4),
    _F32: ("<f", 4),
    _BOOL: ("<?", 1),
    _U64: ("<Q", 8),
    _I64: ("<q", 8),
    _F64: ("<d", 8),
}

# GGML tensor types: name and (block_size, type_size) for payload math.
# Only the entries the serve path can meet need to be exact; unknown ids
# still parse structurally (payload size is not needed for metadata).
_GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    30: ("BF16", 1, 2),
}


class LeanGgufError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LeanTensor:
    name: str
    shape: tuple[int, ...]  # GGUF ne order (innermost first), gguf-py parity
    type_id: int
    type_name: str
    data_offset: int  # absolute file offset (gguf-py ReaderTensor.data_offset)

    @property
    def tensor_type(self) -> str:
        return self.type_name


@dataclass(frozen=True, slots=True)
class _FieldSpan:
    vtype: int
    offset: int  # file offset of the VALUE (after key + vtype)


class LeanGguf:
    """One-pass GGUF metadata index; field values parse on demand."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = open(self.path, "rb")
        header = self._file.read(24)
        if len(header) < 24 or header[:4] != _MAGIC:
            raise LeanGgufError(f"'{self.path}' is not a GGUF file")
        self.version, tensor_count, kv_count = struct.unpack_from("<IQQ", header, 4)
        if self.version < 2:
            raise LeanGgufError(f"GGUF v{self.version} is not supported (need v2+)")

        # Walk the KV section once, recording spans.
        self._fields: dict[str, _FieldSpan] = {}
        pos = 24
        for _ in range(kv_count):
            key, pos = self._read_string(pos)
            vtype = self._read_scalar(_U32, pos)
            pos += 4
            self._fields[key] = _FieldSpan(vtype, pos)
            pos = self._skip_value(vtype, pos)

        # Tensor infos.
        infos: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(tensor_count):
            name, pos = self._read_string(pos)
            ndims = self._read_scalar(_U32, pos)
            pos += 4
            dims = struct.unpack_from(f"<{ndims}Q", self._read_at(pos, 8 * ndims))
            pos += 8 * ndims
            type_id = self._read_scalar(_U32, pos)
            pos += 4
            rel_offset = self._read_scalar(_U64, pos)
            pos += 8
            infos.append((name, tuple(int(d) for d in dims), type_id, rel_offset))

        alignment = int(self.kv("general.alignment") or 32)
        data_start = (pos + alignment - 1) // alignment * alignment
        self.data_start = data_start
        self.tensors: tuple[LeanTensor, ...] = tuple(
            LeanTensor(
                name=name,
                shape=dims,
                type_id=type_id,
                type_name=_GGML_TYPES.get(type_id, (f"TYPE_{type_id}", 0, 0))[0],
                data_offset=data_start + rel_offset,
            )
            for name, dims, type_id, rel_offset in infos
        )
        self._by_name = {t.name: t for t in self.tensors}

    # -- raw access --------------------------------------------------------

    def _read_at(self, offset: int, count: int) -> bytes:
        self._file.seek(offset)
        data = self._file.read(count)
        if len(data) != count:
            raise LeanGgufError(f"truncated GGUF: wanted {count} bytes at {offset}")
        return data

    def _read_scalar(self, vtype: int, offset: int):
        fmt, size = _SCALAR_FORMATS[vtype]
        return struct.unpack(fmt, self._read_at(offset, size))[0]

    def _read_string(self, offset: int) -> tuple[str, int]:
        (length,) = struct.unpack("<Q", self._read_at(offset, 8))
        raw = self._read_at(offset + 8, length)
        return raw.decode("utf-8"), offset + 8 + length

    def _skip_value(self, vtype: int, offset: int) -> int:
        if vtype in _SCALAR_FORMATS:
            return offset + _SCALAR_FORMATS[vtype][1]
        if vtype == _STRING:
            (length,) = struct.unpack("<Q", self._read_at(offset, 8))
            return offset + 8 + length
        if vtype == _ARRAY:
            elem_type = self._read_scalar(_U32, offset)
            count = self._read_scalar(_U64, offset + 4)
            pos = offset + 12
            if elem_type in _SCALAR_FORMATS:
                return pos + count * _SCALAR_FORMATS[elem_type][1]
            if elem_type == _STRING:
                # Length-chained walk; buffered to one read for speed.
                return self._skip_string_array(pos, count)
            raise LeanGgufError(f"unsupported GGUF array element type {elem_type}")
        raise LeanGgufError(f"unsupported GGUF value type {vtype}")

    def _chunked_string_walk(self, pos: int, count: int, collect: bool):
        """Walk `count` length-prefixed strings from `pos` with bounded reads.

        One buffered chunk per ~8MB instead of 2*count seeks; never reads to
        EOF (tensor payloads of multi-GB files must stay untouched).
        """
        chunk_size = 8 << 20
        self._file.seek(pos)
        chunk = self._file.read(chunk_size)
        base = pos  # file offset of chunk[0]
        local = 0
        values: list[str] | None = [] if collect else None
        for _ in range(count):
            while len(chunk) - local < 8:
                chunk = chunk[local:] + self._file.read(chunk_size)
                base += local
                local = 0
            (length,) = struct.unpack_from("<Q", chunk, local)
            while len(chunk) - local < 8 + length:
                chunk = chunk[local:] + self._file.read(max(chunk_size, length + 8))
                base += local
                local = 0
                if len(chunk) < 8 + length:
                    raise LeanGgufError("truncated GGUF string array")
            if values is not None:
                values.append(chunk[local + 8 : local + 8 + length].decode("utf-8"))
            local += 8 + length
        return base + local, values

    def _skip_string_array(self, pos: int, count: int) -> int:
        end, _ = self._chunked_string_walk(pos, count, collect=False)
        return end

    # -- public API --------------------------------------------------------

    def has(self, name: str) -> bool:
        return name in self._fields

    def kv(self, name: str, default=None):
        """Parse one KV field: scalar, string, or (on demand) full array."""
        span = self._fields.get(name)
        if span is None:
            return default
        if span.vtype in _SCALAR_FORMATS:
            return self._read_scalar(span.vtype, span.offset)
        if span.vtype == _STRING:
            return self._read_string(span.offset)[0]
        if span.vtype == _ARRAY:
            return self._parse_array(span.offset)
        raise LeanGgufError(f"unsupported GGUF value type {span.vtype}")

    def _parse_array(self, offset: int):
        elem_type = self._read_scalar(_U32, offset)
        count = self._read_scalar(_U64, offset + 4)
        pos = offset + 12
        if elem_type in _SCALAR_FORMATS:
            fmt, size = _SCALAR_FORMATS[elem_type]
            raw = self._read_at(pos, count * size)
            return list(struct.unpack(f"<{count}{fmt[-1]}", raw))
        if elem_type == _STRING:
            _, values = self._chunked_string_walk(pos, count, collect=True)
            return values
        raise LeanGgufError(f"unsupported GGUF array element type {elem_type}")

    def tensor(self, name: str) -> LeanTensor | None:
        return self._by_name.get(name)

    def get_field(self, name: str):
        """gguf-py GGUFReader facade: object with .contents(), or None.

        Keeps KV consumers (frontend.py, bridge KV helpers) source-compatible
        with both readers."""
        if name not in self._fields:
            return None
        value = self.kv(name)

        class _Field:
            __slots__ = ("_value",)

            def __init__(self, v):
                self._value = v

            def contents(self):
                return self._value

        return _Field(value)

    def payload_view(self, tensor: LeanTensor, mm) -> "object":
        """Byte view (rows, row_bytes) over a memmap — dequantize-compatible."""
        entry = _GGML_TYPES.get(tensor.type_id)
        if entry is None or entry[1] == 0:
            raise LeanGgufError(f"{tensor.name}: unknown GGML type {tensor.type_id}")
        _, block, type_size = entry
        ne0 = tensor.shape[0]
        if ne0 % block != 0:
            raise LeanGgufError(f"{tensor.name}: ne0 {ne0} not divisible by block {block}")
        row_bytes = ne0 // block * type_size
        rows = 1
        for dim in tensor.shape[1:]:
            rows *= dim
        nbytes = rows * row_bytes
        return mm[tensor.data_offset : tensor.data_offset + nbytes].reshape(rows, row_bytes)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "LeanGguf":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["LeanGguf", "LeanGgufError", "LeanTensor"]
