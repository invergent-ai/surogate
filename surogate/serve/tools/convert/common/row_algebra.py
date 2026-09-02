"""Row-only evaluation of recipe expressions.

A recipe expression says where an artifact object's rows come from: a source
tensor, a slice of one, rows gathered from one, several concatenated. When
no node touches the K axis, the expression is a pure row program -- an
ordered list of (source, row) pairs -- and that program can be applied to
anything stored per row: BF16 rows, Q8_0 blocks, NVFP4 codes with their
block scales. That is what lets a quantized source be repacked without ever
decoding it.

This walker was written for the GGUF repack and lifted out of it unchanged
when the compressed-tensors path needed the same enumeration: a fused
parent's constituents, each its own source with its own global scale, are
exactly the segments of its row program.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from surogate.serve.tools.convert.common.recipe import (
    Concat,
    Expression,
    GatherRows,
    Reshape,
    Slice,
    SourceTensor,
    Transpose,
)

#: Global row id = source_index * SOURCE_STRIDE + row. Persisted by the GGUF
#: repack map, so it is a contract, not a tuning knob.
SOURCE_STRIDE = 1 << 40


class RowShapeMismatch(ValueError):
    """A source's stored shape disagrees with the shape the recipe declares."""


@dataclass(frozen=True, slots=True)
class RowProgram:
    """Row-only evaluation of one recipe expression.

    ``rows`` holds global row ids over ``sources`` (in first-use order); the
    flattened order is the object's logical row order because every supported
    expression keeps the trailing K axis untouched.
    """

    sources: tuple[str, ...]
    rows: np.ndarray  # int64 [N]
    k: int

    def segments(self) -> list[tuple[str, np.ndarray]]:
        """The program as runs of rows from one source, in object row order.

        A fused parent ``Concat(q, k, gate, v)`` yields four segments; a
        head-strided gather over one source yields one segment whose rows
        are not consecutive. Consecutive rows from the same source are one
        segment even across concatenation boundaries.
        """

        source_of = self.rows // SOURCE_STRIDE
        row_of = self.rows % SOURCE_STRIDE
        out: list[tuple[str, np.ndarray]] = []
        if self.rows.size == 0:
            return out
        boundaries = np.flatnonzero(np.diff(source_of)) + 1
        for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, self.rows.size]):
            out.append((self.sources[int(source_of[start])], row_of[start:stop]))
        return out


class _Unsupported(Exception):
    pass


def evaluate_rows(
    expression: Expression,
    source_shapes: Mapping[str, Sequence[int]],
    token_ids: np.ndarray | None,
) -> RowProgram | None:
    """Evaluate an expression to global row ids, or None if unsupported.

    ``source_shapes`` maps every source name the caller can supply to its
    stored shape. Values are ndarrays over the row axes only (the trailing K
    axis of every node is asserted equal and dropped); any expression touching
    K or mixing K extents is rejected. A source whose stored shape disagrees
    with the recipe raises ``RowShapeMismatch`` rather than being skipped.
    """

    sources: list[str] = []

    def source_index(name: str) -> int:
        if name not in source_shapes:
            raise _Unsupported()
        try:
            return sources.index(name)
        except ValueError:
            sources.append(name)
            return len(sources) - 1

    def walk(node: Expression) -> tuple[np.ndarray, int]:
        if isinstance(node, SourceTensor):
            if len(node.shape) != 2 or node.name not in source_shapes:
                raise _Unsupported()
            n, k = node.shape
            actual = tuple(source_shapes[node.name])
            if actual != (n, k):
                raise RowShapeMismatch(
                    f"{node.name}: source shape {actual} != recipe shape {(n, k)}"
                )
            index = source_index(node.name)
            ids = np.arange(n, dtype=np.int64) + index * SOURCE_STRIDE
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
                raise RowShapeMismatch(
                    f"GatherRows: {gathered.shape[0]} ids != declared {node.rows}"
                )
            return gathered, k
        raise _Unsupported()  # Cast, DraftHeadTokenIds, unknown nodes

    try:
        ids, k = walk(expression)
    except _Unsupported:
        return None
    return RowProgram(tuple(sources), ids.reshape(-1), k)


def collect_sources(node: Expression, out: set[str]) -> None:
    """Every source tensor name an expression reads, whatever its structure."""

    if isinstance(node, SourceTensor):
        out.add(node.name)
    elif isinstance(node, (Reshape, Slice, Transpose)):
        collect_sources(node.source, out)
    elif isinstance(node, Concat):
        for part in node.sources:
            collect_sources(part, out)
    elif isinstance(node, GatherRows):
        collect_sources(node.source, out)


__all__ = [
    "RowProgram",
    "RowShapeMismatch",
    "SOURCE_STRIDE",
    "collect_sources",
    "evaluate_rows",
]
