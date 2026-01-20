"""Embedding primitives."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import primitive, save


@primitive(impl="kernels.embedding")
def embedding(
    indices: Tensor["B", "T", "int32"],
    weight: Tensor["V", "C"],
) -> Tensor["B", "T", "C"]:
    """Embedding lookup."""
    ...


@embedding.backward
@save("indices")
def embedding_backward(
    d_out: Tensor["B", "T", "C"],
    indices: Tensor["B", "T", "int32"],
    vocab_size: int,
) -> Tensor["V", "C"]:
    """Backward pass for embedding (scatter-add)."""
    ...
