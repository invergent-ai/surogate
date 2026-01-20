"""Matrix multiplication primitives."""

from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import primitive, save
from .common import TransposeMode


@primitive(impl="kernels.matmul")
def matmul(
    A: Tensor["M", "K"],
    B: Tensor["K", "N"],
    *,
    transpose: TransposeMode = TransposeMode.NN,
    accumulate: bool = False,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Tensor["M", "N"]:
    """General matrix multiplication: C = alpha * op(A) @ op(B) + beta * C

    Transpose modes:
    - NN: A[M,K] @ B[K,N]
    - NT: A[M,K] @ B[N,K]
    - TN: A[K,M] @ B[K,N]
    - TT: A[K,M] @ B[N,K]
    """
    ...


@matmul.backward
@save("A", "B")
def matmul_backward(
    d_C: Tensor["M", "N"],
    A: Tensor["M", "K"],
    B: Tensor["K", "N"],
) -> tuple[Tensor["M", "K"], Tensor["K", "N"]]:
    """Backward pass for matmul."""
    ...


@primitive(impl="kernels.batched_matmul")
def batched_matmul(
    A: Tensor["B", "M", "K"],
    B: Tensor["B", "K", "N"],
    *,
    transpose: TransposeMode = TransposeMode.NN,
) -> Tensor["B", "M", "N"]:
    """Batched matrix multiplication."""
    ...
