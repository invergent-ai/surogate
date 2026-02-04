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


@primitive(impl="kernels.matmul_swiglu")
def matmul_swiglu(
    A: Tensor["*", "K"],
    B: Tensor["2*M", "K"],
    *,
    transpose: TransposeMode = TransposeMode.NT,
) -> Tensor["*", "M"]:
    """Fused matmul + SwiGLU activation.

    Computes: swiglu(A @ B^T) where B contains fused [up, gate] weights.

    Output shape is half the matmul output (SwiGLU halves the last dimension).
    This is the common MLP up projection pattern in LLaMA/Qwen models.

    Note: The executor may decompose this into separate matmul + swiglu
    operations if a fused kernel is not available.
    """
    ...


@matmul_swiglu.backward
@save("A", "B")
def matmul_swiglu_backward(
    d_out: Tensor["*", "M"],
    A: Tensor["*", "K"],
    B: Tensor["2*M", "K"],
    up_output: Tensor["*", "2*M"],
) -> tuple[Tensor["*", "K"], Tensor["2*M", "K"]]:
    """Backward pass for fused matmul + swiglu.

    Note: up_output is the intermediate matmul result needed for swiglu backward.
    The executor must save this during forward pass.
    """
    ...
