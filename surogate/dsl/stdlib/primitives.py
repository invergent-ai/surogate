"""
Primitive Operations for Python DSL

Ports the primitives from std/primitives/*.module to Python decorator syntax.
"""

from __future__ import annotations
from enum import Enum

from ..tensor_type import Tensor
from ..decorators import primitive, save


# =============================================================================
# Transpose Mode Enum
# =============================================================================


class TransposeMode(str, Enum):
    """Transpose mode for matmul operations."""
    NN = "NN"  # Neither transposed
    NT = "NT"  # B transposed
    TN = "TN"  # A transposed
    TT = "TT"  # Both transposed


# =============================================================================
# Matrix Operations
# =============================================================================


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


# =============================================================================
# Normalization
# =============================================================================


@primitive(impl="kernels.rmsnorm")
def rmsnorm(
    x: Tensor["*", "C"],
    weight: Tensor["C"],
    *,
    eps: float = 1e-6,
) -> tuple[Tensor["*", "C"], Tensor["*"]]:
    """RMS normalization. Returns (y, rstd)."""
    ...


@rmsnorm.backward
@save("x", "rstd")
def rmsnorm_backward(
    d_y: Tensor["*", "C"],
    x: Tensor["*", "C"],
    weight: Tensor["C"],
    rstd: Tensor["*"],
) -> tuple[Tensor["*", "C"], Tensor["C"]]:
    """Backward pass for RMS norm. Returns (d_x, d_weight)."""
    ...


@primitive(impl="kernels.fused_residual_rmsnorm")
def fused_residual_rmsnorm(
    residual: Tensor["*", "C"],
    x: Tensor["*", "C"],
    weight: Tensor["C"],
    *,
    eps: float = 1e-6,
) -> tuple[Tensor["*", "C"], Tensor["*", "C"], Tensor["*"]]:
    """Fused residual add + RMS norm. Returns (residual_out, y, rstd)."""
    ...


@fused_residual_rmsnorm.backward
@save("residual_out", "rstd")
def fused_residual_rmsnorm_backward(
    d_y: Tensor["*", "C"],
    d_residual_next: Tensor["*", "C"],
    residual_out: Tensor["*", "C"],
    weight: Tensor["C"],
    rstd: Tensor["*"],
) -> tuple[Tensor["*", "C"], Tensor["*", "C"], Tensor["C"]]:
    """Backward pass. Returns (d_residual, d_input, d_weight)."""
    ...


# =============================================================================
# Activations
# =============================================================================


@primitive(impl="kernels.swiglu")
def swiglu(x: Tensor["*", "2M"]) -> Tensor["*", "M"]:
    """SwiGLU activation: silu(gate) * up where x = [gate, up]."""
    ...


@swiglu.backward
@save("x")
def swiglu_backward(
    d_out: Tensor["*", "M"],
    x: Tensor["*", "2M"],
) -> Tensor["*", "2M"]:
    """Backward pass for SwiGLU."""
    ...


@primitive(impl="kernels.silu")
def silu(x: Tensor["*"]) -> Tensor["*"]:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    ...


@silu.backward
@save("x")
def silu_backward(d_out: Tensor["*"], x: Tensor["*"]) -> Tensor["*"]:
    """Backward pass for SiLU."""
    ...


@primitive(impl="kernels.relu2")
def relu2(x: Tensor["*"]) -> Tensor["*"]:
    """ReLU squared activation: max(0, x)^2."""
    ...


@primitive(impl="kernels.silu_mul")
def silu_mul(gate: Tensor["*"], up: Tensor["*"]) -> Tensor["*"]:
    """SiLU(gate) * up activation."""
    ...


# =============================================================================
# Attention
# =============================================================================


@primitive(impl="kernels.flash_attention")
def flash_attention(
    qkv: Tensor["B", "T", "Hq + 2 * Hkv", "D"],
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
    window_size: int | None = None,
) -> tuple[Tensor["B", "T", "Hq", "D"], Tensor["B", "Hq", "T"]]:
    """FlashAttention with packed QKV. Returns (out, lse)."""
    ...


@flash_attention.backward
@save("qkv", "out", "lse")
def flash_attention_backward(
    d_out: Tensor["B", "T", "Hq", "D"],
    out: Tensor["B", "T", "Hq", "D"],
    lse: Tensor["B", "Hq", "T"],
    qkv: Tensor["B", "T", "Hq + 2 * Hkv", "D"],
) -> Tensor["B", "T", "Hq + 2 * Hkv", "D"]:
    """Backward pass for FlashAttention."""
    ...


@primitive(impl="kernels.rope")
def rope(
    qkv: Tensor["B", "T", "H", "D"],
    freqs: Tensor["T", "D // 2", 2, "fp32"],
    position_ids: Tensor["T", "int32"],
    *,
    rotary_dim: int | None = None,
) -> Tensor["B", "T", "H", "D"]:
    """Apply rotary position embedding."""
    ...


@primitive(impl="kernels.qkv_qk_norm_rope")
def qkv_qk_norm_rope(
    qkv: Tensor["B", "T", "Hq + 2 * Hkv", "D"],
    q_norm_weight: Tensor["D"],
    k_norm_weight: Tensor["D"],
    freqs: Tensor["T", "D // 2", 2, "fp32"],
    position_ids: Tensor["T", "int32"],
    *,
    eps: float = 1e-6,
) -> tuple[Tensor["B", "T", "Hq + 2 * Hkv", "D"], Tensor["*"], Tensor["*"]]:
    """Fused QK norm + RoPE. Returns (qkv_out, q_rstd, k_rstd)."""
    ...


# =============================================================================
# Embedding
# =============================================================================


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


# =============================================================================
# Tensor Operations
# =============================================================================


@primitive(impl="kernels.view")
def view(x: Tensor["*"], *, shape: list[int | str]) -> Tensor["*"]:
    """Reshape tensor (no data movement if contiguous)."""
    ...


@primitive(impl="kernels.transpose")
def transpose(
    x: Tensor["D0", "D1"],
    *,
    dim0: int = 0,
    dim1: int = 1,
) -> Tensor["D1", "D0"]:
    """Transpose two dimensions."""
    ...


@transpose.backward
def transpose_backward(d_out: Tensor["D1", "D0"]) -> Tensor["D0", "D1"]:
    """Backward pass for transpose (just transpose back)."""
    ...


@primitive(impl="kernels.concat")
def concat(*tensors: Tensor["*"], dim: int = 0) -> Tensor["*"]:
    """Concatenate tensors along dimension."""
    ...


@primitive(impl="kernels.split")
def split(
    x: Tensor["*"],
    *,
    split_size: int | list[int],
    dim: int = 0,
) -> tuple[Tensor["*"], ...]:
    """Split tensor along dimension."""
    ...


# =============================================================================
# Elementwise Operations
# =============================================================================


@primitive(impl="kernels.add")
def add(a: Tensor["*"], b: Tensor["*"]) -> Tensor["*"]:
    """Element-wise addition."""
    ...


@primitive(impl="kernels.mul")
def mul(a: Tensor["*"], b: Tensor["*"]) -> Tensor["*"]:
    """Element-wise multiplication."""
    ...


@primitive(impl="kernels.scale")
def scale(x: Tensor["*"], *, factor: float) -> Tensor["*"]:
    """Scale tensor by constant factor."""
    ...


@primitive(impl="kernels.bias_add")
def bias_add(x: Tensor["*", "C"], bias: Tensor["C"]) -> Tensor["*", "C"]:
    """Add bias along last dimension."""
    ...


# =============================================================================
# Initialization
# =============================================================================


@primitive(impl="kernels.zeros")
def zeros(*, shape: list[int | str], dtype: str = "bf16") -> Tensor["*"]:
    """Create zero-filled tensor."""
    ...


@primitive(impl="kernels.ones")
def ones(*, shape: list[int | str], dtype: str = "bf16") -> Tensor["*"]:
    """Create one-filled tensor."""
    ...


@primitive(impl="kernels.fill_normal")
def fill_normal(
    *,
    shape: list[int | str],
    mean: float = 0.0,
    std: float = 1.0,
    seed: int = 0,
    dtype: str = "bf16",
) -> Tensor["*"]:
    """Create tensor filled with normal random values."""
    ...


# =============================================================================
# MoE Operations
# =============================================================================


@primitive(impl="kernels.moe_softmax")
def moe_softmax(logits: Tensor["BT", "E"]) -> Tensor["BT", "E"]:
    """MoE router softmax."""
    ...


@primitive(impl="kernels.moe_sigmoid")
def moe_sigmoid(logits: Tensor["BT", "E"]) -> Tensor["BT", "E"]:
    """MoE router sigmoid (for Qwen3 style)."""
    ...


@primitive(impl="kernels.moe_topk")
def moe_topk(
    probs: Tensor["BT", "E"],
    *,
    top_k: int,
    normalize: bool = True,
) -> tuple[Tensor["BT", "top_k"], Tensor["BT", "top_k", "int32"]]:
    """MoE top-k selection. Returns (weights, indices)."""
    ...


@primitive(impl="kernels.moe_permute")
def moe_permute(
    x: Tensor["BT", "C"],
    indices: Tensor["BT", "top_k", "int32"],
    *,
    top_k: int,
) -> Tensor["BT * top_k", "C"]:
    """Permute inputs for expert computation."""
    ...


@primitive(impl="kernels.moe_unpermute")
def moe_unpermute(
    expert_out: Tensor["BT * top_k", "C"],
    weights: Tensor["BT", "top_k"],
    scatter_indices: Tensor["BT * top_k", "int32"],
    *,
    top_k: int,
) -> Tensor["BT", "C"]:
    """Unpermute and combine expert outputs."""
    ...


@primitive(impl="kernels.moe_grouped_gemm_gate_up")
def moe_grouped_gemm_gate_up(
    x: Tensor["BT", "C"],
    weights: Tensor["E", "2 * D", "C"],
    offsets: Tensor["E + 1", "int32"],
) -> Tensor["BT", "2 * D"]:
    """MoE grouped GEMM for gate+up projection."""
    ...


@primitive(impl="kernels.moe_grouped_gemm_down")
def moe_grouped_gemm_down(
    x: Tensor["BT", "D"],
    weights: Tensor["E", "C", "D"],
    offsets: Tensor["E + 1", "int32"],
) -> Tensor["BT", "C"]:
    """MoE grouped GEMM for down projection."""
    ...
