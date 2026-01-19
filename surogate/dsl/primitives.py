"""
Standard Library Primitives

Defines specifications for built-in primitive operations.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .ir import KernelType, PrimitiveSpec
from .types import Dtype, TensorTypeSpec, Shape, SymbolicDim, ConcreteDim, TransposeMode


def _make_tensor_type(dims: List[str], dtype: Dtype = Dtype.BF16) -> TensorTypeSpec:
    """Helper to create tensor type from dimension names."""
    parsed_dims = [SymbolicDim(d) for d in dims]
    return TensorTypeSpec(Shape(parsed_dims), dtype)


# =============================================================================
# Primitive Specifications
# =============================================================================


PRIMITIVES: Dict[str, PrimitiveSpec] = {}


def _register_primitive(spec: PrimitiveSpec):
    """Register a primitive specification."""
    PRIMITIVES[spec.name] = spec


# -----------------------------------------------------------------------------
# Linear Algebra Primitives
# -----------------------------------------------------------------------------

_register_primitive(PrimitiveSpec(
    name="matmul",
    kernel_type=KernelType.MATMUL,
    input_types={
        "A": _make_tensor_type(["M", "K"]),
        "B": _make_tensor_type(["K", "N"]),
    },
    output_types={
        "C": _make_tensor_type(["M", "N"]),
    },
    default_attrs={
        "transpose": TransposeMode.NN,
        "accumulate": False,
        "alpha": 1.0,
        "beta": 0.0,
    },
    save=["A", "B"],
    backward_kernel=KernelType.MATMUL,
    forward_impl="kernels.matmul",
    backward_impl="kernels.matmul",
))

_register_primitive(PrimitiveSpec(
    name="batched_matmul",
    kernel_type=KernelType.BATCHED_MATMUL,
    input_types={
        "A": _make_tensor_type(["B", "M", "K"]),
        "B": _make_tensor_type(["B", "K", "N"]),
    },
    output_types={
        "C": _make_tensor_type(["B", "M", "N"]),
    },
    default_attrs={
        "transpose": TransposeMode.NN,
        "accumulate": False,
    },
    save=["A", "B"],
    backward_kernel=KernelType.BATCHED_MATMUL,
    forward_impl="kernels.batched_matmul",
))

_register_primitive(PrimitiveSpec(
    name="grouped_gemm",
    kernel_type=KernelType.GROUPED_GEMM,
    input_types={
        "input": _make_tensor_type(["total_tokens", "K"]),
        "weights": _make_tensor_type(["num_groups", "N", "K"]),
        "offsets": _make_tensor_type(["num_groups_plus_1"], Dtype.INT32),
    },
    output_types={
        "output": _make_tensor_type(["total_tokens", "N"]),
    },
    default_attrs={
        "transpose": TransposeMode.TN,
    },
    save=["input", "offsets"],
    forward_impl="kernels.moe_grouped_gemm",
    backward_impl="kernels.moe_grouped_gemm_backward",
))

# -----------------------------------------------------------------------------
# Normalization Primitives
# -----------------------------------------------------------------------------

_register_primitive(PrimitiveSpec(
    name="rmsnorm",
    kernel_type=KernelType.RMSNORM,
    input_types={
        "x": _make_tensor_type(["B", "T", "C"]),
        "weight": _make_tensor_type(["C"]),
    },
    output_types={
        "y": _make_tensor_type(["B", "T", "C"]),
        "rstd": _make_tensor_type(["B", "T"]),
    },
    default_attrs={
        "eps": 1e-6,
    },
    save=["x", "rstd"],
    forward_impl="kernels.rmsnorm_forward",
    backward_impl="kernels.rmsnorm_backward",
))

_register_primitive(PrimitiveSpec(
    name="fused_residual_rmsnorm",
    kernel_type=KernelType.FUSED_RESIDUAL_RMSNORM,
    input_types={
        "residual": _make_tensor_type(["B", "T", "C"]),
        "input": _make_tensor_type(["B", "T", "C"]),
        "weight": _make_tensor_type(["C"]),
    },
    output_types={
        "residual_out": _make_tensor_type(["B", "T", "C"]),
        "y": _make_tensor_type(["B", "T", "C"]),
        "rstd": _make_tensor_type(["B", "T"]),
    },
    default_attrs={
        "eps": 1e-6,
    },
    save=["residual_out", "rstd"],
    forward_impl="kernels.fused_residual_rmsnorm_forward",
    backward_impl="kernels.fused_residual_rmsnorm_backward",
))

_register_primitive(PrimitiveSpec(
    name="layernorm",
    kernel_type=KernelType.LAYERNORM,
    input_types={
        "x": _make_tensor_type(["B", "T", "C"]),
        "weight": _make_tensor_type(["C"]),
        "bias": _make_tensor_type(["C"]),
    },
    output_types={
        "y": _make_tensor_type(["B", "T", "C"]),
        "mean": _make_tensor_type(["B", "T"]),
        "rstd": _make_tensor_type(["B", "T"]),
    },
    default_attrs={
        "eps": 1e-5,
    },
    save=["x", "mean", "rstd"],
    forward_impl="kernels.layernorm_forward",
    backward_impl="kernels.layernorm_backward",
))

# -----------------------------------------------------------------------------
# Activation Primitives
# -----------------------------------------------------------------------------

_register_primitive(PrimitiveSpec(
    name="swiglu",
    kernel_type=KernelType.SWIGLU,
    input_types={
        "gate": _make_tensor_type(["B", "T", "D"]),
        "up": _make_tensor_type(["B", "T", "D"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "T", "D"]),
    },
    save=["gate", "up"],
    forward_impl="kernels.silu_mul_forward",
    backward_impl="kernels.silu_mul_backward",
))

_register_primitive(PrimitiveSpec(
    name="geglu",
    kernel_type=KernelType.GEGLU,
    input_types={
        "gate": _make_tensor_type(["B", "T", "D"]),
        "up": _make_tensor_type(["B", "T", "D"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "T", "D"]),
    },
    save=["gate", "up"],
    forward_impl="kernels.gelu_mul_forward",
    backward_impl="kernels.gelu_mul_backward",
))

_register_primitive(PrimitiveSpec(
    name="silu",
    kernel_type=KernelType.SILU,
    input_types={
        "x": _make_tensor_type(["B", "T", "D"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "T", "D"]),
    },
    save=["x"],
    forward_impl="kernels.silu_forward",
    backward_impl="kernels.silu_backward",
))

_register_primitive(PrimitiveSpec(
    name="relu",
    kernel_type=KernelType.RELU,
    input_types={
        "x": _make_tensor_type(["B", "T", "D"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "T", "D"]),
    },
    save=["x"],
    forward_impl="kernels.relu_forward",
    backward_impl="kernels.relu_backward",
))

_register_primitive(PrimitiveSpec(
    name="relu2",
    kernel_type=KernelType.RELU2,
    input_types={
        "x": _make_tensor_type(["B", "T", "D"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "T", "D"]),
    },
    save=["x"],
    forward_impl="kernels.relu2_forward",
    backward_impl="kernels.relu2_backward",
))

_register_primitive(PrimitiveSpec(
    name="gelu",
    kernel_type=KernelType.GELU,
    input_types={
        "x": _make_tensor_type(["B", "T", "D"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "T", "D"]),
    },
    default_attrs={
        "approximate": True,
    },
    save=["x"],
    forward_impl="kernels.gelu_forward",
    backward_impl="kernels.gelu_backward",
))

_register_primitive(PrimitiveSpec(
    name="softmax",
    kernel_type=KernelType.SOFTMAX,
    input_types={
        "x": _make_tensor_type(["B", "T", "D"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "T", "D"]),
    },
    default_attrs={
        "dim": -1,
    },
    save=["out"],  # Softmax backward uses output, not input
    forward_impl="kernels.softmax_forward",
    backward_impl="kernels.softmax_backward",
))

# -----------------------------------------------------------------------------
# Attention Primitives
# -----------------------------------------------------------------------------

_register_primitive(PrimitiveSpec(
    name="flash_attention",
    kernel_type=KernelType.FLASH_ATTENTION,
    input_types={
        "q": _make_tensor_type(["B", "Hq", "T", "D"]),
        "k": _make_tensor_type(["B", "Hkv", "T", "D"]),
        "v": _make_tensor_type(["B", "Hkv", "T", "D"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "Hq", "T", "D"]),
        "lse": _make_tensor_type(["B", "Hq", "T"]),
    },
    default_attrs={
        "causal": True,
        "softmax_scale": None,
        "window_size": None,
    },
    save=["q", "k", "v", "out", "lse"],
    forward_impl="kernels.flash_attention_forward",
    backward_impl="kernels.flash_attention_backward",
))

_register_primitive(PrimitiveSpec(
    name="rope",
    kernel_type=KernelType.ROPE,
    input_types={
        "q": _make_tensor_type(["B", "H", "T", "D"]),
        "k": _make_tensor_type(["B", "Hkv", "T", "D"]),
        "freqs": _make_tensor_type(["T", "D_half", "2"]),
    },
    output_types={
        "q_rot": _make_tensor_type(["B", "H", "T", "D"]),
        "k_rot": _make_tensor_type(["B", "Hkv", "T", "D"]),
    },
    default_attrs={
        "interleaved": False,
    },
    save=[],  # RoPE doesn't need saved activations (freqs are constants)
    forward_impl="kernels.rope_forward",
    backward_impl="kernels.rope_backward",
))

_register_primitive(PrimitiveSpec(
    name="qk_norm",
    kernel_type=KernelType.QK_NORM,
    input_types={
        "q": _make_tensor_type(["B", "H", "T", "D"]),
        "k": _make_tensor_type(["B", "Hkv", "T", "D"]),
        "q_weight": _make_tensor_type(["D"]),
        "k_weight": _make_tensor_type(["D"]),
    },
    output_types={
        "q_norm": _make_tensor_type(["B", "H", "T", "D"]),
        "k_norm": _make_tensor_type(["B", "Hkv", "T", "D"]),
        "q_rstd": _make_tensor_type(["B", "H", "T"]),
        "k_rstd": _make_tensor_type(["B", "Hkv", "T"]),
    },
    default_attrs={
        "eps": 1e-6,
    },
    save=["q", "k", "q_rstd", "k_rstd"],
    forward_impl="kernels.qk_norm_forward",
    backward_impl="kernels.qk_norm_backward",
))

# -----------------------------------------------------------------------------
# Tensor Manipulation Primitives
# -----------------------------------------------------------------------------

_register_primitive(PrimitiveSpec(
    name="split",
    kernel_type=KernelType.SPLIT,
    input_types={
        "x": _make_tensor_type(["B", "T", "N"]),
    },
    output_types={},  # Dynamic based on sizes
    default_attrs={
        "sizes": [],
        "dim": -1,
    },
    save=[],
    forward_impl="pointer_arithmetic",
    backward_impl="concat",
))

_register_primitive(PrimitiveSpec(
    name="concat",
    kernel_type=KernelType.CONCAT,
    input_types={},  # Dynamic based on inputs
    output_types={
        "out": _make_tensor_type(["B", "T", "N"]),
    },
    default_attrs={
        "dim": -1,
    },
    save=[],
    forward_impl="kernels.concat_forward",
    backward_impl="split",
))

_register_primitive(PrimitiveSpec(
    name="view",
    kernel_type=KernelType.VIEW,
    input_types={
        "x": _make_tensor_type(["B", "T", "C"]),
    },
    output_types={
        "out": _make_tensor_type(["shape"]),  # Dynamic shape
    },
    default_attrs={
        "shape": [],
    },
    save=[],
    forward_impl="metadata_only",
    backward_impl="metadata_only",
))

_register_primitive(PrimitiveSpec(
    name="transpose",
    kernel_type=KernelType.TRANSPOSE,
    input_types={
        "x": _make_tensor_type(["D0", "D1"]),
    },
    output_types={
        "out": _make_tensor_type(["D1", "D0"]),
    },
    default_attrs={
        "dim0": 0,
        "dim1": 1,
    },
    save=[],
    forward_impl="kernels.transpose",
    backward_impl="kernels.transpose",
))

_register_primitive(PrimitiveSpec(
    name="permute",
    kernel_type=KernelType.PERMUTE,
    input_types={
        "x": _make_tensor_type(["D0", "D1", "D2"]),
    },
    output_types={
        "out": _make_tensor_type(["D_perm"]),  # Dynamic based on dims
    },
    default_attrs={
        "dims": [],
    },
    save=[],
    forward_impl="kernels.permute",
))

_register_primitive(PrimitiveSpec(
    name="contiguous",
    kernel_type=KernelType.CONTIGUOUS,
    input_types={
        "x": _make_tensor_type(["B", "T", "C"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "T", "C"]),
    },
    save=[],
    forward_impl="kernels.copy",  # Only if not contiguous
))

_register_primitive(PrimitiveSpec(
    name="copy",
    kernel_type=KernelType.COPY,
    input_types={
        "x": _make_tensor_type(["B", "T", "C"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "T", "C"]),
    },
    save=[],
    forward_impl="cudaMemcpy",
))

# -----------------------------------------------------------------------------
# Elementwise Primitives
# -----------------------------------------------------------------------------

_register_primitive(PrimitiveSpec(
    name="add",
    kernel_type=KernelType.ADD,
    input_types={
        "a": _make_tensor_type(["B", "T", "C"]),
        "b": _make_tensor_type(["B", "T", "C"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "T", "C"]),
    },
    save=[],
    forward_impl="kernels.add_forward",
    backward_impl="kernels.add_backward",
))

_register_primitive(PrimitiveSpec(
    name="mul",
    kernel_type=KernelType.MUL,
    input_types={
        "a": _make_tensor_type(["B", "T", "C"]),
        "b": _make_tensor_type(["B", "T", "C"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "T", "C"]),
    },
    save=["a", "b"],
    forward_impl="kernels.mul_forward",
    backward_impl="kernels.mul_backward",
))

_register_primitive(PrimitiveSpec(
    name="scale",
    kernel_type=KernelType.SCALE,
    input_types={
        "x": _make_tensor_type(["B", "T", "C"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "T", "C"]),
    },
    default_attrs={
        "factor": 1.0,
    },
    save=[],
    forward_impl="kernels.scale",
    backward_impl="kernels.scale",
))

_register_primitive(PrimitiveSpec(
    name="add3",
    kernel_type=KernelType.ADD3,
    input_types={
        "a": _make_tensor_type(["B", "T", "C"]),
        "b": _make_tensor_type(["B", "T", "C"]),
        "c": _make_tensor_type(["B", "T", "C"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "T", "C"]),
    },
    save=[],
    forward_impl="kernels.add3_forward",
))

# -----------------------------------------------------------------------------
# Reduction Primitives
# -----------------------------------------------------------------------------

_register_primitive(PrimitiveSpec(
    name="reduce_sum",
    kernel_type=KernelType.REDUCE_SUM,
    input_types={
        "x": _make_tensor_type(["B", "T", "C"]),
    },
    output_types={
        "out": _make_tensor_type(["reduced"]),
    },
    default_attrs={
        "dims": [],
        "keepdim": False,
    },
    save=[],
    forward_impl="kernels.reduce_sum",
    backward_impl="kernels.broadcast",
))

_register_primitive(PrimitiveSpec(
    name="reduce_mean",
    kernel_type=KernelType.REDUCE_MEAN,
    input_types={
        "x": _make_tensor_type(["B", "T", "C"]),
    },
    output_types={
        "out": _make_tensor_type(["reduced"]),
    },
    default_attrs={
        "dims": [],
        "keepdim": False,
    },
    save=[],
    forward_impl="kernels.reduce_mean",
))

_register_primitive(PrimitiveSpec(
    name="reduce_max",
    kernel_type=KernelType.REDUCE_MAX,
    input_types={
        "x": _make_tensor_type(["B", "T", "C"]),
    },
    output_types={
        "values": _make_tensor_type(["reduced"]),
        "indices": _make_tensor_type(["reduced"], Dtype.INT32),
    },
    default_attrs={
        "dim": -1,
        "keepdim": False,
    },
    save=["indices"],
    forward_impl="kernels.reduce_max",
))

# -----------------------------------------------------------------------------
# Embedding Primitives
# -----------------------------------------------------------------------------

_register_primitive(PrimitiveSpec(
    name="embedding",
    kernel_type=KernelType.EMBEDDING,
    input_types={
        "indices": _make_tensor_type(["B", "T"], Dtype.INT32),
        "weight": _make_tensor_type(["V", "D"]),
    },
    output_types={
        "out": _make_tensor_type(["B", "T", "D"]),
    },
    save=["indices"],
    forward_impl="kernels.embedding_forward",
    backward_impl="kernels.embedding_backward",
))

# -----------------------------------------------------------------------------
# MoE Primitives
# -----------------------------------------------------------------------------

_register_primitive(PrimitiveSpec(
    name="moe_router",
    kernel_type=KernelType.MOE_ROUTER,
    input_types={
        "x": _make_tensor_type(["BT", "C"]),
        "gate": _make_tensor_type(["E", "C"]),
    },
    output_types={
        "weights": _make_tensor_type(["BT", "top_k"]),
        "indices": _make_tensor_type(["BT", "top_k"], Dtype.INT32),
        "aux_loss": _make_tensor_type(["1"]),
    },
    default_attrs={
        "top_k": 2,
        "normalize": True,
        "aux_loss_coef": 0.01,
        "use_sigmoid": False,
    },
    save=["x", "indices"],
    forward_impl="kernels.moe_router_forward",
    backward_impl="kernels.moe_router_backward",
))

_register_primitive(PrimitiveSpec(
    name="moe_permute",
    kernel_type=KernelType.MOE_PERMUTE,
    input_types={
        "x": _make_tensor_type(["BT", "C"]),
        "indices": _make_tensor_type(["BT", "top_k"], Dtype.INT32),
        "expert_offsets": _make_tensor_type(["E_plus_1"], Dtype.INT32),
    },
    output_types={
        "permuted": _make_tensor_type(["total_tokens", "C"]),
        "scatter_indices": _make_tensor_type(["total_tokens"], Dtype.INT32),
    },
    save=[],
    forward_impl="kernels.moe_permute_tokens",
))

_register_primitive(PrimitiveSpec(
    name="moe_unpermute",
    kernel_type=KernelType.MOE_UNPERMUTE,
    input_types={
        "expert_outputs": _make_tensor_type(["total_tokens", "C"]),
        "weights": _make_tensor_type(["BT", "top_k"]),
        "scatter_indices": _make_tensor_type(["total_tokens"], Dtype.INT32),
    },
    output_types={
        "combined": _make_tensor_type(["BT", "C"]),
    },
    save=[],
    forward_impl="kernels.moe_unpermute_and_combine",
))

# -----------------------------------------------------------------------------
# Mamba/SSM Primitives
# -----------------------------------------------------------------------------

_register_primitive(PrimitiveSpec(
    name="mamba_conv1d",
    kernel_type=KernelType.MAMBA_CONV1D,
    input_types={
        "x": _make_tensor_type(["B", "D", "T"]),
        "weight": _make_tensor_type(["D", "1", "K"]),
        "bias": _make_tensor_type(["D"]),
    },
    output_types={
        "y": _make_tensor_type(["B", "D", "T"]),
        "conv_state_out": _make_tensor_type(["B", "D", "K_minus_1"]),
    },
    default_attrs={
        "kernel_size": 4,
        "activation": "silu",
    },
    save=["x"],
    forward_impl="kernels.mamba_causal_conv1d_forward",
    backward_impl="kernels.mamba_causal_conv1d_backward",
))

_register_primitive(PrimitiveSpec(
    name="mamba_selective_scan",
    kernel_type=KernelType.MAMBA_SELECTIVE_SCAN,
    input_types={
        "u": _make_tensor_type(["B", "D", "T"]),
        "delta": _make_tensor_type(["B", "D", "T"]),
        "A": _make_tensor_type(["D", "N"]),
        "B": _make_tensor_type(["B", "G", "N", "T"]),
        "C": _make_tensor_type(["B", "G", "N", "T"]),
        "D": _make_tensor_type(["D"]),
        "dt_bias": _make_tensor_type(["D"]),
    },
    output_types={
        "y": _make_tensor_type(["B", "D", "T"]),
        "ssm_state_out": _make_tensor_type(["B", "D", "N"]),
    },
    default_attrs={
        "chunk_size": 256,
    },
    save=["u", "delta", "A", "B", "C", "y"],
    forward_impl="kernels.mamba_selective_scan_forward",
    backward_impl="kernels.mamba_selective_scan_backward",
))

# -----------------------------------------------------------------------------
# Utility Primitives
# -----------------------------------------------------------------------------

_register_primitive(PrimitiveSpec(
    name="zeros",
    kernel_type=KernelType.ZEROS,
    input_types={},
    output_types={
        "out": _make_tensor_type(["shape"]),
    },
    default_attrs={
        "shape": [],
        "dtype": Dtype.BF16,
    },
    save=[],
    forward_impl="cudaMemset",
))

_register_primitive(PrimitiveSpec(
    name="ones",
    kernel_type=KernelType.ONES,
    input_types={},
    output_types={
        "out": _make_tensor_type(["shape"]),
    },
    default_attrs={
        "shape": [],
        "dtype": Dtype.BF16,
    },
    save=[],
))

_register_primitive(PrimitiveSpec(
    name="fill",
    kernel_type=KernelType.FILL,
    input_types={},
    output_types={
        "out": _make_tensor_type(["shape"]),
    },
    default_attrs={
        "shape": [],
        "value": 0.0,
        "dtype": Dtype.BF16,
    },
    save=[],
))


def get_primitive(name: str) -> Optional[PrimitiveSpec]:
    """Get primitive specification by name."""
    return PRIMITIVES.get(name)


def list_primitives() -> List[str]:
    """List all available primitive names."""
    return list(PRIMITIVES.keys())
