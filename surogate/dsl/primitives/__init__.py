"""Primitive Operations for Python DSL"""

from .common import TransposeMode
from .matmul import matmul, batched_matmul
from .normalization import rmsnorm, fused_residual_rmsnorm
from .activations import swiglu, silu, relu2, silu_mul
from .attention import flash_attention, rope, qkv_qk_norm_rope
from .embedding import embedding
from .tensor_ops import view, transpose, concat, split
from .elementwise import add, mul, scale, bias_add
from .initialization import zeros, ones, fill_normal
from .losses import fused_lm_head_loss
from .moe import (
    moe_softmax,
    moe_sigmoid,
    moe_topk,
    moe_permute,
    moe_unpermute,
    moe_grouped_gemm_gate_up,
    moe_grouped_gemm_down,
)
from .mamba import (
    mamba_conv1d,
    mamba_ssm_scan,
    mamba_gated_rmsnorm,
    mamba_split_proj,
    mamba_split_conv_out,
    mamba_combine_scan,
)

__all__ = [
    # Common
    "TransposeMode",
    # Matrix ops
    "matmul",
    "batched_matmul",
    # Normalization
    "rmsnorm",
    "fused_residual_rmsnorm",
    # Activations
    "swiglu",
    "silu",
    "relu2",
    "silu_mul",
    # Attention
    "flash_attention",
    "rope",
    "qkv_qk_norm_rope",
    # Embedding
    "embedding",
    # Tensor ops
    "view",
    "transpose",
    "concat",
    "split",
    # Elementwise
    "add",
    "mul",
    "scale",
    "bias_add",
    # Initialization
    "zeros",
    "ones",
    "fill_normal",
    # Losses
    "fused_lm_head_loss",
    # MoE
    "moe_softmax",
    "moe_sigmoid",
    "moe_topk",
    "moe_permute",
    "moe_unpermute",
    "moe_grouped_gemm_gate_up",
    "moe_grouped_gemm_down",
    # Mamba2 / SSM
    "mamba_conv1d",
    "mamba_ssm_scan",
    "mamba_gated_rmsnorm",
    "mamba_split_proj",
    "mamba_split_conv_out",
    "mamba_combine_scan",
]
