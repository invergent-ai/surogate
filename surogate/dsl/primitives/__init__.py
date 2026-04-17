"""Primitive Operations for Python DSL"""

from .activations import relu2, sigmoid, silu, silu_mul, swiglu
from .attention import flash_attention, mrope, qkv_qk_norm, qkv_qk_norm_rope, rope
from .common import TransposeMode
from .elementwise import add, bias_add, deepstack_inject, mask_scatter, mul, scale
from .embedding import embedding
from .ep import (
    ep_combine,
    ep_dispatch,
)
from .gated_delta_rule import (
    chunk_gated_delta_rule,
)
from .initialization import fill_normal, ones, zeros
from .losses import fused_lm_head_loss
from .mamba import (
    mamba_combine_scan,
    mamba_conv1d,
    mamba_gated_rmsnorm,
    mamba_split_conv_out,
    mamba_split_proj,
    mamba_ssm_scan,
)
from .matmul import batched_matmul, matmul
from .moe import (
    moe_grouped_gemm_down,
    moe_grouped_gemm_gate_up,
    moe_permute,
    moe_sigmoid,
    moe_softmax,
    moe_topk,
    moe_unpermute,
)
from .normalization import fused_residual_rmsnorm, rmsnorm
from .qwen3_5 import (
    qwen3_5_decay,
)
from .tensor_ops import concat, repeat_interleave_heads, split, transpose, view

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
    "sigmoid",
    "relu2",
    "silu_mul",
    # Attention
    "flash_attention",
    "rope",
    "mrope",
    "qkv_qk_norm",
    "qkv_qk_norm_rope",
    # Embedding
    "embedding",
    # Tensor ops
    "view",
    "transpose",
    "concat",
    "split",
    "repeat_interleave_heads",
    # Elementwise
    "add",
    "mul",
    "scale",
    "bias_add",
    "mask_scatter",
    "deepstack_inject",
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
    # Qwen3.5 gated delta rule
    "chunk_gated_delta_rule",
    "qwen3_5_decay",
    # Expert Parallelism
    "ep_dispatch",
    "ep_combine",
]
