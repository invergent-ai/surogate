"""Runtime ``nn.Module`` subclasses used to build DSL blocks and models.

Every class here subclasses ``surogate.dsl.nn.Module`` and implements
``_trace`` to emit graph ops. Class-level ``_hf_mapping_defaults_``
dicts describe HF weight-path discovery and are read by
``surogate.dsl.hf``.
"""

from .attention import (
    Gemma4Attention,
    Gemma4SharedKVAttention,
    GenericGQAttention,
    GptOssAttention,
    GQAAttention,
    NemotronAttention,
    Qwen3_5Attention,
    Qwen3Attention,
    Qwen3VLAttention,
    _resolve_rotary_dim,
)
from .embedding import Embedding, LMHead, ScaledEmbedding
from .gated_delta_rule import ChunkGatedDeltaRule, GatedDeltaNetMixer
from .linear import Linear
from .mamba import Mamba2Mixer
from .mlp import GatedMLP, GenericMLP, SimpleMLP, SwiGLUMLP
from .moe import (
    Gemma4MoEExperts,
    GptOssMoEExperts,
    MoEExpertsGated,
    MoESharedExpert,
    NemotronMoEExperts,
    NemotronSharedExpert,
)
from .rmsnorm import FusedResidualRMSNorm, RMSNorm, RMSNormPlus1
from .short_conv import Lfm2ShortConv

__all__ = [
    # Norms
    "RMSNorm",
    "RMSNormPlus1",
    "FusedResidualRMSNorm",
    # Linear / embedding
    "Linear",
    "Embedding",
    "ScaledEmbedding",
    "LMHead",
    # MLP
    "GenericMLP",
    "SwiGLUMLP",
    "GatedMLP",
    "SimpleMLP",
    "Lfm2ShortConv",
    # Attention
    "GenericGQAttention",
    "GQAAttention",
    "Qwen3Attention",
    "Qwen3VLAttention",
    "Qwen3_5Attention",
    "GptOssAttention",
    "Gemma4Attention",
    "Gemma4SharedKVAttention",
    "NemotronAttention",
    # SSM / linear attention
    "Mamba2Mixer",
    "GatedDeltaNetMixer",
    "ChunkGatedDeltaRule",
    # MoE
    "MoEExpertsGated",
    "MoESharedExpert",
    "GptOssMoEExperts",
    "Gemma4MoEExperts",
    "NemotronMoEExperts",
    "NemotronSharedExpert",
]
