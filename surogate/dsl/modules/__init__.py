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
    LagunaAttention,
    NemotronAttention,
    Qwen3_5Attention,
    Qwen3Attention,
    Qwen3VLAttention,
    _resolve_rotary_dim,
)
from .deepseek_v4 import DeepseekV4Attention, DeepseekV4MoEExperts
from .embedding import Embedding, LMHead, ScaledEmbedding
from .gated_delta_rule import ChunkGatedDeltaRule, GatedDeltaNetMixer
from .glm5_next import (
    Glm5NextHyperConnection,
    Glm5NextHyperConnectionCombine,
    Glm5NextHyperHead,
    Glm5NextKimiDeltaMixer,
    Glm5NextLatentAttention,
)
from .hyper_connection import HyperConnection, HyperConnectionCombine, StreamBroadcast
from .linear import Linear
from .mamba import Mamba2Mixer
from .mlp import GatedMLP, GenericMLP, SimpleMLP, SwiGLUMLP
from .moe import (
    Gemma4MoEExperts,
    GptOssMoEExperts,
    LagunaMoEExperts,
    MoEExpertsGated,
    MoESharedExpert,
    NemotronMoEExperts,
    NemotronSharedExpert,
)
from .rmsnorm import FusedResidualRMSNorm, RMSNorm, RMSNormPlus1
from .vision import VisionTower
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
    # Vision
    "VisionTower",
    # Attention
    "GenericGQAttention",
    "GQAAttention",
    "LagunaAttention",
    "Qwen3Attention",
    "Qwen3VLAttention",
    "Qwen3_5Attention",
    "GptOssAttention",
    "Gemma4Attention",
    "Gemma4SharedKVAttention",
    "NemotronAttention",
    "DeepseekV4Attention",
    # SSM / linear attention
    "Mamba2Mixer",
    "GatedDeltaNetMixer",
    "ChunkGatedDeltaRule",
    # Hyper-connections (qwen4_exp residual streams)
    "HyperConnection",
    # Manifold-constrained hyper-connections + KDA / NoPE-MLA (glm5_next)
    "Glm5NextHyperConnection",
    "Glm5NextHyperConnectionCombine",
    "Glm5NextHyperHead",
    "Glm5NextKimiDeltaMixer",
    "Glm5NextLatentAttention",
    "HyperConnectionCombine",
    "StreamBroadcast",
    # MoE
    "LagunaMoEExperts",
    "MoEExpertsGated",
    "MoESharedExpert",
    "GptOssMoEExperts",
    "Gemma4MoEExperts",
    "NemotronMoEExperts",
    "NemotronSharedExpert",
    "DeepseekV4MoEExperts",
]
