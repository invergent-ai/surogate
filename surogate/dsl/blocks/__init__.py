"""Transformer Blocks for Python DSL"""

from .deepseek_v4 import (
    DeepseekV4CsaHashBlock,
    DeepseekV4CsaMoEBlock,
    DeepseekV4HcaHashBlock,
    DeepseekV4HcaMoEBlock,
    DeepseekV4SlidingHashBlock,
    DeepseekV4SlidingMoEBlock,
)
from .gemma4 import (
    Gemma4FullBlock,
    Gemma4FullMoEBlock,
    Gemma4SharedKVBlock,
    Gemma4SlidingBlock,
    Gemma4SlidingMoEBlock,
)
from .glm5_next import (
    Glm5NextKdaDenseBlock,
    Glm5NextKdaMoEBlock,
    Glm5NextMlaDenseBlock,
    Glm5NextMlaMoEBlock,
)
from .gpt_oss import GptOssBlock
from .laguna import LagunaDenseBlock, LagunaSparseBlock
from .llama import LlamaBlock
from .lfm2 import Lfm2AttentionBlock, Lfm2ConvBlock
from .lfm2_moe import Lfm2MoeAttentionBlock, Lfm2MoeConvBlock
from .nemotron_h import (
    NemotronHAttentionBlock,
    NemotronHMamba2Block,
    NemotronHMLPBlock,
    NemotronHMoEBlock,
)
from .qwen3 import Qwen3Block
from .qwen3_5 import Qwen3_5AttentionBlock, Qwen3_5LinearBlock
from .qwen3_5_moe import Qwen3_5MoEAttentionBlock, Qwen3_5MoELinearBlock
from .qwen3_moe import Qwen3MoEBlock
from .qwen4_exp import Qwen4ExpAttentionBlock, Qwen4ExpLinearBlock
from .qwen3_vl import Qwen3VLBlock

__all__ = [
    "Qwen3Block",
    "Qwen3_5AttentionBlock",
    "Qwen3_5LinearBlock",
    "Qwen3_5MoEAttentionBlock",
    "Qwen3_5MoELinearBlock",
    "Qwen4ExpAttentionBlock",
    "Qwen4ExpLinearBlock",
    "Qwen3VLBlock",
    "Qwen3MoEBlock",
    "GptOssBlock",
    "LagunaDenseBlock",
    "LagunaSparseBlock",
    "LlamaBlock",
    "Lfm2AttentionBlock",
    "Lfm2ConvBlock",
    "Lfm2MoeAttentionBlock",
    "Lfm2MoeConvBlock",
    # GLM-5.3-Flash hybrid blocks (KDA / NoPE-MLA x dense / MoE)
    "Glm5NextKdaDenseBlock",
    "Glm5NextKdaMoEBlock",
    "Glm5NextMlaDenseBlock",
    "Glm5NextMlaMoEBlock",
    # DeepSeek-V4 hybrid blocks (attention schedule x MoE schedule)
    "DeepseekV4SlidingMoEBlock",
    "DeepseekV4SlidingHashBlock",
    "DeepseekV4CsaMoEBlock",
    "DeepseekV4CsaHashBlock",
    "DeepseekV4HcaMoEBlock",
    "DeepseekV4HcaHashBlock",
    # Gemma4 hybrid blocks
    "Gemma4SlidingBlock",
    "Gemma4FullBlock",
    "Gemma4SlidingMoEBlock",
    "Gemma4FullMoEBlock",
    # NemotronH hybrid blocks
    "NemotronHMamba2Block",
    "NemotronHAttentionBlock",
    "NemotronHMLPBlock",
    "NemotronHMoEBlock",
]
