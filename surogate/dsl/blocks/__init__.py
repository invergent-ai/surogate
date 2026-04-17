"""Transformer Blocks for Python DSL"""

from .gemma4 import (
    Gemma4FullBlock,
    Gemma4FullMoEBlock,
    Gemma4SharedKVBlock,
    Gemma4SlidingBlock,
    Gemma4SlidingMoEBlock,
)
from .gpt_oss import GptOssBlock
from .llama import LlamaBlock
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
from .qwen3_vl import Qwen3VLBlock

__all__ = [
    "Qwen3Block",
    "Qwen3_5AttentionBlock",
    "Qwen3_5LinearBlock",
    "Qwen3_5MoEAttentionBlock",
    "Qwen3_5MoELinearBlock",
    "Qwen3VLBlock",
    "Qwen3MoEBlock",
    "GptOssBlock",
    "LlamaBlock",
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
