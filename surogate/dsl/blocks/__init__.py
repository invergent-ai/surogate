"""Transformer Blocks for Python DSL"""

from .common import Activation
from .qwen3 import Qwen3Block
from .qwen3_5 import Qwen3_5AttentionBlock, Qwen3_5LinearBlock
from .qwen3_5_moe import Qwen3_5MoEAttentionBlock, Qwen3_5MoELinearBlock
from .qwen3_vl import Qwen3VLBlock
from .qwen3_moe import Qwen3MoEBlock
from .gpt_oss import GptOssBlock
from .llama import LlamaBlock
from .nemotron_h import (
    NemotronHMamba2Block,
    NemotronHAttentionBlock,
    NemotronHMLPBlock,
    NemotronHMoEBlock,
)

__all__ = [
    "Activation",
    "Qwen3Block",
    "Qwen3_5AttentionBlock",
    "Qwen3_5LinearBlock",
    "Qwen3_5MoEAttentionBlock",
    "Qwen3_5MoELinearBlock",
    "Qwen3VLBlock",
    "Qwen3MoEBlock",
    "GptOssBlock",
    "LlamaBlock",
    # NemotronH hybrid blocks
    "NemotronHMamba2Block",
    "NemotronHAttentionBlock",
    "NemotronHMLPBlock",
    "NemotronHMoEBlock",
]
