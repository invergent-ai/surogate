"""Transformer Blocks for Python DSL"""

from .common import Activation
from .dense import DenseTransformerBlock
from .qwen3 import Qwen3Block
from .qwen3_moe import Qwen3MoEBlock
from .llama import LlamaBlock
from .nemotron_h import (
    NemotronHMamba2Block,
    NemotronHAttentionBlock,
    NemotronHMLPBlock,
    NemotronHMoEBlock,
)

__all__ = [
    "Activation",
    "DenseTransformerBlock",
    "Qwen3Block",
    "Qwen3MoEBlock",
    "LlamaBlock",
    # NemotronH hybrid blocks
    "NemotronHMamba2Block",
    "NemotronHAttentionBlock",
    "NemotronHMLPBlock",
    "NemotronHMoEBlock",
]
