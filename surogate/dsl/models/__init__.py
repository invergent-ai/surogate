"""Model Definitions for Python DSL"""

from .gemma4 import Gemma4CausalModel, Gemma4ConditionalModel
from .gpt_oss import GptOssModel
from .llama import LlamaModel
from .lfm2 import Lfm2Model
from .nemotron_h import NemotronHModel, from_hf_config, parse_hybrid_pattern, to_standard_hybrid_pattern
from .qwen3 import Qwen3Model
from .qwen3_5 import Qwen3_5CausalModel, Qwen3_5ConditionalModel
from .qwen3_5_moe import Qwen3_5MoECausalModel, Qwen3_5MoEConditionalModel
from .qwen3_moe import Qwen3MoEModel
from .qwen3_vl import Qwen3VLModel

__all__ = [
    "Qwen3Model",
    "Qwen3_5CausalModel",
    "Qwen3_5ConditionalModel",
    "Qwen3_5MoECausalModel",
    "Qwen3_5MoEConditionalModel",
    "Qwen3VLModel",
    "Qwen3MoEModel",
    "GptOssModel",
    "LlamaModel",
    "Lfm2Model",
    "Gemma4CausalModel",
    "Gemma4ConditionalModel",
    "NemotronHModel",
    "parse_hybrid_pattern",
    "to_standard_hybrid_pattern",
    "from_hf_config",
]
