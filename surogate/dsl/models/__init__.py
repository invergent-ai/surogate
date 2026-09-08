"""Model Definitions for Python DSL"""

from .deepseek_v4 import DeepseekV4Model
from .gemma3 import Gemma3CausalModel, Gemma3TextModel
from .gemma4 import Gemma4CausalModel, Gemma4ConditionalModel, Gemma4UnifiedModel
from .glm5_next import Glm5NextConditionalModel
from .gpt_oss import GptOssModel
from .laguna import LagunaModel
from .llama import LlamaModel
from .lfm2 import Lfm2Model
from .lfm2_moe import Lfm2MoeModel
from .lfm2_vl import Lfm2VlModel
from .nemotron_h import NemotronHModel, from_hf_config, parse_hybrid_pattern, to_standard_hybrid_pattern
from .qwen3 import Qwen3Model
from .qwen3_5 import Qwen3_5CausalModel, Qwen3_5ConditionalModel
from .qwen3_5_moe import Qwen3_5MoECausalModel, Qwen3_5MoEConditionalModel
from .qwen3_moe import Qwen3MoEModel
from .qwen3_vl import Qwen3VLModel
from .qwen4_exp import Qwen4ExpCausalModel, Qwen4ExpConditionalModel
from .spark2_5 import Spark2_5Model

__all__ = [
    "Spark2_5Model",
    "Qwen3Model",
    "Qwen3_5CausalModel",
    "Qwen3_5ConditionalModel",
    "Qwen3_5MoECausalModel",
    "Qwen3_5MoEConditionalModel",
    "Qwen4ExpCausalModel",
    "Qwen4ExpConditionalModel",
    "Qwen3VLModel",
    "Qwen3MoEModel",
    "Glm5NextConditionalModel",
    "GptOssModel",
    "LagunaModel",
    "LlamaModel",
    "Lfm2Model",
    "Lfm2MoeModel",
    "Lfm2VlModel",
    "DeepseekV4Model",
    "Gemma3CausalModel",
    "Gemma3TextModel",
    "Gemma4CausalModel",
    "Gemma4ConditionalModel",
    "Gemma4UnifiedModel",
    "NemotronHModel",
    "parse_hybrid_pattern",
    "to_standard_hybrid_pattern",
    "from_hf_config",
]
