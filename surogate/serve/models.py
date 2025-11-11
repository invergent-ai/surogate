from swift.llm import Model, MODEL_MAPPING
from swift.llm.model import LLMModelType


def register_additional_model_types():
    MODEL_MAPPING[LLMModelType.qwen3].model_groups[0].models.append(
        Model(None, 'invergent/Qwen3-0.6B-NVFP4')
    )
    MODEL_MAPPING[LLMModelType.qwen3].model_groups[0].models.append(
        Model(None, 'invergent/Qwen3-0.6B-AWQ')
    )
    MODEL_MAPPING[LLMModelType.qwen3_moe].model_groups[0].models.append(
        Model(None, 'invergent/Qwen3-30B-A3B-AWQ')
    )
    MODEL_MAPPING[LLMModelType.gemma3_text].model_groups[0].models.append(
        Model(None, 'invergent/gemma-3-270m-it-NVFP4')
    )
