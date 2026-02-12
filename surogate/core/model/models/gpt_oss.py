from surogate.core.config.enums import LLMModelType, ChatTemplateType
from surogate.core.model.architecture import ModelArchitecture
from surogate.core.model.registry import register_model, ModelTemplate
from surogate.core.model.loader import get_model_tokenizer_with_flash_attn

register_model(
    ModelTemplate(
        LLMModelType.gpt_oss,
        ChatTemplateType.gpt_oss,
        get_model_tokenizer_with_flash_attn,
        architectures=['GptOssForCausalLM'],
        model_arch=ModelArchitecture.gpt_oss))