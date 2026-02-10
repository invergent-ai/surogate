from surogate.core.config.enums import LLMModelType, ChatTemplateType
from surogate.core.model.architecture import ModelArchitecture
from surogate.core.model.registry import register_model, ModelTemplate
from surogate.core.model.loader import get_model_tokenizer_with_flash_attn

register_model(
    ModelTemplate(
        LLMModelType.nemotron_nano,
        ChatTemplateType.nemotron_nano,
        get_model_tokenizer_with_flash_attn,
        architectures=['NemotronHForCausalLM'],
        model_arch=ModelArchitecture.nemotron_nano))