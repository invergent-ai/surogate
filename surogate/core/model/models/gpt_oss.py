from surogate.core.config.enums import ChatTemplateType
from surogate.core.model.registry import register_model, ModelTemplate

register_model(
    ModelTemplate(
        model_type='GptOssForCausalLM',
        chat_templates=[ChatTemplateType.gpt_oss]))