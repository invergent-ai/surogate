from surogate.core.config.enums import ChatTemplateType
from surogate.core.model.registry import register_model, ModelTemplate

register_model(
    ModelTemplate(
        model_type='NemotronHForCausalLM',
        chat_templates=[ChatTemplateType.nemotron_nano]))