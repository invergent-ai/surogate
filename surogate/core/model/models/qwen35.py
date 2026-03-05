from surogate.core.config.enums import ChatTemplateType
from surogate.core.model.registry import register_model, ModelTemplate

"""
- Qwen/Qwen3.5-0.8B
"""
register_model(
    ModelTemplate(
        model_type='Qwen3_5ForConditionalGeneration',
        chat_templates=[ChatTemplateType.qwen3_5],
        is_multimodal=True,
        tags=['vision', 'video']))