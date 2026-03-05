from surogate.core.config.enums import ChatTemplateType
from surogate.core.model.registry import register_model, ModelTemplate

"""
Instruct models:
- Qwen/Qwen2.5-0.5B-Instruct
- Qwen/Qwen2.5-1.5B-Instruct
- Qwen/Qwen2.5-3B-Instruct
- Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen2.5-14B-Instruct
- Qwen/Qwen2.5-32B-Instruct

Base models:
- Qwen/Qwen2.5-0.5B
- Qwen/Qwen2.5-1.5B
- Qwen/Qwen2.5-3B
- Qwen/Qwen2.5-7B
- Qwen/Qwen2.5-14B
- Qwen/Qwen2.5-32B
"""
register_model(
    ModelTemplate(
        model_type='Qwen2ForCausalLM',
        chat_templates=[ChatTemplateType.qwen2_5]))


