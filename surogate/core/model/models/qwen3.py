from surogate.core.config.enums import ChatTemplateType
from surogate.core.model.registry import register_model, ModelTemplate

"""
Instruct models:
- Qwen/Qwen3-0.6B
- Qwen/Qwen3-1.7B
- Qwen/Qwen3-4B
- Qwen/Qwen3-8B
- Qwen/Qwen3-14B
- Qwen/Qwen3-32B

Base models:
- Qwen/Qwen3-0.6B-Base
- Qwen/Qwen3-1.7B-Base
- Qwen/Qwen3-4B-Base
- Qwen/Qwen3-8B-Base
- Qwen/Qwen3-14B-Base
"""
register_model(
    ModelTemplate(
        model_type='Qwen3ForCausalLM',
        chat_templates=[ChatTemplateType.qwen3]))


"""
- Qwen/Qwen3-30B-A3B-Base
- Qwen/Qwen3-30B-A3B
- Qwen/Qwen3-235B-A22B
"""
register_model(
    ModelTemplate(
        model_type='Qwen3MoeForCausalLM',
        chat_templates=[ChatTemplateType.qwen3]))
