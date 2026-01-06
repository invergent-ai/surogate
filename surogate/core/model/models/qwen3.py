from surogate.core.config.enums import LLMModelType, ChatTemplateType
from surogate.core.model.architecture import ModelArchitecture
from surogate.core.model.registry import register_model, ModelTemplate
from surogate.core.model.loader import get_model_tokenizer_with_flash_attn

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
        LLMModelType.qwen3,
        ChatTemplateType.qwen3,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen3ForCausalLM'],
        attention_cls='Qwen3Attention',
        model_arch=ModelArchitecture.llama))


"""
- Qwen/Qwen3-30B-A3B-Base
- Qwen/Qwen3-30B-A3B
- Qwen/Qwen3-235B-A22B
"""
register_model(
    ModelTemplate(
        LLMModelType.qwen3_moe,
        ChatTemplateType.qwen3,
        get_model_tokenizer_with_flash_attn,
        attention_cls='Qwen3MoeAttention',
        architectures=['Qwen3MoeForCausalLM']))

"""
- Qwen/Qwen3-4B-Thinking-2507
"""
register_model(
    ModelTemplate(
        LLMModelType.qwen3_thinking,
        ChatTemplateType.qwen3_thinking,
        get_model_tokenizer_with_flash_attn,
        attention_cls='Qwen3Attention',
        architectures=['Qwen3ForCausalLM']))

"""
- Qwen/Qwen3-30B-A3B-Instruct-2507
- Qwen/Qwen3-235B-A22B-Instruct-2507
"""
register_model(
    ModelTemplate(
        LLMModelType.qwen3_nothinking,
        ChatTemplateType.qwen3_nothinking,
        get_model_tokenizer_with_flash_attn,
        attention_cls='Qwen3MoeAttention',
        architectures=['Qwen3MoeForCausalLM', 'Qwen3ForCausalLM']))