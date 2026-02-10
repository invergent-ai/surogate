from surogate.core.config.enums import LLMModelType, ChatTemplateType
from surogate.core.model.architecture import register_model_architecture, LLMComponents, LLMArchitecture, \
    MLLMComponents, MLLMArchitecture, ModelArchitecture
from surogate.core.model.registry import register_model, ModelTemplate
from surogate.core.model.loader import get_model_tokenizer_with_flash_attn

register_model_architecture(
    LLMComponents(
        LLMArchitecture.qwen,
        module_list='transformer.h',
        mlp='transformer.h.{}.mlp',
        down_proj='transformer.h.{}.mlp.c_proj',
        attention='transformer.h.{}.attn',
        o_proj='transformer.h.{}.attn.c_proj',
        qkv_proj='transformer.h.{}.attn.c_attn',
        embedding='transformer.wte',
        lm_head='lm_head'))

register_model_architecture(
    MLLMComponents(
        MLLMArchitecture.qwen2_vl,
        language_model=['model.language_model', 'lm_head'],
        aligner='model.visual.merger',
        vision_tower='model.visual'))

register_model_architecture(
    MLLMComponents(
        MLLMArchitecture.qwen3_vl,
        language_model=['model.language_model', 'lm_head'],
        aligner=['model.visual.merger', 'model.visual.deepstack_merger_list'],
        vision_tower='model.visual'))

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
        LLMModelType.qwen2_5,
        ChatTemplateType.qwen2_5,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen2ForCausalLM'],
        model_arch=ModelArchitecture.llama))


