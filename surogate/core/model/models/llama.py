from surogate.core.config.enums import LLMModelType, ChatTemplateType
from surogate.core.model.architecture import register_model_architecture, LLMComponents, LLMArchitecture, \
    ModelArchitecture
from surogate.core.model.registry import register_model, ModelTemplate
from surogate.core.model.loader import get_model_tokenizer_with_flash_attn

register_model_architecture(
    LLMComponents(
        LLMArchitecture.llama,
        module_list='model.layers',
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.down_proj',
        attention='model.layers.{}.self_attn',
        o_proj='model.layers.{}.self_attn.o_proj',
        q_proj='model.layers.{}.self_attn.q_proj',
        k_proj='model.layers.{}.self_attn.k_proj',
        v_proj='model.layers.{}.self_attn.v_proj',
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ))

"""
Instruct:
- meta-llama/Meta-Llama-3-8B-Instruct

Base:
- meta-llama/Meta-Llama-3-8B
"""
register_model(
    ModelTemplate(
        LLMModelType.llama3,
        ChatTemplateType.llama3,
        get_model_tokenizer_with_flash_attn,
        attention_cls='LlamaAttention',
        architectures=['LlamaForCausalLM'],
        model_arch=ModelArchitecture.llama,
    )
)

"""
Instruct:
- meta-llama/Meta-Llama-3.1-8B-Instruct
- nvidia/Llama-3.1-Nemotron-70B-Instruct-HF

Base:
- meta-llama/Meta-Llama-3.1-8B
"""
register_model(
    ModelTemplate(
        LLMModelType.llama3_1,
        ChatTemplateType.llama3_2,
        get_model_tokenizer_with_flash_attn,
        attention_cls='LlamaAttention',
        architectures=['LlamaForCausalLM'],
        model_arch=ModelArchitecture.llama,
    )
)

"""
Instruct:
- meta-llama/Llama-3.2-1B-Instruct
- meta-llama/Llama-3.2-3B-Instruct

Base:
- meta-llama/Llama-3.2-1B
- meta-llama/Llama-3.2-3B
"""
register_model(
    ModelTemplate(
        LLMModelType.llama3_2,
        ChatTemplateType.llama3_2,
        get_model_tokenizer_with_flash_attn,
        attention_cls='LlamaAttention',
        architectures=['LlamaForCausalLM'],
        model_arch=ModelArchitecture.llama,
    )
)
