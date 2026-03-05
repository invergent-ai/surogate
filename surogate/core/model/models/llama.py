from surogate.core.config.enums import ChatTemplateType
from surogate.core.model.registry import register_model, ModelTemplate


"""
Instruct:
- meta-llama/Meta-Llama-3-8B-Instruct
- meta-llama/Meta-Llama-3.1-8B-Instruct
- nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
- meta-llama/Llama-3.2-1B-Instruct
- meta-llama/Llama-3.2-3B-Instruct

Base:
- meta-llama/Meta-Llama-3-8B
- meta-llama/Meta-Llama-3.1-8B
- meta-llama/Llama-3.2-1B
- meta-llama/Llama-3.2-3B
"""
register_model(
    ModelTemplate(
        model_type='LlamaForCausalLM',
        chat_templates=[ChatTemplateType.llama3, ChatTemplateType.llama3_2]))
