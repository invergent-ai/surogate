from enum import Enum

class SurogateDatasetType(str, Enum):
    text = "text"
    instruction = "instruction"
    conversation = "conversation"

class InstructionDatasetSystemPromptType(str, Enum):
    fixed = "fixed"
    field = "field"

class LLMChatTemplateType:
    chatml = 'chatml'
    dummy = 'dummy'
    qwen2_5 = 'qwen2_5'
    qwen3 = 'qwen3'
    qwen3_thinking = 'qwen3_thinking'
    qwen3_nothinking = 'qwen3_nothinking'
    llama3 = 'llama3'
    llama3_2 = 'llama3_2'
    gpt_oss = 'gpt_oss'

class MLLMChatTemplateType:
    qwen2_5_vl = 'qwen2_5_vl'
    qwen3_vl = 'qwen3_vl'

class ChatTemplateType(LLMChatTemplateType, MLLMChatTemplateType):
    pass

class LLMModelType:
    qwen2_5 = 'qwen2_5'

    qwen3 = 'qwen3'
    qwen3_thinking = 'qwen3_thinking'
    qwen3_nothinking = 'qwen3_nothinking'
    qwen3_moe = 'qwen3_moe'
    qwen3_moe_thinking = 'qwen3_moe_thinking'

    llama3 = 'llama3'
    llama3_1 = 'llama3_1'
    llama3_2 = 'llama3_2'

    gpt_oss = 'gpt_oss'

class MLLMModelType:
    qwen2_5_vl = 'qwen2_5_vl'
    qwen3_vl = 'qwen3_vl'