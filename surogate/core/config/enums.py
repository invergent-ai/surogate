from enum import Enum

class SurogateDatasetType(str, Enum):
    text = "text"
    instruction = "instruction"
    conversation = "conversation"
    auto = "auto"

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
    llama = 'llama'  # llama2
    llama3 = 'llama3'
    llama3_2 = 'llama3_2'
    gpt_oss = 'gpt_oss'
    nemotron_nano = 'nemotron_nano'

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

    llama3 = 'llama3'
    llama3_1 = 'llama3_1'
    llama3_2 = 'llama3_2'

    gpt_oss = 'gpt_oss'
    
    nemotron_nano = 'nemotron_nano'

class MLLMModelType:
    qwen2_5_vl = 'qwen2_5_vl'
    qwen3_vl = 'qwen3_vl'

class RMModelType:
    internlm2_reward = 'internlm2_reward'
    qwen2_reward = 'qwen2_reward'
    qwen2_5_prm = 'qwen2_5_prm'
    qwen2_5_math_reward = 'qwen2_5_math_reward'
    llama3_2_reward = 'llama3_2_reward'
    gemma_reward = 'gemma_reward'

class RerankerModelType:
    bge_reranker = 'bge_reranker'
    modern_bert_gte_reranker = 'modern_bert_gte_reranker'
    jina_reranker_m0 = 'jina_reranker_m0'
    qwen3_reranker = 'qwen3_reranker'