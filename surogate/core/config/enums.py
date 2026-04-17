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
    chatml = "chatml"
    dummy = "dummy"
    qwen2_5 = "qwen2_5"
    qwen3 = "qwen3"
    qwen3_thinking = "qwen3_thinking"
    qwen3_nothinking = "qwen3_nothinking"
    llama = "llama"  # llama2
    llama3 = "llama3"
    llama3_2 = "llama3_2"
    gpt_oss = "gpt_oss"
    seed_oss = "seed_oss"
    minimax_m2 = "minimax_m2"
    nemotron_nano = "nemotron_nano"


class MLLMChatTemplateType:
    qwen2_5_vl = "qwen2_5_vl"
    qwen3_vl = "qwen3_vl"
    qwen3_5 = "qwen3_5"


class ChatTemplateType(LLMChatTemplateType, MLLMChatTemplateType):
    pass


class LLMModelType:
    qwen2_5 = "qwen2_5"
    qwen3 = "qwen3"
    qwen3_moe = "qwen3_moe"
    llama = "llama"
    gpt_oss = "gpt_oss"
    seed_oss = "seed_oss"
    nemotron_nano = "nemotron_nano"
    minimax_m2 = "minimax_m2"


class MLLMModelType:
    qwen2_5_vl = "qwen2_5_vl"
    qwen3_vl = "qwen3_vl"
    qwen3_vl_moe = "qwen3_vl_moe"
    qwen3_5 = "qwen3_5"
    qwen3_5_moe = "qwen3_5_moe"
