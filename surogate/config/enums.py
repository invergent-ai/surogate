from enum import Enum

class SurogateDatasetType(str, Enum):
    text = "text"
    instruction = "instruction"
    conversation = "conversation"

class InstructionDatasetSystemPromptType(str, Enum):
    fixed = "fixed"
    field = "field"

class ChatTemplateType(str, Enum):
    tokenizer_default = "tokenizer_default"
    jinja = "jinja"
