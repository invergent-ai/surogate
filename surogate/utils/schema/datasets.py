from pydantic import BaseModel, Field

from .enums import SurogateDatasetType, InstructionDatasetSystemPromptType, ChatTemplateType


class BaseDataset(BaseModel):
    path: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "HuggingFace dataset repo | s3:// | gs:// | path to local file or directory"
        },
    )

    subset: str | None = Field(
        default=None,
        json_schema_extra={"description": "subset of dataset to load"},
    )

    split: str | None = Field(
        default=None,
        json_schema_extra={"description": "name of dataset split to load from"},
    )

    type: SurogateDatasetType | None = Field(
        default=None,
        json_schema_extra={
            "description": "The type of prompt to use"
        },
    )

    samples: int | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of samples to use"
        }
    )


class TextDataset(BaseDataset, BaseModel):
    text_column: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The name of the column in your dataset that contains the raw text."
        }
    )

class InstructionDataset(BaseDataset, BaseModel):
    chat_template: ChatTemplateType | None = Field(
        default='tokenizer_default',
        json_schema_extra={
            "description": "The Jinja chat template to use for formatting the prompt: 'tokenizer_default' (Uses the chat template available in the model's tokenizer_config.json file. If the chat template is not available in the tokenizer, it will raise an error) | 'jinja' (Custom Jinja chat template) "
        }
    )

    chat_template_jinja: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Custom Jinja chat template path. Required if 'chat_template' is 'jinja'."
        }
    )

    system_prompt_type: InstructionDatasetSystemPromptType | None = Field(
        default=None,
        json_schema_extra={
            "description": "The type of system prompt to use: 'field': Use a dataset field specified in the 'system_field' config field | 'fixed': Use the value from the 'system_prompt' config field"
        }
    )

    system_field: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The name of the column in your dataset that contains the System Prompt"
        }
    )

    system_prompt: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The system prompt"
        }
    )

    instruction_field: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The name of the column in your dataset that contains the instruction"
        }
    )

    input_field: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The name of the column in your dataset that contains the input"
        }
    )

    output_field: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The name of the column in your dataset that contains the output"
        }
    )



class ConversationDataset(BaseDataset, BaseModel):
    chat_template: ChatTemplateType | None = Field(
        default='tokenizer_default',
        json_schema_extra={
            "description": "The Jinja chat template to use for formatting the prompt: 'tokenizer_default' (Uses the chat template available in the model's tokenizer_config.json file. If the chat template is not available in the tokenizer, it will raise an error) | 'jinja' (Custom Jinja chat template) "
        }
    )

    chat_template_jinja: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Custom Jinja chat template path. Required if 'chat_template' is 'jinja'."
        }
    )

    system_field: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The name of the column in your dataset that contains the System Prompt"
        }
    )

    messages_field: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The name of the column in your dataset that contains the conversation (e.g. 'messages')"
        }
    )

    tools_field: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "If the chat template supports tools, this should be the name of the column in your dataset that contains the available tools."
        }
    )

    message_property_mappings: dict[str, str] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Mapping of properties from the input dataset to the chat template. (default: message_property_mappings={'role':'role', 'content':'content'}). If a property exists in the template but not in this mapping, the system will attempt to load it directly from the message using the property name as the key. Example: In the mapping below, 'from' is loaded from input dataset and used as 'role', while 'value' is loaded and used as 'content' in the chat template."
        },
    )

