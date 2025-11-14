from typing import Union, Any

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
    text_field: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "The name of the column in your dataset that contains the raw text."
        }
    )

    @staticmethod
    def validate_fields(ds_cfg: 'InstructionDataset', columns: list[str]):
        if ds_cfg.text_field is None:
            raise ValueError("'text_field' must be specified for TextDataset.")
        if ds_cfg.text_field not in columns:
            raise ValueError(
                f"Text field '{ds_cfg.text_field}' is missing from the dataset. Dataset columns: {columns}."
            )


class InstructionDataset(BaseDataset, BaseModel):
    system_prompt_type: InstructionDatasetSystemPromptType | None = Field(
        default=None,
        json_schema_extra={
            "description": "The type of system prompt to use: 'field': Use a dataset field specified in the 'system_field' config field | 'fixed': Use the value from the 'system_prompt' config field"
        }
    )

    system_prompt_field: str | None = Field(
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

    prompt_format: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Format of the prompt as a Python string template. Use {system}, {instruction}, {input}, and {output} as placeholders."
        }
    )

    prompt_format_no_input: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Format of the prompt as a Python string template when there is no 'input'. Use {system}, {instruction} and {output} as placeholders."
        }
    )

    @staticmethod
    def validate_fields(ds_cfg: 'InstructionDataset', columns: list[str]):
        if ds_cfg.instruction_field is None:
            raise ValueError("'instruction_field' must be specified for InstructionDataset.")
        if ds_cfg.output_field is None:
            raise ValueError("'output_field' must be specified for InstructionDataset.")

        if ds_cfg.instruction_field not in columns:
            raise ValueError(
                f"Instruction field '{ds_cfg.instruction_field}' is missing from the dataset. Dataset columns: {columns}."
            )
        if ds_cfg.output_field not in columns:
            raise ValueError(
                f"Output field '{ds_cfg.output_field}' is missing from the dataset. Dataset columns: {columns}."
            )
        if ds_cfg.system_prompt_type == InstructionDatasetSystemPromptType.fixed and len(ds_cfg.system_prompt or "") == 0:
            raise ValueError(
                "'system_prompt' must be a non-empty string when 'system_prompt_type' is 'fixed'."
            )
        if ds_cfg.system_prompt_type == InstructionDatasetSystemPromptType.field and ds_cfg.system_prompt_field is None:
            raise ValueError(
                "'system_prompt_field' must be specified when 'system_prompt_type' is 'field'."
            )
        if ds_cfg.system_prompt_type == InstructionDatasetSystemPromptType.field and ds_cfg.system_prompt_field not in columns:
            raise ValueError(
                f"System prompt field '{ds_cfg.system_prompt_field}' is missing from the dataset. Dataset columns: {columns}."
            )


class ConversationDataset(BaseDataset, BaseModel):
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

    @staticmethod
    def validate_fields(ds_cfg: 'ConversationDataset', columns: list[str]):
        if ds_cfg.messages_field is None:
            raise ValueError("'messages_field' must be specified for ConversationDataset.")
        if ds_cfg.messages_field not in columns:
            raise ValueError(
                f"Messages field '{ds_cfg.messages_field}' is missing from the dataset. Dataset columns: {columns}."
            )

        if ds_cfg.tools_field is not None and ds_cfg.tools_field not in columns:
            raise ValueError(
                f"Tools field '{ds_cfg.tools_field}' is missing from the dataset. Dataset columns: {columns}."
            )

        if ds_cfg.system_field is not None and ds_cfg.system_field not in columns:
            raise ValueError(
                f"System field '{ds_cfg.system_field}' is missing from the dataset. Dataset columns: {columns}."
            )


SurogateDataset = InstructionDataset | ConversationDataset | TextDataset