from abc import ABC
from dataclasses import dataclass
from typing import Optional, Literal

from surogate.core.config.enums import SurogateDatasetType, InstructionDatasetSystemPromptType
from surogate.utils.dict import DictDefault


@dataclass
class DatasetConfig:
    """
    DatasetConfig class is a dataclass that holds configuration parameters for a dataset.

    Args:
        path (Optional[str]): HuggingFace dataset repo | s3:// | gs:// | path to local file or directory.
        subset (Optional[str]): HuggingFace subset of dataset to load
        split (Optional[str]): Name of dataset split to load from. Defaults to 'train'.
        type (Optional[Literal['text', 'instruction', 'conversation']]): The type of dataset.
        samples (Optional[int]): Number of samples to use.
    """
    path: str | None = None
    subset: Optional[str] = None
    split: str | None = None
    type: Literal[SurogateDatasetType.text, SurogateDatasetType.instruction, SurogateDatasetType.conversation, SurogateDatasetType.auto] | None = None
    samples: Optional[int] = None

    def __init__(self, cfg: DictDefault):
        self.path = cfg['path']
        self.subset = cfg['subset']
        self.split = cfg['split'] or 'train'
        self.type = cfg['type']
        self.samples = cfg['samples']

    def __post_init__(self):
        if self.type not in SurogateDatasetType.__dict__.values():
            raise ValueError(f"Dataset type {self.type} is not supported.")

    def validate_columns(self, columns: list[str]):
        pass

class TextDatasetConfig(DatasetConfig):
    """
    TextDatasetConfig class is a dataclass that holds configuration parameters for a text dataset.

    Args:
        text_field (str): The name of the column in your dataset that contains the raw text.
    """
    text_field: Optional[str] = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.text_field = cfg['text_field'] or 'text'
        self.__post_init__()

    def __post_init__(self):
        super().__post_init__()
        if self.text_field is None:
            raise ValueError("'text_field' must be specified for TextDataset.")

    def validate_columns(self, columns: list[str]):
        if self.text_field not in columns:
            raise ValueError(
                f"Instruction field '{self.text_field}' is missing from the dataset. Dataset columns: {columns}."
            )


class InstructionDatasetConfig(DatasetConfig):
    """
    InstructionDatasetConfig class is a dataclass that holds configuration parameters for an instruction dataset.

    Args:
        system_prompt_type (Optional[Literal['field', 'fixed']]): The type of system prompt to use: 'field': Use a dataset field specified in the 'system_field' config field | 'fixed': Use the value from the 'system_prompt' config field
        system_prompt_field (Optional[str]): The name of the column in your dataset that contains the system prompt when system_prompt_type is 'field'.
        system_prompt (Optional[str]): The fixed system prompt to use when system_prompt_type is 'fixed'.
        instruction_field (Optional[str]): The name of the column in your dataset that contains the instruction
        input_field (Optional[str]): The name of the column in your dataset that contains the input.
        output_field (Optional[str]): The name of the column in your dataset that contains the output
        prompt_format (Optional[str]): Format of the prompt as a Python string template. Use {system}, {instruction}, {input}, and {output} as placeholders.
        prompt_format_no_input (Optional[str]): Format of the prompt as a Python string template when there is no 'input'. Use {system}, {instruction} and {output} as placeholders.
    """

    system_prompt_type: Optional[
        Literal[InstructionDatasetSystemPromptType.field, InstructionDatasetSystemPromptType.fixed]] = None
    system_prompt_field: Optional[str] = None
    system_prompt: Optional[str] = None

    instruction_field: str | None = None
    input_field: Optional[str] = None
    output_field: str | None = None

    prompt_format: Optional[str] = None
    prompt_format_no_input: Optional[str] = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.system_prompt_type = cfg['system_prompt_type']
        self.system_prompt_field = cfg['system_prompt_field']
        self.system_prompt = cfg['system_prompt']
        self.instruction_field = cfg['instruction_field']
        self.input_field = cfg['input_field']
        self.output_field = cfg['output_field']
        self.prompt_format = cfg['prompt_format']
        self.prompt_format_no_input = cfg['prompt_format_no_input']
        self.__post_init__()

    def __post_init__(self):
        super().__post_init__()

        if self.instruction_field is None:
            raise ValueError("'instruction_field' must be specified for InstructionDataset.")
        if self.output_field is None:
            raise ValueError("'output_field' must be specified for InstructionDataset.")
        if self.system_prompt_type == InstructionDatasetSystemPromptType.fixed and len(self.system_prompt or "") == 0:
            raise ValueError(
                "'system_prompt' must be a non-empty string when 'system_prompt_type' is 'fixed'."
            )
        if self.system_prompt_type == InstructionDatasetSystemPromptType.field and self.system_prompt_field is None:
            raise ValueError(
                "'system_prompt_field' must be specified when 'system_prompt_type' is 'field'."
            )

    def validate_columns(self, columns: list[str]):
        if self.instruction_field not in columns:
            raise ValueError(
                f"Instruction field '{self.instruction_field}' is missing from the dataset. Dataset columns: {columns}."
            )
        if self.output_field not in columns:
            raise ValueError(
                f"Output field '{self.output_field}' is missing from the dataset. Dataset columns: {columns}."
            )
        if self.system_prompt_type == InstructionDatasetSystemPromptType.field and self.system_prompt_field not in columns:
            raise ValueError(
                f"System prompt field '{self.system_prompt_field}' is missing from the dataset. Dataset columns: {columns}."
            )


class ConversationDatasetConfig(DatasetConfig):
    """
    ConversationDatasetConfig class is a dataclass that holds configuration parameters for a conversation dataset.

    Args:
        messages_field (str): Column containing message list. Defaults to "messages".
        completion_field (Optional[str]): If set, messages are built by concatenating
            messages_field + completion_field. Useful for datasets that split prompt
            and completion into separate columns (e.g. prompt=[{role:user,...}],
            completion=[{role:assistant,...}]).
    """
    system_field: Optional[str] = None
    messages_field: Optional[str] = None
    completion_field: Optional[str] = None
    tools_field: Optional[str] = None
    message_property_mappings: Optional[dict[str, str]] = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.system_field = cfg['system_field']
        self.messages_field = cfg['messages_field'] or "messages"
        self.completion_field = cfg['completion_field']
        self.tools_field = cfg['tools_field']
        self.message_property_mappings = cfg['message_property_mappings'] or {
            "role": "role",
            "content": "content",
            "tool_calls": "tool_calls",
        }
        self.__post_init__()

    def __post_init__(self):
        super().__post_init__()

        if self.messages_field is None:
            raise ValueError("'messages_field' must be specified for ConversationDataset.")

    def validate_columns(self, columns: list[str]):
        if self.messages_field not in columns:
            raise ValueError(
                f"Messages field '{self.messages_field}' is missing from the dataset. Dataset columns: {columns}."
            )

        if self.tools_field is not None and self.tools_field not in columns:
            raise ValueError(
                f"Tools field '{self.tools_field}' is missing from the dataset. Dataset columns: {columns}."
            )

        if self.system_field is not None and self.system_field not in columns:
            raise ValueError(
                f"System field '{self.system_field}' is missing from the dataset. Dataset columns: {columns}."
            )


SurogateDatasetConfig = TextDatasetConfig | InstructionDatasetConfig | ConversationDatasetConfig

def create_dataset_config(ds_cfg: DictDefault):
    ds_type = ds_cfg['type']
    if ds_type == 'text':
        return TextDatasetConfig(ds_cfg)
    elif ds_type == 'instruction':
        return InstructionDatasetConfig(ds_cfg)
    elif ds_type == 'conversation':
        return ConversationDatasetConfig(ds_cfg)
    else:
        return DatasetConfig(ds_cfg)