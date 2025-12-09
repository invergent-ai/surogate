from typing import Dict, Any, Optional

from surogate.core.config.dataset_config import InstructionDatasetConfig
from surogate.core.config.enums import InstructionDatasetSystemPromptType
from surogate.core.datasets.preprocessor.row import RowPreprocessor
from surogate.core.model.chat_templates.utils import history_to_messages
from surogate.utils.logger import get_logger

logger = get_logger()


class InstructionPreprocessor(RowPreprocessor):
    default_prompt_format = "{instruction}\n{input}"
    default_prompt_no_input_format = "{instruction}"

    def __init__(self, dataset_config: InstructionDatasetConfig):
        super().__init__()
        self.ds_cfg = dataset_config

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        instruction = row.pop(self.ds_cfg.instruction_field, None)
        if instruction is None:
            raise ValueError(
                f"Instruction field '{self.ds_cfg.instruction_field}' is missing from the dataset."
            )
        input = row.pop(self.ds_cfg.input_field, None)
        output = row.pop(self.ds_cfg.output_field, None)
        if output is None:
            raise ValueError(
                f"Output field '{self.ds_cfg.output_field}' is missing from the dataset."
            )

        if self.ds_cfg.system_prompt_type == InstructionDatasetSystemPromptType.field:
            system_prompt = row.pop(self.ds_cfg.system_prompt_field, None)
        else:
            system_prompt = self.ds_cfg.system_prompt

        turn_format = self.ds_cfg.prompt_format if self.ds_cfg.prompt_format \
            else self.default_prompt_format
        turn_no_input_format = self.ds_cfg.prompt_format_no_input if self.ds_cfg.prompt_format_no_input \
            else self.default_prompt_no_input_format

        if input:
            query = turn_format.format(instruction=instruction, input=input)
        else:
            query = turn_no_input_format.format(instruction=instruction)

        history = [query, output]
        row.update({'messages': history_to_messages([history], system_prompt)})
        return row
