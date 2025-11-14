from typing import Dict, Any, Optional

from swift.llm.dataset import RowPreprocessor
from swift.llm import history_to_messages
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger
from surogate.utils.schema.datasets import InstructionDataset
from surogate.utils.schema.enums import InstructionDatasetSystemPromptType

logger = get_logger()


class InstructionPreprocessor(RowPreprocessor):
    default_system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    default_system_prompt_no_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    default_prompt_format = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    default_prompt_no_input_format = "### Instruction:\n{instruction}\n\n### Response:\n"

    def __init__(self, cfg: DictDefault, dataset_config: InstructionDataset):
        super().__init__()
        self.cfg = cfg
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
            query = (
                        system_prompt if system_prompt else self.default_system_prompt
                    ) + turn_format.format(instruction=instruction, input=input)
        else:
            query = (
                        system_prompt if system_prompt else self.default_system_prompt_no_input
                    ) + turn_no_input_format.format(instruction=instruction)

        history = [query, output]
        row.update({'messages': history_to_messages([history], system_prompt)})
        return row
