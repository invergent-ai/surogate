from typing import Dict, Any, Optional, List

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

        # Cache format strings (optimization: avoid checking on every row)
        self.turn_format = dataset_config.prompt_format or self.default_prompt_format
        self.turn_no_input_format = dataset_config.prompt_format_no_input or self.default_prompt_no_input_format

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        instruction = row.get(self.ds_cfg.instruction_field, None)
        if instruction is None:
            raise ValueError(
                f"Instruction field '{self.ds_cfg.instruction_field}' is missing from the dataset."
            )
        input = row.get(self.ds_cfg.input_field, None)
        output = row.get(self.ds_cfg.output_field, None)
        if output is None:
            raise ValueError(
                f"Output field '{self.ds_cfg.output_field}' is missing from the dataset."
            )

        if self.ds_cfg.system_prompt_type == InstructionDatasetSystemPromptType.field:
            system_prompt = row.get(self.ds_cfg.system_prompt_field, None)
        else:
            system_prompt = self.ds_cfg.system_prompt

        if input:
            query = self.turn_format.format(instruction=instruction, input=input)
        else:
            query = self.turn_no_input_format.format(instruction=instruction)

        history = [query, output]
        row.update({'messages': history_to_messages([history], system_prompt)})
        return row

    def preprocess_batch(self, rows: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        """Process a batch of instruction dataset rows efficiently.

        1. Batches field extraction operations
        2. Batches string formatting operations
        3. Reduces function call overhead

        Args:
            rows: List of input row dictionaries

        Returns:
            List of processed dictionaries with 'messages' field
        """
        results = []

        # Batch extract fields
        instructions = []
        inputs = []
        outputs = []
        system_prompts = []
        remaining_fields = []  # Fields to preserve after extraction

        for row in rows:
            instruction = row.get(self.ds_cfg.instruction_field)
            if instruction is None:
                raise ValueError(
                    f"Instruction field '{self.ds_cfg.instruction_field}' is missing from the dataset."
                )

            input_val = row.get(self.ds_cfg.input_field)
            output = row.get(self.ds_cfg.output_field)
            if output is None:
                raise ValueError(
                    f"Output field '{self.ds_cfg.output_field}' is missing from the dataset."
                )

            if self.ds_cfg.system_prompt_type == InstructionDatasetSystemPromptType.field:
                system_prompt = row.get(self.ds_cfg.system_prompt_field)
            else:
                system_prompt = self.ds_cfg.system_prompt

            instructions.append(instruction)
            inputs.append(input_val)
            outputs.append(output)
            system_prompts.append(system_prompt)

            # Collect remaining fields (excluding extracted ones)
            extracted_fields = {
                self.ds_cfg.instruction_field,
                self.ds_cfg.input_field,
                self.ds_cfg.output_field
            }
            if self.ds_cfg.system_prompt_type == InstructionDatasetSystemPromptType.field:
                extracted_fields.add(self.ds_cfg.system_prompt_field)

            remaining = {k: v for k, v in row.items() if k not in extracted_fields}
            remaining_fields.append(remaining)

        # Batch format queries
        queries = []
        for instruction, input_val in zip(instructions, inputs):
            if input_val:
                query = self.turn_format.format(instruction=instruction, input=input_val)
            else:
                query = self.turn_no_input_format.format(instruction=instruction)
            queries.append(query)

        # Batch create messages
        for query, output, system_prompt, remaining in zip(queries, outputs, system_prompts, remaining_fields):
            history = [query, output]
            result = {'messages': history_to_messages([history], system_prompt)}
            result.update(remaining)
            results.append(result)

        return results
