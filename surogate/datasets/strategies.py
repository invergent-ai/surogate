import abc
import json
from collections import defaultdict
from typing import Optional, Callable, Any, Dict, List, Tuple

from transformers import PreTrainedTokenizer, BatchEncoding

from surogate.datasets.prompters import InstructionPrompterWithChatTemplate, InstructionPrompter
from surogate.datasets.prompters import Prompter, ChatTemplatePrompter
from surogate.utils.dict import remove_none_values
from surogate.utils.logger import get_logger
from surogate.utils.schema.enums import InstructionDatasetSystemPromptType

logger = get_logger()

IGNORE_INDEX = -100

class PromptTokenizingStrategy(abc.ABC):
    """
    Abstract class for tokenizing strategies
    """
    filter_rows: Optional[Callable] = None

    def __init__(
            self,
            prompter: Prompter,
            tokenizer,
            sequence_len: int | None,
    ):
        self.prompter = prompter
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.sequence_len = sequence_len

    @abc.abstractmethod
    def tokenize_prompt(self, prompt):
        pass

    @property
    def supports_batched(self):
        return False

    def _tokenize(
            self, prompt: str, add_eos_token: bool = True, strip_bos_token: bool = False
    ) -> BatchEncoding:
        empty = BatchEncoding(data={"input_ids": [], "attention_mask": []})
        if not prompt:
            logger.warning_once("Empty text requested for tokenization.")
            return empty

        result = self.tokenizer(
            prompt,
            truncation=True,
            padding=False,
            return_tensors=None,
            max_length=self.sequence_len,
        )

        if len(result["input_ids"]) == 0:
            logger.warning("Tokenizer result is empty. You may want to audit your dataset")
            return empty

        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.sequence_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if result["input_ids"][0] == self.tokenizer.bos_token_id and strip_bos_token:
            result["input_ids"] = result["input_ids"][1:]
            result["attention_mask"] = result["attention_mask"][1:]

        result["labels"] = result["input_ids"].copy()

        return result


class ChatTemplateStrategy(PromptTokenizingStrategy):
    def __init__(
            self,
            prompter: "ChatTemplatePrompter",
            tokenizer,
            sequence_len: int | None,
    ):
        super().__init__(prompter, tokenizer, sequence_len)
        self.prompter: ChatTemplatePrompter = prompter

    @property
    def supports_batched(self) -> bool:
        # Let calling code know we can handle lists of examples
        return True

    def is_prompt_batched(self, prompt: dict[str, Any]) -> bool:
        try:
            return all(isinstance(v, list) for v in prompt.values()) and all(
                isinstance(v, list) for v in prompt[self.prompter.messages_field]
            )
        except KeyError:
            return False

    def tokenize_prompt(self, prompt: dict[str, Any]):
        prompt = remove_none_values(prompt)

        if not self.is_prompt_batched(prompt) or not self.supports_batched:
            return self._tokenize_single_prompt(prompt)

        res = defaultdict(lambda: [])
        feature_names = list(prompt.keys())

        # Process each prompt individually
        for row in zip(*prompt.values(), strict=False):
            tokenized_prompt = self._tokenize_single_prompt(
                dict(zip(feature_names, row, strict=False))
            )
            for key, val in tokenized_prompt.items():
                res[key].append(val)

        # If there are no examples left, return an empty dictionary
        if not res:
            return {}

        return dict(res)

    def _tokenize_single_prompt(self, prompt: dict) -> Dict[str, List[int]]:
        turns = self.get_conversation_thread(prompt)
        tools = self._get_tools(prompt)
        input_ids = self.prompter.build_prompt(turns, tools=tools)  # type: ignore
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }

    def get_conversation_thread(self, prompt):
        turns = []
        messages = self._get_messages(prompt)
        possible_sys_turn = self.transform_message(messages[0])

        if (
                possible_sys_turn["role"] != "system"
                and self.prompter.system_field in prompt
        ):
            turn = {"role": "system", "content": prompt[self.prompter.system_field]}
            turns.append(turn)

        for message in messages:
            turns.append(self.transform_message(message))

        return turns

    def transform_message(self, message: dict) -> dict:
        transformed_message = {}
        for key, value in self.prompter.message_property_mappings.items():
            if message.get(value) is not None:
                transformed_message[key] = message[value]
            else:
                logger.debug(
                    f"Could not find value for property {value} in message: {message}"
                )

        # Map the role if necessary
        if "role" in transformed_message:
            transformed_message["role"] = self.prompter.roles.get(
                transformed_message["role"], transformed_message["role"]
            )

        if "tool_calls" in transformed_message and transformed_message["tool_calls"]:
            for tool_call in transformed_message["tool_calls"]:
                if "function" in tool_call and "arguments" in tool_call["function"]:
                    args = tool_call["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            tool_call["function"]["arguments"] = json.loads(args)
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Error parsing tool_calls arguments as JSON. "
                                f"Function: {tool_call.get('function', {}).get('name', 'unknown')}, "
                                f"Arguments string: {args!r}, "
                                f"Error: {e}"
                            )
                            raise

        return transformed_message

    def _get_tools(self, prompt) -> list[dict] | None:
        """Get tools from prompt if available."""
        tools = prompt.get(self.prompter.tools_field, None)
        if tools is None:
            return None

        if isinstance(tools, list):
            return tools

        raise ValueError(
            "Unknown tools format. Please convert it into a list[dict].\n"
            f"Current format: {type(tools)}"
        )

    def _get_messages(self, prompt):
        messages = prompt.get(self.prompter.messages_field, None)
        if messages is None:
            raise ValueError("Messages is null. Please check `messages_field`.")

        if isinstance(messages, list):
            return messages

        raise ValueError(
            "Unknown messages format. Please convert it into a list[dict].\n"
            f"Current format: {type(messages)}"
        )


class InstructionStrategy(PromptTokenizingStrategy):
    def __init__(
            self,
            prompter: "InstructionPrompter",
            tokenizer,
            sequence_len: int | None,
    ):
        super().__init__(prompter, tokenizer, sequence_len)
        self.prompter: InstructionPrompter = prompter

    def supports_batched(self):
        return True

    def is_prompt_batched(self, prompt: dict[str, Any]) -> bool:
        return all(isinstance(v, list) for v in prompt.values())

    def tokenize_prompt(self, prompt: dict[str, Any]):
        prompt = remove_none_values(prompt)

        if not self.is_prompt_batched(prompt):
            return self._tokenize_single_prompt(prompt)

        res: dict[str, list] = defaultdict(list)
        feature_names = list(prompt.keys())

        # Batched case: each value must be a list of equal length
        for row in zip(*prompt.values(), strict=True):
            example = dict(zip(feature_names, row, strict=False))
            tokenized = self._tokenize_single_prompt(example)
            for k, v in tokenized.items():
                res[k].append(v)

        return dict(res)

    def _tokenize_single_prompt(self, prompt: dict) -> BatchEncoding:
        instruction = prompt.get(self.prompter.instruction_field)
        if instruction is None:
            raise ValueError(
                f"Instruction field '{self.prompter.instruction_field}' is missing from the dataset."
            )
        input = prompt.get(self.prompter.input_field)
        output = prompt.get(self.prompter.output_field)
        if output is None:
            raise ValueError(
                f"Output field '{self.prompter.output_field}' is missing from the dataset."
            )
        if self.prompter.system_prompt_type == InstructionDatasetSystemPromptType.field:
            system_prompt = prompt.get(self.prompter.system_prompt_field)
        else:
            system_prompt = self.prompter.system_prompt

        user_prompt = self.prompter.build_prompt(instruction, None, input, system_prompt)
        tokenized_prompt = self._tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_prompt["input_ids"])
        tokenized_prompt["labels"] = [IGNORE_INDEX] * user_prompt_len

        tokenized_response_prompt = self._tokenize(
            output, strip_bos_token=True, add_eos_token=True
        )

        tokenized_prompt["input_ids"] += tokenized_response_prompt["input_ids"]
        tokenized_prompt["attention_mask"] += tokenized_response_prompt["attention_mask"]
        tokenized_prompt["labels"] += tokenized_response_prompt["input_ids"]

        return tokenized_prompt


class InstructionStrategyWithChatTemplate(InstructionStrategy):
    def __init__(
            self,
            prompter: "InstructionPrompterWithChatTemplate",
            tokenizer,
            sequence_len: int | None,
    ):
        super().__init__(prompter, tokenizer, sequence_len)
        self.prompter: InstructionPrompterWithChatTemplate = prompter

    def _tokenize_single_prompt(self, prompt: dict) -> Dict[str, List[int]]:
        instruction = prompt.get(self.prompter.instruction_field)
        if instruction is None:
            raise ValueError(
                f"Instruction field '{self.prompter.instruction_field}' is missing from the dataset."
            )
        input = prompt.get(self.prompter.input_field)
        output = prompt.get(self.prompter.output_field)
        if output is None:
            raise ValueError(
                f"Output field '{self.prompter.output_field}' is missing from the dataset."
            )
        if self.prompter.system_prompt_type == InstructionDatasetSystemPromptType.field:
            system_prompt = prompt.get(self.prompter.system_prompt_field)
        else:
            system_prompt = self.prompter.system_prompt

        input_ids = self.prompter.build_prompt(instruction, output, input, system_prompt)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
