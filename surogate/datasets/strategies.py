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

        self.roles_to_train = ["assistant"]
        self.train_on_eos = "turn"
        self.train_on_eot = self.train_on_eos

        if (
            hasattr(self.tokenizer, "eos_token")
            and self.tokenizer.eos_token is not None
        ):
            self.eot_tokens = [self.tokenizer.eos_token]


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
        input_ids = self.prompter.build_prompt(turns, tools=tools)
        labels = [IGNORE_INDEX] * len(input_ids)

        for index, turn in enumerate(turns):
            role = turn.get("role")
            should_train = role in self.roles_to_train

            if not should_train and (
                self.train_on_eos == "turn"
            ):
                if index == len(turns) - 1:
                    logger.warning(
                        "Last turn is not trainable, skipping having to find the turn indices. "
                        "This may cause incorrect last EOS token to be unmasked."
                        "This is likely a dataset design issue. Please ensure last turn is trainable."
                    )
                continue

            turn_start_idx, turn_end_idx = self.find_turn(
                turns=turns, turn_idx=index, tools=tools
            )
            if should_train and turn_start_idx != -1 and turn_end_idx != -1:
                labels[turn_start_idx:turn_end_idx] = input_ids[
                    turn_start_idx:turn_end_idx
                ]

            # Find and handle EOT and EOS tokens
            token_idx = self.find_first_eot_token(input_ids, start_idx=turn_end_idx)
            if (
                    token_idx != -1 and abs(token_idx - turn_end_idx) <= 3
            ):
                # Set labels if needed for this turn
                if self.train_on_eot == "turn" and should_train:
                    labels[token_idx] = input_ids[token_idx]
            else:
                problem_span = self.tokenizer.decode(input_ids[turn_end_idx:token_idx + 1])
                logger.warning(
                    f"EOT token missing after turn {turn}. eot_idx: {token_idx}, turn_end_idx: {turn_end_idx}. Problematic span: {problem_span!r}"
                )

            token_idx = self.find_first_eos_token(input_ids, start_idx=turn_end_idx)
            if (
                    token_idx != -1 and abs(token_idx - turn_end_idx) <= 3
            ):
                # Set labels if needed for this turn
                if self.train_on_eos == "turn" and should_train:
                    labels[token_idx] = input_ids[token_idx]
            else:
                problem_span = self.tokenizer.decode(input_ids[turn_end_idx:token_idx + 1])
                logger.warning(
                    f"EOS token missing after turn {turn}. eos_idx: {token_idx}, turn_end_idx: {turn_end_idx}. Problematic span: {problem_span!r}"
                )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    def find_first_eos_token(self, input_ids, start_idx):
        eos_token_id = self.tokenizer.eos_token_id
        for i in range(start_idx, len(input_ids)):
            if input_ids[i] == eos_token_id:
                return i
        return -1

    def find_first_eot_token(self, input_ids, start_idx):
        """Find the first EOT token in the input_ids starting from start_idx."""
        # Get token IDs for all EOT tokens
        eot_token_ids = []
        for token in self.eot_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"EOT token '{token}' is encoded as multiple tokens: {token_ids}. Please add it under `tokens: ` in the config."
                )

            eot_token_ids.append(token_ids[0])  # Use the last token ID if multiple

        # Search for any of the EOT token IDs
        for i in range(start_idx, len(input_ids)):
            if input_ids[i] in eot_token_ids:
                return i
        return -1

    def find_turn(
        self, turns: list[dict], turn_idx: int, tools: list[dict] | None = None
    ):
        """
        Locate the starting and ending indices of the specified turn in a conversation.
        """

        if turn_idx >= len(turns):
            raise ValueError(f"Turn index {turn_idx} out of range")

        # mistral/gemma3 does not output message if it contains only system message
        if (
            turn_idx == 0
            and turns[0].get("role") == "system"
            and ("mistral" in self.tokenizer.name_or_path.lower())
        ):
            return -1, -1

        empty_turn = {
            "role": turns[turn_idx].get("role"),
            "content": "[[dummy_message]]",
        }

        # Create conversation versions
        turns_with_empty = turns[:turn_idx] + [empty_turn]
        turns_with_content = turns[: turn_idx + 1]

        # Generate the conversation up to the turn, with final turn replaced with dummy content
        dummy_ids = self.prompter.build_prompt(turns_with_empty, tools=tools)  # type: ignore

        # Generate the conversation up to the turn, with final turn included
        full_ids = self.prompter.build_prompt(turns_with_content, tools=tools)  # type: ignore

        if not full_ids or not dummy_ids:
            logger.warning(f"Empty template generated for turn {turn_idx}")
            return -1, -1

        # Find first difference (start of content)
        start_idx = None
        min_len = min(len(dummy_ids), len(full_ids))
        for i in range(min_len):
            if dummy_ids[i] != full_ids[i]:
                start_idx = i
                break

        if start_idx is None:
            logger.warning(f"Could not find content start boundary for turn {turn_idx}")
            return -1, -1

        # Find last difference (end of content)
        end_idx = None
        for i in range(min_len):
            dummy_pos = len(dummy_ids) - 1 - i
            full_pos = len(full_ids) - 1 - i
            if dummy_ids[dummy_pos] != full_ids[full_pos]:
                end_idx = full_pos + 1  # Add one to include the last token when slice
                break

        if end_idx is None:
            logger.warning(f"Could not find content end boundary for turn {turn_idx}")
            return -1, -1

        if end_idx < start_idx:
            logger.warning(
                f"Content end boundary is before start boundary for turn {turn_idx}"
            )
            return -1, -1

        if end_idx == start_idx:
            logger.warning(
                f"Content end boundary is the same as start boundary for turn {turn_idx}. This is likely an empty turn."
            )
            return -1, -1

        return start_idx, end_idx

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
