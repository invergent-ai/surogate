from typing import Any, Set

from surogate.datasets.jinja_template_analyzer import JinjaTemplateAnalyzer
from surogate.utils.schema.enums import InstructionDatasetSystemPromptType


class Prompter:
    """
    Base prompter class for all prompters
    """


class ChatTemplatePrompter(Prompter):
    def __init__(
            self,
            tokenizer,
            sequence_len: int | None,
            chat_template: str,
            message_property_mappings: dict[str, str] | None = None,
            messages_field: str = "messages",
            system_field: str = "system",
            tools_field: str = "tools",
            roles: dict[str, list[str]] | None = None,
            template_thinking_key: str | None = "reasoning_content",
    ):
        if message_property_mappings is None or (not message_property_mappings):
            message_property_mappings = {
                "role": "role",
                "content": "content",
            }

        if roles:
            self.roles = {s: t for t, sources in roles.items() for s in sources}
        else:
            self.roles = {
                "human": "user",
                "user": "user",
                "assistant": "assistant",
                "gpt": "assistant",
                "system": "system",
                "tool": "tool",
            }

        self._chat_template_msg_variables = get_chat_template_msg_variables(
            chat_template, messages_field
        )

        self.sequence_len = sequence_len
        self.message_property_mappings = message_property_mappings
        self.messages_field = messages_field
        self.system_field = system_field
        self.tools_field = tools_field
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.template_thinking_key: str = template_thinking_key or "reasoning_content"

    @property
    def chat_template_msg_variables(self) -> Set[str]:
        return self._chat_template_msg_variables

    def build_prompt(
            self,
            conversation: list[dict],
            add_generation_prompt=False,
            tools=None,
    ):
        """
        Build a prompt from a conversation.

        Args:
            conversation: A list of messages.
            add_generation_prompt: Whether to add a generation prompt.
            tools: A list of tools. (optional)
        """
        chat_template_kwargs = {
            "chat_template": self.chat_template,
            "add_generation_prompt": add_generation_prompt,
        }

        if tools:
            chat_template_kwargs["tools"] = tools

        return self.tokenizer.apply_chat_template(
            conversation,
            **chat_template_kwargs,
        )

class InstructionPrompter(Prompter):
    default_system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    default_system_prompt_no_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    default_prompt_format = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    default_prompt_no_input_format = "### Instruction:\n{instruction}\n\n### Response:\n"

    def __init__(
            self,
            tokenizer,
            system_prompt_type: InstructionDatasetSystemPromptType,
            system_prompt_field: str,
            system_prompt: str,
            instruction_field: str,
            input_field: str,
            output_field: str,
            sequence_len: int | None,
            prompt_format: str | None = None,
            prompt_format_no_input: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.instruction_field = instruction_field
        self.input_field = input_field
        self.output_field = output_field
        self.prompt_format = prompt_format
        self.prompt_format_no_input = prompt_format_no_input
        self.system_prompt_type = system_prompt_type
        self.system_prompt_field = system_prompt_field
        self.system_prompt = system_prompt

    def build_prompt(
            self,
            instruction: str,
            output: str | None,
            input: str | None = None,
            system_prompt: str | None = None,
    ):
        turn_format = self.prompt_format if self.prompt_format \
            else self.default_prompt_format
        turn_no_input_format = self.prompt_format_no_input if self.prompt_format_no_input \
            else self.default_prompt_no_input_format

        if input:
            res = (
                      system_prompt if system_prompt else self.default_system_prompt
                  ) + turn_format.format(instruction=instruction, input=input)
        else:
            res = (
                      system_prompt if system_prompt else self.default_system_prompt_no_input
                  ) + turn_no_input_format.format(instruction=instruction)

        if output:
            res = f"{res}{output}"

        return res

class InstructionPrompterWithChatTemplate(InstructionPrompter):
    def __init__(
            self,
            tokenizer,
            sequence_len: int | None,
            chat_template: str,
            system_prompt_type: InstructionDatasetSystemPromptType,
            system_prompt_field: str,
            system_prompt: str,
            instruction_field: str,
            input_field: str,
            output_field: str
    ):
        super().__init__(
            tokenizer, system_prompt_type, system_prompt_field, system_prompt,
            instruction_field, input_field, output_field, sequence_len
        )

        self.chat_template = chat_template
        self._chat_template_msg_variables = get_chat_template_msg_variables(chat_template, "messages")

    def build_prompt(
            self,
            instruction: str,
            output: str,
            input: str | None = None,
            system_prompt: str | None = None,
            add_generation_prompt=False,
    ):
        conversation = []

        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})

        user_content = instruction
        if input:
            user_content += f"\n\n{input}"
        conversation.append({"role": "user", "content": user_content})
        conversation.append({"role": "assistant", "content": output})

        return self.tokenizer.apply_chat_template(
            conversation,
            chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt,
        )


def get_chat_template_msg_variables(
        chat_template: str, field_messages: str
) -> Set[str]:
    template_analyzer = JinjaTemplateAnalyzer(chat_template)
    return template_analyzer.get_message_vars(field_messages)
