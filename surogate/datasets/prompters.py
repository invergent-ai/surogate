from typing import Any, Set

from surogate.datasets.jinja_template_analyzer import JinjaTemplateAnalyzer


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

        self._chat_template_msg_variables = self.get_chat_template_msg_variables(
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
            images: A list of images. (optional)
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

    def get_chat_template_msg_variables(
            self, chat_template: str, field_messages: str
    ) -> Set[str]:
        template_analyzer = JinjaTemplateAnalyzer(chat_template)
        return template_analyzer.get_message_vars(field_messages)


class UnsupportedPrompter(Prompter):
    """
    A dummy class for custom prompters
    """

    def __init__(self) -> None:
        pass

    def __repr__(self):
        return "Pre-tokenized or custom dataset types are unsupported for logging"
