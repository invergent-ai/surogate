from dataclasses import dataclass, field
from typing import Optional, List

from surogate.core.config.enums import ChatTemplateType
from .base import Word, register_chat_template
from .thinking import ThinkingChatTemplateProcessor
from .chatml import ChatmlChatTemplate, DEFAULT_SYSTEM


@dataclass
class QwenChatTemplate(ChatmlChatTemplate):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])

@dataclass
class Qwen25ChatTemplate(QwenChatTemplate):
    default_system: Optional[str] = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'

register_chat_template(Qwen25ChatTemplate(ChatTemplateType.qwen2_5))

class Qwen3ChatTemplateProcessor(ThinkingChatTemplateProcessor):
    no_think_prefix = '<think>\n\n</think>\n\n'

register_chat_template(
    QwenChatTemplate(
        ChatTemplateType.qwen3, default_system=None, template_processor_cls=Qwen3ChatTemplateProcessor))

register_chat_template(
    QwenChatTemplate(
        ChatTemplateType.qwen3_thinking, default_system=None, response_prefix='<think>\n',
        template_processor_cls=ThinkingChatTemplateProcessor))

register_chat_template(
    QwenChatTemplate(
        ChatTemplateType.qwen3_nothinking, default_system=None))