from dataclasses import dataclass, field
from typing import Optional, List

from surogate.core.config.enums import ChatTemplateType
from .base import Word, register_chat_template
from .chatml import ChatmlChatTemplate, DEFAULT_SYSTEM


@dataclass
class QwenChatTemplate(ChatmlChatTemplate):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    agent_template: str = 'hermes'

@dataclass
class Qwen25ChatTemplate(QwenChatTemplate):
    default_system: Optional[str] = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'

register_chat_template(Qwen25ChatTemplate(ChatTemplateType.qwen2_5))

@dataclass
class Qwen3MixedTemplateMeta(QwenChatTemplate):
    default_system: Optional[str] = None
    non_thinking_prefix: str = '<think>\n\n</think>\n\n'

register_chat_template(Qwen3MixedTemplateMeta(ChatTemplateType.qwen3, is_thinking=True))
register_chat_template(Qwen3MixedTemplateMeta(ChatTemplateType.qwen3_thinking, default_system=None, is_thinking=True, thinking_prefix='<think>\n'))
register_chat_template(Qwen3MixedTemplateMeta(ChatTemplateType.qwen3_nothinking, default_system=None))
