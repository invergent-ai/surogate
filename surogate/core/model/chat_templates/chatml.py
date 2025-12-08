from dataclasses import dataclass, field
from typing import Optional

from surogate.core.config.enums import ChatTemplateType
from .base import ChatTemplate, Prompt, register_chat_template

DEFAULT_SYSTEM = 'You are a helpful assistant.'

@dataclass
class ChatmlChatTemplate(ChatTemplate):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>\n'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'])
    auto_add_bos: bool = True

register_chat_template(ChatmlChatTemplate(ChatTemplateType.chatml))
