from dataclasses import dataclass, field
from typing import List

from surogate.core.config.enums import ChatTemplateType
from .base import Prompt, Word, register_chat_template
from .chatml import ChatmlChatTemplate


@dataclass
class NemotronNanoChatTemplate(ChatmlChatTemplate):
    # Model always emits system block, even when no system message is provided
    prefix: Prompt = field(default_factory=lambda: ['<|im_start|>system\n<|im_end|>\n'])
    auto_add_bos: bool = False
    non_thinking_prefix: str = '<think></think>'
    stop_words: List[Word] = field(default_factory=lambda: ['<|im_end|>'])

register_chat_template(NemotronNanoChatTemplate(ChatTemplateType.nemotron_nano, is_thinking=True))
