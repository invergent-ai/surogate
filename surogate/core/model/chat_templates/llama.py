from dataclasses import dataclass, field
from typing import Optional
import datetime as dt

from .base import ChatTemplate, Prompt,register_chat_template
from ...config.enums import ChatTemplateType

# ref: https://github.com/facebookresearch/llama/blob/main/llama/generation.py
LLAMA_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest assistant. '
    'Always answer as helpfully as possible, while being safe. '
    'Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information.")

register_chat_template(
    ChatTemplate(
        ChatTemplateType.llama, ['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s><s>[INST] '], ['</s>'],
        default_system=LLAMA_DEFAULT_SYSTEM,
        system_prefix=['<s>[INST] <<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))

@dataclass
class Llama3ChatTemplate(ChatTemplate):
    prefix: Prompt = field(default_factory=lambda: ['<|begin_of_text|>'])
    prompt: Prompt = field(default_factory=lambda: [
        '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    ])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|eot_id|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|eot_id|>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>'])
    agent_template: str = 'llama3'

register_chat_template(Llama3ChatTemplate(ChatTemplateType.llama3))

def _get_llama32_prefix() -> Prompt:
    now = dt.datetime.now()
    date_string = now.strftime('%d %b %Y')
    date_prompt = f'Cutting Knowledge Date: December 2023\nToday Date: {date_string}'
    return [f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{date_prompt}\n\n' '{{SYSTEM}}<|eot_id|>']

@dataclass
class Llama32ChatTemplate(Llama3ChatTemplate):
    prefix: Prompt = field(default_factory=lambda: _get_llama32_prefix())
    system_prefix: Optional[Prompt] = None

register_chat_template(Llama32ChatTemplate(ChatTemplateType.llama3_2))