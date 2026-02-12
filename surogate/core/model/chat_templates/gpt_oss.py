from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from surogate.core.config.enums import ChatTemplateType
from surogate.core.model.chat_templates.inputs import StdChatTemplateInputs
from surogate.core.model.chat_templates.processor import ChatTemplateProcessor
from surogate.core.model.utils import Prompt
from .base import ChatTemplate, register_chat_template

@dataclass
class GptOssTemplateMeta(ChatTemplate):
    prefix: Prompt = field(default_factory=lambda: ['{{SYSTEM}}'])
    prompt: Prompt = field(default_factory=lambda: ['<|start|>user<|message|>{{QUERY}}<|end|><|start|>assistant'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|end|>'])
    suffix: Prompt = field(default_factory=lambda: ['<|return|>'])

class GptTemplate(ChatTemplateProcessor):
    support_padding_free = False

    def _get_gpt_oss_prefix(self):
        today = datetime.now().strftime('%Y-%m-%d')
        return ('<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n'
                f'Knowledge cutoff: 2024-06\nCurrent date: {today}\n\nReasoning: medium\n\n'
                '# Valid channels: analysis, commentary, final. '
                'Channel must be included for every message.<|end|>')

    def _swift_prepare_inputs(self, inputs: StdChatTemplateInputs):
        super()._swift_prepare_inputs(inputs)
        messages = inputs.messages
        if self.use_chat_template:
            if inputs.system is None:
                inputs.system = self._get_gpt_oss_prefix()
            elif not inputs.system.startswith('<|start|>'):
                inputs.system = self._get_gpt_oss_prefix() + (
                    f'<|start|>developer<|message|># Instructions\n\n{inputs.system}<|end|>')
            for i, message in enumerate(messages):
                if message['role'] == 'assistant' and isinstance(message['content'], str):
                    if not message['content'].startswith('<|channel|>'):
                        message['content'] = '<|channel|>final<|message|>' + message['content']



register_chat_template(
    GptOssTemplateMeta(ChatTemplateType.gpt_oss, template_processor_cls=GptTemplate))
