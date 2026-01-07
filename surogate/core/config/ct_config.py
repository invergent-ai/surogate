from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Literal

from surogate.core.model.chat_templates.base import CHAT_TEMPLATE_MAPPING
from surogate.core.model.chat_templates.processor import ChatTemplateProcessor, get_chat_template_processor
from surogate.core.model.registry import ModelTemplate
from surogate.core.model.utils import Processor
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class ChatTemplateConfig(ABC):
    # from parent config
    model_template: ModelTemplate = None

    """
    Configuration for chat templates.

    Args:
        template (Optional[str]): The chat template to use. Choices are defined in CHAT_TEMPLATE_MAPPING.
    """
    template: Optional[str] = field(
        default=None, metadata={'help': f'template choices: {list(CHAT_TEMPLATE_MAPPING.keys())}'})
    system: Optional[str] = None  # Override the default_system in the template.
    max_length: Optional[int] = None

    truncation_strategy: Optional[Literal['delete', 'left', 'right', 'split']] = None
    max_pixels: Optional[int] = None
    agent_template: Optional[str] = None
    norm_bbox: Literal['norm1000', 'none', None] = None
    use_chat_template: Optional[bool] = None

    # train
    padding_free: Optional[bool] = None
    padding_side: Optional[Literal['left', 'right']] = None
    loss_scale: Optional[str] = None
    sequence_parallel_size: Optional[int] = None

    # infer/deploy
    # thinking
    response_prefix: Optional[str] = None
    enable_thinking: Optional[bool] = None
    add_non_thinking_prefix: bool = True
    
    def __init__(self, cfg: DictDefault):
        self.template = cfg.get('template', None)
        self.system = cfg.get('system', None)
        self.max_length = cfg.get('max_length', None)
        self.truncation_strategy = cfg.get('truncation_strategy', 'delete')
        self.max_pixels = cfg.get('max_pixels', None)
        self.agent_template = cfg.get('agent_template', None)
        self.norm_bbox = cfg.get('norm_bbox', None)
        self.use_chat_template = cfg.get('use_chat_template', True)
        self.padding_free = cfg.get('padding_free', False)
        self.padding_side = cfg.get('padding_side', None)
        self.loss_scale = cfg.get('loss_scale', 'default')
        self.sequence_parallel_size = cfg.get('sequence_parallel_size', 1)
        self.response_prefix = cfg.get('response_prefix', None)

    def __post_init__(self):
        if self.template is None and getattr(self, 'model_template', None):
            self.template = self.model_template.chat_template
        if self.system is not None:
            self.system = self.system.replace('\\n', '\n')
        if self.response_prefix is not None:
            self.response_prefix = self.response_prefix.replace('\\n', '\n')
        if self.truncation_strategy is None:
            self.truncation_strategy = 'delete'
        if self.padding_side is None:
            self.padding_side = 'right'
                

    def get_template_kwargs(self):
        from surogate.core.config.sft_config import SFTConfig

        truncation_strategy = self.truncation_strategy
        if truncation_strategy == 'delete':
            truncation_strategy = 'raise'

        return {
            'default_system': self.system,
            'max_length': self.max_length,
            'truncation_strategy': truncation_strategy,
            'max_pixels': self.max_pixels,
            'agent_template': self.agent_template,
            'norm_bbox': self.norm_bbox,
            'use_chat_template': self.use_chat_template,
            # train
            'padding_free': self.padding_free,
            'padding_side': self.padding_side,
            'loss_scale': self.loss_scale,
            'sequence_parallel_size': self.sequence_parallel_size,
            # infer/deploy
            'response_prefix': self.response_prefix,
            'enable_thinking': self.enable_thinking,
            'add_non_thinking_prefix': self.add_non_thinking_prefix,
        }

    def get_template_processor(
            self, processor: Optional['Processor'],
            template_type: Optional[str] = None
    ) -> 'ChatTemplateProcessor':
        template_kwargs = self.get_template_kwargs()
        template_type = template_type or self.template
        return get_chat_template_processor(template_type, processor, **template_kwargs)
