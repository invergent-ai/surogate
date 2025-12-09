from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Tuple, List, Type

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from surogate.core.config.enums import MLLMModelType, RMModelType, RerankerModelType

MODEL_MAPPING: Dict[str, 'ModelTemplate'] = {}

GetModelTokenizerFunction = Callable[..., Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]]

@dataclass
class ModelTemplate:
    model_type: Optional[str]
    chat_template: Optional[str]
    get_function: GetModelTokenizerFunction
    model_arch: Optional[str] = None
    architectures: List[str] = field(default_factory=list)
    additional_saved_files: List[str] = field(default_factory=list)
    torch_dtype: Optional[torch.dtype] = None
    is_multimodal: bool = False
    is_reward: bool = False
    is_reranker: bool = False
    tags: List[str] = field(default_factory=list)
    task_type: Optional[str] = None
    attention_cls: Type[nn.Module] = None

    def __post_init__(self):
        if self.chat_template is None:
            self.chat_template = 'dummy'

        if self.model_type in MLLMModelType.__dict__:
         self.is_multimodal = True
        if self.model_type in RMModelType.__dict__:
            self.is_reward = True
        if self.model_type in RerankerModelType.__dict__:
            self.is_reranker = True


def register_model(model_template: ModelTemplate) -> None:
    from .architecture import get_model_architecture
    model_type = model_template.model_type
    if model_type in MODEL_MAPPING:
        raise ValueError(f'The `{model_type}` has already been registered in the MODEL_MAPPING.')
    if model_template.model_arch:
        model_template.model_arch = get_model_architecture(model_template.model_arch)
    MODEL_MAPPING[model_type] = model_template
