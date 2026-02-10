from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Tuple, List, Type, Any, Union

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from surogate.core.config.enums import MLLMModelType, RMModelType, RerankerModelType
from surogate.core.model.architecture import ModelArchitecture, MLLMComponents

MODEL_MAPPING: Dict[str, 'ModelTemplate'] = {}

@dataclass
class ModelTemplate:
    model_type: Optional[str]
    chat_template: Optional[str]
    get_function: Callable[..., Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]]
    model_arch: Optional[ModelArchitecture] = None
    model_components: Optional[MLLMComponents] = None
    architectures: List[str] = field(default_factory=list)
    additional_saved_files: List[str] = field(default_factory=list)
    torch_dtype: Optional[torch.dtype] = None
    is_multimodal: bool = False
    tags: List[str] = field(default_factory=list)
    task_type: Optional[str] = None

    def __post_init__(self):
        if self.chat_template is None:
            self.chat_template = 'dummy'

        if self.model_type in MLLMModelType.__dict__:
         self.is_multimodal = True


def register_model(model_template: ModelTemplate) -> None:
    from .architecture import get_model_architecture
    model_type = model_template.model_type
    if model_type in MODEL_MAPPING:
        raise ValueError(f'The `{model_type}` has already been registered in the MODEL_MAPPING.')
    if model_template.model_arch:
        model_template.model_components = get_model_architecture(model_template.model_arch)
    MODEL_MAPPING[model_type] = model_template

def get_matched_model_types(architectures: Optional[List[str]]) -> List[str]:
    """Get possible model_type."""
    architectures = architectures or ['null']
    if architectures:
        architectures = architectures[0]
    arch_mapping = _get_arch_mapping()
    return arch_mapping.get(architectures) or []

def _get_arch_mapping():
    res = {}
    for model_type, model_meta in MODEL_MAPPING.items():
        architectures = model_meta.architectures
        if not architectures:
            architectures.append('null')
        for arch in architectures:
            if arch not in res:
                res[arch] = []
            res[arch].append(model_type)
    return res