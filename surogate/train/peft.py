import json
import os
from dataclasses import dataclass, field, asdict
from types import MethodType
from typing import Optional, Dict

import peft
import torch
import transformers
from peft import LoraModel
from peft.config import PeftConfigMixin
from peft.tuners import lora

from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class LoraConfig(peft.LoraConfig):
    lora_dtype: Optional[str] = field(
        default=None, metadata={'help': 'The lora dtype, default None means following the original layer\'s dtype'})

    def to_peft_config(self) -> peft.LoraConfig:
        _dict = asdict(self)
        _dict.pop('lora_dtype')
        return peft.LoraConfig(**_dict)

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        self.to_peft_config().save_pretrained(save_directory, **kwargs)
        additional_args = {
            'lora_dtype': self.lora_dtype,
        }
        with open(os.path.join(save_directory, 'additional_config.json'), 'w', encoding='utf-8') as f:
            json.dump(additional_args, f)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, subfolder: Optional[str] = None, **kwargs):
        if hasattr(PeftConfigMixin, 'from_pretrained_origin'):
            self = PeftConfigMixin.from_pretrained_origin(pretrained_model_name_or_path, subfolder, **kwargs)
        else:
            self = super(LoraConfig, cls).from_pretrained(pretrained_model_name_or_path, subfolder, **kwargs)

        if type(self) == peft.LoraConfig:
            self = LoraConfig(**self.to_dict())

        if os.path.isfile(os.path.join(pretrained_model_name_or_path, 'additional_config.json')):
            with open(
                    os.path.join(pretrained_model_name_or_path, 'additional_config.json'), 'r', encoding='utf-8') as f:
                _json = json.load(f)
                for key, value in _json.items():
                    setattr(self, key, value)

        return self


def _create_and_replace_hook(self, peft_config, adapter_name, target, *args, **kwargs):
    all_supported_names = ('linear',)
    all_supported_types = (torch.nn.Embedding, torch.nn.Conv2d, transformers.pytorch_utils.Conv1D, lora.Linear)
    target_modules = getattr(peft_config, 'target_modules', None)
    target_parameters = getattr(peft_config, 'target_parameters', None)
    if target is None:
        return

    if isinstance(target_modules, str) and not any(
            [name in target.__class__.__name__.lower()
             for name in all_supported_names]) and not any([isinstance(target, type_)
                                                            for type_ in
                                                            all_supported_types]) and not target_parameters:
        return

    if target.__class__.__name__ == 'NonDynamicallyQuantizableLinear':
        return

    return self._create_and_replace_origin(peft_config, adapter_name, target, *args, **kwargs)


def keep_device_forward(self, *args, **kwargs):
    x = args[0]
    weight = self.weight if hasattr(self, 'weight') else self.weight0  # compat megatron
    if weight.device != x.device:
        return self.forward_origin(x.to(weight.device), *args[1:], **kwargs)
    else:
        return self.forward_origin(*args, **kwargs)


def _convert_dtype(target: torch.nn.Module, adapter_name: str, lora_dtype: str):
    if lora_dtype is not None:
        torch_dtype = getattr(torch, lora_dtype)
        if hasattr(target, 'lora_A') and adapter_name in target.lora_A:
            target.lora_A[adapter_name].to(torch_dtype)
            target.lora_B[adapter_name].to(torch_dtype)
        if hasattr(target, 'lora_embedding_A') and adapter_name in target.lora_embedding_A:
            target.lora_embedding_A[adapter_name].to(torch_dtype)
            target.lora_embedding_B[adapter_name].to(torch_dtype)


def hot_patch_peft_module():
    from peft.tuners.lora import LoraLayer
    if hasattr(LoraModel, '_create_and_replace_origin'):
        return

    # Fix Lora does not support NonDynamicallyQuantizableLinear
    LoraModel._create_and_replace_origin = LoraModel._create_and_replace
    LoraModel._create_and_replace = _create_and_replace_hook

    def __new_init__(self, model: torch.nn.Module, config: Dict[str, LoraConfig], adapter_name: str):
        self.__init_origin__(model, config, adapter_name)
        active_adapters = self.active_adapter
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]

        for active_adapter in active_adapters:
            active_config = config[active_adapter] if isinstance(config, dict) else config
            if hasattr(active_config, 'lora_dtype'):
                for name, module in model.named_modules():
                    if isinstance(module, LoraLayer):
                        _convert_dtype(module, active_adapter, active_config.lora_dtype)
                        for lora in list(module.lora_A.values()) + list(module.lora_B.values()):
                            if not hasattr(lora, 'forward_origin'):
                                lora.forward_origin = lora.forward
                                lora.forward = MethodType(keep_device_forward, lora)

    LoraModel.__init_origin__ = LoraModel.__init__
    LoraModel.__init__ = __new_init__


hot_patch_peft_module()
