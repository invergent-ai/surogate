import inspect
from functools import wraps
from types import MethodType
from typing import Optional, List, Any

import torch
import torch.utils.checkpoint
from peft import PeftModel
from transformers import Trainer

from surogate.core.model.hf_config import HfConfigFactory
from surogate.core.model.registry import ModelTemplate
from surogate.core.model.utils import find_module_list
from surogate.utils.dist import is_dist
from surogate.utils.logger import get_logger
from surogate.utils.utils import deep_getattr

logger = get_logger()

class GradientCheckpointingMixin(Trainer):
    def _fix_gradient_checkpointing(self):
        # fix use_reentrant
        if hasattr(torch.utils.checkpoint, '_old_checkpoint'):  # avoid double patching
            return
        args = self.args
        if args.gradient_checkpointing_kwargs:
            use_reentrant_ = args.gradient_checkpointing_kwargs.get('use_reentrant')
        else:
            use_reentrant_ = None
        if use_reentrant_ is None:
            if is_dist() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                use_reentrant_ = False
            else:
                use_reentrant_ = True

        _old_checkpoint = torch.utils.checkpoint.checkpoint

        @wraps(_old_checkpoint)
        def _new_checkpoint(*args, use_reentrant=None, **kwargs):
            return _old_checkpoint(*args, use_reentrant=use_reentrant_, **kwargs)

        torch.utils.checkpoint._old_checkpoint = _old_checkpoint
        torch.utils.checkpoint.checkpoint = _new_checkpoint
        try:
            # Fix the old version of transformers.
            import transformers.modeling_utils
            transformers.modeling_utils.checkpoint = _new_checkpoint
        except (ImportError, AttributeError):
            pass

    def _prepare_gradient_checkpointing(self, model) -> None:
        HfConfigFactory.set_model_config_attr(model, 'use_cache', False)
        args = self.args

        if args.gradient_checkpointing:
            self.dynamic_gradient_checkpointing(model)

        gc_kwargs = {}
        parameters = inspect.signature(model.gradient_checkpointing_enable).parameters
        if 'gradient_checkpointing_kwargs' in parameters:
            gc_kwargs['gradient_checkpointing_kwargs'] = args.gradient_checkpointing_kwargs

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(**gc_kwargs)
            model.enable_input_require_grads()

        model_template = model.model_template
        model_arch = model_template.model_arch
        if model_template.is_multimodal and model_arch:
            for vision_tower_name in model_arch.vision_tower:
                vision_tower = deep_getattr(model, vision_tower_name)
                if hasattr(vision_tower, 'enable_input_require_grads'):
                    try:
                        if args.vit_gradient_checkpointing:
                            vision_tower.gradient_checkpointing_enable(**gc_kwargs)
                            vision_tower.enable_input_require_grads()
                        else:
                            vision_tower.gradient_checkpointing_disable()
                            vision_tower.disable_input_require_grads()
                    except (NotImplementedError, AttributeError) as e:
                        logger.warning(f'prepare gradient_checkpointing failed: {e}')

        # Avoid vit_gradient_checkpointing being overwritten by transformers.Trainer.gradient_checkpointing_enable.
        self.args.gradient_checkpointing = False

    def dynamic_gradient_checkpointing(self, model, including_vit: bool = True) -> None:
        if isinstance(model, PeftModel):
            model = model.model

        model_template: ModelTemplate = getattr(model, 'model_template', None)
        if model_template is not None and model_template.is_multimodal and model_template.model_arch:
            tower_names = model_template.model_arch.language_model.copy()
            if including_vit:
                tower_names += model_template.model_arch.vision_tower
        else:
            tower_names = [None]

        model.supports_gradient_checkpointing = True
        for tower_name in tower_names:
            if tower_name is None:
                model_tower = model
            else:
                model_tower = deep_getattr(model, tower_name)
            model_tower.supports_gradient_checkpointing = True
            module_list = find_module_list(model_tower)
            if module_list is None:
                continue
            self._add_gradient_checkpointing(module_list)
            logger.info(f'Automatically add gradient_checkpointing to {model_tower.__class__}.')

    def _add_gradient_checkpointing(self, module_list):
        requires_grad = None

        def _new_forward(self, *args, **kwargs):
            nonlocal requires_grad
            if requires_grad is None:
                requires_grad = any(p.requires_grad for p in self.parameters())

            new_args = self._kwargs_to_args(self.__old_forward, args, kwargs)
            if new_args is not None and self.gradient_checkpointing and self.training:
                if new_args and isinstance(new_args[0], torch.Tensor) and requires_grad and not new_args[0].requires_grad:
                    new_args[0].requires_grad_(True)
                layer_ret = self._gradient_checkpointing_func(self.__old_forward, *new_args)
                logger.info_once('Successfully using dynamic gradient checkpointing.')
            else:
                layer_ret = self.__old_forward(*args, **kwargs)
            return layer_ret

        for module in module_list:
            module.gradient_checkpointing = False
            if hasattr(module, '_old_forward'):  # device_map
                __old_forward = module._old_forward
                module._old_forward = MethodType(_new_forward, module)
            else:
                __old_forward = module.forward
                module.forward = MethodType(_new_forward, module)
            module.__old_forward = __old_forward

    def _kwargs_to_args(self, func, args, kwargs) -> Optional[List[Any]]:
        parameters = inspect.signature(func).parameters
        args = list(args)
        parameters = list(parameters.items())[len(args):]
        for key, param in parameters:
            if key in kwargs:
                args.append(kwargs[key])
            elif param.default != param.empty:
                args.append(param.default)
            else:
                return
        return args
