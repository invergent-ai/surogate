import json
import os
import re
from typing import Union, List, Callable, Optional

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel, get_peft_model, PeftModelForCausalLM
from peft.utils import CONFIG_NAME
from torch import nn

from surogate.core.config.sft_config import SFTConfig
from surogate.core.model.peft_patcher import patch_peft_model
from surogate.train.cross_entropy import apply_cross_entropy_patch
from surogate.train.peft import LoraConfig
from surogate.utils.logger import get_logger
from surogate.utils.utils import deep_getattr

logger = get_logger()


class TrainUtils:
    @classmethod
    def prepare_model(
            cls,
            config: SFTConfig,
            model,
            *,
            task_type=None
    ):
        apply_cross_entropy_patch(config)

        if config.resume_from_checkpoint:
            model = cls.from_pretrained(model, config.resume_from_checkpoint, is_trainable=True)
        else:
            model = cls.prepare_adapter(config, model, task_type=task_type)
            model = patch_peft_model(model)

        # fix bug: Attempting to unscale FP16 gradients.
        #   peft: https://github.com/huggingface/peft/issues/1249
        for p in model.parameters():
            if p.requires_grad and p.dtype == torch.float16:
                logger.info_once('Convert trainable parameters from fp16 to fp32.')
                p.data = p.data.to(dtype=torch.float32)

        return model

    @classmethod
    def from_pretrained(
            cls,
            model: Union[nn.Module, PeftModel],
            model_id: str = None,
            **kwargs
    ):
        if not os.path.exists(model_id):
            model_id = snapshot_download(model_id)

        is_peft_model = False
        if os.path.exists(os.path.join(model_id, CONFIG_NAME)):
            with open(os.path.join(model_id, CONFIG_NAME), 'r', encoding='utf-8') as f:
                _json = json.load(f)
                is_peft_model = True

        if is_peft_model:
            peft_model = cls._load_peft_model(model, model_id, 'default', **kwargs)
            for _dir in os.listdir(model_id):
                if os.path.isdir(os.path.join(model_id, _dir)) and \
                        os.path.exists(os.path.join(model_id, _dir, CONFIG_NAME)):
                    peft_model = cls._load_peft_model(peft_model, model_id, _dir, **kwargs)
            return peft_model
        else:
            return model

    @classmethod
    def _load_peft_model(cls, model, model_id, adapter_name, **kwargs):
        import peft

        if not isinstance(model, peft.PeftModel):
            model_id = os.path.join(model_id, adapter_name) \
                if adapter_name != 'default' and os.path.exists(os.path.join(model_id, adapter_name)) \
                else model_id
            return PeftModel.from_pretrained(model, model_id, adapter_name=adapter_name, **kwargs)
        else:
            model_id = os.path.join(model_id, adapter_name) \
                if adapter_name != 'default' and os.path.exists(os.path.join(model_id, adapter_name)) \
                else model_id
            model.load_adapter(model_id, adapter_name)
            return model

    @classmethod
    def prepare_adapter(cls, config: SFTConfig, model, *, task_type=None):
        task_type = task_type.upper() if task_type is not None else None

        target_modules = cls._get_target_modules(config, model)
        modules_to_save = cls._get_modules_to_save(config, model, task_type)
        lora_kwargs = {
            'r': config.lora_rank,
            'target_modules': target_modules,
            'lora_alpha': config.lora_alpha,
            'lora_dropout': config.lora_dropout,
            'bias': config.lora_bias,
            'modules_to_save': modules_to_save,
        }

        if task_type == 'EMBEDDING':
            task_type = None
        elif task_type == 'RERANKER':
            task_type = 'SEQ_CLS'
        elif task_type == 'GENERATIVE_RERANKER':
            task_type = 'CAUSAL_LM'

        lora_config = LoraConfig(task_type=task_type, lora_dtype=None, **lora_kwargs)
        model = get_peft_model(model, lora_config)
        return model

    @classmethod
    def _get_target_modules(cls, config: SFTConfig, model) -> Union[str, List[str]]:
        """Replace all-linear to actual modules"""
        model_template = config.model_template
        if isinstance(config.lora_target_modules, str):
            return config.lora_target_modules
        target_modules = config.lora_target_modules.copy()
        if 'all-linear' in target_modules:
            if model_template.is_multimodal:
                return cls._get_multimodal_target_regex(
                    config,
                    model,
                    freeze_llm=False,
                    freeze_vit=True,
                    freeze_aligner=True,
                    include_embedding='all-embedding' in target_modules)
            else:
                target_modules.remove('all-linear')
                target_modules += cls._find_all_linears(config, model)
        if 'all-embedding' in target_modules:
            target_modules.remove('all-embedding')
            target_modules += cls._find_embedding(model)
        return target_modules

    @classmethod
    def _get_modules_to_save(cls, config: SFTConfig, model, task_type=None):
        modules_to_save = config.modules_to_save.copy()
        if 'all-embedding' in config.modules_to_save:
            modules_to_save.remove('all-embedding')
            modules_to_save += cls._find_embedding(model)
        if 'all-norm' in config.modules_to_save:
            modules_to_save.remove('all-norm')
            modules_to_save += cls._find_norm(model)
        if task_type and task_type.lower() == 'seq_cls':  # reward_model
            modules_to_save.append('v_head')
        return modules_to_save

    @classmethod
    def _get_multimodal_target_regex(
            cls,
            config: SFTConfig,
            model,
            *,
            freeze_llm: bool = False,
            freeze_vit: bool = True,
            freeze_aligner: bool = True,
            include_embedding: bool = False,
            exclude_router: bool = False,
    ) -> str:
        model_arch = config.model_template.model_arch
        modules = []
        if not freeze_llm:
            modules += model_arch.language_model
        if not freeze_vit:
            modules += model_arch.vision_tower
        if not freeze_aligner:
            modules += model_arch.aligner
        assert len(modules) > 0, f'modules: {modules}'

        extra_layers = []
        if include_embedding:
            extra_layers.append(nn.Embedding)
        res = []
        for module in modules:
            rejected_modules = []
            if not freeze_vit or not freeze_llm:
                for aligner in model_arch.aligner:
                    if aligner.startswith(f'{module}.'):
                        rejected_modules.append(aligner)

            sub_module = deep_getattr(model, module)
            if isinstance(sub_module, nn.Linear) and module.endswith('lm_head'):
                target_modules = []
            else:
                target_modules = cls._find_all_linears(config, sub_module, model_arch, extra_layers)
            if exclude_router and model.model_info.is_moe_model:
                target_modules = [tm for tm in target_modules if tm not in {'gate'}]
            if not target_modules:
                continue
            target_modules = [tm for tm in target_modules if tm]
            target_pattern = rf'.*\.({"|".join(target_modules)})' if target_modules else ''
            rejected_pattern = rf'(?!({"|".join(rejected_modules)}))' if rejected_modules else ''
            res.append(rf'{rejected_pattern}{module}{target_pattern}')

        return rf'^({"|".join(res)})$'

    @classmethod
    def _find_all_linears(cls, config: SFTConfig, model, model_arch=None, extra_layers=None, sub_module=None):
        if model_arch is None:
            model_arch = config.model_template.model_arch
        # lm_head
        if model_arch and model_arch.lm_head:
            output = model_arch.lm_head
            idx = output.rfind('.')
            lm_head_name = output[idx + 1:]
        else:
            lm_head_name = 'lm_head'
        # 'score', 'classifier': classification model
        # 'v_head': reward model
        ignore_layers = [lm_head_name, 'score', 'v_head', 'classifier'] + ['lora_A', 'lora_B', 'base_layer']
        ignore_linear_cls = [
            'glulinear'  # phi4-mm
        ]

        def _cond(name, module):
            module_name = module.__class__.__name__.lower()
            if (extra_layers and isinstance(module, tuple(extra_layers)) or
                ('linear' in module_name and all(linear_cls not in module_name
                                                 for linear_cls in ignore_linear_cls))) and all(layer not in name
                                                                                                for layer in
                                                                                                ignore_layers):
                return True
            return False

        return cls._find_layers(model, _cond, sub_module=sub_module)

    @classmethod
    def _find_layers(
            cls,
            model: nn.Module,
            cond: Callable[[str, nn.Module], bool],
            sub_module: Optional[str] = None,
            min_name_len: Optional[int] = None,
    ) -> List[str]:
        # The content of target_module_names cannot exist in inner_nodes.
        sub_module_str = sub_module
        if sub_module is None:
            sub_module = model
        else:
            sub_module = deep_getattr(model, sub_module)
        inner_nodes = set()
        for name, module in model.named_modules():
            name = re.sub(r'\d+\.', '{}.', name)
            if not cond(name, module):
                inner_nodes.add(name)
        target_module_names = set()
        for name, module in sub_module.named_modules():
            if sub_module_str:
                name = f'{sub_module_str}.{name}' if name else sub_module_str
            if cond(name, module):
                module_name_list = name.split('.')
                module_name = module_name_list.pop()
                i = 1
                for inner_node in inner_nodes:
                    while module_name_list and inner_node.endswith(re.sub(
                            r'\d+\.', '{}.', module_name)) or min_name_len and i < min_name_len:
                        module_name = f'{module_name_list.pop()}.{module_name}'
                        i += 1
                target_module_names.add(module_name)
        return list(target_module_names)

    @classmethod
    def _find_embedding(cls, model: nn.Module) -> List[str]:
        return cls._find_layers(model, lambda name, module: isinstance(module, torch.nn.Embedding))

    @classmethod
    def _find_norm(cls, model: nn.Module) -> List[str]:
        # find_layer_norm
        return cls._find_layers(
            model,
            lambda name, module: isinstance(module,
                                            torch.nn.LayerNorm) or 'rmsnorm' in module.__class__.__name__.lower())
