import copy
import inspect
import os
import re
from contextlib import contextmanager
from functools import wraps
from types import MethodType
from typing import Dict, Optional, Union, Tuple

import accelerate
import torch
import transformers
from peft import PeftModel
from torch import nn
from torch.distributed._composable.replicate import DDP
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from transformers import PreTrainedModel, trainer, dynamic_module_utils
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from surogate.core.model.hf_config import HfConfigFactory
from surogate.core.model.model_info import ModelInfo
from surogate.core.model.registry import ModelTemplate
from surogate.utils.dist import is_mp_ddp, get_device_count, get_dist_setting, safe_ddp_context, is_mp
from surogate.utils.logger import get_logger
from surogate.utils.tensor import _get_max_memory, _sync_max_memory
from surogate.utils.utils import deep_getattr

logger = get_logger()

def patch_get_input_embeddings(model, embedding_keys: str):
    def get_input_embeddings(self) -> nn.Module:
        return deep_getattr(model, embedding_keys)

    model.get_input_embeddings = MethodType(get_input_embeddings, model)


def patch_awq_compat(model_info: ModelInfo):
    if model_info.quant_method != 'awq':
        return

    try:
        # compat transformers>=4.50 (autoawq)
        from transformers.quantizers.quantizer_awq import AwqQuantizer
        from transformers.integrations import get_keys_to_not_convert
        _process_model_before_weight_loading = AwqQuantizer._process_model_before_weight_loading

        def _new_process_model_before_weight_loading(self, model, *args, **kwargs):
            modules_to_not_convert = self.quantization_config.modules_to_not_convert
            if modules_to_not_convert is not None:
                self.quantization_config.modules_to_not_convert = list(
                    modules_to_not_convert) + get_keys_to_not_convert(model)
            return _process_model_before_weight_loading(self, model, *args, **kwargs)

        AwqQuantizer._process_model_before_weight_loading = _new_process_model_before_weight_loading
    except Exception:
        pass


@contextmanager
def patch_automodel_for_sequence_classification(
        model_info: ModelInfo = None,
        model_template: ModelTemplate = None,
        patch_from_pretrained=True,
        patch_missing_init=True,
        **kwargs
):
    """
    Context manager for patching AutoModel sequence classification.

    Args:
        model_info: Model information
        model_template: Model metadata
        patch_from_pretrained (bool): Whether to patch PreTrainedModel.from_pretrained
        patch_missing_init (bool): Whether to patch missing __init__ methods
        **kwargs: Additional keyword arguments
    """
    model_config = kwargs.get('model_config', None)
    from_pretrained = PreTrainedModel.from_pretrained.__func__

    # Patch 1: from_pretrained method
    _new_from_pretrained = None
    if patch_from_pretrained:

        @classmethod
        def _new_from_pretrained(cls, *args, **kwargs):
            __init__ = cls.__init__

            def __new_init__(self, *args, **kwargs):
                __init__(self, *args, **kwargs)
                _patch_sequence_classification(self, model_template)

            cls.__init__ = __new_init__
            if hasattr(cls, '_tp_plan'):  # fix tp_plan
                cls._tp_plan = cls._tp_plan or {}
            res = from_pretrained(cls, *args, **kwargs)
            cls.__init__ = __init__
            return res

    # Patch 2: missing __init__ methods
    patched_classes = []
    if patch_missing_init:

        def get_all_subclasses(cls, include_root=True):
            subclass_list = []

            def recurse(cl):
                for subclass in cl.__subclasses__():
                    subclass_list.append(subclass)
                    recurse(subclass)

            recurse(cls)

            ret = set(subclass_list)
            if include_root:
                ret.add(cls)
            return ret

        def create_default_init(cls):
            """Create a default __init__ method that calls super().__init__"""

            def default_init(self, *args, **kwargs):
                super(cls, self).__init__(*args, **kwargs)

            return default_init

        if model_config is not None:
            # we should import in advance so that get_all_subclasses can find the class
            archs = model_config.architectures
            for arch in archs:
                try:
                    getattr(transformers, arch)
                except AttributeError:
                    continue

        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            if '__init__' not in subclass.__dict__:
                subclass.__init__ = create_default_init(subclass)
                patched_classes.append(subclass)

    if patch_from_pretrained:
        PreTrainedModel.from_pretrained = _new_from_pretrained

    try:
        yield
    finally:
        # Restore patches
        if patch_from_pretrained:
            PreTrainedModel.from_pretrained = classmethod(from_pretrained)

        if patch_missing_init:
            for subclass in patched_classes:
                try:
                    if '__init__' in subclass.__dict__:
                        del subclass.__init__
                except (AttributeError, TypeError):
                    pass


def _patch_sequence_classification(model, model_meta):
    hidden_size = HfConfigFactory.get_config_attr(model.config, 'hidden_size')
    initializer_range = HfConfigFactory.get_config_attr(model.config, 'initializer_range')

    lm_heads = ['lm_head', 'output', 'embed_out', 'output_layer']
    llm_model = get_lm_head_model(model, model_meta, lm_heads)
    llm_model.num_labels = model.config.num_labels
    for lm_head in lm_heads:
        if hasattr(llm_model, lm_head):
            hidden_size = getattr(llm_model, lm_head).in_features
            setattr(llm_model, lm_head, nn.Identity())
            break
    llm_model.score = nn.Linear(hidden_size, llm_model.num_labels, bias=False, dtype=llm_model.dtype)
    if llm_model.score.weight.device == torch.device('meta'):
        llm_model.score.to_empty(device='cpu')
    llm_model.score.weight.data.normal_(mean=0.0, std=initializer_range)

    origin_forward = llm_model.forward

    @wraps(origin_forward.__func__)
    def new_forward(self, *args, **kwargs):
        return transformers_seq_cls_forward(self, *args, origin_forward=origin_forward, **kwargs)

    llm_model.forward = MethodType(new_forward, llm_model)


def transformers_seq_cls_forward(self, *args, origin_forward, padding_side=None, **kwargs):
    labels = kwargs.pop('labels', None)
    return_dict = kwargs.pop('return_dict', None)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    input_ids = kwargs.get('input_ids')
    inputs_embeds = kwargs.get('inputs_embeds')

    output = origin_forward(*args, **kwargs)
    if hasattr(output, 'logits'):
        output.logits = output.logits.to(self.score.weight.dtype)
    elif 'last_hidden_state' in output:
        output.logits = output['last_hidden_state'].to(self.score.weight.dtype)
    logits = self.score(output.logits)
    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]

    if padding_side == 'left':
        pooled_logits = logits[:, -1]
    else:
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError('Cannot handle batch sizes > 1 if no padding token is defined.')
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if output.get('attention_mask') is not None:
                # When use padding_free in seq_cls tasks, `revert_padding_free` will add a attention_mask in the output
                batch_size = output.get('attention_mask').shape[0]
                sequence_lengths = output.get('attention_mask').sum(dim=1) - 1
            elif input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            elif kwargs.get('attention_mask') is not None:
                sequence_lengths = kwargs['attention_mask'].sum(dim=1) - 1
            else:
                sequence_lengths = -1
        if isinstance(sequence_lengths, torch.Tensor):
            sequence_lengths = sequence_lengths.to(logits.device)

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

    loss = None
    if labels is not None:
        labels = labels.to(logits.device)
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = 'regression'
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = 'single_label_classification'
            else:
                self.config.problem_type = 'multi_label_classification'

        if self.config.problem_type == 'regression':
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(pooled_logits, labels)
        elif self.config.problem_type == 'single_label_classification':
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == 'multi_label_classification':
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(pooled_logits, labels)
    if not return_dict:
        output = (pooled_logits,) + output[1:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutputWithPast(
        loss=loss,
        logits=pooled_logits,
        past_key_values=output.past_key_values,
        hidden_states=output.hidden_states,
        attentions=output.attentions,
    )


def get_lm_head_model(model, model_template: ModelTemplate = None, lm_heads=None):
    if isinstance(model, PeftModel):
        model = model.model
    model_template = model_template or model.model_template
    lm_heads = lm_heads or ['lm_head']
    llm_prefix_list = getattr(model_template.model_arch, 'language_model', None)
    prefix_list = []
    if llm_prefix_list:
        prefix_list = llm_prefix_list[0].split('.')

    current_model = model
    for prefix in prefix_list:
        current_model = getattr(current_model, prefix)
        for lm_head in lm_heads:
            if hasattr(current_model, lm_head):
                return current_model
    return model


@contextmanager
def patch_automodel(model_info, model_template, automodel_class, return_dummy_model, **kwargs):
    from_pretrained = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def _new_from_pretrained(cls, *args, **kwargs):
        if 'AutoAWQFor' in automodel_class.__name__:
            kwargs.pop('use_cache', None)
        if model_info.quant_method == 'gptq':
            cls.main_input_name = 'input_ids'
        if hasattr(cls, '_tp_plan'):  # fix tp_plan
            cls._tp_plan = cls._tp_plan or {}
        if return_dummy_model:
            origin_torch_dtype = torch.get_default_dtype()
            torch.set_default_dtype(kwargs['config'].torch_dtype)
            model = cls(copy.deepcopy(kwargs['config']))
            torch.set_default_dtype(origin_torch_dtype)
        else:
            model = from_pretrained(cls, *args, **kwargs)
        return model

    PreTrainedModel.from_pretrained = _new_from_pretrained

    try:
        yield
    finally:
        PreTrainedModel.from_pretrained = classmethod(from_pretrained)


def patch_output_normalizer(module: torch.nn.Module, model_template: ModelTemplate):
    def lm_head_forward(self, hidden_states):
        return hidden_states

    lm_heads = ['lm_head', 'output', 'embed_out', 'output_layer']
    llm_model = get_lm_head_model(module, model_template=model_template)

    found = False
    for lm_head in lm_heads:
        if hasattr(llm_model, lm_head):
            getattr(llm_model, lm_head).forward = MethodType(lm_head_forward, getattr(llm_model, lm_head))
            found = True
            break

    assert found, 'Cannot find the proper lm_head name'

    def _output_embedding_hook(module, args, kwargs, output):
        attention_mask = kwargs['attention_mask']
        hidden_states = output.logits
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            embeddings = hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            embeddings = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return {
            'last_hidden_state': embeddings.contiguous(),
        }

    llm_model.register_forward_hook(_output_embedding_hook, with_kwargs=True)

def deepspeed_set_z3_leaf_modules(model, z3_leaf_modules):
    if not is_deepspeed_zero3_enabled():
        return
    try:
        architecture = model.config.architectures[0]
    except Exception:
        return
    if z3_leaf_modules is None:
        if architecture == 'Qwen3VLMoeForConditionalGeneration':
            from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock
            z3_leaf_modules = [Qwen3VLMoeTextSparseMoeBlock]
        elif architecture == 'Qwen3OmniMoeForConditionalGeneration':
            from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextSparseMoeBlock
            z3_leaf_modules = [Qwen3OmniMoeThinkerTextSparseMoeBlock]
        elif architecture == 'Qwen2MoeForCausalLM':
            from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
            z3_leaf_modules = [Qwen2MoeSparseMoeBlock]
        elif architecture == 'Qwen3MoeForCausalLM':
            from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
            z3_leaf_modules = [Qwen3MoeSparseMoeBlock]
        elif architecture == 'Glm4MoeForCausalLM':
            from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE
            z3_leaf_modules = [Glm4MoeMoE]
        elif architecture == 'Glm4vMoeForConditionalGeneration':
            from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeTextMoE
            z3_leaf_modules = [Glm4vMoeTextMoE]
        elif architecture == 'GptOssForCausalLM':
            from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
            z3_leaf_modules = [GptOssMLP]
        elif architecture == 'Llama4ForCausalLM':
            from transformers.models.llama4.modeling_llama4 import Llama4TextMoe
            z3_leaf_modules = [Llama4TextMoe]
        elif architecture == 'Qwen3NextForCausalLM':
            from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock
            z3_leaf_modules = [Qwen3NextSparseMoeBlock]

    if z3_leaf_modules:
        from deepspeed.utils import set_z3_leaf_modules
        set_z3_leaf_modules(model, z3_leaf_modules)
        logger.info(f'Setting z3_leaf_modules: {z3_leaf_modules}')


_mp_ddp_patched = False

def patch_mp_ddp():
    """Patch ddp with device_map.
    After patching, the ddp can run with the device_map.
    This should be called before any training starts.
    """
    global _mp_ddp_patched
    if _mp_ddp_patched:
        return
    _mp_ddp_patched = True
    if is_mp_ddp():
        from accelerate.utils.modeling import get_balanced_memory, infer_auto_device_map

        @wraps(infer_auto_device_map)
        def _infer_auto_device_map_patch(model: nn.Module,
                                         max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
                                         **kwargs) -> Dict[str, Union[int, str, torch.device]]:
            """The auxiliary function for supports MP + DDP. Monkey Patching.
            add feat in accelerate to support MP + DDP"""
            verbose = kwargs.pop('verbose', False)
            n_gpu = get_device_count()
            _, local_rank, _, local_world_size = get_dist_setting()
            device_ids = list(range(local_rank, n_gpu, local_world_size))
            max_memory = _get_max_memory(device_ids)
            max_memory = _sync_max_memory(max_memory)
            max_memory = get_balanced_memory(model, max_memory, low_zero=False, **kwargs)
            max_memory = {k: v for k, v in max_memory.items() if v > 0}
            return infer_auto_device_map(model, max_memory, verbose=verbose, **kwargs)

        _old_ddp_init = DDP.__init__
        accelerate.accelerator.torch.nn.parallel.DistributedDataParallel.__init__ = (
            lambda self, model, device_ids, output_device, *args, **kwargs: _old_ddp_init(self, model, *args, **kwargs))
        transformers.modeling_utils.get_balanced_memory = lambda *args, **kwargs: {}
        transformers.modeling_utils.infer_auto_device_map = _infer_auto_device_map_patch

        _old_accelerator_init = trainer.Accelerator.__init__
        trainer.Accelerator.__init__ = (lambda self, device_placement=False, *args, **kwargs: _old_accelerator_init(
            self, device_placement=device_placement, *args, **kwargs))
        trainer.Accelerator.verify_device_map = lambda *args, **kwargs: False

@contextmanager
def patch_get_dynamic_module():
    origin_get_cached_module_file = dynamic_module_utils.get_cached_module_file

    def new_get_cached_module_file(pretrained_model_name_or_path, *args, **kwargs):
        with safe_ddp_context(hash_id=str(pretrained_model_name_or_path)):
            return origin_get_cached_module_file(pretrained_model_name_or_path, *args, **kwargs)

    dynamic_module_utils.get_cached_module_file = new_get_cached_module_file
    try:
        yield
    finally:
        dynamic_module_utils.get_cached_module_file = origin_get_cached_module_file

@contextmanager
def patch_tp_plan(load_model: bool):
    if not load_model or not is_mp() or 'WORLD_SIZE' not in os.environ:
        yield
        return
    logger.info_once('Patch tp_plan.')
    WORLD_SIZE = os.environ.get('WORLD_SIZE')
    os.environ['_PATCH_WORLD_SIZE'] = WORLD_SIZE
    os.environ.pop('WORLD_SIZE')
    yield
    os.environ['WORLD_SIZE'] = WORLD_SIZE

def patch_getattr(obj_cls, item_name: str):
    if hasattr(obj_cls, '_patch'):  # avoid double patch
        return

    def __new_getattr__(self, key: str):
        try:
            return super(self.__class__, self).__getattr__(key)
        except AttributeError:
            if item_name in dir(self):
                item = getattr(self, item_name)
                return getattr(item, key)
            raise

    obj_cls.__getattr__ = __new_getattr__
    obj_cls._patch = True

def detab_code(code: str) -> Tuple[str, str]:
    try:
        spaces = re.match(r"([\s\t]{1,})", code).group(0)
        code = re.sub(r"^" + spaces, "", code, flags=re.MULTILINE)
    except AttributeError:
        return code, ""
    return code, spaces


@contextmanager
def patch_attach_align_device_hook_on_blocks():
    from accelerate import big_modeling
    origin_attach_align_device_hook_on_blocks = big_modeling.attach_align_device_hook_on_blocks

    def attach_align_device_hook_on_blocks(*args, **kwargs):
        return

    big_modeling.attach_align_device_hook_on_blocks = attach_align_device_hook_on_blocks
    try:
        yield
    finally:
        big_modeling.attach_align_device_hook_on_blocks = origin_attach_align_device_hook_on_blocks

# Patches models to add RoPE Scaling
def patch_linear_scaling(
        model_name = None,
        rope_module = None,
        scaled_rope_module = None,
        attention_module = None,
):
    assert rope_module is not None and scaled_rope_module is not None
    assert attention_module is not None

    model_filepath = f"transformers.models.{model_name}.modeling_{model_name}"
    exec_code = (
        f"import torch.nn as nn\n"
        f"from typing import Union, Optional, List, Any, Callable, Tuple\n"
        f"from {model_filepath} import logger, "
        f"{model_name.title()}Attention, {model_name.title()}Config"
    )

    try:
        function = inspect.getsource(attention_module.__init__)
    except:
        # Most likely already patched!
        return None, None
    where = function.find("def")
    function = function.split("\n")
    function = "\n".join(x[where:] for x in function)
    init_name = f"{model_name.title()}Attention__init__"
    function = function.replace("def __init__", f"def {init_name}")
    function = function.replace(
        "super().__init__()",
        f"super({model_name.title()}Attention, self).__init__()",
    )
    fix_rope_function = """
    if getattr(self.config, "rope_scaling", None) is None:
        self.rotary_emb = {rope_function}(
            dim = self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    else:
        scaling_type = self.config.rope_scaling["type"]
        scaling_factor = self.config.rope_scaling["factor"]
        if scaling_type == "linear":
            self.rotary_emb = {scaled_rope_function}(
                dim = self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {{scaling_type}}")
    pass
    """
    fix_rope_function = fix_rope_function.format(
        rope_function = rope_module.__name__,
        scaled_rope_function = scaled_rope_module.__name__,
    )
    rotary_emb = re.findall(
        r"self\.rotary\_emb \= .+?\)",
        function,
        flags = re.DOTALL | re.MULTILINE,
    )
    if len(rotary_emb) == 0:
        return None, exec_code + "\n\n" + function

    rotary_emb = rotary_emb[0]
    function = function.replace(rotary_emb, fix_rope_function, 1)
    function = exec_code + "\n\n" + function
    return init_name, function