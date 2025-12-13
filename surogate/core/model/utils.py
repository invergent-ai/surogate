import os
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, GenerationConfig, BaseImageProcessor, \
    FeatureExtractionMixin
from transformers import ProcessorMixin as HfProcessorMixin
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_torch_bf16_gpu_available, is_torch_cuda_available

from surogate.core.model.hf_config import HfConfigFactory
from surogate.core.model.registry import ModelTemplate
from surogate.utils.dist import get_dist_setting, is_mp
from surogate.utils.logger import get_logger
from surogate.utils.utils import deep_getattr

logger = get_logger()

Processor = Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, HfProcessorMixin]
Prompt = List[Union[str, List[int], List[str]]]
Word = Union[str, List[int]]
Context = Word


class ContextType:
    RESPONSE = 'response'
    SUFFIX = 'suffix'
    OTHER = 'other'

def get_default_torch_dtype(torch_dtype: Optional[torch.dtype]):
    # torch_dtype: torch_dtype in config.json
    if torch_dtype is not None:
        return torch_dtype

    try:
        is_bf16_available = is_torch_bf16_gpu_available()
    except:  # noqa
        is_bf16_available = False

    if is_torch_cuda_available():
        if is_bf16_available:
            return torch.bfloat16
        else:
            return torch.float16
    else:
        # cpu
        return torch.float32


def get_default_device_map():
    if is_deepspeed_zero3_enabled() or os.environ.get('ACCELERATE_USE_FSDP', 'False') == 'true':
        return None
    local_rank = get_dist_setting()[1]
    if local_rank == -1:
        local_rank = 0
    if is_torch_cuda_available():
        return 'auto' if is_mp() else f'cuda:{local_rank}'
    else:
        return 'cpu'


def fix_do_sample_warning(generation_config: GenerationConfig) -> None:
    # Use the default values of temperature/top_p/top_k in generation_config.
    if generation_config.temperature == 0:
        generation_config.do_sample = False
    if generation_config.do_sample is False:
        generation_config.temperature = 1.
        generation_config.top_p = 1.
        generation_config.top_k = 50


def find_module_list(model) -> Optional[nn.ModuleList]:
    module_lists = []
    for m in model.modules():
        if hasattr(m, 'gradient_checkpointing') or m.__class__.__name__ == 'CheckpointWrapper':
            return
        if (isinstance(m, (nn.ModuleList, nn.Sequential)) and len(m) >= 10
                and 'mlp' not in m[0].__class__.__name__.lower()):  # fix moe
            module_lists.append(m)
    if module_lists:
        return max(module_lists, key=lambda x: len(x))


def update_generation_config_eos_token(generation_config, template: 'ChatTemplateProcessor'):
    if generation_config is None:
        return
    stop_words = template.chat_template.stop_words
    eos_token_id = generation_config.eos_token_id
    if eos_token_id is None:
        eos_token_id = []
    elif isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    modified = False
    for stop_word in stop_words:
        if stop_word is None:
            continue
        if isinstance(stop_word, str):
            stop_word = template._tokenize(stop_word)
        if isinstance(stop_word, (list, tuple)) and len(stop_word) == 1 and stop_word[0] not in eos_token_id:
            eos_token_id.append(stop_word[0])
            modified = True
    if modified:
        generation_config.eos_token_id = eos_token_id


def get_causal_lm_model_cls_prefix(model_type: str) -> Tuple[str, str]:
    if model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        causal_lm_cls = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type]
        causal_lm_cls_prefix = causal_lm_cls
        for suffix in [
            "ForCausalLM",
            "ForConditionalGeneration",
            "LMHeadModel",
            "GenerationDecoder",
        ]:
            causal_lm_cls_prefix = causal_lm_cls_prefix.replace(suffix, "")
        return causal_lm_cls_prefix, causal_lm_cls
    causal_lm_cls_prefix = "".join(
        [part.capitalize() for part in model_type.split("_")]
    )
    return causal_lm_cls_prefix, f"{causal_lm_cls_prefix}ForCausalLM"


def check_tie_word_embeddings(model):
    config = model.config
    try:
        from peft.utils import ModulesToSaveWrapper
        if not HfConfigFactory.get_config_attr(config, 'tie_word_embeddings'):
            return
        for module in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if not isinstance(module, ModulesToSaveWrapper):
                return
        HfConfigFactory.set_config_attr(config, 'tie_word_embeddings', False)
    except Exception:
        pass


def get_llm_module_from_multimodal(model: torch.nn.Module, model_template: ModelTemplate = None, inner_backbone=True):
    """Get LLM model, this function can be used to get the llm module from a multi-modal model.

    Args:
        model: The model instance
        model_meta: The model_meta information
        inner_backbone: Get inner backbone model, like `QwenModel` or `LlamaModel`

    Returns:

    """
    from peft import PeftModel
    from accelerate.utils import extract_model_from_parallel
    model = extract_model_from_parallel(model)

    if isinstance(model, PeftModel):
        model = model.model
    if model_template is None:
        model_template = model.model_template

    llm_prefix = getattr(model_template.model_arch, 'language_model', None)
    if llm_prefix:
        llm_model = deep_getattr(model, llm_prefix[0])
    else:
        llm_model = model

    if inner_backbone:
        if hasattr(llm_model, 'thinker'):
            llm_model = llm_model.thinker.model
        elif hasattr(llm_model, 'model'):
            llm_model = llm_model.model
    return llm_model
