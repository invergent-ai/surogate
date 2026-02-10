import math
import os
from functools import partial
from typing import Optional, List, Union, Any, Dict, Tuple, Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig, AutoConfig, AutoTokenizer, AutoModelForCausalLM

from surogate.core.hub.huggingface import HuggingFaceHub
from surogate.core.model.attn import AttnImpl
from surogate.core.model.hf_config import HfConfigFactory
from surogate.core.model.model_info import ModelInfo
from surogate.core.model.registry import MODEL_MAPPING, ModelTemplate
from surogate.core.model.utils import fix_do_sample_warning, get_default_torch_dtype, \
    get_causal_lm_model_cls_prefix
from surogate.core.model.patcher import get_lm_head_model, \
    patch_getattr, \
    patch_automodel, patch_awq_compat
from surogate.utils.logger import get_logger

logger = get_logger()

def safe_snapshot_download(
        model_id_or_path: str,
        revision: Optional[str] = None,
        download_model: bool = True,
        hub_token: Optional[str] = None,
        ignore_patterns: Optional[List[str]] = None,
        check_local: bool = False,
        **kwargs
) -> str:
    if check_local:
        model_suffix = model_id_or_path.rsplit('/', 1)[-1]

        if os.path.exists(model_suffix):
            model_dir = os.path.abspath(os.path.expanduser(model_suffix))
            logger.info(f'Loading the model from local path: {model_dir}')
            return model_dir

    if ignore_patterns is None:
        ignore_patterns = [
            '*.zip', '*.gguf', '*.pth', '*.pt', 'consolidated*', 'onnx/*', '*.safetensors.md', '*.msgpack', '*.onnx',
            '*.ot', '*.h5'
        ]

    if not download_model:
        ignore_patterns += ['*.bin', '*.safetensors']

    hub = HuggingFaceHub()

    if model_id_or_path.startswith('~'):
        model_id_or_path = os.path.abspath(os.path.expanduser(model_id_or_path))
    model_path_to_check = '/'.join(model_id_or_path.split(':', 1))
    if os.path.exists(model_id_or_path):
        model_dir = model_id_or_path
        sub_folder = None
    elif os.path.exists(model_path_to_check):
        model_dir = model_path_to_check
        sub_folder = None
    else:
        if model_id_or_path.startswith('/'):  # startswith
            raise ValueError(f"path: '{model_id_or_path}' not found")
        model_id_or_path = model_id_or_path.split(':', 1)  # get sub_folder
        if len(model_id_or_path) == 1:
            model_id_or_path = [model_id_or_path[0], None]
        model_id_or_path, sub_folder = model_id_or_path
        if sub_folder is not None:
            kwargs['allow_patterns'] = [f"{sub_folder.rstrip('/')}/*"]

        model_dir = hub.download_model(model_id_or_path, revision, ignore_patterns, token=hub_token, **kwargs)

        logger.debug(f'Loading model from local path: {model_dir}')

    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    if sub_folder:
        model_dir = os.path.join(model_dir, sub_folder)
    assert os.path.isdir(model_dir), f'model_dir: {model_dir}'
    return model_dir


def get_model_and_tokenizer_from_local(
        model_dir: str,
        model_info: ModelInfo,
        model_kwargs: Dict[str, Any],
        load_model: bool = True,
        *,
        tokenizer=None,
        model_config=None,
        automodel_class=None,
        **kwargs
) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
    """Load the model and tokenizer from the local model_dir."""
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # fix prediction_step (internvl2, ovis, ...)
    if not hasattr(model_config, 'keys_to_ignore_at_inference'):
        model_config.keys_to_ignore_at_inference = []
    if 'past_key_values' not in model_config.keys_to_ignore_at_inference:
        model_config.keys_to_ignore_at_inference.append('past_key_values')

    torch_dtype = model_info.torch_dtype
    HfConfigFactory.set_config_attr(model_config, 'torch_dtype', torch_dtype, include_vit=True)
    rope_scaling = kwargs.get('rope_scaling')
    max_model_len = kwargs.get('max_model_len')
    model_template = kwargs.get('model_template')
    if rope_scaling:
        HfConfigFactory.set_config_attr(model_config, 'rope_scaling', rope_scaling)
    if max_model_len:
        HfConfigFactory.set_max_model_len(model_config, max_model_len)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)

    if model_info.quant_method == 'fp8':
        torch_dtype = 'auto'

    model_kwargs['dtype'] = torch_dtype

    model = None
    if load_model:
        logger.info(f'model_kwargs: {model_kwargs}')
        automodel_class = automodel_class or AutoModelForCausalLM
        context_kwargs = {
            'model_info': model_info,
            'model_meta': model_template,
            'automodel_class': automodel_class,
        }
        if model is None:
            context = partial(patch_automodel, **context_kwargs)
            with context():
                model = automodel_class.from_pretrained(
                    model_dir, config=model_config, trust_remote_code=True, **model_kwargs)

        # fix not save modeling_xxx.py (transformers 4.45)
        # https://github.com/huggingface/transformers/issues/24737
        has_remote_code = hasattr(model_config, 'auto_map') and automodel_class.__name__ in model_config.auto_map
        if has_remote_code and model._auto_class is None:
            model._auto_class = automodel_class.__name__

    model_info.config = model_config if model is None else model.config

    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = tokenizer.eos_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token
    assert tokenizer.eos_token_id is not None
    assert tokenizer.pad_token_id is not None

    if model is not None:
        # fix seq classification task
        HfConfigFactory.set_model_config_attr(model, 'pad_token_id', pad_token)

    return model, tokenizer


def get_model_info_and_template(
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        *,
        download_model: bool = True,
        model_type: Optional[str] = None,
        quantization_config=None,
        **kwargs
) -> Tuple[ModelInfo, ModelTemplate]:
    """
    Get ModeInfo and ModelTemplate on the provided parameters.

    Args:
        model_id_or_path: The model ID or local path.
        torch_dtype: The desired torch dtype.
        download_model: Whether to download the model or not.
        model_type: The type of the model from LLMModelType
        quantization_config: The quantization configuration.
    """
    model_dir = safe_snapshot_download(
        model_id_or_path,
        check_local=True,
        download_model=download_model)

    model_info = ModelInfo.create(model_dir, model_type, quantization_config=quantization_config)

    if model_type is None and model_info.model_type is not None:
        model_type = model_info.model_type
        logger.info(f'Setting model_type: {model_type}')

    if model_type is not None:
        model_template = MODEL_MAPPING[model_type]
    else:
        model_template = ModelTemplate(None, 'dummy', get_model_and_tokenizer_from_local, model_arch=None)
        logger.info(f'Temporarily create model_meta: {model_template}')

    if torch_dtype is None:
        torch_dtype = model_template.torch_dtype or get_default_torch_dtype(model_info.torch_dtype)
        logger.debug(f'Setting torch_dtype: {torch_dtype}')

    model_info.torch_dtype = torch_dtype

    return model_info, model_template


def get_model_and_tokenizer_from_local(
        model_dir: str,
        model_info: ModelInfo,
        model_kwargs: Dict[str, Any],
        load_model: bool = True,
        *,
        tokenizer=None,
        model_config=None,
        automodel_class=None,
        **kwargs
):
    """
    Load the model and tokenizer from a local directory.
    """
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    torch_dtype = model_info.torch_dtype
    HfConfigFactory.set_config_attr(model_config, 'torch_dtype', torch_dtype, include_vit=True)
    rope_scaling = kwargs.get('rope_scaling')
    max_model_len = kwargs.get('max_model_len')

    if rope_scaling:
        HfConfigFactory.set_config_attr(model_config, 'rope_scaling', rope_scaling)

    if max_model_len:
        HfConfigFactory.set_max_model_len(model_config, max_model_len)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
 
    model_kwargs['dtype'] = torch_dtype

    model = None

    if load_model:
        patch_awq_compat(model_info)
        automodel_class = automodel_class or AutoModelForCausalLM
        model_template: ModelTemplate = kwargs['model_template']
        context_kwargs = {
            'model_info': model_info,
            'model_template': model_template,
            'automodel_class': automodel_class,
        }

        if model is None:
            context = partial(patch_automodel, **context_kwargs)
            with context():
                model = automodel_class.from_pretrained(
                        model_dir, config=model_config, trust_remote_code=True, **model_kwargs)
                    

        # fix not save modeling_xxx.py (transformers 4.45)
        # https://github.com/huggingface/transformers/issues/24737
        has_remote_code = hasattr(model_config, 'auto_map') and automodel_class.__name__ in model_config.auto_map
        if has_remote_code and model._auto_class is None:
            model._auto_class = automodel_class.__name__

    model_info.config = model_config if model is None else model.config

    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = tokenizer.eos_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token
    assert tokenizer.eos_token_id is not None
    assert tokenizer.pad_token_id is not None

    if model is not None:
        # fix seq classification task
        HfConfigFactory.set_model_config_attr(model, 'pad_token_id', pad_token)

    return model, tokenizer


def get_model_tokenizer_with_flash_attn(
        model_dir: str,
        model_info: ModelInfo,
        model_kwargs: Dict[str, Any],
        load_model: bool = True,
        **kwargs
):
    model_config = kwargs.get('model_config')
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    AttnImpl.update_attn_impl(model_config, kwargs.get('attn_impl'), kwargs.get('attn_impl_keys'))
    kwargs['model_config'] = model_config
    return get_model_and_tokenizer_from_local(model_dir, model_info, model_kwargs, load_model, **kwargs)


def get_model_tokenizer_multimodal(model_dir: str, *args, **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['tokenizer'] = processor.tokenizer
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    return model, processor

def get_model_info_and_tokenizer(
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Union[str, Dict[str, Any], None] = None,
        *,
        load_model: bool = True,
        # hub
        hub_token: Optional[str] = None,
        revision: Optional[str] = None,
        download_model: Optional[bool] = None,
        # model kwargs
        model_type: Optional[str] = None,
        quantization_config=None,
        max_memory: Union[str, Dict[str, Any]] = None,
        attn_impl: Optional[str] = None,
        new_special_tokens: Optional[List[str]] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_model_len: Optional[int] = None,
        automodel_class=None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs) -> Tuple[ModelInfo, ModelTemplate, Optional[PreTrainedModel], PreTrainedTokenizerBase]:
    """
    model_id_or_path: The path to the model or the model_id from modelscope/huggingface (controlled by `use_hf`).
    torch_dtype: If you pass `None`, it will retrieve the torch_dtype from the config.json file.
    model_kwargs: Passed to `automodel_class.from_pretrained`.
    load_model: Whether to load the model. If set to False, the model will return `None`.
    use_hf: Indicates whether the model download hub is modelscope or huggingface.
    model_type: If it is not possible to uniquely determine the model_type from the architecture in config.json,
        it needs to be provided.
    attn_impl: If set to 'flash_attn': It will automatically convert names based on the model.
        If set to None : It will be automatically selected between sdpa and eager.
    download_model: Whether to download the model weights. If `None`, it will be selected based on load_model.
    tokenizer_path: The path to the tokenizer. If `None`, it will use the tokenizer from the model.
    """
    if model_kwargs is None:
        model_kwargs = {}
    if download_model is None:
        download_model = load_model

    model_info, model_template = get_model_info_and_template(
        model_id_or_path,
        torch_dtype,
        use_hf=True,
        hub_token=hub_token,
        revision=revision,
        download_model=download_model,
        model_type=model_type,
        quantization_config=quantization_config)

    model_kwargs['device_map'] = device_map
    if quantization_config:
        model_kwargs['quantization_config'] = quantization_config
    if max_memory:
        model_kwargs['max_memory'] = max_memory
    model_dir = model_info.model_dir
    get_function = model_template.get_function
    kwargs['automodel_class'] = automodel_class
    kwargs['attn_impl'] = attn_impl
    kwargs['rope_scaling'] = rope_scaling
    kwargs['model_template'] = model_template
    kwargs['max_model_len'] = max_model_len

    model, processor = get_function(model_dir, model_info, model_kwargs, load_model, **kwargs)

    if not isinstance(processor, PreTrainedTokenizerBase) and hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
        patch_getattr(processor.__class__, 'tokenizer')
    else:
        tokenizer = processor
    
    if new_special_tokens:
        num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
        if num_new_tokens > 0:
            logger.info(f'Added {num_new_tokens} new special tokens.')

            if model is not None:
                llm_model = get_lm_head_model(model, model_template)
                origin_vocab_size = HfConfigFactory.get_config_attr(llm_model.config, 'vocab_size')
                if origin_vocab_size < len(tokenizer):
                    vocab_size = math.ceil(len(tokenizer) / 128) * 128
                    llm_model.resize_token_embeddings(vocab_size)
                    # fix transformers==4.52.4 qwen2.5-vl
                    HfConfigFactory.set_config_attr(llm_model.config, 'vocab_size', vocab_size)

    tokenizer.model_info = model_info
    tokenizer.model_template = model_template

    if model is not None:
        model.model_info = model_info
        model.model_template = model_template
        model.model_dir = model_dir

        # generation_config
        generation_config_path = os.path.join(model_dir, 'generation_config.json')
        if not hasattr(model, 'generation_config') and os.path.isfile(generation_config_path):
            model.generation_config = GenerationConfig.from_pretrained(model_dir)
        # fix llama2 warning
        if getattr(model, 'generation_config', None):
            fix_do_sample_warning(model.generation_config)

    if processor is not None:
        processor.model_info = model_info
        processor.model_template = model_template

    return model_info, model_template, model, processor
