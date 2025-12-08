import math
import os
from contextlib import nullcontext
from functools import partial
from typing import Tuple, Optional, Union, Dict, Any, List, Literal

import torch
from swift.llm.model import fix_do_sample_warning, get_default_device_map, \
    ModelMeta, get_matched_model_meta
from swift.llm.model.patcher import (get_lm_head_model, patch_attach_align_device_hook_on_blocks,
                                     patch_get_dynamic_module, patch_mp_ddp,
                                     patch_tp_plan, patch_automodel_for_sequence_classification, patch_automodel)
from swift.llm.model.register import MODEL_MAPPING, get_default_torch_dtype, \
    deepspeed_set_z3_leaf_modules, get_matched_model_types
from swift.llm.model.utils import HfConfigFactory, ModelInfo, safe_snapshot_download, InitModelStrategy
from swift.utils import patch_getattr
from transformers import PreTrainedTokenizerBase, PreTrainedModel, GenerationConfig, AutoConfig, \
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, PretrainedConfig

from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()


def load_model_and_tokenizer(model_id: str, model_type: str, args: DictDefault, load_model = True) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
    if model_id is None:
        raise ValueError("'model' must be specified in config.")

    model, tokenizer = get_model_and_tokenizer(
        model_id,
        use_hf=True,
        hub_token=args.get('hub_token'),
        load_model=load_model,
        model_type=model_type,
    )

    return model, tokenizer


def get_model_and_tokenizer(
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
        task_type: Literal['causal_lm', 'seq_cls', 'reranker', 'generative_reranker'] = None,
        num_labels: Optional[int] = None,
        return_dummy_model: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
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
    if load_model:
        patch_mp_ddp()
    if model_kwargs is None:
        model_kwargs = {}
    if download_model is None:
        download_model = load_model and not return_dummy_model

    model_info, model_meta = get_model_info_and_meta(
        model_id_or_path,
        torch_dtype,
        use_hf=True,
        hub_token=hub_token,
        revision=revision,
        download_model=download_model,
        model_type=model_type,
        quantization_config=quantization_config,
        task_type=task_type,
        num_labels=num_labels)

    if device_map is None:
        device_map = get_default_device_map()
    model_kwargs['device_map'] = device_map
    if quantization_config:
        model_kwargs['quantization_config'] = quantization_config
    if max_memory:
        model_kwargs['max_memory'] = max_memory
    model_dir = model_info.model_dir
    get_function = model_meta.get_function
    kwargs['automodel_class'] = automodel_class
    kwargs['attn_impl'] = attn_impl
    kwargs['rope_scaling'] = rope_scaling
    kwargs['model_meta'] = model_meta
    kwargs['max_model_len'] = max_model_len
    kwargs['return_dummy_model'] = return_dummy_model
    patch_offload = kwargs.pop('patch_offload', False)
    patch_offload_context = patch_attach_align_device_hook_on_blocks() if patch_offload else nullcontext()
    with patch_get_dynamic_module(), patch_tp_plan(load_model), patch_offload_context:
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

            if model is not None and not return_dummy_model:
                llm_model = get_lm_head_model(model, model_meta)
                origin_vocab_size = HfConfigFactory.get_config_attr(llm_model.config, 'vocab_size')
                if origin_vocab_size < len(tokenizer):
                    vocab_size = math.ceil(len(tokenizer) / 128) * 128
                    llm_model.resize_token_embeddings(vocab_size)
                    # fix transformers==4.52.4 qwen2.5-vl
                    HfConfigFactory.set_config_attr(llm_model.config, 'vocab_size', vocab_size)

    tokenizer.model_info = model_info
    tokenizer.model_meta = model_meta

    if model is not None:
        model.model_info = model_info
        model.model_meta = model_meta
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
        processor.model_meta = model_meta

    return model, processor


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
    HfConfigFactory.compat_zero3(model_config)
    leaf_modules = kwargs.get('leaf_modules')
    rope_scaling = kwargs.get('rope_scaling')
    max_model_len = kwargs.get('max_model_len')
    return_dummy_model = kwargs.get('return_dummy_model')
    model_meta = kwargs.get('model_meta')
    if rope_scaling:
        HfConfigFactory.set_config_attr(model_config, 'rope_scaling', rope_scaling)
    if max_model_len:
        HfConfigFactory.set_max_model_len(model_config, max_model_len)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    num_labels = model_info.num_labels or getattr(model_config, 'num_labels', None)
    if num_labels and model_info.task_type in ['seq_cls', 'reranker']:
        model_info.num_labels = num_labels
        model_config.num_labels = num_labels

    if model_info.task_type == 'seq_cls':
        problem_type = kwargs.get('problem_type')
        if problem_type is None:
            if model_info.num_labels == 1 or model_meta.is_reward:
                problem_type = 'regression'
            else:
                problem_type = 'single_label_classification'
        model_config.problem_type = problem_type

    if model_info.quant_method == 'fp8':
        torch_dtype = 'auto'

    model_kwargs['dtype'] = torch_dtype

    model = None
    if load_model:
        logger.info(f'model_kwargs: {model_kwargs}')
        if model_info.task_type in {'seq_cls', 'reranker'} and automodel_class is None and not return_dummy_model:
            with patch_automodel_for_sequence_classification(model_config=model_config, patch_from_pretrained=False):
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_dir, config=model_config, trust_remote_code=True, **model_kwargs)
                except ValueError:
                    model = None

        automodel_class = automodel_class or AutoModelForCausalLM
        context_kwargs = {
            'model_info': model_info,
            'model_meta': model_meta,
            'automodel_class': automodel_class,
            'return_dummy_model': return_dummy_model,
        }
        if model is None:
            if return_dummy_model:
                context = partial(patch_automodel, **context_kwargs)
            elif model_info.task_type == 'seq_cls' and not model_meta.is_reward:
                context = partial(patch_automodel_for_sequence_classification, **context_kwargs)
            elif model_info.task_type == 'seq_cls' and model_meta.is_reward and model_config.num_labels > 1:
                logger.warning('You are using a reward model for seq_cls task and num_labels > 1, '
                               'ignore_mismatched_sizes will be set to True')
                model_kwargs['ignore_mismatched_sizes'] = True
                context = partial(patch_automodel_for_sequence_classification, **context_kwargs)
            elif model_info.task_type == 'reranker':
                # For reranker task, patch CausalLM to SequenceClassification with num_labels=1
                logger.info('Converting CausalLM to SequenceClassification for reranker task with num_labels=1')
                context = partial(patch_automodel_for_sequence_classification, **context_kwargs)
            elif model_info.task_type == 'generative_reranker':
                # For generative reranker, keep CausalLM structure unchanged
                logger.info('Loading model as CausalLM for generative_reranker task')
                context = partial(patch_automodel, **context_kwargs)
            else:
                context = partial(patch_automodel, **context_kwargs)
            with context():
                model = automodel_class.from_pretrained(
                    model_dir, config=model_config, trust_remote_code=True, **model_kwargs)

        # fix not save modeling_xxx.py (transformers 4.45)
        # https://github.com/huggingface/transformers/issues/24737
        has_remote_code = hasattr(model_config, 'auto_map') and automodel_class.__name__ in model_config.auto_map
        if has_remote_code and model._auto_class is None:
            model._auto_class = automodel_class.__name__

        if model_info.task_type == 'embedding' and automodel_class.__name__ != 'AutoModel':
            from swift.llm.model.patcher import patch_output_normalizer
            patch_output_normalizer(model, model_meta=model_meta)

        init_strategy = kwargs.get('init_strategy')
        if init_strategy is not None:
            InitModelStrategy.init_parameters(model, init_strategy)

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
        if leaf_modules is not None or model_info.is_moe_model:
            # deepspeed zero3
            deepspeed_set_z3_leaf_modules(model)

    return model, tokenizer


def _get_model_info(
        model_dir: str,
        model_type: Optional[str],
        quantization_config
) -> ModelInfo:
    try:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        config = PretrainedConfig.get_config_dict(model_dir)[0]
    if quantization_config is not None:
        HfConfigFactory.set_config_attr(config, 'quantization_config', quantization_config)
    quant_info = HfConfigFactory.get_quant_info(config) or {}
    torch_dtype = HfConfigFactory.get_torch_dtype(config, quant_info)
    max_model_len = HfConfigFactory.get_max_model_len(config)
    rope_scaling = HfConfigFactory.get_config_attr(config, 'rope_scaling')
    is_moe_model = HfConfigFactory.is_moe_model(config)

    if model_type is None:
        architectures = HfConfigFactory.get_config_attr(config, 'architectures')
        model_types = get_matched_model_types(architectures)
        if len(model_types) > 1:
            raise ValueError('Please explicitly pass the model_type. For reference, '
                             f'the available model_types: {model_types}.')
        elif len(model_types) == 1:
            model_type = model_types[0]
    elif model_type not in MODEL_MAPPING:
        raise ValueError(f"model_type: '{model_type}' not in {list(MODEL_MAPPING.keys())}")

    res = ModelInfo(
        model_type,
        model_dir,
        torch_dtype,
        max_model_len,
        quant_info.get('quant_method'),
        quant_info.get('quant_bits'),
        rope_scaling=rope_scaling,
        is_moe_model=is_moe_model,
    )
    return res
