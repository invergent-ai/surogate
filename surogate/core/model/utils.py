import os
from functools import partial
from typing import Optional, Dict, Any, Tuple, Literal, Union, List

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForCausalLM, PreTrainedTokenizerBase, PreTrainedModel, GenerationConfig, BaseImageProcessor, \
    FeatureExtractionMixin
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import is_torch_bf16_gpu_available, is_torch_cuda_available
from transformers import ProcessorMixin as HfProcessorMixin

from surogate.core.model.attn import AttnImpl
from surogate.core.model.hf_config import HfConfigFactory
from surogate.core.model.init_strategy import InitModelStrategy
from surogate.core.model.loader import safe_snapshot_download
from surogate.core.model.model_info import ModelInfo
from surogate.core.model.patcher import patch_awq_compat, patch_automodel_for_sequence_classification, patch_automodel, \
    patch_output_normalizer, deepspeed_set_z3_leaf_modules, patch_mp_ddp, patch_get_dynamic_module, patch_tp_plan, \
    patch_getattr
from surogate.core.model.registry import ModelTemplate, MODEL_MAPPING
from surogate.utils.dist import get_dist_setting, is_mp
from surogate.utils.logger import get_logger

logger = get_logger()

Processor = Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, HfProcessorMixin]
Prompt = List[Union[str, List[int], List[str]]]
Word = Union[str, List[int]]
Context = Word

class ContextType:
    RESPONSE = 'response'
    SUFFIX = 'suffix'
    OTHER = 'other'

def get_model_info_and_template(
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        *,
        download_model: bool = False,
        model_type: Optional[str] = None,
        quantization_config=None,
        task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'reranker', 'generative_reranker', None] = None,
        num_labels=None,
        **kwargs
) -> Tuple[ModelInfo, ModelTemplate]:
    model_dir = safe_snapshot_download(
        model_id_or_path,
        download_model=download_model)

    model_info = ModelInfo.create(model_dir, model_type, quantization_config=quantization_config)

    if model_type is None and model_info.model_type is not None:
        model_type = model_info.model_type
        logger.info(f'Setting model_type: {model_type}')

    if model_type is not None:
        model_template = MODEL_MAPPING[model_type]
    else:
        model_template = ModelTemplate(None, 'dummy', get_model_tokenizer_from_local, model_arch=None)
        logger.info(f'Temporarily create model_meta: {model_template}')

    if torch_dtype is None:
        torch_dtype = model_template.torch_dtype or get_default_torch_dtype(model_info.torch_dtype)
        logger.info(f'Setting torch_dtype: {torch_dtype}')

    model_info.torch_dtype = torch_dtype
    if task_type is None:
        if model_template.is_reward:
            num_labels = 1
        if num_labels is None:
            task_type = 'causal_lm'
        else:
            task_type = 'seq_cls'

        if model_template.task_type is not None:
            task_type = model_template.task_type

    # Handle reranker task type
    if task_type == 'reranker':
        if num_labels is None:
            num_labels = 1  # Default to 1 for reranker tasks
        logger.info(f'Setting reranker task with num_labels={num_labels}')
    elif task_type == 'generative_reranker':
        # Generative reranker doesn't need num_labels as it uses CausalLM structure
        num_labels = None
        logger.info('Setting generative_reranker task (no num_labels needed)')
    elif task_type == 'seq_cls':
        assert num_labels is not None, 'Please pass the parameter `num_labels`.'

    model_info.task_type = task_type
    model_info.num_labels = num_labels

    if model_template.attention_cls is None:
        try:
            # Dynamically import the module and attention class
            module_path = f"transformers.models.{model_type}.modeling_{model_type}"
            model_cls_prefix, _ = get_causal_lm_model_cls_prefix(model_type)
            module = __import__(module_path, fromlist=[f"{model_cls_prefix}Attention"])
            attention_cls = getattr(module, f"{model_cls_prefix}Attention")
            model_template.attention_cls = attention_cls
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not determine attention class for model type '{model_type}': {e}")

    return model_info, model_template


def get_model_tokenizer_from_local(
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
    HfConfigFactory.compat_zero3(model_config)
    leaf_modules = kwargs.get('leaf_modules')
    rope_scaling = kwargs.get('rope_scaling')
    max_model_len = kwargs.get('max_model_len')
    return_dummy_model = kwargs.get('return_dummy_model')
    model_template = kwargs.get('model_template')

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
            if model_info.num_labels == 1 or model_template.is_reward:
                problem_type = 'regression'
            else:
                problem_type = 'single_label_classification'
        model_config.problem_type = problem_type

    if model_info.quant_method == 'fp8':
        torch_dtype = 'auto'

    model_kwargs['dtype'] = torch_dtype

    model = None

    if load_model:
        patch_awq_compat(model_info)
        if model_info.task_type in {'seq_cls', 'reranker'} and automodel_class is None and not return_dummy_model:
            with patch_automodel_for_sequence_classification(model_config=model_config, patch_from_pretrained=False):
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_dir, config=model_config, trust_remote_code=True, **model_kwargs)
                except ValueError:
                    model = None

        automodel_class = automodel_class or AutoModelForCausalLM
        model_template = kwargs['model_template']
        context_kwargs = {
            'model_info': model_info,
            'model_template': model_template,
            'automodel_class': automodel_class,
            'return_dummy_model': return_dummy_model,
        }

        if model is None:
            if return_dummy_model:
                context = partial(patch_automodel, **context_kwargs)
            elif model_info.task_type == 'seq_cls' and not model_template.is_reward:
                context = partial(patch_automodel_for_sequence_classification, **context_kwargs)
            elif model_info.task_type == 'seq_cls' and model_template.is_reward and model_config.num_labels > 1:
                logger.warning('You are using a reward model for seq_cls task and num_labels > 1, '
                               'ignore_mismatched_sizes will be set to True')
                model_kwargs['ignore_mismatched_sizes'] = True
                context = partial(patch_automodel_for_sequence_classification, **context_kwargs)
            elif model_info.task_type == 'reranker' and not model_template.is_reranker:
                # For reranker task, patch CausalLM to SequenceClassification with num_labels=1
                logger.info('Converting CausalLM to SequenceClassification for reranker task with num_labels=1')
                context = partial(patch_automodel_for_sequence_classification, **context_kwargs)
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
            patch_output_normalizer(model, model_template=model_template)

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
            deepspeed_set_z3_leaf_modules(model, leaf_modules)

    return model, tokenizer


def get_model_tokenizer_with_flash_attn(model_dir: str,
                                        model_info: ModelInfo,
                                        model_kwargs: Dict[str, Any],
                                        load_model: bool = True,
                                        **kwargs):
    model_config = kwargs.get('model_config')
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    AttnImpl.update_attn_impl(model_config, kwargs.get('attn_impl'), kwargs.get('attn_impl_keys'))
    kwargs['model_config'] = model_config
    return get_model_tokenizer_from_local(model_dir, model_info, model_kwargs, load_model, **kwargs)


def get_model_tokenizer_multimodal(model_dir: str, *args, **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['tokenizer'] = processor.tokenizer
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    return model, processor


def get_model_tokenizer(
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Union[str, Dict[str, Any], None] = None,
        *,
        load_model: bool = True,
        download_model: Optional[bool] = None,
        model_type: Optional[str] = None,
        quantization_config=None,
        max_memory: Union[str, Dict[str, Any]] = None,
        attn_impl: Optional[str] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_model_len: Optional[int] = None,
        automodel_class=None,
        task_type: Literal['causal_lm', 'seq_cls', 'reranker', 'generative_reranker'] = None,
        num_labels: Optional[int] = None,
        return_dummy_model: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
    if load_model:
        patch_mp_ddp()
    if model_kwargs is None:
        model_kwargs = {}
    if download_model is None:
        download_model = load_model and not return_dummy_model

    model_info, model_template = get_model_info_and_template(
        model_id_or_path,
        torch_dtype,
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
    get_function = model_template.get_function

    kwargs['automodel_class'] = automodel_class
    kwargs['attn_impl'] = attn_impl
    kwargs['rope_scaling'] = rope_scaling
    kwargs['model_template'] = model_template
    kwargs['max_model_len'] = max_model_len
    kwargs['return_dummy_model'] = return_dummy_model

    with patch_get_dynamic_module(), patch_tp_plan(load_model):
        model, processor = get_function(model_dir, model_info, model_kwargs, load_model, **kwargs)

    if not isinstance(processor, PreTrainedTokenizerBase) and hasattr(processor, 'tokenizer'):
        tokenizer = processor.tokenizer
        patch_getattr(processor.__class__, 'tokenizer')
    else:
        tokenizer = processor

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

    return model, processor


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