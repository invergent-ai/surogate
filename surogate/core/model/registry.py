from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace as dataclass_replace
import math
import os
import re
from typing import Dict, Literal, Optional, Tuple, List, Type, Any, Union
from packaging import version
from peft import PeftModel
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig
import transformers
from surogate.core.model.hf_config import HfConfigFactory
from surogate.core.model.patcher import patch_getattr
from surogate.core.model.utils import Processor, get_default_torch_dtype
from surogate.utils.logger import get_logger
from surogate.core.hub.huggingface import HuggingFaceHub
 
logger = get_logger()
MODEL_MAPPING: Dict[str, 'ModelTemplate'] = {}
CHAT_TEMPLATE_MAPPING: Dict[str, 'ModelTemplate'] = {}

class BaseModelLoader(ABC):
    @abstractmethod
    def __init__(self, model_info, model_template: 'ModelTemplate', *args, **kwargs):
        pass

    @abstractmethod
    def load(self) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
        pass
    
@dataclass
class ModelTemplate:
    model_type: Optional[str]  # HuggingFace architecture string, e.g. 'LlamaForCausalLM'
    chat_templates: Optional[List[str]]
    loader: Optional[Type[BaseModelLoader]] = None
    additional_saved_files: List[str] = field(default_factory=list)
    torch_dtype: Optional[torch.dtype] = None
    is_multimodal: bool = False
    tags: List[str] = field(default_factory=list)
    task_type: Optional[str] = None
    # Resolved single chat template (set after template selection)
    chat_template: Optional[str] = None

    def __post_init__(self):
        if self.loader is None:
            self.loader = ModelLoader

        if self.chat_templates is None:
            self.chat_templates = ['dummy']

        # Auto-resolve when there is exactly one option
        if self.chat_template is None and len(self.chat_templates) == 1:
            self.chat_template = self.chat_templates[0]


def register_model(model_template: ModelTemplate) -> None:
    model_type = model_template.model_type
    MODEL_MAPPING[model_type] = model_template
    for ct in (model_template.chat_templates or []):
        CHAT_TEMPLATE_MAPPING[ct] = model_template

def get_matched_template_types(architectures: Optional[List[str]]) -> List[str]:
    """Return the model_type (architecture string) if it is registered, else empty list."""
    arch = (architectures or ['null'])[0]
    return [arch] if arch in MODEL_MAPPING else []


class ModelLoader(BaseModelLoader):
    def __init__(
        self,
        model_info: 'ModelInfo',
        model_template: ModelTemplate,
        *,
        load_model: bool = False,
        # model kwargs
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_model_len: Optional[int] = None,
        auto_model_cls=None,
        return_dummy_model: bool = False,
        new_special_tokens: Optional[List[str]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.model_info = model_info
        self.model_template = model_template
        self.load_model = load_model
        self.rope_scaling = rope_scaling
        self.max_model_len = max_model_len
        self.auto_model_cls = auto_model_cls
        self.auto_config_cls = None
        self.auto_tokenizer_cls = None
        self.return_dummy_model = return_dummy_model
        self.new_special_tokens = new_special_tokens
        self.model_kwargs = model_kwargs
        self.patch_offload = kwargs.pop('patch_offload', False)
        self.init_strategy = kwargs.get('init_strategy')
        self.local_repo_path = kwargs.get('local_repo_path')
        self.leaf_modules = None
        self.pad_token = None
        if model_info.quant_method == 'fp8':
            self.torch_dtype = 'auto'
        else:
            self.torch_dtype = model_info.torch_dtype
        if version.parse(transformers.__version__) >= version.parse('4.56'):
            model_kwargs['dtype'] = self.torch_dtype
        else:
            model_kwargs['torch_dtype'] = self.torch_dtype
        logger.info(f'model_kwargs: {model_kwargs}')
    
    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        auto_tokenizer_cls = self.auto_tokenizer_cls
        if auto_tokenizer_cls is None:
            if os.path.exists(os.path.join(model_dir, 'preprocessor_config.json')):
                from transformers import AutoProcessor
                auto_tokenizer_cls = AutoProcessor
            else:
                auto_tokenizer_cls = AutoTokenizer
        return auto_tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
    
    def _get_model_processor(self, model_dir, config):
        processor = self.get_processor(model_dir, config)
        model = None
        if self.load_model:
            model = self.get_model(model_dir, config, processor, self.model_kwargs.copy())
        return model, processor
    
    def _get_tokenizer(self, processor):
        if not isinstance(processor, PreTrainedTokenizerBase) and hasattr(processor, 'tokenizer'):
            tokenizer = processor.tokenizer
            patch_getattr(processor.__class__, 'tokenizer')
        else:
            tokenizer = processor
        return tokenizer
    
    def _postprocess_processor(self, processor: Processor):
        tokenizer = self._get_tokenizer(processor)
        pad_token = tokenizer.pad_token_id
        if pad_token is None:
            pad_token = tokenizer.eos_token_id
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = pad_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = pad_token
        assert tokenizer.eos_token_id is not None
        assert tokenizer.pad_token_id is not None
        self.pad_token = pad_token
        tokenizer.model_info = self.model_info
        tokenizer.model_template = self.model_template
    
    def _add_new_special_tokens(self, model, processor, config):
        if not self.new_special_tokens:
            return
        tokenizer = self._get_tokenizer(processor)
        num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': self.new_special_tokens})
        if num_new_tokens > 0:
            logger.info(f'Added {num_new_tokens} new special tokens.')
            origin_vocab_size = HfConfigFactory.get_config_attr(config, 'vocab_size')
            if origin_vocab_size < len(tokenizer):
                vocab_size = math.ceil(len(tokenizer) / 128) * 128
                # fix transformers==4.52.4 qwen2.5-vl
                HfConfigFactory.set_config_attr(config, 'vocab_size', vocab_size)
                if model is not None and not self.return_dummy_model:
                    if isinstance(llm_model, PeftModel):
                        llm_model = llm_model.model        
                    llm_model.resize_token_embeddings(vocab_size)
    
    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        auto_model_cls = self.auto_model_cls or AutoModelForCausalLM
        model = auto_model_cls.from_pretrained(model_dir, config=config, trust_remote_code=True, **model_kwargs)
        has_remote_code = hasattr(config, 'auto_map') and auto_model_cls.__name__ in config.auto_map
        if has_remote_code and model._auto_class is None:
            model._auto_class = auto_model_cls.__name__
        self._compat_transformers5(model)
        return model
    
    def _postprocess_model(self, model_dir, model):
        model.model_info = self.model_info
        model.model_template = self.model_template
        model.model_dir = model_dir
        HfConfigFactory.set_model_config_attr(model, 'pad_token_id', self.pad_token)
   
    def _compat_transformers5(self, model):
        if self.model_template.is_multimodal:
            for key in ['language_model', 'vision_tower', 'multi_modal_projector', 'visual', 'vision_model']:
                _set_property(model, key)
    
    def get_config(self, model_dir: str) -> PretrainedConfig:
        auto_config_cls = self.auto_config_cls or AutoConfig
        return auto_config_cls.from_pretrained(model_dir, trust_remote_code=True)
    
    def _postprocess_config(self, config):
        torch_dtype = self.model_info.torch_dtype
        HfConfigFactory.set_config_attr(config, 'torch_dtype', torch_dtype, include_vit=True)
        if self.rope_scaling:
            rope_parameters = HfConfigFactory.get_config_attr(config, 'rope_parameters') or {}
            for key in ['rope_theta', 'partial_rotary_factor']:
                if self.rope_scaling.get(key) is None and rope_parameters.get(key) is not None:
                    self.rope_scaling[key] = rope_parameters[key]
            HfConfigFactory.set_config_attr(config, 'rope_scaling', self.rope_scaling)
        if self.max_model_len:
            HfConfigFactory.set_max_model_len(config, self.max_model_len)
        self.model_info.config = config
        return config
                    
    def load(self) -> Tuple[Optional[PreTrainedModel], Processor]:
        model_dir = self.model_info.model_dir
        config = self.get_config(model_dir)
        self._postprocess_config(config)
        model, processor = self._get_model_processor(model_dir, config)
        self._postprocess_processor(processor)
        if model:
            self._postprocess_model(model_dir, model)
        self._add_new_special_tokens(model, processor, config)
        return model, processor
    
    
@dataclass
class ModelInfo:
    template_type: str
    model_dir: str
    torch_dtype: torch.dtype
    max_model_len: int
    quant_method: Literal['gptq', 'awq', 'bnb', 'prequant_fp8', 'prequant_nvfp4', 'prequant_mxfp4', None]
    quant_bits: int

    # extra
    rope_scaling: Optional[Dict[str, Any]] = None
    is_moe_model: bool = False
    is_multimodal: bool = False
    config: Optional[PretrainedConfig] = None
    quant_info: Optional[Dict[str, Any]] = None
    # Resolved chat template (set when user passes a specific chat template name)
    chat_template: Optional[str] = None

    def __post_init__(self):
        self.model_name = ModelInfo.get_model_name(self.model_dir)

    @staticmethod
    def get_model_name(model_id_or_path: str) -> Optional[str]:
        assert isinstance(model_id_or_path, str), f'model_id_or_path: {model_id_or_path}'
        # compat hf hub
        model_id_or_path = model_id_or_path.rstrip('/')
        match_ = re.search('/models--.+?--(.+?)/snapshots/', model_id_or_path)
        if match_ is not None:
            return match_.group(1)

        model_name = model_id_or_path.rsplit('/', 1)[-1]
        # compat modelscope snapshot_download
        model_name = model_name.replace('___', '.')
        return model_name

    @staticmethod
    def create(model_dir: str, template_type: Optional[str], quantization_config) -> 'ModelInfo':
        try:
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        except Exception:
            config = PretrainedConfig.get_config_dict(model_dir)[0]

        if quantization_config is not None:
            HfConfigFactory.set_config_attr(config, 'quantization_config', quantization_config)

        quant_info = HfConfigFactory.get_quant_info(config) or {}
        # Fallback: some ModelOpt models store quant info in hf_quant_config.json
        if not quant_info:
            quant_info = HfConfigFactory.get_quant_info_from_hf_quant_config(model_dir) or {}
        torch_dtype = HfConfigFactory.get_torch_dtype(config, quant_info)
        max_model_len = HfConfigFactory.get_max_model_len(config)
        rope_scaling = HfConfigFactory.get_config_attr(config, 'rope_scaling')
        is_moe_model = HfConfigFactory.is_moe_model(config)
        is_multimodal = HfConfigFactory.is_multimodal(config)

        chat_template = None
        if template_type is None:
            architectures = HfConfigFactory.get_config_attr(config, 'architectures')
            model_types = get_matched_template_types(architectures)
            if len(model_types) > 1:
                raise ValueError(
                    'Failed to automatically match `template_type`. '
                    f'Please explicitly pass the `template_type` for `{model_dir}`. '
                    f'Possible values: {model_types}.')
            elif len(model_types) == 1:
                model_type = model_types[0]
            else:
                model_type = None
        elif template_type in CHAT_TEMPLATE_MAPPING:
            # User specified a specific chat template name (e.g. "llama3", "llama3_2")
            mt = CHAT_TEMPLATE_MAPPING[template_type]
            model_type = mt.model_type
            chat_template = template_type
        elif template_type in MODEL_MAPPING:
            model_type = template_type
        else:
            raise ValueError(
                f"template_type: '{template_type}' is not recognized. "
                f"Valid model types: {list(MODEL_MAPPING.keys())}. "
                f"Valid chat templates: {list(CHAT_TEMPLATE_MAPPING.keys())}.")

        return ModelInfo(
            model_type,
            model_dir,
            torch_dtype,
            max_model_len,
            quant_info.get('quant_method'),
            quant_info.get('quant_bits'),
            rope_scaling=rope_scaling,
            is_moe_model=is_moe_model,
            is_multimodal=is_multimodal,
            quant_info=quant_info or None,
            chat_template=chat_template,
        )

def _set_property(model, key):
    if not hasattr(model, 'model'):
        return
    text_model = model.model
    if not hasattr(text_model, key):
        return

    def _value(self):
        return getattr(text_model, key)

    setattr(model.__class__, key, property(_value))
    
    
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


def _get_model_info_and_template(
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        *,
        download_model: bool = True,
        hub_token: Optional[str] = None,
        template_type: Optional[str] = None,
        quantization_config=None,

) -> Tuple[ModelInfo, ModelTemplate]:
    """
    Get ModeInfo and ModelTemplate on the provided parameters.

    Args:
        model_id_or_path: The model ID or local path.
        torch_dtype: The desired torch dtype.
        download_model: Whether to download the model or not.
        template_type: The type of the chat template from LLMModelType
        quantization_config: The quantization configuration.
    """
    model_dir = safe_snapshot_download(
        model_id_or_path,
        check_local=True,
        hub_token=hub_token,
        download_model=download_model)

    model_info = ModelInfo.create(model_dir, template_type, quantization_config=quantization_config)

    if model_info.template_type is None:
        if model_info.is_multimodal:
            raise ValueError(
                f'Model "{model_id_or_path}" is not supported: no suitable `template_type` was found. '
                f'Please specify `template_type` explicitly.')
        else:
            model_template = ModelTemplate(None, ['dummy'], ModelLoader)
            logger.info(f'Temporarily create model_meta: {model_template}')
    else:
        logger.info(f'Setting template_type: {model_info.template_type}')
        model_template = MODEL_MAPPING[model_info.template_type]

        # Resolve the specific chat template
        if model_info.chat_template is not None:
            # Already resolved: user passed a specific chat template name (e.g. "llama3")
            resolved_chat_template = model_info.chat_template
        elif len(model_template.chat_templates) == 0:
            raise ValueError(
                f'No chat templates registered for model type "{model_info.template_type}".')
        elif len(model_template.chat_templates) == 1:
            resolved_chat_template = model_template.chat_templates[0]
        else:
            raise ValueError(
                f'Model type "{model_info.template_type}" has multiple chat templates: '
                f'{model_template.chat_templates}. '
                f'Please set `template_type` to one of: {model_template.chat_templates}.')

        model_template = dataclass_replace(model_template, chat_template=resolved_chat_template)

    if torch_dtype is None:
        torch_dtype = model_template.torch_dtype or get_default_torch_dtype(model_info.torch_dtype)
        logger.debug(f'Setting torch_dtype: {torch_dtype}')

    model_info.torch_dtype = torch_dtype
    if model_template.is_multimodal:
        model_info.is_multimodal = True
    return model_info, model_template


def get_model_info_and_tokenizer(
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Union[str, Dict[str, Any], None] = None,
        *,
        load_model: bool = True,
        # hub
        hub_token: Optional[str] = None,
        download_model: Optional[bool] = None,
        # model kwargs
        template_type: Optional[str] = None,
        quantization_config=None,
        max_memory: Union[str, Dict[str, Any]] = None,
        new_special_tokens: Optional[List[str]] = None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_model_len: Optional[int] = None,
        auto_model_cls: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs) -> Tuple[ModelInfo, ModelTemplate, Optional[PreTrainedModel], PreTrainedTokenizerBase]:
    """Load a pretrained model and its tokenizer from a model hub or local path."""
    if model_kwargs is None:
        model_kwargs = {}
    if download_model is None:
        download_model = load_model

    model_info, model_template = _get_model_info_and_template(
        model_id_or_path,
        torch_dtype,
        hub_token=hub_token,
        download_model=download_model,
        template_type=template_type,
        quantization_config=quantization_config)

    model_kwargs['device_map'] = device_map
    if quantization_config:
        model_kwargs['quantization_config'] = quantization_config
    if max_memory:
        model_kwargs['max_memory'] = max_memory
        
        
    loader = model_template.loader(
        model_info,
        model_template,
        load_model=load_model,
        rope_scaling=rope_scaling,
        max_model_len=max_model_len,
        auto_model_cls=auto_model_cls,
        new_special_tokens=new_special_tokens,
        model_kwargs=model_kwargs,
        **kwargs)
    
    model, tokenizer = loader.load()

    return model_info, model_template, model, tokenizer
