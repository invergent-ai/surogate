"""Model registry — loads model config and tokenizer from HuggingFace.

All model metadata (is_multimodal, is_moe, dtype, etc.) is derived directly
from config.json via HfConfigFactory.  No MODEL_MAPPING or chat-template
registration is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Dict, Optional, Tuple, List, Any

from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class ModelInfo:
    """Metadata about a model, derived from config.json."""
    model_dir: str
    torch_dtype: Any  # torch.dtype
    max_model_len: int
    quant_method: Optional[str]
    quant_bits: Optional[int]

    rope_scaling: Optional[Dict[str, Any]] = None
    is_moe_model: bool = False
    is_multimodal: bool = False
    config: Optional[Any] = None  # PretrainedConfig or dict
    quant_info: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.model_name = ModelInfo.get_model_name(self.model_dir)

    @staticmethod
    def get_model_name(model_id_or_path: str) -> Optional[str]:
        assert isinstance(model_id_or_path, str), f'model_id_or_path: {model_id_or_path}'
        model_id_or_path = model_id_or_path.rstrip('/')
        match_ = re.search('/models--.+?--(.+?)/snapshots/', model_id_or_path)
        if match_ is not None:
            return match_.group(1)
        model_name = model_id_or_path.rsplit('/', 1)[-1]
        model_name = model_name.replace('___', '.')
        return model_name

    @staticmethod
    def create(model_dir: str, quantization_config=None) -> 'ModelInfo':
        from transformers import AutoConfig, PretrainedConfig
        from surogate.core.model.hf_config import HfConfigFactory

        try:
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        except Exception:
            config = PretrainedConfig.get_config_dict(model_dir)[0]

        if quantization_config is not None:
            HfConfigFactory.set_config_attr(config, 'quantization_config', quantization_config)

        quant_info = HfConfigFactory.get_quant_info(config) or {}
        if not quant_info:
            quant_info = HfConfigFactory.get_quant_info_from_hf_quant_config(model_dir) or {}
        torch_dtype = HfConfigFactory.get_torch_dtype(config, quant_info)
        max_model_len = HfConfigFactory.get_max_model_len(config)
        rope_scaling = HfConfigFactory.get_config_attr(config, 'rope_scaling')
        is_moe_model = HfConfigFactory.is_moe_model(config)
        is_multimodal = HfConfigFactory.is_multimodal(config)

        return ModelInfo(
            model_dir,
            torch_dtype,
            max_model_len,
            quant_info.get('quant_method'),
            quant_info.get('quant_bits'),
            rope_scaling=rope_scaling,
            is_moe_model=is_moe_model,
            is_multimodal=is_multimodal,
            quant_info=quant_info or None,
        )


def _load_tokenizer(model_dir: str):
    """Load tokenizer (or processor for multimodal) from model_dir."""
    from transformers import AutoTokenizer
    if os.path.exists(os.path.join(model_dir, 'preprocessor_config.json')):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def _load_config(model_dir: str):
    """Load HF config from model_dir."""
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(model_dir, trust_remote_code=True)


def _postprocess_config(model_info: ModelInfo, config, rope_scaling=None, max_model_len=None):
    """Apply dtype / rope / max_model_len overrides to the loaded config."""
    from surogate.core.model.hf_config import HfConfigFactory

    HfConfigFactory.set_config_attr(config, 'torch_dtype', model_info.torch_dtype, include_vit=True)
    if rope_scaling:
        rope_parameters = HfConfigFactory.get_config_attr(config, 'rope_parameters') or {}
        for key in ['rope_theta', 'partial_rotary_factor']:
            if rope_scaling.get(key) is None and rope_parameters.get(key) is not None:
                rope_scaling[key] = rope_parameters[key]
        HfConfigFactory.set_config_attr(config, 'rope_scaling', rope_scaling)
    if max_model_len:
        HfConfigFactory.set_max_model_len(config, max_model_len)
    model_info.config = config
    return config


def _postprocess_tokenizer(tokenizer):
    """Ensure pad_token_id and eos_token_id are set. Unwrap processor to raw tokenizer."""
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer = tokenizer.tokenizer

    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = tokenizer.eos_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = pad_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token
    assert tokenizer.eos_token_id is not None
    assert tokenizer.pad_token_id is not None
    return tokenizer


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

    from surogate.core.hub.huggingface import HuggingFaceHub
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
        if model_id_or_path.startswith('/'):
            raise ValueError(f"path: '{model_id_or_path}' not found")
        model_id_or_path = model_id_or_path.split(':', 1)
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


def get_model_info_and_tokenizer(
        model_id_or_path: str,
        torch_dtype=None,
        *,
        load_model: bool = False,
        hub_token: Optional[str] = None,
        download_model: Optional[bool] = None,
        template_type: Optional[str] = None,
        quantization_config=None,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_model_len: Optional[int] = None,
        **kwargs
) -> Tuple[ModelInfo, Any]:
    """Load model info and tokenizer.

    Returns (model_info, tokenizer).
    """
    if download_model is None:
        download_model = load_model

    model_dir = safe_snapshot_download(
        model_id_or_path,
        check_local=True,
        hub_token=hub_token,
        download_model=download_model)

    model_info = ModelInfo.create(model_dir, quantization_config=quantization_config)

    if torch_dtype is None:
        from surogate.core.model.utils import get_default_torch_dtype
        torch_dtype = get_default_torch_dtype(model_info.torch_dtype)
        logger.debug(f'Setting torch_dtype: {torch_dtype}')
    model_info.torch_dtype = torch_dtype

    # Load config and apply overrides
    config = _load_config(model_dir)
    _postprocess_config(model_info, config, rope_scaling=rope_scaling, max_model_len=max_model_len)

    # Load tokenizer (unwraps processor → tokenizer if needed)
    tokenizer = _load_tokenizer(model_dir)
    tokenizer = _postprocess_tokenizer(tokenizer)

    return model_info, tokenizer
