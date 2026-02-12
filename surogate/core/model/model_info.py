import re
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, List

import torch
from transformers import PretrainedConfig, AutoConfig

from surogate.core.model.hf_config import HfConfigFactory
from surogate.core.model.registry import MODEL_MAPPING
from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class ModelInfo:
    model_type: str
    native_model_type: str
    model_dir: str
    torch_dtype: torch.dtype
    max_model_len: int
    quant_method: Literal['gptq', 'awq', 'bnb', 'prequant_fp8', 'prequant_nvfp4', 'prequant_mxfp4', None]
    quant_bits: int

    # extra
    rope_scaling: Optional[Dict[str, Any]] = None
    is_moe_model: bool = False
    config: Optional[PretrainedConfig] = None
    quant_info: Optional[Dict[str, Any]] = None

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
    def create(model_dir: str, model_type: Optional[str], quantization_config) -> 'ModelInfo':
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
        native_model_type = HfConfigFactory.get_config_attr(config, 'model_type')

        if model_type is None:
            architectures = HfConfigFactory.get_config_attr(config, 'architectures')
            model_types = get_matched_model_types(architectures)
            if len(model_types) > 1:
                if "Qwen3ForCausalLM" in architectures:
                    model_type = "qwen3"
                else:
                    raise ValueError('Failed to automatically match `model_type`. '
                                 f'Please explicitly pass the `model_type` for `{model_dir}`. '
                                 f'Recommended `model_types` include: {model_types}.')
            elif len(model_types) == 1:
                model_type = model_types[0]
        elif model_type not in MODEL_MAPPING:
            raise ValueError(f"model_type: '{model_type}' not in {list(MODEL_MAPPING.keys())}")

        return ModelInfo(
            model_type,
            native_model_type,
            model_dir,
            torch_dtype,
            max_model_len,
            quant_info.get('quant_method'),
            quant_info.get('quant_bits'),
            rope_scaling=rope_scaling,
            is_moe_model=is_moe_model,
            quant_info=quant_info or None,
        )

def get_matched_model_types(architectures: Optional[List[str]]) -> List[str]:
    """Get possible model_type."""
    architectures = architectures or ['null']
    if architectures:
        architectures = architectures[0]
    arch_mapping = _get_arch_mapping()
    return arch_mapping.get(architectures) or []

def _get_arch_mapping():
    res = {}
    for model_type, model_meta in MODEL_MAPPING.items():
        architectures = model_meta.architectures
        if not architectures:
            architectures.append('null')
        for arch in architectures:
            if arch not in res:
                res[arch] = []
            res[arch].append(model_type)
    return res