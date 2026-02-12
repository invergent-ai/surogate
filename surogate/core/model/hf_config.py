import json
import os
from typing import Union, Dict, Any, Optional, List, Tuple

import torch
from transformers import PretrainedConfig

from surogate.utils.utils import deep_getattr


class HfConfigFactory:
    @staticmethod
    def get_torch_dtype(config: Union[PretrainedConfig, Dict[str, Any]],
                        quant_info: Dict[str, Any]) -> Optional[torch.dtype]:
        for key in ['torch_dtype', 'params_dtype']:
            torch_dtype = HfConfigFactory.get_config_attr(config, key)
            if torch_dtype is not None:
                break
        torch_dtype = HfConfigFactory.to_torch_dtype(torch_dtype)
        if torch_dtype is None:
            torch_dtype = quant_info.get('torch_dtype')
        return torch_dtype

    @staticmethod
    def is_moe_model(config) -> bool:
        if 'Moe' in config.__class__.__name__:
            return True
        for key in ['num_experts', 'num_experts_per_tok', 'moe_intermediate_size']:
            if HfConfigFactory.get_config_attr(config, key):
                return True
        return False

    @staticmethod
    def get_max_model_len(config: Union[PretrainedConfig, Dict[str, Any]]) -> Optional[int]:
        """Get the max length supported by the model"""
        INF = int(1e9)
        max_model_len = INF

        possible_keys = [
            'seq_length',  # qwen, chatglm
            'max_position_embeddings',  # qwen1.5, llama2
            'n_positions',  # polylm, phi-2
            'model_max_length',  # baichuan2
            # others
            'seq_len',
            'max_seq_len',
            'max_sequence_length',
            'max_seq_length',
        ]
        for key in possible_keys:
            max_len_key = HfConfigFactory.get_config_attr(config, key)
            if max_len_key is not None:
                max_model_len = min(max_model_len, max_len_key)
        if max_model_len == INF:
            max_model_len = None
        return max_model_len

    @staticmethod
    def set_max_model_len(config: Union[PretrainedConfig, Dict[str, Any]], value: int):
        """Set the max length supported by the model"""

        possible_keys = [
            'seq_length',  # qwen, chatglm
            'max_position_embeddings',  # qwen1.5, llama2
            'n_positions',  # polylm, phi-2
            'model_max_length',  # baichuan2
            # others
            'seq_len',
            'max_seq_len',
            'max_sequence_length',
            'max_seq_length',
        ]
        for key in possible_keys:
            max_len_value = HfConfigFactory.get_config_attr(config, key)
            if max_len_value is not None:
                HfConfigFactory.set_config_attr(config, key, value)

    @staticmethod
    def get_quant_info(config: Union[PretrainedConfig, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get quant_method, quant_bits, dtype. not support hqq/eetq now, support awq/gptq/bnb/aqlm"""
        if isinstance(config, dict):
            quantization_config = config.get('quantization_config')
        else:
            quantization_config = getattr(config, 'quantization_config', None)
        if quantization_config is None:
            return
        quantization_config = dict(quantization_config)
        quant_method = quantization_config.get('quant_method')
        res = {}
        if quant_method in {'gptq', 'awq', 'aqlm'}:
            res['quant_method'] = quant_method
            res['torch_dtype'] = torch.float16
            quant_bits = quantization_config.get('bits')
            if quant_bits is not None:
                res['quant_bits'] = quant_bits
        elif quant_method == 'bitsandbytes':
            res['quant_method'] = 'bnb'
            load_in_4bit = quantization_config.get('_load_in_4bit')
            load_in_8bit = quantization_config.get('_load_in_8bit')
            bnb_4bit_compute_dtype = quantization_config.get('bnb_4bit_compute_dtype')
            if load_in_4bit:
                res['quant_bits'] = 4
            elif load_in_8bit:
                res['quant_bits'] = 8
            res['torch_dtype'] = HfConfigFactory.to_torch_dtype(bnb_4bit_compute_dtype)
        elif quant_method == 'hqq':
            res['quant_method'] = quant_method
            res['quant_bits'] = quantization_config['quant_config']['weight_quant_params']['nbits']
        elif quant_method == 'fp8':
            # Fine-grained FP8 (e.g., DeepSeek-V3/R1): per-block (128x128) FP8 E4M3
            res['quant_method'] = 'prequant_fp8'
            res['quant_bits'] = 8
            res['weight_block_size'] = quantization_config.get('weight_block_size', [128, 128])
            res['modules_to_not_convert'] = quantization_config.get(
                'modules_to_not_convert', quantization_config.get('ignore', []))
        elif quant_method == 'modelopt':
            quant_algo = quantization_config.get('quant_algo', '')
            if quant_algo == 'NVFP4':
                # NVIDIA ModelOpt NVFP4: packed FP4 + FP8 block scales + FP32 global
                res['quant_method'] = 'prequant_nvfp4'
                res['quant_bits'] = 4
                res['modules_to_not_convert'] = quantization_config.get(
                    'ignore', quantization_config.get('modules_to_not_convert', []))
            else:
                res['quant_method'] = quant_method
        elif quant_method == 'mxfp4':
            # Microscaling FP4: packed FP4 E2M1 + E8M0 shared exponents per 32 elements
            res['quant_method'] = 'prequant_mxfp4'
            res['quant_bits'] = 4
            res['modules_to_not_convert'] = quantization_config.get(
                'modules_to_not_convert', quantization_config.get('ignore', []))
        elif quant_method is not None:
            res['quant_method'] = quant_method
        return res or None

    @staticmethod
    def get_quant_info_from_hf_quant_config(model_dir: str) -> Optional[Dict[str, Any]]:
        """Fallback: read quantization info from hf_quant_config.json (ModelOpt convention).

        Some ModelOpt-quantized models (e.g., Nemotron NVFP4) store quantization
        metadata in a separate hf_quant_config.json instead of config.json.
        """
        # Try local path first
        path = os.path.join(model_dir, 'hf_quant_config.json')
        if not os.path.isfile(path):
            # Try resolving from HuggingFace Hub cache
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(model_dir, filename='hf_quant_config.json')
            except Exception:
                return None

        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            return None

        quant = data.get('quantization', {})
        quant_algo = quant.get('quant_algo', '')
        if quant_algo == 'NVFP4':
            return {
                'quant_method': 'prequant_nvfp4',
                'quant_bits': 4,
                'modules_to_not_convert': quant.get('exclude_modules', []),
            }
        elif quant_algo == 'FP8':
            return {
                'quant_method': 'prequant_fp8',
                'quant_bits': 8,
                'modules_to_not_convert': quant.get('exclude_modules', []),
            }
        return None

    @staticmethod
    def to_torch_dtype(torch_dtype: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
        if torch_dtype is None:
            return None
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
        return torch_dtype

    @staticmethod
    def get_config_attr(config: Union[PretrainedConfig, Dict[str, Any]],
                        attr_name: str,
                        include_vit: bool = False) -> Optional[Any]:
        """Get the value of the attribute named attr_name."""
        attrs = HfConfigFactory._get_config_attrs(config, attr_name, include_vit)
        if len(attrs) == 0:
            return None
        else:
            return attrs[0][1]

    @staticmethod
    def set_config_attr(config: Union[PretrainedConfig, Dict[str, Any]],
                        attr_name: str,
                        value: Any,
                        include_vit: bool = False,
                        ensure_set: bool = True) -> int:
        """Set all the attr_name attributes to value."""
        attrs = HfConfigFactory._get_config_attrs(config, attr_name, include_vit)
        if ensure_set and len(attrs) == 0:
            attrs.append((config, None))
        for config, _ in attrs:
            if isinstance(config, dict):
                config[attr_name] = value
            else:
                setattr(config, attr_name, value)
        return len(attrs)

    @staticmethod
    def set_model_config_attr(model, attr_name: str, value: Any) -> None:
        for module in model.modules():
            if getattr(module, 'config', None) and getattr(module.config, attr_name, value) != value:
                setattr(module.config, attr_name, value)

    @staticmethod
    def _get_config_attrs(
            config: Union[PretrainedConfig, Dict[str, Any]],
            attr_name: str,
            include_vit: bool = False,
            parent_key: Optional[str] = None
    ) -> List[Tuple[PretrainedConfig, Any]]:
        res = []
        if isinstance(config, dict):
            keys = config.keys()
        elif isinstance(config, PretrainedConfig):
            keys = dir(config)
        else:
            return []
        config_keys = [None, 'language_config', 'llm_config', 'text_config']
        if include_vit:
            config_keys += ['vit_config', 'vision_config', 'audio_config']
        if attr_name in keys and parent_key in config_keys:
            res.append((config, deep_getattr(config, attr_name)))

        for k in keys:
            if k.endswith('_config'):
                if isinstance(config, dict):
                    v = config[k]
                else:
                    v = getattr(config, k)
                res += HfConfigFactory._get_config_attrs(v, attr_name, include_vit, k)
        return res
