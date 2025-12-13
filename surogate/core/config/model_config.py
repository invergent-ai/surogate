import ast
import math
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Literal, Union, Dict, Any

import torch
from transformers.utils.quantization_config import QuantizationConfigMixin

from surogate.core.model.hf_config import HfConfigFactory
from surogate.core.model.loader import get_model_info_and_template, get_model_tokenizer
from surogate.utils.dict import DictDefault
from surogate.utils.dist import get_dist_setting
from surogate.utils.jsonl import json_parse_to_dict
from surogate.utils.logger import get_logger
from surogate.utils.model import get_model_name

logger = get_logger()


@dataclass
class ModelConfig(ABC):
    task_type: Optional[str] = None

    """
    ModelConfig class is a dataclass that holds configuration parameters for quantizing a model using SurogatePtq.

    Args:
        model (Optional[str]): model_id or model_path. Default is None.
        model_type (Optional[str]): Type of the model group. Default is None.
        torch_dtype (Literal['bfloat16', 'float16', 'float32', None]): Model parameter dtype. Default is None.
        max_model_len (Optional[int]): Maximum model length for rope scaling. Default is None.
        rope_scaling (Literal): Type of RoPE scaling. You can pass a string such as 'linear', 'dynamic', 'yarn',
            along with max_model_len and Surogate will automatically configure the corresponding rope_scaling,
            overriding the value in config.json. Alternatively, pass a JSON string like '{"factor": 2.0, "type": "yarn"}',
            which will directly replace the rope_scaling in config.json. Default is None.
        attn_impl (Optional[str]): Attention implementation to use.
            Options include 'flash_attention_2', 'flash_attention_3','flex_attention' etc.
            Default is None, reading from config.json.
        device_map (Optional[Union[dict, str]]): Device placement configuration for the model, e.g., 'auto', 'cpu', a JSON string,
            or a JSON file path. This argument is passed through to the from_pretrained method in Transformers.
            Default is None, automatically determined based on available devices and distributed training setup.
        max_memory: Optional[Union[dict, str]] = None: When device_map is set to 'auto' or 'sequential',
            model weights are allocated across devices according to max_memory,
            e.g., --max_memory '{0: "20GB", 1: "20GB"}'.
            Default is None. Passed through to the from_pretrained interface in Transformers.
        quant_method: Optional[Literal['bnb_4bit', 'falqon']]: On-the-fly quantization method used when loading the model.
            Default is None.
    """
    model: Optional[str] = None
    model_type: Optional[str] = None
    torch_dtype: Optional[
        Union[Literal['bfloat16', 'float16', 'float32'], Union[torch.bfloat16, torch.float16, torch.float32]]] = None
    max_model_len: Optional[int] = None
    rope_scaling: Optional[str] = None
    attn_impl: Optional[str] = None
    device_map: Optional[Union[dict, str]] = None
    max_memory: Optional[Union[dict, str]] = None
    quant_method: Optional[Literal['bnb_4bit', 'falqon']] = None
    num_labels: Optional[int] = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.model = cfg['model']
        self.model_type = cfg['model_type']
        self.torch_dtype = cfg['torch_dtype']
        self.max_model_len = cfg['max_model_len']
        self.rope_scaling = cfg['rope_scaling']
        self.attn_impl = cfg['attn_impl']
        self.device_map = cfg['device_map']
        self.max_memory = cfg['max_memory']
        self.quant_method = cfg['quant_method']

    def __post_init__(self):
        self.model_suffix = get_model_name(self.model)
        self._init_device_map()
        self._init_max_memory()
        self._init_torch_dtype()

    def _init_max_memory(self):
        logger.debug("preparing max_memory...")
        if isinstance(self.max_memory, str):
            try:
                self.max_memory = ast.literal_eval(self.max_memory)
            except Exception:
                pass
        self.max_memory = json_parse_to_dict(self.max_memory)
        # mp&ddp
        _, local_rank, _, local_world_size = get_dist_setting()
        if local_world_size > 1 and isinstance(self.max_memory, dict) and local_rank > 0:
            for k in list(self.max_memory.keys()):
                if isinstance(k, int):
                    self.max_memory[k + local_rank] = self.max_memory.pop(k)

    def _init_torch_dtype(self) -> None:
        logger.debug("preparing torch_dtype...")
        self.torch_dtype: Optional[torch.dtype] = HfConfigFactory.to_torch_dtype(self.torch_dtype)
        self.torch_dtype: torch.dtype = self._init_model_info()

    def _init_device_map(self):
        logger.debug("preparing device_map...")
        if self.device_map:
            self.device_map: Union[str, Dict[str, Any], None] = json_parse_to_dict(self.device_map, strict=False)
        # mp&ddp
        _, local_rank, _, local_world_size = get_dist_setting()
        if local_world_size > 1 and isinstance(self.device_map, dict) and local_rank > 0:
            for k, v in self.device_map.items():
                if isinstance(v, int):
                    self.device_map[k] += local_rank

    def _init_model_info(self) -> torch.dtype:
        logger.debug("init model info and template...")
        self.model_info, self.model_template = get_model_info_and_template(**self.get_model_kwargs())
        self.task_type = self.model_info.task_type
        self.num_labels = self.model_info.num_labels
        self.model_dir = self.model_info.model_dir
        self.model_type = self.model_info.model_type

        if self.model_info.rope_scaling and self.max_model_len is not None:
            self._init_rope_scaling()

        return self.model_info.torch_dtype

    def _init_rope_scaling(self):
        logger.debug("preparing rope_scaling...")
        if self.rope_scaling:
            rope_scaling: dict = json_parse_to_dict(self.rope_scaling, strict=False)
            if isinstance(rope_scaling, str):
                assert rope_scaling in ['linear', 'dynamic', 'yarn']
                rope_scaling = {'type': rope_scaling}
        else:
            rope_scaling = self.model_info.rope_scaling
            # reset the factor
            rope_scaling.pop('factor', None)

        if 'factor' not in rope_scaling and self.max_model_len is None:
            self.rope_scaling = rope_scaling
            logger.info(f'Setting args.rope_scaling: {rope_scaling}')
            return

        origin_max_model_len = None
        if rope_scaling and rope_scaling.get('original_max_position_embeddings') is not None:
            origin_max_model_len = rope_scaling['original_max_position_embeddings']
        elif self.model_info.rope_scaling:
            if self.model_info.rope_scaling.get('original_max_position_embeddings') is not None:
                origin_max_model_len = self.model_info.rope_scaling['original_max_position_embeddings']
            elif self.model_info.rope_scaling.get('factor') is not None:
                origin_max_model_len = self.model_info.max_model_len // self.model_info.rope_scaling['factor']
        if origin_max_model_len is None:
            origin_max_model_len = self.model_info.max_model_len
        assert origin_max_model_len is not None, '`origin_max_model_len` from model config is not set'
        rope_scaling['original_max_position_embeddings'] = origin_max_model_len

        if 'factor' not in rope_scaling:
            rope_scaling['factor'] = max(float(math.ceil(self.max_model_len / origin_max_model_len)), 1.0)
        rope_model_len = int(origin_max_model_len * rope_scaling['factor'])
        if self.max_model_len is None:
            self.max_model_len = rope_model_len
        elif self.max_model_len > rope_model_len:
            logger.warning(f'rope config ({rope_model_len} = {rope_scaling["factor"]} * '
                           f'{origin_max_model_len}) should be bigger than max_model_len '
                           f'from command line ({self.max_model_len})')
        self.rope_scaling = rope_scaling
        logger.info(f'Setting args.rope_scaling: {rope_scaling}')
        logger.info(f'Setting args.max_model_len: {self.max_model_len}')

    def _init_quantization_config(self) -> Optional[QuantizationConfigMixin]:
        if self.quant_method is None:
            return None

        assert self.quant_method in {'bnb_4bit', 'falqon'}

        if self.quant_method == 'bnb_4bit':
            from transformers import BitsAndBytesConfig
            bnb_4bit_compute_dtype = None
            if self.torch_dtype in {torch.float16, torch.float32}:
                bnb_4bit_compute_dtype = torch.float32
            elif self.torch_dtype == torch.bfloat16:
                bnb_4bit_compute_dtype = torch.bfloat16

            bnb_4bit_compute_dtype: torch.dtype = HfConfigFactory.to_torch_dtype(bnb_4bit_compute_dtype)

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_quant_storage=torch.bfloat16,
                llm_int8_skip_modules=self._get_modules_to_skip_quant())
        else:
            raise ValueError(f'Unsupported quantization method: {self.quant_method}')

        return quantization_config

    def _get_modules_to_skip_quant(self):
        model_arch = self.model_template.model_arch
        res = []
        if self.model_info.is_moe_model:
            res += ['mlp.gate', 'mlp.shared_expert_gate']
        if model_arch is not None:
            for key in ['vision_tower', 'aligner']:
                value = getattr(model_arch, key, None)
                if value:
                    res += value
        if not res:
            return None
        res.append('lm_head')
        return res

    def get_model_kwargs(
            self,
            model_id_or_path: Optional[str] = None,
            torch_dtype: Optional[torch.dtype] = None,
            model_type: Optional[str] = None,
            device_map: Optional[Dict[str, torch.device]] = None,
            max_memory: Optional[Dict[str, str]] = None,
            quantization_config: Optional[QuantizationConfigMixin] = None,
            attn_impl: Optional[str] = None,
            rope_scaling: Optional[Union[str, dict]] = None,
            max_model_len: Optional[int] = None,
            task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'reranker', 'generative_reranker', None] = None,
            num_labels: Optional[int] = None,
    ) -> dict[str, Any]:
        return {
            'model_id_or_path': model_id_or_path or self.model,
            'torch_dtype': torch_dtype or self.torch_dtype,
            'model_type': model_type or self.model_type,
            'device_map': device_map or self.device_map,
            'max_memory': max_memory or self.max_memory,
            'quantization_config': quantization_config or self._init_quantization_config(),
            'attn_impl': attn_impl or self.attn_impl,
            'rope_scaling': rope_scaling or self.rope_scaling,
            'max_model_len': max_model_len or self.max_model_len,
            'task_type': task_type or self.task_type,
            'num_labels': num_labels or self.num_labels,
        }
