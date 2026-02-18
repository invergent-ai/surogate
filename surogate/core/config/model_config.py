import math
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Union, Any

import torch
from transformers.utils.quantization_config import QuantizationConfigMixin

from surogate.core.model.loader import get_model_info_and_tokenizer
from surogate.utils.dict import DictDefault
from surogate.utils.jsonl import json_parse_to_dict
from surogate.utils.logger import get_logger
from surogate.utils.model import get_model_name

@dataclass
class ModelConfig(ABC):
    """
    ModelConfig class is a dataclass that holds configuration parameters for quantizing a model using SurogatePtq.

    Args:
        model (Optional[str]): model_id or model_path. Default is None.
        model_type (Optional[str]): Type of the model group. Default is None.
        max_model_len (Optional[int]): Maximum model length for rope scaling. Default is None.
        rope_scaling (Literal): Type of RoPE scaling. Only relevant for vision-language models — it is applied
            to the HuggingFace vision model loaded in Python for multi-modal preprocessing (on_the_fly_mm.py).
            Has no effect for pure LLM fine-tuning: the C++ training engine reads rope config directly from
            config.json on disk and ignores this value. You can pass a string such as 'linear', 'dynamic', 'yarn',
            or a JSON string like '{"factor": 2.0, "type": "yarn"}'. Default is None.
    """
    model: Optional[str] = None
    model_type: Optional[str] = None
    torch_dtype: Optional[Union[torch.bfloat16, torch.float16, torch.float32]] = None
    max_model_len: Optional[int] = None
    rope_scaling: Optional[str] = None
    quantization_config: Optional[QuantizationConfigMixin] = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.model = cfg['model']
        self.model_type = cfg['model_type']
        self.torch_dtype = cfg['torch_dtype']
        self.max_model_len = cfg['max_model_len']
        self.rope_scaling = cfg['rope_scaling']

    def __post_init__(self):
        self.model_suffix = get_model_name(self.model)
        self.torch_dtype = self._init_model_info()


    def _init_model_info(self) -> torch.dtype:
        logger = get_logger()
        logger.debug("init model info and template...")
        self.model_info, self.model_template, self._model, self.tokenizer = get_model_info_and_tokenizer(**self.get_model_kwargs(), load_model=False, download_model=True)
        self.model_dir = self.model_info.model_dir
        self.model_type = self.model_info.model_type

        if self.model_info.rope_scaling and self.max_model_len is not None:
            self._init_rope_scaling()

        return self.model_info.torch_dtype

    def _init_rope_scaling(self):
        logger = get_logger()
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


    def _get_modules_to_skip_quant(self):
        return [
            'lm_head',
            'multi_modal_projector', 'modality_projection', 'vision_tower', 'aligner', 'merger',  # multi-modal
            'router', 'mlp.gate', 'mlp.shared_expert_gate', 'block_sparse_moe.gate',  # MoE
            'mamba'
        ]

    def get_model_kwargs(
            self,
            model_id_or_path: Optional[str] = None,
            torch_dtype: Optional[torch.dtype] = None,
            model_type: Optional[str] = None,
            rope_scaling: Optional[Union[str, dict]] = None,
            max_model_len: Optional[int] = None,
    ) -> dict[str, Any]:
        return {
            'model_id_or_path': model_id_or_path or self.model,
            'torch_dtype': torch_dtype or self.torch_dtype,
            'model_type': model_type or self.model_type,
            'rope_scaling': rope_scaling or self.rope_scaling,
            'max_model_len': max_model_len or self.max_model_len,
        }
