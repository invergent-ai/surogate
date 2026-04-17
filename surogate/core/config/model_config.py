from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger


@dataclass
class ModelConfig(ABC):
    """
    ModelConfig class holds configuration for model loading and setup.

    Args:
        model (Optional[str]): model_id or model_path.
        max_model_len (Optional[int]): Maximum model length for rope scaling.
        rope_scaling: RoPE scaling config string (e.g. 'linear', 'yarn', or JSON).
    """

    model: str | None = None
    torch_dtype: torch.bfloat16 | torch.float16 | torch.float32 | None = None
    max_model_len: int | None = None
    rope_scaling: str | None = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.model = cfg["model"]
        self.torch_dtype = cfg["torch_dtype"]
        self.max_model_len = cfg["max_model_len"]
        self.rope_scaling = cfg["rope_scaling"]

    def __post_init__(self):
        from surogate.utils.model import get_model_name

        self.model_suffix = get_model_name(self.model)
        self.torch_dtype = self._init_model_info()

    def _init_model_info(self):
        from surogate.core.model.registry import get_model_info_and_tokenizer

        logger = get_logger()
        logger.debug("init model info...")
        self.model_info, self.tokenizer = get_model_info_and_tokenizer(
            model_id_or_path=self.model,
            torch_dtype=self.torch_dtype,
            rope_scaling=self.rope_scaling,
            max_model_len=self.max_model_len,
            load_model=False,
            download_model=True,
        )
        self.model_dir = self.model_info.model_dir
        self.is_multimodal = self.model_info.is_multimodal

        if self.model_info.rope_scaling and self.max_model_len is not None:
            self._init_rope_scaling()

        return self.model_info.torch_dtype

    def _init_rope_scaling(self):
        logger = get_logger()
        logger.debug("preparing rope_scaling...")
        if self.rope_scaling:
            from surogate.utils.jsonl import json_parse_to_dict

            rope_scaling: dict = json_parse_to_dict(self.rope_scaling, strict=False)
            if isinstance(rope_scaling, str):
                assert rope_scaling in ["linear", "dynamic", "yarn"]
                rope_scaling = {"type": rope_scaling}
        else:
            rope_scaling = self.model_info.rope_scaling
            rope_scaling.pop("factor", None)

        if "factor" not in rope_scaling and self.max_model_len is None:
            self.rope_scaling = rope_scaling
            logger.info(f"Setting args.rope_scaling: {rope_scaling}")
            return

        origin_max_model_len = None
        if rope_scaling and rope_scaling.get("original_max_position_embeddings") is not None:
            origin_max_model_len = rope_scaling["original_max_position_embeddings"]
        elif self.model_info.rope_scaling:
            if self.model_info.rope_scaling.get("original_max_position_embeddings") is not None:
                origin_max_model_len = self.model_info.rope_scaling["original_max_position_embeddings"]
            elif self.model_info.rope_scaling.get("factor") is not None:
                origin_max_model_len = self.model_info.max_model_len // self.model_info.rope_scaling["factor"]
        if origin_max_model_len is None:
            origin_max_model_len = self.model_info.max_model_len
        assert origin_max_model_len is not None, "`origin_max_model_len` from model config is not set"
        rope_scaling["original_max_position_embeddings"] = origin_max_model_len

        if "factor" not in rope_scaling:
            rope_scaling["factor"] = max(float(math.ceil(self.max_model_len / origin_max_model_len)), 1.0)
        rope_model_len = int(origin_max_model_len * rope_scaling["factor"])
        if self.max_model_len is None:
            self.max_model_len = rope_model_len
        elif self.max_model_len > rope_model_len:
            logger.warning(
                f"rope config ({rope_model_len} = {rope_scaling['factor']} * "
                f"{origin_max_model_len}) should be bigger than max_model_len "
                f"from command line ({self.max_model_len})"
            )
        self.rope_scaling = rope_scaling
        logger.info(f"Setting args.rope_scaling: {rope_scaling}")
        logger.info(f"Setting args.max_model_len: {self.max_model_len}")
