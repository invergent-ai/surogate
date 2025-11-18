from abc import ABC
from dataclasses import dataclass
from typing import Optional

from swift.llm import MODEL_MAPPING

from surogate.loaders.loader import get_model_info_and_meta
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class ModelConfig(ABC):
    """
    PTQConfig class is a dataclass that holds configuration parameters for quantizing a model using SurogatePtq.

    Args:
        model (Optional[str]): model_id or model_path. Default is None.
        model_type (Optional[str]): Type of the model group. Default is None.
    """
    model: Optional[str] = None
    model_type: Optional[str] = None

    def __init__(self, cfg: DictDefault):
        self.model = cfg['model']
        self.model_type = cfg['model_type']

    def __post_init__(self):
        if self.model is None:
            raise ValueError(f'Please set model: <model_id_or_path')
        if self.model_type is not None and self.model_type not in MODEL_MAPPING:
            raise ValueError(
                f'Unsupported model_type: {self.model_type}. Supported model_type: {list(MODEL_MAPPING.keys())}')

        self._init_model_info_and_torch_dtype()

    def _init_model_info_and_torch_dtype(self):
        self.model_info, self.model_meta = get_model_info_and_meta(
            model_id_or_path=self.model,
            model_type=self.model_type,
            use_hf=True
        )
        self.torch_dtype = self.model_info.torch_dtype


