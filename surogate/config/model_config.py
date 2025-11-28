from abc import ABC
from dataclasses import dataclass
from typing import Optional

from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()

@dataclass
class ModelConfig(ABC):
    """
    ModelConfig class is a dataclass that holds configuration parameters for quantizing a model using SurogatePtq.

    Args:
        model (Optional[str]): model_id or model_path. Default is None.
        model_type (Optional[str]): Type of the model group. Default is None.
    """
    model: Optional[str] = None
    model_type: Optional[str] = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.model = cfg['model']
        self.model_type = cfg['model_type']


