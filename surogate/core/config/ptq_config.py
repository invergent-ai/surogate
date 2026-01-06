from dataclasses import dataclass
from typing import Literal, List, Optional

from surogate.core.config.dataset_config import SurogateDatasetConfig, create_dataset_config
from surogate.core.config.model_config import ModelConfig
from surogate.utils.dict import DictDefault
from surogate.utils.seed import RAND_SEED

@dataclass
class PTQConfig(ModelConfig):
    """
    PTQConfig class is a dataclass that holds configuration parameters for quantizing a model using SurogatePtq.

    Args:
        scheme (Literal['fp8', 'gptq_int4', 'gptq_int8', 'awq', 'nvfp4']): Quantization scheme to use.
        sequence_len (int): The sequence length to use for calibration. Default is 2048.
        datasets (Optional[List[DatasetConfig]]): List of dataset configurations for calibration. Default is None.
        ignore_layers (Optional[List[str]]): List of layer names to ignore during quantization.
        seed (Optional[int]): Random seed for reproducibility. Default is 1234.
        save_path (Optional[str]): Path to save the quantized model. Default is "./output".
    """
    scheme: Optional[Literal['fp8', 'gptq_int4', 'gptq_int8', 'awq', 'nvfp4']] = None
    sequence_len: Optional[int] = None
    datasets: Optional[List[SurogateDatasetConfig]] = None
    ignore_layers: Optional[List[str]] = None
    seed: Optional[int] = None
    save_path: Optional[str] = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.scheme = cfg['scheme']
        self.sequence_len = cfg['sequence_len'] or 2048
        self.datasets = [create_dataset_config(ds_cfg) for ds_cfg in cfg.get('datasets', [])]
        self.ignore_layers = cfg['ignore_layers'] or []
        self.seed = cfg['seed'] or RAND_SEED
        self.save_path = cfg['save_path'] or "./output"
        self.__post_init__()


    def __post_init__(self):
        super().__post_init__()
        if self.scheme in ['gptq_int4', 'gptq_int8', 'awq', 'nvfp4']:
            if not self.datasets:
                raise ValueError(f'Calibration datasets are required for quantization scheme: {self.scheme}')
