from abc import ABC
from dataclasses import dataclass
from typing import Optional, List

from surogate.core.config.dataset_config import SurogateDatasetConfig, create_dataset_config
from surogate.core.datasets.datasets import get_default_process_count
from surogate.utils.dict import DictDefault
from surogate.utils.seed import RAND_SEED


@dataclass
class TrainDatasetConfig(ABC):
    """
    SFTConfig class is a dataclass that holds configuration parameters for Supervised Fine-Tuning (SFT)

    Args:
        train_seed (Optional[int], defaults to 1234):
            Seed for the training dataloader
        eval_seed (Optional[int], defaults to 5678):
            Seed for the evaluation dataloader
        datasets (Optional[List[DatasetConfig]]):
            List of datasets for training. Default is None.
        validation_datasets (Optional[List[DatasetConfig]]):
            List of datasets for validation during training. Default is None.
        validation_split_ratio (Optional[float]):
            Ratio of training data to use for validation if no validation_datasets are provided. Default is 0.1.
        dataloader_num_workers (Optional[int], defaults to 0):
            Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the
            main process.
        sample_packing (Optional[bool], defaults to True):
            Whether to enable sample packing to fit more data samples into a single sequence.
            Packing reduces the number of samples in the dataset; please adjust the gradient accumulation steps and
            learning rate accordingly.
        sequence_len (Optional[int], defaults to None):
            Maximum token length after tokenizer.encode for a single data sample (to prevent OOM during training).
            Samples exceeding this limit are truncated to this length.
            Default is None, meaning it’s set to the model’s maximum supported sequence length (i.e., max_model_len).
    """
    train_seed: Optional[int]  = RAND_SEED
    eval_seed: Optional[int]  = RAND_SEED
    datasets: Optional[List[SurogateDatasetConfig]] = None
    validation_datasets: Optional[List[SurogateDatasetConfig]] = None
    validation_split_ratio: Optional[float] = 0.1
    dataloader_num_workers: Optional[int] = None
    sample_packing: Optional[bool] = True
    sequence_len: Optional[int] = 1024

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.train_seed = cfg.get('train_seed', self.train_seed)
        self.eval_seed = cfg.get('eval_seed', self.eval_seed)
        self.datasets = [create_dataset_config(ds_cfg) for ds_cfg in cfg.get('datasets', [])]
        self.validation_datasets = [create_dataset_config(ds_cfg) for ds_cfg in cfg.get('validation_datasets', [])]
        self.validation_split_ratio = cfg.get('validation_split_ratio', self.validation_split_ratio)
        self.dataloader_num_workers = min(cfg.get('dataloader_num_workers', get_default_process_count()), 8)
        self.sample_packing = cfg.get('sample_packing', self.sample_packing)
        self.sequence_len = cfg.get('sequence_len', self.sequence_len)

    def __post_init__(self):
        super().__post_init__()