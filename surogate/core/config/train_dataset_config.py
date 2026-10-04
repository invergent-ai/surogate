import os
from abc import ABC
from dataclasses import dataclass

from surogate.core.config.dataset_config import SurogateDatasetConfig, create_dataset_config
from surogate.utils.dict import DictDefault


def _get_default_process_count():
    if dataset_processes := os.environ.get("SUROGATE_DATASET_PROCESSES"):
        return int(dataset_processes)
    if runpod_cpu_count := os.environ.get("RUNPOD_CPU_COUNT"):
        return int(runpod_cpu_count)
    return os.cpu_count()


from surogate.utils.seed import RAND_SEED


def _parse_sample_packing(value) -> bool | str:
    """`sample_packing` is true, false or "bin"; YAML strings like "false" mean the boolean."""
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "false"):
        return text == "true"
    if text == "bin":
        return "bin"
    raise ValueError(f"sample_packing must be true, false or bin, got {value!r}")


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
        sample_packing (Optional[bool | str], defaults to True):
            Whether to enable sample packing to fit more data samples into a single sequence.
            Packing reduces the number of samples in the dataset; please adjust the gradient accumulation steps and
            learning rate accordingly. `true` concatenates samples and cuts the stream every `sequence_len` tokens
            (a sample can be split across two sequences); `bin` packs whole samples with Next-k-Fit
            (surogate/train/bin_packing.py); `false` gives each sample its own padded sequence.
        sample_packing_bins (Optional[int], defaults to 64):
            Open sequences `sample_packing: bin` fills at once (the k of Next-k-Fit). Larger values pad less.
        sequence_len (Optional[int], defaults to None):
            Maximum token length after tokenizer.encode for a single data sample (to prevent OOM during training).
            Samples exceeding this limit are truncated to this length.
            Default is None, meaning it’s set to the model’s maximum supported sequence length (i.e., max_model_len).
    """

    train_seed: int | None = RAND_SEED
    eval_seed: int | None = RAND_SEED
    datasets: list[SurogateDatasetConfig] | None = None
    validation_datasets: list[SurogateDatasetConfig] | None = None
    validation_split_ratio: float | None = 0.1
    dataloader_num_workers: int | None = None
    sample_packing: bool | str | None = True
    sample_packing_bins: int | None = 64
    sequence_len: int | None = 1024

    def __init__(self, cfg: DictDefault):
        self.train_seed = cfg.get("train_seed", self.train_seed)
        self.eval_seed = cfg.get("eval_seed", self.eval_seed)
        self.datasets = [create_dataset_config(ds_cfg) for ds_cfg in cfg.get("datasets", [])]
        self.validation_datasets = [create_dataset_config(ds_cfg) for ds_cfg in cfg.get("validation_datasets", [])]
        self.validation_split_ratio = cfg.get("validation_split_ratio", self.validation_split_ratio)
        self.dataloader_num_workers = min(cfg.get("dataloader_num_workers", _get_default_process_count()), 8)
        self.sample_packing = _parse_sample_packing(cfg.get("sample_packing", self.sample_packing))
        self.sample_packing_bins = int(cfg.get("sample_packing_bins", self.sample_packing_bins))
        if self.sample_packing_bins < 1:
            raise ValueError(f"sample_packing_bins must be >= 1, got {self.sample_packing_bins}")
        self.sequence_len = cfg.get("sequence_len", self.sequence_len)

    def __post_init__(self):
        pass
