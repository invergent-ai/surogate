import os
from contextlib import contextmanager
from typing import List

from datasets import Dataset, IterableDataset, DatasetDict, IterableDatasetDict, interleave_datasets, disable_caching, \
    enable_caching

from surogate.core.config.dataset_config import DatasetConfig, ConversationDatasetConfig, InstructionDatasetConfig, \
    TextDatasetConfig
from surogate.core.config.enums import SurogateDatasetType
from surogate.core.datasets.preprocessor.conversation import ConversationPreprocessor
from surogate.core.datasets.preprocessor.instruction import InstructionPreprocessor
from surogate.core.datasets.loader import load_dataset_with_config
from surogate.core.datasets.lock import FileLockLoader
from surogate.core.datasets.preprocessor.text import TextPreprocessor
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()


def load_datasets(cfg: List[DatasetConfig], args: DictDefault, temp_path: str, seed: int) -> Dataset | IterableDataset:
    # Prepare datasets (with file locking logic for multiple ranks)
    loader = FileLockLoader(temp_path)
    try:
        disable_caching()
        dataset = loader.load(lambda: _load_and_prepare_datasets(cfg, args, seed))
    finally:
        loader.cleanup()
        enable_caching()

    return dataset


def _load_and_prepare_datasets(cfg: List[DatasetConfig], args: DictDefault, seed: int) -> Dataset | IterableDataset:
    datasets = []
    for dataset_config in cfg:
        dataset_wrapper = _load_and_prepare_single_dataset(args, dataset_config)
        datasets.append(dataset_wrapper)

    dataset = merge_datasets(datasets, seed)

    return dataset


def _load_and_prepare_single_dataset(args: DictDefault, ds_cfg: DatasetConfig) -> Dataset | IterableDataset:
    dataset = load_dataset_with_config(ds_cfg, args)

    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        if ds_cfg.split and ds_cfg.split in dataset:
            dataset = dataset[ds_cfg.split]
        else:
            raise ValueError(
                f"no {ds_cfg.split} split found for dataset {ds_cfg.path}, you may "
                "specify a split with 'split: ...'"
            )

    if ds_cfg.samples:
        dataset = dataset.select(range(min(dataset.num_rows, ds_cfg.samples)))

    return wrap_dataset(ds_cfg=ds_cfg, dataset=dataset)


def wrap_dataset(ds_cfg: DatasetConfig, dataset: Dataset | IterableDataset) -> Dataset | IterableDataset:
    ds_columns = dataset.column_names
    ds_cfg.validate_columns(ds_columns)

    if ds_cfg.type == SurogateDatasetType.conversation:
        ds_cfg = ds_cfg if isinstance(ds_cfg, ConversationDatasetConfig) else ConversationDatasetConfig(
            **ds_cfg.__dict__)
        processor = ConversationPreprocessor(ds_cfg)
    elif ds_cfg.type == SurogateDatasetType.instruction:
        ds_cfg = ds_cfg if isinstance(ds_cfg, InstructionDatasetConfig) else InstructionDatasetConfig(
            **ds_cfg.__dict__)
        processor = InstructionPreprocessor(ds_cfg)
    elif ds_cfg.type == SurogateDatasetType.text:
        ds_cfg = ds_cfg if isinstance(ds_cfg, TextDatasetConfig) else TextDatasetConfig(
            **ds_cfg.__dict__)
        processor = TextPreprocessor(ds_cfg)
    else:
        raise ValueError(f"Unsupported dataset type: {ds_cfg.type}")

    return processor(dataset, num_proc=get_default_process_count(), load_from_cache_file=False, strict=False)


def merge_datasets(datasets: list[Dataset], seed: int) -> Dataset:
    """Merge multiple datasets into one with optional shuffling.

    Args:
        datasets: List of datasets to merge.
        seed: Random seed for shuffling.

    Returns:
        Merged dataset.
    """
    if len(datasets) == 1:
        ds = datasets[0]
        return ds.shuffle(seed=seed)

    logger.info("Interleaving datasets...")
    merged_dataset = interleave_datasets(datasets)
    merged_dataset = merged_dataset.shuffle(seed=seed)
    return merged_dataset


@contextmanager
def disable_datasets_caching():
    try:
        disable_caching()
        yield
    finally:
        enable_caching()


def get_default_process_count():
    if dataset_processes := os.environ.get("SUROGATE_DATASET_PROCESSES"):
        return int(dataset_processes)
    if runpod_cpu_count := os.environ.get("RUNPOD_CPU_COUNT"):
        return int(runpod_cpu_count)
    return os.cpu_count()
