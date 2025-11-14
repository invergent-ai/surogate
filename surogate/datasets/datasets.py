import os
from contextlib import contextmanager

from datasets import Dataset, IterableDataset, DatasetDict, IterableDatasetDict, interleave_datasets, disable_caching, \
    enable_caching
from transformers import PreTrainedTokenizerBase

from surogate.datasets.conversation import ConversationPreprocessor
from surogate.datasets.instruction import InstructionPreprocessor
from surogate.datasets.loader import load_dataset_with_config
from surogate.datasets.lock import FileLockLoader
from surogate.datasets.text import TextPreprocessor
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger
from surogate.utils.schema.datasets import SurogateDataset, InstructionDataset, ConversationDataset, TextDataset
from surogate.utils.schema.enums import SurogateDatasetType

logger = get_logger()


def load_datasets(cfg: DictDefault) -> Dataset | IterableDataset:
    # Prepare datasets (with file locking logic for multiple ranks)
    loader = FileLockLoader(cfg)
    try:
        disable_caching()
        dataset = loader.load(lambda: _load_and_prepare_datasets(cfg))
    finally:
        loader.cleanup()
        enable_caching()

    return dataset


def _load_and_prepare_datasets(cfg: DictDefault) -> Dataset | IterableDataset:
    datasets_configs = cfg.get('datasets')
    datasets = []
    for dataset_config in datasets_configs:
        dataset_wrapper = _load_and_prepare_single_dataset(
            cfg, dataset_config
        )
        datasets.append(dataset_wrapper)

    dataset = merge_datasets(datasets, cfg)

    return dataset


def _load_and_prepare_single_dataset(cfg: DictDefault, ds_cfg: SurogateDataset) -> Dataset | IterableDataset:
    dataset = load_dataset_with_config(
        ds_cfg, False
    )

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

    return wrap_dataset(
        cfg=cfg,
        ds_cfg=ds_cfg,
        dataset=dataset,
    )


def wrap_dataset(
        cfg: DictDefault,
        ds_cfg: SurogateDataset,
        dataset: Dataset | IterableDataset
) -> Dataset | IterableDataset:
    ds_columns = dataset.column_names
    if ds_cfg.type == SurogateDatasetType.conversation:
        ConversationDataset.validate_fields(ds_cfg, ds_columns)
        processor = ConversationPreprocessor(cfg, ds_cfg)
    elif ds_cfg.type == SurogateDatasetType.instruction:
        InstructionDataset.validate_fields(ds_cfg, ds_columns)
        processor = InstructionPreprocessor(cfg, ds_cfg)
    elif ds_cfg.type == SurogateDatasetType.text:
        TextDataset.validate_fields(ds_cfg, ds_columns)
        processor = TextPreprocessor(cfg, ds_cfg)
    else:
        raise ValueError(f"Unsupported dataset type: {ds_cfg.type}")

    return processor(dataset, num_proc=get_default_process_count(), load_from_cache_file=False, strict=False)


def merge_datasets(datasets: list[Dataset], cfg: DictDefault) -> Dataset:
    """Merge multiple datasets into one with optional shuffling.

    Args:
        datasets: List of datasets to merge.
        cfg: Configuration object containing shuffle settings.

    Returns:
        Merged dataset.
    """
    if len(datasets) == 1:
        ds = datasets[0]
        return ds.shuffle(seed=cfg.seed)

    logger.info("Interleaving datasets...")
    merged_dataset = interleave_datasets(datasets)
    merged_dataset = merged_dataset.shuffle(seed=cfg.seed)
    return merged_dataset


@contextmanager
def disable_datasets_caching():
    try:
        disable_caching()
        yield
    finally:
        enable_caching()


def get_default_process_count():
    if axolotl_dataset_processes := os.environ.get("SUROGATE_DATASET_PROCESSES"):
        return int(axolotl_dataset_processes)
    if runpod_cpu_count := os.environ.get("RUNPOD_CPU_COUNT"):
        return int(runpod_cpu_count)
    return os.cpu_count()
