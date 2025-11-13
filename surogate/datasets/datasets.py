from contextlib import contextmanager

from datasets import Dataset, IterableDataset, DatasetDict, IterableDatasetDict, interleave_datasets, disable_caching, \
    enable_caching
from transformers import PreTrainedTokenizer

from surogate.datasets.loader import load_dataset_with_config
from surogate.datasets.lock import FileLockLoader
from surogate.datasets.wrapper import get_dataset_wrapper
from surogate.loaders.tokenizer import load_tokenizer
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()


def load_datasets(cfg: DictDefault) -> Dataset:
    tokenizer = load_tokenizer(cfg)

    # Prepare datasets (with file locking logic for multiple ranks)
    loader = FileLockLoader(cfg)
    try:
        disable_caching()
        dataset = loader.load(lambda: _load_and_prepare_datasets(cfg, tokenizer))
    finally:
        loader.cleanup()
        enable_caching()

    return dataset


def _load_and_prepare_datasets(cfg: DictDefault, tokenizer: PreTrainedTokenizer) -> Dataset | IterableDataset:
    datasets_configs = cfg.get('datasets')
    datasets = []
    for dataset_config in datasets_configs:
        dataset = load_dataset_with_config(
            dataset_config, False
        )

        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            if dataset_config.split and dataset_config.split in dataset:
                dataset = dataset[dataset_config.split]
            else:
                raise ValueError(
                    f"no {dataset_config.split} split found for dataset {dataset_config.path}, you may "
                    "specify a split with 'split: ...'"
                )

        if dataset_config.samples:
            dataset = dataset.select(range(min(dataset.num_rows, dataset_config.samples)))

        dataset_wrapper = get_dataset_wrapper(
            cfg=cfg,
            dataset_config=dataset_config,
            tokenizer=tokenizer,
            dataset=dataset,
        )

        datasets.append(dataset_wrapper)

    dataset = merge_datasets(datasets, cfg)

    return dataset


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
