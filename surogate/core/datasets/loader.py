from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from adlfs import AzureBlobFileSystem
from datasets import Dataset as HfDataset
from datasets import IterableDataset, Dataset, DatasetDict, IterableDatasetDict, load_from_disk, load_dataset, \
    concatenate_datasets
from gcsfs import GCSFileSystem
from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError, HFValidationError
from ocifs import OCIFileSystem
from s3fs import S3FileSystem

from surogate.core.config.dataset_config import DatasetConfig
from surogate.core.config.enums import SurogateDatasetType
from surogate.core.datasets.preprocessor.auto import AutoPreprocessor
from surogate.core.datasets.preprocessor.conversation import ConversationPreprocessor
from surogate.core.datasets.preprocessor.instruction import InstructionPreprocessor
from surogate.core.datasets.preprocessor.text import TextPreprocessor
from surogate.core.datasets.utils import DATASET_TYPE, sample_dataset
from surogate.utils.logger import get_logger
from surogate.utils.np_utils import get_seed

logger = get_logger()

EXTENSIONS_TO_DATASET_TYPES = {
    ".parquet": "parquet",
    ".arrow": "arrow",
    ".csv": "csv",
    ".txt": "text",
}


def _shard_dataset_for_node(
    dataset: Dataset,
    node_rank: int,
    num_nodes: int,
) -> Dataset:
    """
    Deterministically shard a dataset for a specific node in distributed training.

    Each node gets a contiguous 1/num_nodes portion of the dataset.
    The last node takes any remainder samples.

    Args:
        dataset: The full dataset to shard.
        node_rank: This node's rank (0-indexed).
        num_nodes: Total number of nodes.

    Returns:
        A subset of the dataset for this node.
    """
    total_samples = len(dataset)

    if total_samples < num_nodes:
        logger.warning(
            f"Dataset has fewer samples ({total_samples}) than nodes ({num_nodes}). "
            f"Some nodes will have no data. Consider reducing num_nodes."
        )
        # Give one sample per node until we run out
        if node_rank < total_samples:
            return dataset.select([node_rank])
        else:
            # Return empty dataset for nodes beyond sample count
            return dataset.select([])

    samples_per_node = total_samples // num_nodes
    start_idx = node_rank * samples_per_node
    # Last node takes remainder
    end_idx = total_samples if node_rank == num_nodes - 1 else start_idx + samples_per_node

    logger.info(f"Node {node_rank}/{num_nodes}: Loading samples {start_idx}-{end_idx} ({end_idx - start_idx} samples)")
    return dataset.select(range(start_idx, end_idx))


def load_dataset_with_config(
        dataset_config: DatasetConfig,
        *,
        streaming=False,
        num_workers: int = 1,
        node_rank: Optional[int] = None,
        num_nodes: Optional[int] = None,
) -> Dataset | IterableDataset:
    load_dataset_kwargs = {
        "split": dataset_config.split if dataset_config.split else None,
        "name": dataset_config.subset,
        "streaming": streaming,
        "num_proc": num_workers,
    }

    dataset = None

    if Path(dataset_config.path).exists():
        dataset = _load_from_local_path(dataset_config, load_dataset_kwargs)
    else:
        is_hub_dataset = _check_if_hub_dataset(dataset_config.path)
        if is_hub_dataset:
            dataset = _load_from_hub(dataset_config, load_dataset_kwargs)
        else:
            remote_fs, storage_options = _get_remote_filesystem(dataset_config.path)
            is_cloud_dataset = False
            if remote_fs:
                try:
                    is_cloud_dataset = remote_fs.exists(dataset_config.path)
                except (FileNotFoundError, ConnectionError):
                    pass

            if is_cloud_dataset:
                dataset = _load_from_cloud(
                    dataset_config, remote_fs, storage_options, load_dataset_kwargs
                )
            elif dataset_config.path.startswith("https://"):
                dataset = _load_from_url(dataset_config, load_dataset_kwargs)

    if dataset is None:
        raise ValueError(
            f"The dataset could not be loaded. This could be due to a misconfigured dataset path "
            f"({dataset_config.path}). Try double-check your path / name. "
            f"This is not caused by the dataset type."
        )

    # Apply node-based sharding for distributed training
    if node_rank is not None and num_nodes is not None and num_nodes > 1:
        if streaming:
            logger.warning(
                "Node-based sharding is not supported for streaming datasets. "
                "Each node will see the full dataset."
            )
        elif isinstance(dataset, Dataset):
            dataset = _shard_dataset_for_node(dataset, node_rank, num_nodes)
        else:
            logger.warning(
                f"Node-based sharding is not supported for dataset type {type(dataset)}. "
                f"Each node will see the full dataset."
            )

    return dataset


def _check_if_hub_dataset(path: str) -> bool:
    """Check if a dataset exists on the HuggingFace Hub."""
    try:
        logger.info_once("Fetching dataset...")
        snapshot_download(
            repo_id=path,
            repo_type="dataset",
            ignore_patterns=["*"]
        )
        return True
    except (
            RepositoryNotFoundError,
            RevisionNotFoundError,
            FileNotFoundError,
            ConnectionError,
            HFValidationError,
            ValueError,
    ):
        return False


def _load_from_local_path(
        dataset_config: DatasetConfig, load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from a local path."""
    local_path = Path(dataset_config.path)

    if local_path.is_dir():
        try:
            logger.info(f"Loading dataset from {local_path}...")
            return load_from_disk(dataset_config.path)
        except FileNotFoundError:
            return load_dataset(dataset_config.path, **load_dataset_kwargs)
    elif local_path.is_file():
        dataset_type = get_dataset_type(dataset_config)
        logger.info(f"Loading dataset from {local_path}...")
        return load_dataset(
            dataset_type,
            data_files=dataset_config.path,
            **load_dataset_kwargs,
        )
    else:
        raise ValueError(
            "Unhandled dataset load: local path exists, but is neither a directory or a file"
        )


def _load_from_hub(
        dataset_config: DatasetConfig, load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from the HuggingFace Hub."""
    return load_dataset(
        dataset_config.path,
        **load_dataset_kwargs,
    )


def _load_from_cloud(
        dataset_config: DatasetConfig,
        remote_fs: S3FileSystem | GCSFileSystem | AzureBlobFileSystem | OCIFileSystem,
        storage_options: dict,
        load_dataset_kwargs: dict,
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from cloud storage."""
    if remote_fs.isdir(dataset_config.path):
        return load_from_disk(
            dataset_config.path,
            storage_options=storage_options,
        )

    if remote_fs.isfile(dataset_config.path):
        dataset_type = get_dataset_type(dataset_config)
        return load_dataset(
            dataset_type,
            data_files=dataset_config.path,
            storage_options=storage_options,
            **load_dataset_kwargs,
        )

    raise ValueError(
        f"Cloud path {dataset_config.path} is neither a directory nor a file"
    )


def _load_from_url(
        dataset_config: DatasetConfig, load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from a URL."""
    dataset_type = get_dataset_type(dataset_config)
    return load_dataset(
        dataset_type,
        data_files=dataset_config.path,
        **load_dataset_kwargs,
    )


def _get_remote_filesystem(
        path: str,
) -> tuple[
    S3FileSystem | GCSFileSystem | AzureBlobFileSystem | OCIFileSystem | None, dict
]:
    """Get the appropriate filesystem for a remote path."""
    if path.startswith("s3://"):
        try:
            import s3fs

            storage_options = {"anon": False}
            return s3fs.S3FileSystem(**storage_options), storage_options
        except ImportError as exc:
            raise ImportError("s3:// paths require s3fs to be installed") from exc

    elif path.startswith(("gs://", "gcs://")):
        try:
            import gcsfs

            storage_options = {"token": None}  # type: ignore
            return gcsfs.GCSFileSystem(**storage_options), storage_options
        except ImportError as exc:
            raise ImportError(
                "gs:// or gcs:// paths require gcsfs to be installed"
            ) from exc

    elif path.startswith(("adl://", "abfs://", "az://")):
        try:
            import adlfs

            storage_options = {"anon": False}
            return adlfs.AzureBlobFileSystem(**storage_options), storage_options
        except ImportError as exc:
            raise ImportError(
                "adl:// or abfs:// paths require adlfs to be installed"
            ) from exc

    elif path.startswith("oci://"):
        try:
            import ocifs

            storage_options = {}
            return ocifs.OCIFileSystem(**storage_options), storage_options
        except ImportError as exc:
            raise ImportError("oci:// paths require ocifs to be installed") from exc

    return None, {}


def get_dataset_type(dataset_config: DatasetConfig) -> str:
    """Get the dataset type from the path if it's not specified."""
    for extension, dataset_type in EXTENSIONS_TO_DATASET_TYPES.items():
        if extension in dataset_config.path:
            return dataset_type

    return "json"


def pre_process(
        dataset: DATASET_TYPE,
        ds_config: DatasetConfig,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        strict: bool = False,
):
    preprocessor = AutoPreprocessor()
    if ds_config.type == SurogateDatasetType.instruction:
        preprocessor = InstructionPreprocessor(ds_config)
    elif ds_config.type == SurogateDatasetType.conversation:
        preprocessor = ConversationPreprocessor(ds_config)
    elif ds_config.type == SurogateDatasetType.text:
        preprocessor = TextPreprocessor(ds_config)

    preprocessor.dataset_sample = ds_config.samples
    return preprocessor(dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict)


def post_process(
        train_dataset: DATASET_TYPE,
        *,
        dataset_sample: Optional[int] = None,
        split_dataset_ratio: float = 0.,
        streaming: bool = False,
        shuffle: bool = True,
        random_state: Optional[np.random.RandomState] = None,
) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
    """Split into train/val datasets and perform dataset sampling."""
    assert dataset_sample is None or dataset_sample > 0
    assert 0 <= split_dataset_ratio <= 1
    val_dataset = None
    if streaming:
        if dataset_sample is None:
            if split_dataset_ratio == 0:
                val_dataset = None
            elif split_dataset_ratio == 1:
                train_dataset, val_dataset = None, train_dataset
            else:
                raise ValueError('The IterableDataset does not support splitting the training set '
                                 'and validation set when dataset_sample is None.')
        else:
            # not shuffle
            train_dataset = train_dataset.take(dataset_sample)
            val_sample = int(dataset_sample * split_dataset_ratio)
            val_dataset = None if val_sample == 0 else train_dataset.take(val_sample)
            if val_sample:
                train_dataset = train_dataset.skip(val_sample)
    else:
        if dataset_sample is None:
            dataset_sample = len(train_dataset)
        if split_dataset_ratio == 0:
            train_dataset = sample_dataset(train_dataset, dataset_sample, shuffle, random_state)
            val_dataset = None
        elif split_dataset_ratio == 1:
            train_dataset, val_dataset = None, train_dataset
            val_sample = dataset_sample
            # Avoid duplication in the val_dataset.
            assert val_sample <= len(val_dataset), f'val_sample: {val_sample}, len(val_dataset): {len(val_dataset)}'
            val_dataset = sample_dataset(val_dataset, val_sample, shuffle, random_state)
        else:
            # Avoid duplication in the val_dataset.
            train_len = min(len(train_dataset), dataset_sample)
            val_sample = max(int(train_len * split_dataset_ratio), 1)
            train_sample = dataset_sample - val_sample
            assert train_sample > 0
            train_dataset, val_dataset = train_dataset.train_test_split(
                test_size=val_sample, shuffle=shuffle, seed=get_seed(random_state)).values()
            train_dataset = sample_dataset(train_dataset, train_sample, shuffle, random_state)
    return train_dataset, val_dataset


def concat_datasets(datasets: List[HfDataset]) -> Optional[HfDataset]:
    """Concatenate datasets, normalizing schemas to handle type mismatches.

    When datasets have columns with the same name but different types (e.g., 'id' as
    string vs int64), this function removes conflicting columns to allow concatenation.
    Only columns essential for training (messages, conversations, text, input, output,
    instruction, response) are preserved.
    """
    if len(datasets) == 0:
        return None
    if len(datasets) == 1:
        return datasets[0]

    # Essential columns that should be kept for training
    # These are the columns that preprocessors actually use
    essential_columns = {
        # Conversation format
        'messages', 'conversations', 'conversation',
        # Instruction format
        'instruction', 'input', 'output', 'response', 'system',
        # Text format
        'text', 'content',
        # Common useful fields
        'question', 'answer', 'query', 'chosen', 'rejected',
    }

    # Find columns present in all datasets
    all_columns = [set(ds.column_names) for ds in datasets]
    common_columns = set.intersection(*all_columns) if all_columns else set()

    # Check for type conflicts in common columns
    def get_feature_type(feature):
        """Get a comparable type representation for a feature."""
        if hasattr(feature, 'dtype'):
            return str(feature.dtype)
        return str(type(feature).__name__)

    conflicting_columns = set()
    for col in common_columns:
        types = set()
        for ds in datasets:
            if col in ds.features:
                types.add(get_feature_type(ds.features[col]))
        if len(types) > 1:
            conflicting_columns.add(col)

    if conflicting_columns:
        logger.warning(
            f"Found columns with type conflicts across datasets: {conflicting_columns}. "
            f"These columns will be removed before concatenation."
        )

    # Determine which columns to keep:
    # 1. Must be in all datasets (common)
    # 2. Must not have type conflicts, OR must be essential
    # 3. Essential columns with conflicts will be removed (can't reconcile types)
    columns_to_keep = common_columns - conflicting_columns

    # Also keep essential columns that are common and don't conflict
    essential_present = essential_columns & common_columns - conflicting_columns

    if not columns_to_keep:
        raise ValueError(
            "No common columns remain after removing type-conflicting columns. "
            "Datasets are incompatible for concatenation."
        )

    # Remove non-essential columns from each dataset to normalize schemas
    normalized_datasets = []
    for ds in datasets:
        cols_to_remove = [c for c in ds.column_names if c not in columns_to_keep]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)
        normalized_datasets.append(ds)

    logger.info(f"Concatenating {len(datasets)} datasets with columns: {sorted(columns_to_keep)}")
    return concatenate_datasets(normalized_datasets)


def shuffle_dataset(dataset, seed: int, buffer_size: int = 1000):
    if isinstance(dataset, HfDataset):
        return dataset.shuffle(seed=seed)
    else:
        return dataset.shuffle(seed=seed, buffer_size=buffer_size)
