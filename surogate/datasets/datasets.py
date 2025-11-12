from pathlib import Path

from adlfs import AzureBlobFileSystem
from gcsfs import GCSFileSystem
from ocifs import OCIFileSystem
from s3fs import S3FileSystem

from datasets import Dataset, IterableDataset, DatasetDict, IterableDatasetDict, load_from_disk, load_dataset, \
    concatenate_datasets
from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError, HFValidationError
from transformers import PreTrainedTokenizer

from surogate.datasets.lock import FileLockLoader
from surogate.datasets.wrapper import get_dataset_wrapper
from surogate.loaders.tokenizer import load_tokenizer
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger
from surogate.utils.schema.datasets import ConversationDataset, InstructionDataset, TextDataset, BaseDataset

logger = get_logger()

EXTENSIONS_TO_DATASET_TYPES = {
    ".parquet": "parquet",
    ".arrow": "arrow",
    ".csv": "csv",
    ".txt": "text",
}


def _load_and_prepare_datasets(cfg: DictDefault, tokenizer: PreTrainedTokenizer) -> Dataset | IterableDataset:
    datasets_configs = cfg.get('datasets')
    datasets = []
    for dataset_config in datasets_configs:
        dataset_wrapper = _load_and_process_single_dataset(
            dataset_config=dataset_config,
            cfg=cfg,
            tokenizer=tokenizer,
        )
        datasets.append(dataset_wrapper)

    dataset = merge_datasets(datasets, cfg)

    return dataset


def _load_and_process_single_dataset(
        dataset_config: ConversationDataset | InstructionDataset | TextDataset,
        cfg: DictDefault,
        tokenizer: PreTrainedTokenizer,
) -> Dataset | IterableDataset:
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
        dataset = dataset.select(range(dataset_config.samples))

    return get_dataset_wrapper(
        cfg=cfg,
        dataset_config=dataset_config,
        tokenizer=tokenizer,
        dataset=dataset,
    )


def load_dataset_with_config(
        dataset_config: BaseDataset,
        use_auth_token: bool,
        streaming=False
) -> Dataset | IterableDataset:
    load_dataset_kwargs = {
        "split": dataset_config.split if dataset_config.split else None,
        "name": dataset_config.subset,
        "streaming": streaming,
        "trust_remote_code": True,
    }

    if Path(dataset_config.path).exists():
        return _load_from_local_path(dataset_config, load_dataset_kwargs)

    is_hub_dataset = _check_if_hub_dataset(dataset_config, use_auth_token)
    if is_hub_dataset:
        return _load_from_hub(dataset_config, use_auth_token, load_dataset_kwargs)

    remote_fs, storage_options = _get_remote_filesystem(dataset_config.path)
    is_cloud_dataset = False
    if remote_fs:
        try:
            is_cloud_dataset = remote_fs.exists(dataset_config.path)
        except (FileNotFoundError, ConnectionError):
            pass

    if is_cloud_dataset:
        return _load_from_cloud(
            dataset_config, remote_fs, storage_options, load_dataset_kwargs
        )
    if dataset_config.path.startswith("https://"):
        return _load_from_url(dataset_config, load_dataset_kwargs)

    raise ValueError(
        f"The dataset could not be loaded. This could be due to a misconfigured dataset path "
        f"({dataset_config.path}). Try double-check your path / name. "
        f"This is not caused by the dataset type."
    )


def _check_if_hub_dataset(dataset_config: BaseDataset, use_auth_token: bool) -> bool:
    """Check if a dataset exists on the HuggingFace Hub."""
    try:
        snapshot_download(
            repo_id=dataset_config.path,
            repo_type="dataset",
            token=use_auth_token,
            ignore_patterns=["*"],
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
        dataset_config: BaseDataset, load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from a local path."""
    local_path = Path(dataset_config.path)

    if local_path.is_dir():
        try:
            return load_from_disk(dataset_config.path)
        except FileNotFoundError:
            return load_dataset(dataset_config.path, **load_dataset_kwargs)
    elif local_path.is_file():
        dataset_type = get_dataset_type(dataset_config)
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
        dataset_config: BaseDataset, use_auth_token: bool, load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from the HuggingFace Hub."""
    return load_dataset(
        dataset_config.path,
        token=use_auth_token,
        **load_dataset_kwargs,
    )


def _load_from_cloud(
        dataset_config: BaseDataset,
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
        dataset_config: BaseDataset, load_dataset_kwargs: dict
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


def get_dataset_type(dataset_config: BaseDataset) -> str:
    """Get the dataset type from the path if it's not specified."""
    if dataset_config.ds_type:
        return dataset_config.ds_type

    for extension, dataset_type in EXTENSIONS_TO_DATASET_TYPES.items():
        if extension in dataset_config.path:
            return dataset_type

    return "json"


def load_datasets(cfg: DictDefault) -> Dataset:
    tokenizer = load_tokenizer(cfg)

    def _load_datasets() -> Dataset | IterableDataset:
        return _load_and_prepare_datasets(cfg, tokenizer)

    # Prepare datasets (with file locking logic for multiple ranks)
    loader = FileLockLoader(cfg)
    try:
        dataset = loader.load(_load_datasets)
    finally:
        loader.cleanup()

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

    logger.info("Merging datasets...")
    merged_dataset = concatenate_datasets(datasets)
    merged_dataset = merged_dataset.shuffle(seed=cfg.seed)
    return merged_dataset
