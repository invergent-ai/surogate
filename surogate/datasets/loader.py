from pathlib import Path
from typing import Optional, Literal, Dict

from datasets import IterableDataset, Dataset, DatasetDict, IterableDatasetDict, load_from_disk, load_dataset
from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError, HFValidationError
from swift.llm import DatasetMeta, RowPreprocessor
from swift.llm.dataset.loader import DatasetSyntax
from datasets import Dataset as HfDataset
from surogate.config.dataset_config import DatasetConfig
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

from adlfs import AzureBlobFileSystem
from gcsfs import GCSFileSystem
from ocifs import OCIFileSystem
from s3fs import S3FileSystem

logger = get_logger()

EXTENSIONS_TO_DATASET_TYPES = {
    ".parquet": "parquet",
    ".arrow": "arrow",
    ".csv": "csv",
    ".txt": "text",
}


def swift_load_dataset(
        dataset_syntax: Optional[DatasetSyntax] = None,
        dataset_meta: Optional[DatasetMeta] = None,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        streaming: bool = False,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        strict: bool = False,
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
        columns: Optional[Dict[str, str]] = None,
        remove_unused_columns: bool = True,
        dataset_config: DatasetConfig,
        sg_args: DictDefault,
) -> HfDataset:
    dataset = load_dataset_with_config(dataset_config, hub_token, streaming)
    dataset = dataset_meta.preprocess_func(
        dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict)
    if remove_unused_columns:
        dataset = RowPreprocessor.remove_useless_columns(dataset)
    return dataset


def load_dataset_with_config(
        dataset_config: DatasetConfig,
        hub_token: Optional[str] = None,
        streaming=False
) -> Dataset | IterableDataset:
    load_dataset_kwargs = {
        "split": dataset_config.split if dataset_config.split else None,
        "name": dataset_config.subset,
        "streaming": streaming
    }

    logger.info("Loading dataset from path: %s", dataset_config.path)

    if Path(dataset_config.path).exists():
        return _load_from_local_path(dataset_config, load_dataset_kwargs)

    is_hub_dataset = _check_if_hub_dataset(dataset_config.path, hub_token)
    if is_hub_dataset:
        return _load_from_hub(dataset_config, hub_token, load_dataset_kwargs)

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


def _check_if_hub_dataset(path: str, hub_token: Optional[str]) -> bool:
    """Check if a dataset exists on the HuggingFace Hub."""
    try:
        snapshot_download(
            repo_id=path,
            repo_type="dataset",
            token=hub_token,
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
        dataset_config: DatasetConfig, load_dataset_kwargs: dict
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
        dataset_config: DatasetConfig, hub_token: Optional[str], load_dataset_kwargs: dict
) -> Dataset | IterableDataset | DatasetDict | IterableDatasetDict:
    """Load a dataset from the HuggingFace Hub."""
    return load_dataset(
        dataset_config.path,
        token=hub_token,
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
