import os
from pathlib import Path

from transformers.utils import strtobool

from surogate.core.hub.base_hub import HubOperation
from surogate.utils.logger import get_logger

logger = get_logger()

from huggingface_hub.hf_api import api


class HuggingFaceHub(HubOperation):
    @classmethod
    def try_login(cls, key: str | None = None, secret: str | None = None) -> bool:
        pass

    @classmethod
    def create_model_repo(
        cls,
        repo_id: str,
        private: bool = False,
        key: str | None = None,
        secret: str | None = None,
    ):
        return api.create_repo(repo_id, token=key, private=private)

    @classmethod
    def push_to_hub(
        cls,
        repo_id: str,
        folder_path: str | Path,
        path_in_repo: str | None = None,
        commit_message: str | None = None,
        commit_description: str | None = None,
        private: bool = False,
        revision: str | None = "master",
        ignore_patterns: list[str] | str | None = None,
        key: str | None = None,
        secret: str | None = None,
        **kwargs,
    ):
        cls.create_model_repo(repo_id, key, private)
        if revision is None or revision == "master":
            revision = "main"
        return api.upload_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            commit_description=commit_description,
            token=key,
            revision=revision,
            ignore_patterns=ignore_patterns,
            **kwargs,
        )

    @classmethod
    def load_dataset(
        cls,
        dataset_id: str,
        subset_name: str,
        split: str,
        streaming: bool = False,
        revision: str | None = None,
        key: str | None = None,
        secret: str | None = None,
    ):
        from datasets import load_dataset

        return load_dataset(dataset_id, name=subset_name, split=split, streaming=streaming, token=key)

    @classmethod
    def download_model(
        cls,
        model_id_or_path: str | None = None,
        revision: str | None = None,
        ignore_patterns: list[str] | None = None,
        key: str | None = None,
        secret: str | None = None,
        **kwargs,
    ):
        if revision is None or revision == "master":
            revision = "main"

        use_hf_transfer = strtobool(os.environ.get("USE_HF_TRANSFER", "1"))
        if use_hf_transfer:
            from huggingface_hub import _snapshot_download

            _snapshot_download.HF_HUB_ENABLE_HF_TRANSFER = True
        from huggingface_hub import snapshot_download

        return snapshot_download(
            model_id_or_path, repo_type="model", revision=revision, ignore_patterns=ignore_patterns, token=key, **kwargs
        )
