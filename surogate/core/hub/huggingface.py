import os
from pathlib import Path
from typing import Optional, Union, List, Literal

from transformers.utils import strtobool

from surogate.core.hub.base_hub import HubOperation
from surogate.utils.logger import get_logger

logger = get_logger()

from huggingface_hub.hf_api import api

class HuggingFaceHub(HubOperation):
    @classmethod
    def try_login(cls, key: Optional[str] = None, secret: Optional[str] = None) -> bool:
        pass

    @classmethod
    def create_model_repo(
        cls, repo_id: str, 
        private: bool = False,
        key: Optional[str] = None, secret: Optional[str] = None, 
    ):
        return api.create_repo(repo_id, token=key, private=private)

    @classmethod
    def push_to_hub(
        cls,
        repo_id: str,
        folder_path: Union[str, Path],
        path_in_repo: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        private: bool = False,
        revision: Optional[str] = 'master',
        ignore_patterns: Optional[Union[List[str], str]] = None,
        key: Optional[str] = None, secret: Optional[str] = None,
        **kwargs
    ):
        cls.create_model_repo(repo_id, key, private)
        if revision is None or revision == 'master':
            revision = 'main'
        return api.upload_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            commit_description=commit_description,
            token=key,
            revision=revision,
            ignore_patterns=ignore_patterns,
            **kwargs)

    @classmethod
    def load_dataset(
        cls,
        dataset_id: str,
        subset_name: str,
        split: str,
        streaming: bool = False,
        revision: Optional[str] = None,
        key: Optional[str] = None, secret: Optional[str] = None,
    ):
        from datasets import load_dataset
        return load_dataset(
            dataset_id,
            name=subset_name,
            split=split,
            streaming=streaming,
            token=key)


    @classmethod
    def download_model(
        cls,
        model_id_or_path: Optional[str] = None,
        revision: Optional[str] = None,
        ignore_patterns: Optional[List[str]] = None,
        key: Optional[str] = None, secret: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs
    ):
        if revision is None or revision == 'master':
            revision = 'main'

        use_hf_transfer = strtobool(os.environ.get('USE_HF_TRANSFER', '1'))
        if use_hf_transfer:
            from huggingface_hub import _snapshot_download
            _snapshot_download.HF_HUB_ENABLE_HF_TRANSFER = True
        from huggingface_hub import snapshot_download
        return snapshot_download(
            model_id_or_path, repo_type='model', revision=revision,
            ignore_patterns=ignore_patterns, token=token or key, **kwargs)