import os
from pathlib import Path
from typing import Optional, Union, List, Literal

from transformers.utils import strtobool

from surogate.core.datasets.progress import create_hfhub_tqdm
from surogate.core.hub.base_hub import HubOperation
from surogate.utils.logger import get_logger

logger = get_logger()

from huggingface_hub.hf_api import api
class HuggingFaceHub(HubOperation):
    @classmethod
    def try_login(cls, token: Optional[str] = None) -> bool:
        pass

    @classmethod
    def create_model_repo(cls, repo_id: str, token: Optional[str] = None, private: bool = False) -> str:
        return api.create_repo(repo_id, token=token, private=private)

    @classmethod
    def push_to_hub(cls,
                    repo_id: str,
                    folder_path: Union[str, Path],
                    path_in_repo: Optional[str] = None,
                    commit_message: Optional[str] = None,
                    commit_description: Optional[str] = None,
                    token: Optional[Union[str, bool]] = None,
                    private: bool = False,
                    revision: Optional[str] = 'master',
                    ignore_patterns: Optional[Union[List[str], str]] = None,
                    **kwargs):
        cls.create_model_repo(repo_id, token, private)
        if revision is None or revision == 'master':
            revision = 'main'
        return api.upload_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            revision=revision,
            ignore_patterns=ignore_patterns,
            **kwargs)

    @classmethod
    def load_dataset(cls,
                     dataset_id: str,
                     subset_name: str,
                     split: str,
                     streaming: bool = False,
                     download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
                     num_proc: Optional[int] = None,
                     **kwargs):
        from datasets import load_dataset
        return load_dataset(
            dataset_id,
            name=subset_name,
            split=split,
            streaming=streaming,
            download_mode=download_mode,
            num_proc=num_proc)


    @classmethod
    def download_model(cls,
                       model_id_or_path: Optional[str] = None,
                       revision: Optional[str] = None,
                       ignore_patterns: Optional[List[str]] = None,
                       **kwargs):
        if revision is None or revision == 'master':
            revision = 'main'

        use_hf_transfer = strtobool(os.environ.get('USE_HF_TRANSFER', '1'))
        if use_hf_transfer:
            from huggingface_hub import _snapshot_download
            _snapshot_download.HF_HUB_ENABLE_HF_TRANSFER = True
        from huggingface_hub import snapshot_download
        return snapshot_download(
            model_id_or_path, repo_type='model', revision=revision, ignore_patterns=ignore_patterns,
            tqdm_class=create_hfhub_tqdm('Downloading model: '), **kwargs)