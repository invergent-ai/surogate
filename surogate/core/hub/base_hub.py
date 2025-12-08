from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union, List


class HubOperation:

    @classmethod
    @contextmanager
    def patch_hub(cls):
        yield

    @classmethod
    def try_login(cls, token: Optional[str] = None) -> bool:
        """Try to login to the hub

        Args:
            token: The hub token to use

        Returns:
            bool: Whether login is successful
        """
        raise NotImplementedError

    @classmethod
    def create_model_repo(cls, repo_id: str, token: Optional[str] = None, private: bool = False):
        """Create a model repo on the hub

        Args:
            repo_id: The model id of the hub
            token: The hub token to use
            private: If is a private repo
        """
        raise NotImplementedError

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
        """Push a model-like folder to the hub

        Args:
            repo_id: The repo id
            folder_path: The local folder path
            path_in_repo: Which remote folder to put the local files in
            commit_message: The commit message of git
            commit_description: The commit description
            token: The hub token
            private: Private hub or not
            revision: The revision to push to
            ignore_patterns: The ignore file patterns
        """
        raise NotImplementedError

    @classmethod
    def load_dataset(cls,
                     dataset_id: str,
                     subset_name: str,
                     split: str,
                     streaming: bool = False,
                     revision: Optional[str] = None):
        """Load a dataset from the repo

        Args:
            dataset_id: The dataset id
            subset_name: The subset name of the dataset
            split: The split info
            streaming: Streaming mode
            revision: The revision of the dataset

        Returns:
            The Dataset instance
        """
        raise NotImplementedError

    @classmethod
    def download_model(cls,
                       model_id_or_path: Optional[str] = None,
                       revision: Optional[str] = None,
                       download_model: bool = True,
                       ignore_patterns: Optional[List[str]] = None,
                       **kwargs):
        """Download model from the hub

        Args:
            model_id_or_path: The model id
            revision: The model revision
            download_model: Whether downloading bin/safetensors files, this is usually useful when only
                using tokenizer
            ignore_patterns: Custom ignore pattern
            **kwargs:

        Returns:
            The local dir
        """
        raise NotImplementedError