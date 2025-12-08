import os
from typing import Optional, List

from surogate.core.hub.huggingface import HuggingFaceHub
from surogate.utils.dist import safe_ddp_context
from surogate.utils.logger import get_logger

logger = get_logger()

def safe_snapshot_download(
        model_id_or_path: str,
        revision: Optional[str] = None,
        download_model: bool = True,
        hub_token: Optional[str] = None,
        ignore_patterns: Optional[List[str]] = None,
        check_local: bool = False,
        **kwargs
) -> str:
    if check_local:
        model_suffix = model_id_or_path.rsplit('/', 1)[-1]

        if os.path.exists(model_suffix):
            model_dir = os.path.abspath(os.path.expanduser(model_suffix))
            logger.info(f'Loading the model from local path: {model_dir}')
            return model_dir

    if ignore_patterns is None:
        ignore_patterns = [
            '*.zip', '*.gguf', '*.pth', '*.pt', 'consolidated*', 'onnx/*', '*.safetensors.md', '*.msgpack', '*.onnx',
            '*.ot', '*.h5'
        ]

    if not download_model:
        ignore_patterns += ['*.bin', '*.safetensors']

    hub = HuggingFaceHub()

    if model_id_or_path.startswith('~'):
        model_id_or_path = os.path.abspath(os.path.expanduser(model_id_or_path))
    model_path_to_check = '/'.join(model_id_or_path.split(':', 1))
    if os.path.exists(model_id_or_path):
        model_dir = model_id_or_path
        sub_folder = None
    elif os.path.exists(model_path_to_check):
        model_dir = model_path_to_check
        sub_folder = None
    else:
        if model_id_or_path.startswith('/'):  # startswith
            raise ValueError(f"path: '{model_id_or_path}' not found")
        model_id_or_path = model_id_or_path.split(':', 1)  # get sub_folder
        if len(model_id_or_path) == 1:
            model_id_or_path = [model_id_or_path[0], None]
        model_id_or_path, sub_folder = model_id_or_path
        if sub_folder is not None:
            kwargs['allow_patterns'] = [f"{sub_folder.rstrip('/')}/*"]

        with safe_ddp_context(hash_id=model_id_or_path):
            model_dir = hub.download_model(model_id_or_path, revision, ignore_patterns, token=hub_token, **kwargs)

        logger.info(f'Loading model from local path: {model_dir}')

    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    if sub_folder:
        model_dir = os.path.join(model_dir, sub_folder)
    assert os.path.isdir(model_dir), f'model_dir: {model_dir}'
    return model_dir