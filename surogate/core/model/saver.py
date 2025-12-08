import os
import shutil
from typing import Optional, Union, List

from transformers import PreTrainedModel

from surogate.core.model.utils import Processor
from surogate.utils.fs import copy_files_by_pattern


def save_checkpoint(
    model: Optional[PreTrainedModel],
    processor: 'Processor',
    output_dir: str,
    *,
    safe_serialization: bool = True,
    max_shard_size: Union[int, str] = '5GB',
    model_dirs: List[str] = None,
    additional_saved_files: Optional[List[str]] = None
) -> None:
    if model is not None:
        if model.__class__.__name__ != 'SentenceTransformer':
            model.save_pretrained(output_dir, safe_serialization=safe_serialization, max_shard_size=max_shard_size)
        else:
            model.save_pretrained(output_dir, safe_serialization=safe_serialization)
            copy_files_by_pattern(model.model_dir, output_dir, '*.py')
            copy_files_by_pattern(model.model_dir, output_dir, '*.json')

    processor.save_pretrained(output_dir)

    if model_dirs is None:
        model_dirs = []
    else:
        model_dirs = model_dirs.copy()

    if model and model.model_dir and model.model_dir not in model_dirs:
        model_dirs.append(model.model_dir)

    for src_file in (additional_saved_files or []) + ['preprocessor_config.json', 'args.json']:
        tgt_path = os.path.join(output_dir, src_file)
        if os.path.exists(tgt_path) and src_file == 'args.json':
            continue
        for model_dir in model_dirs:
            src_path: str = os.path.join(model_dir, src_file)
            if os.path.isfile(src_path):
                shutil.copy(src_path, tgt_path)
                break
            elif os.path.isdir(src_path):
                shutil.copytree(src_path, tgt_path)
                break