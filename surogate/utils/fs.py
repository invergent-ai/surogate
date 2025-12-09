import fnmatch
import glob
import os
import shutil
from pathlib import Path
from typing import Union, List, Dict


def get_cache_dir():
    default_cache_dir = Path.home().joinpath('.cache', 'surogate')
    base_path = os.getenv('SUROGATE_CACHE_DIR', default_cache_dir)
    return base_path

def to_abspath(path: Union[str, List[str], None], check_path_exist: bool = False) -> Union[str, List[str], None]:
    """Check the path for validity and convert it to an absolute path.

    Args:
        path: The path to be checked/converted
        check_path_exist: Whether to check if the path exists

    Returns:
        Absolute path
    """
    if path is None:
        return
    elif isinstance(path, str):
        # Remove user path prefix and convert to absolute path.
        path = os.path.abspath(os.path.expanduser(path))
        if check_path_exist and not os.path.exists(path):
            raise FileNotFoundError(f"path: '{path}'")
        return path
    assert isinstance(path, list), f'path: {path}'
    res = []
    for v in path:
        res.append(to_abspath(v, check_path_exist))
    return res


def raise_nofile_limit():
    try:
        import resource  # POSIX only
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 8192 if soft < 8192 else soft
        if hard < target:
            target = hard  # cannot exceed hard limit
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        # Silently ignore on unsupported platforms
        pass

def copy_files_by_pattern(source_dir, dest_dir, patterns, exclude_patterns=None):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if isinstance(patterns, str):
        patterns = [patterns]

    if exclude_patterns is None:
        exclude_patterns = []
    elif isinstance(exclude_patterns, str):
        exclude_patterns = [exclude_patterns]

    def should_exclude_file(file_path, file_name):
        for exclude_pattern in exclude_patterns:
            if fnmatch.fnmatch(file_name, exclude_pattern):
                return True
            rel_file_path = os.path.relpath(file_path, source_dir)
            if fnmatch.fnmatch(rel_file_path, exclude_pattern):
                return True
        return False

    for pattern in patterns:
        pattern_parts = pattern.split(os.path.sep)
        if len(pattern_parts) > 1:
            subdir_pattern = os.path.sep.join(pattern_parts[:-1])
            file_pattern = pattern_parts[-1]

            for root, dirs, files in os.walk(source_dir):
                rel_path = os.path.relpath(root, source_dir)
                if rel_path == '.' or (rel_path != '.' and not fnmatch.fnmatch(rel_path, subdir_pattern)):
                    continue

                for file in files:
                    if fnmatch.fnmatch(file, file_pattern):
                        file_path = os.path.join(root, file)

                        if should_exclude_file(file_path, file):
                            continue

                        target_dir = os.path.join(dest_dir, rel_path)
                        if not os.path.exists(target_dir):
                            os.makedirs(target_dir)
                        dest_file = os.path.join(target_dir, file)

                        if not os.path.exists(dest_file):
                            shutil.copy2(file_path, dest_file)
        else:
            search_path = os.path.join(source_dir, pattern)
            matched_files = glob.glob(search_path)

            for file_path in matched_files:
                if os.path.isfile(file_path):
                    file_name = os.path.basename(file_path)

                    if should_exclude_file(file_path, file_name):
                        continue

                    destination = os.path.join(dest_dir, file_name)
                    if not os.path.exists(destination):
                        shutil.copy2(file_path, destination)

