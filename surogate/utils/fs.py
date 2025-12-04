import os
from typing import Union, List


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