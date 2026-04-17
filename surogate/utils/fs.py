import os
from pathlib import Path


def get_cache_dir():
    default_cache_dir = Path.home().joinpath(".cache", "surogate")
    base_path = os.getenv("SUROGATE_CACHE_DIR", default_cache_dir)
    return base_path


def to_abspath(path: str | list[str] | None, check_path_exist: bool = False) -> str | list[str] | None:
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
    assert isinstance(path, list), f"path: {path}"
    res = []
    for v in path:
        res.append(to_abspath(v, check_path_exist))
    return res
