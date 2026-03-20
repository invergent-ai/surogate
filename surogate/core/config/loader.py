from __future__ import annotations

import json
import os
import re
from typing import Any, Type, TypeVar
from urllib.request import urlopen
import yaml
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()

TConfig = TypeVar("TConfig")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(
    config_cls: Type[TConfig],
    path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> TConfig:
    cfg_dict: dict[str, Any] = {}

    # Check if path is an HTTP(S) URL
    if path:
        if path.startswith(('http://', 'https://')):
            logger.info(f"Fetching config from URL: {path}")
            with urlopen(path) as response:
                content = response.read().decode('utf-8')
                cfg_dict = yaml.safe_load(content)
        else:
            with open(path, encoding="utf-8") as file:
                cfg_dict = yaml.safe_load(file)

    if cfg_dict is None:
        cfg_dict = {}
    if not isinstance(cfg_dict, dict):
        raise ValueError(
            f"Invalid config payload for {config_cls.__name__}: expected mapping at top level"
        )

    if overrides:
        cfg_dict = _deep_merge(cfg_dict, dict(overrides))

    # Expand environment variables
    cfg_dict = _expand_env_vars(cfg_dict)
    cfg: DictDefault = DictDefault(cfg_dict)
    
    config = config_cls(cfg)
    cfg.config_path = path or "<cli-overrides>"

    cfg_to_log = {
        k: v for k, v in cfg.items() if v is not None
    }

    logger.debug(
        "config:\n%s",
        json.dumps(cfg_to_log, indent=2, default=str, sort_keys=True),
    )

    return config

def _expand_env_vars(obj: Any) -> Any:
    """
    Recursively expand environment variables in config.
    Supports ${VAR_NAME} syntax.
    """
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([^}]+)\}'

        def replace_env_var(match):
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                logger.warning(f"Environment variable not found: {var_name}")
                return match.group(0)  # Return original if not found
            logger.debug(f"Expanded ${{{var_name}}} (length: {len(value)})")
            return value

        return re.sub(pattern, replace_env_var, obj)
    else:
        return obj

