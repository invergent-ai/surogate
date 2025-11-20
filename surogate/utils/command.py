import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

from surogate.utils.logger import get_logger
from swift.utils import seed_everything

logger = get_logger()

from surogate.utils.dict import DictDefault


class SurogateCommand(ABC):
    def __init__(self, config_cls: Any, **kwargs):
        self.args = DictDefault(kwargs)
        self.config = self.load_config(config_cls)

        if hasattr(self.config, 'seed') and self.config.seed:
            seed_everything(self.config.seed)

    def _expand_env_vars(self, obj: Any) -> Any:
        """
        Recursively expand environment variables in config.
        Supports ${VAR_NAME} syntax.
        """
        if isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
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

    def load_config(self, config_cls):
        if isinstance(self.args['config'], (str, Path)):
            with open(self.args['config'], encoding="utf-8") as file:
                cfg_dict = yaml.safe_load(file)

                # Expand environment variables
                cfg_dict = self._expand_env_vars(cfg_dict)

                cfg: DictDefault = DictDefault(cfg_dict)
            cfg.config_path = self.args['config']

        config = config_cls(cfg)

        cfg_to_log = {
            k: v for k, v in cfg.items() if v is not None
        }

        logger.debug(
            "config:\n%s",
            json.dumps(cfg_to_log, indent=2, default=str, sort_keys=True),
        )

        return config

    @abstractmethod
    def run(self):
        pass