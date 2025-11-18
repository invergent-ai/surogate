import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

from surogate.utils.logger import get_logger

logger = get_logger()

from surogate.utils.dict import DictDefault


class SurogateCommand(ABC):
    def __init__(self, config_cls: Any, **kwargs):
        self.args = DictDefault(kwargs)
        self.config = self.load_config(config_cls)

    def load_config(self, config_cls):
        if isinstance(self.args['config'], (str, Path)):
            with open(self.args['config'], encoding="utf-8") as file:
                cfg: DictDefault = DictDefault(yaml.safe_load(file))
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
